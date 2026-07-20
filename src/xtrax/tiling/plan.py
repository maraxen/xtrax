"""BatchPlan and BatchPlanner for composable axis tiling strategy selection."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax

from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.roles import AmbiguousAxisError, AxisRole
from xtrax.tiling.strategy import (
    AxisStrategy,
    Bucket,
    SafeMap,
    Scan,
    ScanTransition,
    Vmap,
    WhileBodyFn,
    WhileCarry,
)

if TYPE_CHECKING:
    from xtrax.tiling.carry import CarrySpec
    from xtrax.tiling.dedup import DedupSpec


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a single axis to be tiled.

    Attributes:
        name: Human-readable axis name (e.g., "batch", "sequence").
        cardinality: Number of elements along this axis.
        default_batch_size: Default batch size threshold and chunk size for SafeMap.
        tile_granularity: Alignment granularity (default 1, no constraint).
        heterogeneous: Whether elements have different sizes (default False).
        dedup_eligible: Whether this axis is eligible for deduplication (default False).
        bucket_boundaries: Optional sorted, strictly-ascending bucket sizes. When
            provided, the planner selects the Bucket strategy (length-padding to the
            nearest boundary) instead of the cardinality-based rules. None disables
            bucketing (default).
    """

    name: str
    cardinality: int
    default_batch_size: int
    tile_granularity: int = 1
    heterogeneous: bool = False
    dedup_eligible: bool = False
    bucket_boundaries: tuple[int, ...] | list[int] | None = None
    role: AxisRole = AxisRole.KNOWN

    def __post_init__(self) -> None:
        """Normalize and validate bucket_boundaries when provided."""
        if self.bucket_boundaries is None:
            return
        boundaries = tuple(self.bucket_boundaries)
        # Coerce to tuple so the frozen dataclass stays hashable even if a list
        # was passed for ergonomics.
        object.__setattr__(self, "bucket_boundaries", boundaries)
        if len(boundaries) == 0:
            raise ValueError(f"AxisSpec(name={self.name!r}): bucket_boundaries must be non-empty.")
        if any(b <= 0 for b in boundaries):
            raise ValueError(
                f"AxisSpec(name={self.name!r}): bucket_boundaries must be positive, "
                f"got {boundaries}."
            )
        if list(boundaries) != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
            raise ValueError(
                f"AxisSpec(name={self.name!r}): bucket_boundaries must be strictly "
                f"ascending, got {boundaries}."
            )

    def __getattr__(self, name: str):
        """Provide deprecation shim for old field names."""
        if name == "batch_size":
            warnings.warn(
                "AxisSpec.batch_size is deprecated; use AxisSpec.default_batch_size",
                DeprecationWarning,
                stacklevel=2,
            )
            return object.__getattribute__(self, "default_batch_size")
        if name == "granularity":
            warnings.warn(
                "AxisSpec.granularity is deprecated; use AxisSpec.tile_granularity",
                DeprecationWarning,
                stacklevel=2,
            )
            return object.__getattribute__(self, "tile_granularity")
        raise AttributeError(f"type object 'AxisSpec' has no attribute {name!r}")


@dataclass(frozen=True)
class AxisDecision:
    """Decision for how to tile a single axis.

    Attributes:
        spec: The AxisSpec that was analyzed.
        batch_size: Final batch size used (from spec).
        reasoning: Human-readable explanation of the decision.
        strategy: Selected AxisStrategy (Bucket, Vmap, SafeMap, or DedupGather).
    """

    spec: AxisSpec
    batch_size: int
    reasoning: str
    strategy: AxisStrategy


@dataclass(frozen=True)
class BatchPlan:
    """Complete tiling plan for all axes.

    Attributes:
        decisions: Tuple of AxisDecision objects, one per input spec.
    """

    decisions: tuple[AxisDecision, ...]


class BatchPlanner:
    """Planner that selects tiling strategies based on axis properties.

    Selection rules (in priority order):
    1. bucket_boundaries is not None → Bucket (length-padding)
    2. dedup_eligible=True → DedupGather
    3. cardinality <= batch_size → Vmap
    4. cardinality > batch_size AND divisible → SafeMap
    5. non-divisible → SafeMap with warning (deferred-failure contract)

    When memory_estimator is provided, it overrides rule 3/4 decisions
    to prefer SafeMap if estimated Vmap memory exceeds device limit.

    When budget is provided (joint-budget mode), rules 3-5 are replaced for
    non-bucket axes: every eligible axis starts at Vmap, then axes with
    cardinality > default_batch_size are greedily demoted to SafeMap — in the
    order specs were given — until budget.estimate() over the whole plan fits
    budget.bytes. Callers express demotion priority by spec order (axes they
    are most willing to sequentialize first). Estimator exceptions propagate;
    an unfittable plan raises BudgetInfeasibleError.
    """

    def __init__(
        self,
        memory_estimator: Callable[[AxisSpec], int] | None = None,
        carry_specs: list[CarrySpec] | None = None,
        dedup_specs: list[DedupSpec] | None = None,
        heterogeneous_axes: set[str] | None = None,
        budget: MemoryBudget | None = None,
    ) -> None:
        """Initialize the planner.

        Args:
            memory_estimator: Optional function that estimates Vmap memory (bytes)
                for a given AxisSpec. If provided and estimate exceeds device limit,
                SafeMap is preferred over Vmap. If the estimator raises an exception,
                falls back to default rules silently. Mutually exclusive with budget.
            carry_specs: Optional list of CarrySpec objects declaring which axes
                should use Scan strategy (Phase 0 pre-demotion), or WhileCarry
                when CarrySpec.collect_outputs=False.
            dedup_specs: Optional list of DedupSpec objects declaring which axes
                should use DedupGather strategy (Phase 0b pre-demotion).
            heterogeneous_axes: Optional set of axis names (strings) that contain
                heterogeneous elements (variable shapes). These axes cannot use Scan
                strategy. Default None (no heterogeneous constraints).
            budget: Optional MemoryBudget enabling joint-budget planning (greedy
                demotion until the whole-plan estimate fits). Mutually exclusive
                with memory_estimator: budget mode is strict (no silent fallback,
                no implicit device-limit read) by design.

        Raises:
            ValueError: If both budget and memory_estimator are provided.
        """
        if budget is not None and memory_estimator is not None:
            raise ValueError(
                "BatchPlanner: budget and memory_estimator are mutually exclusive; "
                "budget mode replaces the per-axis memory override."
            )
        self.memory_estimator = memory_estimator
        self.carry_specs = carry_specs or []
        self.dedup_specs = dedup_specs or []
        self.heterogeneous_axes = heterogeneous_axes or set()
        self.budget = budget

    def plan(self, specs: Sequence[AxisSpec]) -> BatchPlan:
        """Generate a tiling plan for the given specs.

        Phase 0: Pre-demote axes with declared CarrySpec to Scan (or, when
        CarrySpec.collect_outputs=False, to WhileCarry instead).
        Phase 0b: Pre-demote axes with declared DedupSpec to DedupGather.
        Phases 1+: Apply standard strategy selection rules to remaining axes —
        or, when a MemoryBudget is set, greedy joint-budget demotion (see
        _plan_joint_budget) for all non-bucket remaining axes.

        Spec order is preserved in decisions (Phase 0/0b then remaining rules,
        all in the order specs were provided).

        Args:
            specs: Sequence of AxisSpec objects to plan.

        Returns:
            BatchPlan with decisions for each spec.

        Raises:
            ValueError: If a CarrySpec targets a heterogeneous axis.
            AmbiguousAxisError: If an axis has an unresolved UNKNOWN role.
            BudgetInfeasibleError: In budget mode, if demoting every candidate
                still leaves the joint estimate over budget. Budget-mode
                estimator exceptions also propagate unchanged.
        """
        carry_by_name = {cs.axis_name: cs for cs in self.carry_specs}
        dedup_by_name = {ds.axis_name: ds for ds in self.dedup_specs}
        decisions: list[AxisDecision | None] = []
        pending: list[int] = []

        # Process specs in order, applying Phase 0/0b rules first, then standard rules
        for spec in specs:
            # Phase 0: CarrySpec pre-demotion
            if spec.name in carry_by_name:
                cs = carry_by_name[spec.name]
                # Validate: Scan/WhileCarry are invalid on heterogeneous axes
                if spec.name in self.heterogeneous_axes:
                    raise ValueError(
                        f"Cannot create a carry strategy for axis '{spec.name}': "
                        f"axis is heterogeneous (shapes vary per element), "
                        f"but Scan/WhileCarry require static carry shape. "
                        f"Remove this axis from CarrySpec or remove it from heterogeneous_axes."
                    )
                # cs.transition's static type is `ScanTransition | WhileBodyFn` -- which
                # concrete shape it actually is depends on cs.collect_outputs, a runtime
                # bool a type checker can't narrow the union on. The two Protocols share
                # a `__call__` name but incompatible arity, so cast rather than isinstance.
                if cs.collect_outputs:
                    carry_strategy: AxisStrategy = Scan(
                        init=cs.init,
                        transition=cast("ScanTransition", cs.transition),
                        ordered_sinks=cs.ordered_sinks,
                    )
                    reasoning = f"carry-bearing scan (CarrySpec declared for '{spec.name}')"
                else:
                    carry_strategy = WhileCarry(
                        init=cs.init,
                        body=cast("WhileBodyFn", cs.transition),
                        cond=cs.cond,
                    )
                    reasoning = (
                        f"carry-only while-loop (CarrySpec declared for '{spec.name}', "
                        "collect_outputs=False)"
                    )
                decisions.append(
                    AxisDecision(
                        spec=spec,
                        batch_size=1,
                        reasoning=reasoning,
                        strategy=carry_strategy,
                    ),
                )
                continue

            # Phase 0b: DedupSpec pre-demotion
            if spec.name in dedup_by_name:
                ds = dedup_by_name[spec.name]
                dg_strategy = ds.to_dedup_gather()
                decisions.append(
                    AxisDecision(
                        spec=spec,
                        batch_size=ds.k,
                        reasoning=(
                            f"dedup-gather (DedupSpec for '{spec.name}', "
                            f"k={ds.k}, k_bucket={dg_strategy.k_bucket})"
                        ),
                        strategy=dg_strategy,
                    ),
                )
                continue

            # E1.3b KEYSTONE guard (AC3): fail loud on unresolved UNKNOWN-role axes.
            # Phase-0/0b axes have already `continue`d above, so they never reach here.
            # Every existing hand-written AxisSpec defaults to KNOWN and passes through.
            if spec.role == AxisRole.UNKNOWN:
                raise AmbiguousAxisError(
                    f"axis '{spec.name}' has an unresolved role; declare it with "
                    f"@axis_config or provide an override before planning."
                )

            # Joint-budget mode: bucket axes are fixed via Rule 1 as usual;
            # everything else is deferred to the greedy Phase 2 below.
            if self.budget is not None and spec.bucket_boundaries is None:
                pending.append(len(decisions))
                decisions.append(None)
                continue

            # Standard rules for remaining axes
            decision = self._decide_strategy(spec)
            decisions.append(decision)

        if self.budget is not None:
            self._plan_joint_budget(specs, decisions, pending)

        return BatchPlan(decisions=tuple(d for d in decisions if d is not None))

    def _plan_joint_budget(
        self,
        specs: Sequence[AxisSpec],
        decisions: list[AxisDecision | None],
        pending: list[int],
    ) -> None:
        """Resolve pending axes under the joint MemoryBudget (greedy demotion).

        Fills decisions[idx] in place for every idx in pending. Every pending
        axis starts at Vmap; axes with cardinality > default_batch_size are
        demoted to SafeMap one at a time — in the order given — until
        budget.estimate() over the full plan (fixed decisions included) fits
        budget.bytes.

        Args:
            specs: Full spec sequence (indices align with decisions).
            decisions: Decision list with None placeholders at pending indices.
            pending: Indices of axes awaiting joint-budget resolution.

        Raises:
            BudgetInfeasibleError: If all candidates are demoted and the
                estimate still exceeds the budget.
        """
        budget = self.budget
        if budget is None:  # pragma: no cover - plan() only calls with budget set
            raise RuntimeError("_plan_joint_budget requires a MemoryBudget")

        for idx in pending:
            spec = specs[idx]
            decisions[idx] = AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning="joint-budget: Vmap (pending final estimate)",
                strategy=Vmap(),
            )

        def _snapshot() -> tuple[AxisDecision, ...]:
            return tuple(d for d in decisions if d is not None)

        candidates = [
            idx for idx in pending if specs[idx].cardinality > specs[idx].default_batch_size
        ]
        estimate = budget.estimate(_snapshot())
        step = 0
        for idx in candidates:
            if estimate <= budget.bytes:
                break
            spec = specs[idx]
            if spec.cardinality % spec.default_batch_size != 0:
                warnings.warn(
                    f"AxisSpec(name={spec.name!r}): cardinality={spec.cardinality} "
                    f"is not divisible by batch_size={spec.default_batch_size}. "
                    f"This plan will raise ValueError at make_axis_dispatch time.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            step += 1
            before = estimate
            decisions[idx] = AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning="joint-budget: demoted (pending final estimate)",
                strategy=SafeMap(batch_size=spec.default_batch_size),
            )
            estimate = budget.estimate(_snapshot())
            decisions[idx] = AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning=(
                    f"joint-budget: demoted to SafeMap(batch_size="
                    f"{spec.default_batch_size}) at step {step} "
                    f"(estimate {before} -> {estimate} B, budget {budget.bytes} B)"
                ),
                strategy=SafeMap(batch_size=spec.default_batch_size),
            )

        if estimate > budget.bytes:
            state_desc = ", ".join(
                f"{d.spec.name}={type(d.strategy).__name__}" for d in _snapshot()
            )
            raise BudgetInfeasibleError(
                f"plan cannot fit MemoryBudget: estimate {estimate} B > budget "
                f"{budget.bytes} B after demoting all {len(candidates)} candidate "
                f"axes; final strategies: {state_desc}"
            )

        # Finalize reasoning for pending axes that kept Vmap.
        for idx in pending:
            decision = decisions[idx]
            if decision is None or not isinstance(decision.strategy, Vmap):
                continue
            spec = specs[idx]
            if spec.cardinality <= spec.default_batch_size:
                reasoning = (
                    f"joint-budget: Vmap (cardinality {spec.cardinality} <= "
                    f"batch_size {spec.default_batch_size}; demotion would be a "
                    f"no-op; final estimate {estimate} B <= budget {budget.bytes} B)"
                )
            else:
                reasoning = (
                    f"joint-budget: Vmap retained "
                    f"(final estimate {estimate} B <= budget {budget.bytes} B)"
                )
            decisions[idx] = AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning=reasoning,
                strategy=decision.strategy,
            )

    def _decide_strategy(self, spec: AxisSpec) -> AxisDecision:
        """Decide strategy for a single AxisSpec following selection rules."""

        # Rule 1: explicit bucket_boundaries → Bucket (host-side length-padding).
        # This is the strongest, most explicit signal and wins over dedup/cardinality
        # rules: the caller has declared the variable-length axis and its buckets.
        # Bucket is a host plan descriptor — padding happens before the JIT boundary
        # via select_bucket()/bucketize(), not in make_axis_dispatch.
        if spec.bucket_boundaries is not None:
            boundaries = tuple(spec.bucket_boundaries)
            strategy = Bucket(boundaries=boundaries)
            return AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning=(f"bucket_boundaries={boundaries} → Bucket (host-side padding)"),
                strategy=strategy,
            )

        # Rule 2: dedup_eligible → skip (handled via Phase 0b DedupSpec)
        # Note: DedupGather now requires explicit unique_indices, index_map, k via DedupSpec.
        # Rule-based dedup_eligible without explicit DedupSpec falls through to standard rules.
        if spec.dedup_eligible:
            # No special handling: fall through to cardinality-based rules
            pass

        # Check memory estimate before deciding between Vmap and SafeMap
        should_prefer_safemap_for_memory = False
        if self.memory_estimator is not None:
            try:
                estimated_bytes = self.memory_estimator(spec)
                # Get device memory limit (default 4 GiB)
                try:
                    device_limit = jax.devices()[0].memory_stats().get("bytes_limit", 4 * (2**30))
                except Exception:
                    device_limit = 4 * (2**30)

                if estimated_bytes > device_limit:
                    should_prefer_safemap_for_memory = True
            except Exception:
                # Fall back silently to default rules
                pass

        # Rule 3: cardinality <= batch_size → Vmap (unless memory override)
        if spec.cardinality <= spec.default_batch_size:
            if should_prefer_safemap_for_memory:
                # Memory estimator overrides: use SafeMap
                strategy = SafeMap(batch_size=spec.default_batch_size)
                reasoning = "cardinality <= batch_size but memory_estimator override → SafeMap"
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            else:
                strategy = Vmap()
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning="cardinality <= batch_size → Vmap",
                    strategy=strategy,
                )

        # Rule 4 & 5: cardinality > batch_size
        # Check divisibility
        if spec.cardinality % spec.default_batch_size == 0:
            # Rule 4: divisible → SafeMap (unless memory estimator allows Vmap)
            if should_prefer_safemap_for_memory:
                # Memory estimate exceeds limit: use SafeMap
                strategy = SafeMap(batch_size=spec.default_batch_size)
                reasoning = (
                    "cardinality > batch_size and divisible but memory_estimator override → SafeMap"
                )
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            elif self.memory_estimator is not None:
                # Memory estimator is provided and under limit: prefer Vmap
                strategy = Vmap()
                reasoning = "cardinality > batch_size and divisible but memory safe → Vmap"
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            else:
                # No memory estimator: use default SafeMap
                strategy = SafeMap(batch_size=spec.default_batch_size)
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning="cardinality > batch_size and divisible → SafeMap",
                    strategy=strategy,
                )
        else:
            # Rule 5: non-divisible → SafeMap + warning (deferred-failure contract)
            warnings.warn(
                f"AxisSpec(name={spec.name!r}): cardinality={spec.cardinality} "
                f"is not divisible by batch_size={spec.default_batch_size}. "
                f"This plan will raise ValueError at make_axis_dispatch time.",
                RuntimeWarning,
                stacklevel=3,
            )
            strategy = SafeMap(batch_size=spec.default_batch_size)
            reasoning = "cardinality > batch_size but not divisible → SafeMap (deferred failure)"
            return AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning=reasoning,
                strategy=strategy,
            )

"""BatchPlan and BatchPlanner for composable axis tiling strategy selection."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.estimators import lowered_memory_estimate
from xtrax.tiling.roles import AmbiguousAxisError, AxisRole
from xtrax.tiling.strategy import (
    AxisStrategy,
    Bucket,
    ChunkedMap,
    Scan,
    ScanTransition,
    Vmap,
    WhileBodyFn,
    WhileCarry,
)

if TYPE_CHECKING:
    from xtrax.tiling.carry import CarrySpec
    from xtrax.tiling.dedup import DedupSpec

logger = logging.getLogger(__name__)

_default_device_limit_logged = False


def _memory_estimator_device_limit() -> int:
    """Byte limit the per-axis ``memory_estimator`` override compares against.

    A device that reports ``memory_stats()["bytes_limit"]`` contributes that
    full limit (``device_memory_budget(fraction=1.0)``), which is the comparison
    this path has always made. When the runtime cannot answer, log once and
    return the documented 4 GiB default. ``device_memory_budget`` itself still
    raises; the fallback lives here so planning on CPU keeps a defined limit.
    """
    global _default_device_limit_logged
    from xtrax.tiling.estimators import DEFAULT_DEVICE_MEMORY_BYTES, device_memory_budget

    try:
        return device_memory_budget(fraction=1.0)
    except Exception:
        if not _default_device_limit_logged:
            _default_device_limit_logged = True
            logger.info(
                "BatchPlanner memory_estimator: device did not report "
                "memory_stats()['bytes_limit']; using the documented default "
                "of %s bytes (4 GiB).",
                DEFAULT_DEVICE_MEMORY_BYTES,
            )
        return DEFAULT_DEVICE_MEMORY_BYTES


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a single axis to be tiled.

    Attributes:
        name: Human-readable axis name (e.g., "batch", "sequence").
        cardinality: Number of elements along this axis.
        default_batch_size: Default batch size threshold and chunk size for ChunkedMap.
        tile_granularity: Alignment granularity (default 1, no constraint).
        heterogeneous: Whether elements have different sizes (default False).
        dedup_eligible: Whether this axis is eligible for deduplication (default False).
        bucket_boundaries: Optional sorted, strictly-ascending bucket sizes. When
            provided, the planner selects the Bucket strategy (length-padding to the
            nearest boundary) instead of the cardinality-based rules. None disables
            bucketing (default).
        element_input_bytes: Per-element input size in bytes, when the caller
            knows it. A per-axis ``memory_estimator`` result below this value
            raises ``ValueError``. None means the spec has no element shape or
            dtype, so the planner skips that check.
        varying_inputs: Named inputs that vary along this axis. Every other named
            input is invariant along it. Empty means all named inputs are invariant.
            ``declare_varying_inputs`` checks the names against the known axes and
            inputs. The field lives here because a plan stores this spec on each
            decision, and plan/explain already print spec fields.
    """

    name: str
    cardinality: int
    default_batch_size: int
    tile_granularity: int = 1
    heterogeneous: bool = False
    dedup_eligible: bool = False
    bucket_boundaries: tuple[int, ...] | list[int] | None = None
    role: AxisRole = AxisRole.KNOWN
    element_input_bytes: int | None = None
    varying_inputs: tuple[str, ...] | list[str] | str = ()

    def __post_init__(self) -> None:
        """Validate element_input_bytes and varying_inputs; normalize bucket_boundaries."""
        if self.element_input_bytes is not None:
            value = self.element_input_bytes
            if isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"AxisSpec(name={self.name!r}): element_input_bytes must be a "
                    f"non-negative int, got {value!r}."
                )
        varying = self.varying_inputs
        if isinstance(varying, str):
            raise ValueError(
                f"AxisSpec(name={self.name!r}): varying_inputs must be a sequence of "
                f"input names, not the string {varying!r}."
            )
        object.__setattr__(self, "varying_inputs", tuple(varying))
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


def declare_varying_inputs(
    specs: Sequence[AxisSpec],
    varying: Mapping[str, Sequence[str]],
    input_names: Sequence[str],
) -> tuple[AxisSpec, ...]:
    """Return specs annotated with the inputs that vary along each axis.

    Inputs absent from an axis's list are invariant along that axis. Axes
    absent from ``varying`` keep the ``varying_inputs`` already stored on the
    spec. Duplicate names in one list are dropped, first occurrence kept.

    Args:
        specs: Axis specs the declaration may name.
        varying: Map from axis name to the input names that vary along it.
        input_names: Input names the declaration may use.

    Returns:
        Specs in input order, with ``varying_inputs`` set from ``varying``.

    Raises:
        ValueError: ``varying`` names an axis absent from ``specs``, or an
            input name absent from ``input_names``. The message quotes the
            unknown name and lists the known names.
    """
    known_axes = {spec.name for spec in specs}
    unknown_axes = [axis for axis in varying if axis not in known_axes]
    if unknown_axes:
        known = ", ".join(repr(spec.name) for spec in specs) or "(none)"
        raise ValueError(f"unknown axis {unknown_axes[0]!r}; known axes: {known}")
    known_inputs = set(input_names)
    for names in varying.values():
        for name in names:
            if name not in known_inputs:
                known = ", ".join(repr(n) for n in input_names) or "(none)"
                raise ValueError(f"unknown input {name!r}; known inputs: {known}")
    updated: list[AxisSpec] = []
    for spec in specs:
        if spec.name not in varying:
            updated.append(spec)
            continue
        deduped = tuple(dict.fromkeys(varying[spec.name]))
        updated.append(replace(spec, varying_inputs=deduped))
    return tuple(updated)


@dataclass(frozen=True)
class AxisDecision:
    """Decision for how to tile a single axis.

    Attributes:
        spec: The AxisSpec that was analyzed.
        batch_size: Final batch size used (from spec).
        reasoning: Human-readable explanation of the decision.
        strategy: Selected AxisStrategy (Bucket, Vmap, ChunkedMap, or DedupGather).
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

    def decision_for(self, axis_name: str) -> AxisDecision:
        """Return the decision for ``axis_name``.

        Args:
            axis_name: Axis name to look up (``AxisSpec.name``).

        Returns:
            The ``AxisDecision`` for that axis, including its strategy.

        Raises:
            KeyError: If no decision in this plan has ``spec.name == axis_name``.
                The message names the missing axis and the axes that are present.
        """
        for decision in self.decisions:
            if decision.spec.name == axis_name:
                return decision
        known = ", ".join(repr(decision.spec.name) for decision in self.decisions)
        known_desc = known if known else "(none)"
        raise KeyError(
            f"BatchPlan has no decision for axis {axis_name!r}; known axes: {known_desc}"
        )


class BatchPlanner:
    """Planner that selects tiling strategies based on axis properties.

    Selection rules (in priority order):
    1. bucket_boundaries is not None → Bucket (length-padding)
    2. dedup_eligible=True → DedupGather
    3. cardinality <= batch_size → Vmap
    4. cardinality > batch_size → ChunkedMap. Divisibility does not matter: a
       ragged final chunk is handled by the chunked map itself (#5565).

    When memory_estimator is provided, it overrides rule 3/4 decisions
    to prefer ChunkedMap if estimated Vmap memory exceeds the device
    allocator limit. The limit is ``device_memory_budget(fraction=1.0)``
    (the full ``bytes_limit``). If the device does not report one, the
    comparison uses the documented 4 GiB default and logs that once.
    An estimator that raises fails ``plan()`` with ``RuntimeError`` naming
    the axis; the original exception is chained. A missing device limit is
    not an estimator failure.

    When ``AxisSpec.element_input_bytes`` is set, that value is the
    per-element input-byte lower bound: an estimate below it raises
    ``ValueError`` naming the axis, the estimate, and the bound. When the
    field is None, the spec has no element shape or dtype, so no lower
    bound is derivable and the check is skipped.

    When memory_estimator is None, rules 3 and 4 stay as written. When rule 3
    then selects Vmap with no memory estimate, the planner logs a warning once
    per planner and axis.

    When budget is provided (joint-budget mode), rules 3-4 are replaced for
    non-bucket axes: every eligible homogeneous axis starts at Vmap, then axes
    with cardinality > default_batch_size are greedily demoted to ChunkedMap —
    in the order specs were given — until budget.estimate() over the whole plan
    fits budget.bytes. A heterogeneous axis is not eligible for Vmap: element
    shapes vary, so it is fixed to ChunkedMap(batch_size=default_batch_size)
    for the whole joint plan and is never a demotion candidate. Callers express
    demotion priority by spec order (axes they are most willing to sequentialize
    first). Estimator exceptions propagate; an unfittable plan raises
    BudgetInfeasibleError.
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
                for a given AxisSpec. If provided and the estimate exceeds the device
                limit, ChunkedMap is preferred over Vmap. The limit comes from
                ``device_memory_budget(fraction=1.0)``; when the device reports no
                ``bytes_limit``, a documented 4 GiB default is logged once and used.
                If the estimator raises, ``plan()`` raises ``RuntimeError`` naming
                the axis (the original exception is chained). When this argument
                is None, cardinality rules are unchanged; a Vmap chosen without an
                estimate logs a warning once per planner and axis. An estimate below
                ``AxisSpec.element_input_bytes``, when that field is set, raises
                ``ValueError``. Mutually exclusive with budget.
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
        self._missing_estimator_warned_axes: set[str] = set()

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
            ValueError: If a CarrySpec targets a heterogeneous axis, or if a
                per-axis memory estimate is below ``AxisSpec.element_input_bytes``.
            RuntimeError: If ``memory_estimator`` raises. The message names the
                axis and the original exception is chained.
            AmbiguousAxisError: If an axis has an unresolved UNKNOWN role.
            BudgetInfeasibleError: In budget mode, if demoting every candidate
                -- and, as a last resort, planning dedup axes without dedup --
                still leaves the joint estimate over budget. Budget-mode
                estimator exceptions also propagate unchanged.
            DedupSpecCollisionError: If two DedupSpecs name the same axis (#5175).
        """
        from xtrax.tiling.dedup_synthesis import merge_dedup_specs

        carry_by_name = {cs.axis_name: cs for cs in self.carry_specs}
        # #5175: route through merge_dedup_specs so two DedupSpecs for one axis raise
        # DedupSpecCollisionError here too, instead of silently keeping the last one.
        dedup_by_name = merge_dedup_specs(*({ds.axis_name: ds} for ds in self.dedup_specs))
        decisions: list[AxisDecision | None] = []
        pending: list[int] = []
        dedup_indices: list[int] = []

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
                dedup_indices.append(len(decisions))
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
            self._plan_joint_budget(specs, decisions, pending, dedup_indices)

        return BatchPlan(decisions=tuple(d for d in decisions if d is not None))

    def _plan_joint_budget(
        self,
        specs: Sequence[AxisSpec],
        decisions: list[AxisDecision | None],
        pending: list[int],
        dedup_indices: Sequence[int] = (),
    ) -> None:
        """Resolve pending axes under the joint MemoryBudget (greedy demotion).

        Fills decisions[idx] in place for every idx in pending. Every pending
        homogeneous axis starts at Vmap; heterogeneous axes are fixed to
        ChunkedMap (Vmap is invalid when element shapes vary). Homogeneous
        axes with cardinality > default_batch_size are demoted to ChunkedMap
        one at a time — in the order given — until budget.estimate() over the
        full plan (fixed decisions included) fits budget.bytes.

        DedupGather axes (Phase 0b) are fixed decisions, so on their own they
        could turn a plan that fits the budget without a DedupSpec into an
        infeasible one (#5175). If every ordinary demotion still leaves the plan
        over budget, dedup axes are handed back to ordinary budget planning one
        at a time, in spec order, until it fits -- a last resort, each with a
        RuntimeWarning and the fallback recorded in its decision's reasoning --
        before BudgetInfeasibleError is raised. Adding a
        DedupSpec therefore never makes a feasible plan infeasible. Note the
        fallback changes the axis's numerics at the ~1e-6 level: gather-per-
        canonical and per-row outputs are equal only up to float tolerance
        (see verify_dedup_outputs).

        Args:
            specs: Full spec sequence (indices align with decisions).
            decisions: Decision list with None placeholders at pending indices.
            pending: Indices of axes awaiting joint-budget resolution.
            dedup_indices: Indices of Phase-0b DedupGather decisions.

        Raises:
            BudgetInfeasibleError: If all candidates are demoted -- dedup axes
                included -- and the estimate still exceeds the budget.
        """
        budget = self.budget
        if budget is None:  # pragma: no cover - plan() only calls with budget set
            raise RuntimeError("_plan_joint_budget requires a MemoryBudget")

        estimate, n_candidates = self._greedy_demote(specs, decisions, pending, budget)
        dropped: list[int] = []
        # Release dedup axes ONE AT A TIME, in spec order, stopping at the first fit: the
        # minimal change, so a DedupSpec another axis's release already made room for keeps
        # its dedup (and its numerics).
        for idx in dedup_indices:
            if estimate <= budget.bytes:
                break
            spec = specs[idx]
            if spec.role == AxisRole.UNKNOWN:
                # The Phase-0b `continue` skipped this guard; ordinary planning needs it.
                raise AmbiguousAxisError(
                    f"axis '{spec.name}' has an unresolved role; declare it with "
                    f"@axis_config or provide an override before planning."
                )
            warnings.warn(
                f"MemoryBudget of {budget.bytes} B cannot be met with DedupGather on axis "
                f"{spec.name!r}; planning it without dedup instead (outputs then match the "
                "dedup path only up to float tolerance).",
                RuntimeWarning,
                stacklevel=3,
            )
            dropped.append(idx)
            if spec.bucket_boundaries is None:
                decisions[idx] = None
                pending = sorted({*pending, idx})
            else:
                decisions[idx] = self._decide_strategy(spec)
            estimate, n_candidates = self._greedy_demote(specs, decisions, pending, budget)

        def _snapshot() -> tuple[AxisDecision, ...]:
            return tuple(d for d in decisions if d is not None)

        if estimate > budget.bytes:
            state_desc = ", ".join(
                f"{d.spec.name}={type(d.strategy).__name__}" for d in _snapshot()
            )
            also = " (DedupGather already dropped)" if dropped else ""
            raise BudgetInfeasibleError(
                f"plan cannot fit MemoryBudget: estimate {estimate} B > budget "
                f"{budget.bytes} B after demoting all {n_candidates} candidate "
                f"axes{also}; final strategies: {state_desc}"
            )

        self._finalize_budget_reasoning(specs, decisions, pending, estimate, budget)
        for idx in dropped:
            decision = decisions[idx]
            if decision is None:  # pragma: no cover - every dropped axis was re-decided
                continue
            decisions[idx] = AxisDecision(
                spec=decision.spec,
                batch_size=decision.batch_size,
                reasoning=(
                    f"{decision.reasoning}; DedupSpec dropped: MemoryBudget "
                    f"{budget.bytes} B infeasible with DedupGather (#5175)"
                ),
                strategy=decision.strategy,
            )

    def _greedy_demote(
        self,
        specs: Sequence[AxisSpec],
        decisions: list[AxisDecision | None],
        pending: list[int],
        budget: MemoryBudget,
    ) -> tuple[int, int]:
        """Assign initial strategies, then demote candidates until the joint estimate fits.

        Homogeneous pending axes start at Vmap. Heterogeneous pending axes start
        at ChunkedMap and are excluded from demotion: element shapes vary, so
        Vmap is invalid for the whole joint plan, including when the budget
        would otherwise keep the axis mapped. Returns (final estimate, number
        of demotion candidates).
        """
        for idx in pending:
            decisions[idx] = self._budget_initial_decision(specs[idx])

        def _snapshot() -> tuple[AxisDecision, ...]:
            return tuple(d for d in decisions if d is not None)

        candidates = [
            idx
            for idx in pending
            if not specs[idx].heterogeneous
            and specs[idx].cardinality > specs[idx].default_batch_size
        ]
        estimate = budget.estimate(_snapshot())
        step = 0
        for idx in candidates:
            if estimate <= budget.bytes:
                break
            spec = specs[idx]
            step += 1
            before = estimate
            decisions[idx] = AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning="joint-budget: demoted (pending final estimate)",
                strategy=ChunkedMap(batch_size=spec.default_batch_size),
            )
            estimate = budget.estimate(_snapshot())
            decisions[idx] = AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning=(
                    f"joint-budget: demoted to ChunkedMap(batch_size="
                    f"{spec.default_batch_size}) at step {step} "
                    f"(estimate {before} -> {estimate} B, budget {budget.bytes} B)"
                ),
                strategy=ChunkedMap(batch_size=spec.default_batch_size),
            )

        return estimate, len(candidates)

    def _finalize_budget_reasoning(
        self,
        specs: Sequence[AxisSpec],
        decisions: list[AxisDecision | None],
        pending: list[int],
        estimate: int,
        budget: MemoryBudget,
    ) -> None:
        """Finalize reasoning for pending axes that kept their initial strategy."""
        for idx in pending:
            decision = decisions[idx]
            if decision is None:
                continue
            spec = specs[idx]
            if spec.heterogeneous and isinstance(decision.strategy, ChunkedMap):
                decisions[idx] = AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning=(
                        "joint-budget: heterogeneous axis fixed to "
                        f"ChunkedMap(batch_size={spec.default_batch_size}); "
                        "Vmap is invalid when element shapes vary "
                        f"(final estimate {estimate} B, budget {budget.bytes} B)"
                    ),
                    strategy=decision.strategy,
                )
                continue
            if not isinstance(decision.strategy, Vmap):
                continue
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

    def _warn_missing_memory_estimator(self, spec: AxisSpec) -> None:
        """Log once when cardinality rules run with no per-axis estimator."""
        if spec.name in self._missing_estimator_warned_axes:
            return
        self._missing_estimator_warned_axes.add(spec.name)
        logger.warning(
            "BatchPlanner memory_estimator is None for axis %r; "
            "cardinality rules select Vmap when cardinality <= default_batch_size.",
            spec.name,
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

        # Check memory estimate before deciding between Vmap and ChunkedMap.
        # A missing device limit is not an estimator failure; that path logs the
        # documented 4 GiB default inside _memory_estimator_device_limit.
        should_prefer_safemap_for_memory = False
        if self.memory_estimator is not None:
            try:
                estimated_bytes = self.memory_estimator(spec)
            except Exception as exc:
                raise RuntimeError(
                    f"BatchPlanner memory_estimator failed for axis {spec.name!r}: {exc}"
                ) from exc
            bound = spec.element_input_bytes
            if bound is not None and estimated_bytes < bound:
                raise ValueError(
                    f"memory_estimator for axis {spec.name!r} returned {estimated_bytes} "
                    f"bytes, below the per-element input lower bound of {bound} bytes"
                )
            if estimated_bytes > _memory_estimator_device_limit():
                should_prefer_safemap_for_memory = True

        # Rule 3: cardinality <= batch_size → Vmap (unless memory override)
        if spec.cardinality <= spec.default_batch_size:
            if should_prefer_safemap_for_memory:
                # Memory estimator overrides: use ChunkedMap
                strategy = ChunkedMap(batch_size=spec.default_batch_size)
                reasoning = "cardinality <= batch_size but memory_estimator override → ChunkedMap"
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            else:
                if self.memory_estimator is None:
                    self._warn_missing_memory_estimator(spec)
                strategy = Vmap()
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.default_batch_size,
                    reasoning="cardinality <= batch_size → Vmap",
                    strategy=strategy,
                )

        # Rule 4: cardinality > batch_size. Divisibility no longer changes the decision
        # (#5565): the chunked map runs a ragged final chunk itself, so the former Rule 5
        # (non-divisible -> ChunkedMap + a warning that dispatch would raise) is gone.
        remainder = spec.cardinality % spec.default_batch_size
        shape = "divisible" if remainder == 0 else f"ragged final chunk of {remainder}"
        if should_prefer_safemap_for_memory:
            # Memory estimate exceeds limit: use ChunkedMap
            strategy = ChunkedMap(batch_size=spec.default_batch_size)
            reasoning = (
                f"cardinality > batch_size ({shape}) but memory_estimator override → ChunkedMap"
            )
        elif self.memory_estimator is not None:
            # Memory estimator is provided and under limit: prefer Vmap
            strategy = Vmap()
            reasoning = f"cardinality > batch_size ({shape}) but memory safe → Vmap"
        else:
            # No memory estimator: use default ChunkedMap
            strategy = ChunkedMap(batch_size=spec.default_batch_size)
            reasoning = f"cardinality > batch_size ({shape}) → ChunkedMap"
        return AxisDecision(
            spec=spec,
            batch_size=spec.default_batch_size,
            reasoning=reasoning,
            strategy=strategy,
        )

    def _budget_initial_decision(self, spec: AxisSpec) -> AxisDecision:
        """Initial joint-budget strategy for one pending axis.

        Heterogeneous axes are fixed to ChunkedMap. Homogeneous axes start at
        Vmap and may be demoted later.
        """
        if spec.heterogeneous:
            return AxisDecision(
                spec=spec,
                batch_size=spec.default_batch_size,
                reasoning=(
                    "joint-budget: heterogeneous axis fixed to "
                    f"ChunkedMap(batch_size={spec.default_batch_size}) "
                    "(pending final estimate)"
                ),
                strategy=ChunkedMap(batch_size=spec.default_batch_size),
            )
        return AxisDecision(
            spec=spec,
            batch_size=spec.default_batch_size,
            reasoning="joint-budget: Vmap (pending final estimate)",
            strategy=Vmap(),
        )


def _live_extent(decision: AxisDecision) -> int:
    """Elements live at once: full cardinality under Vmap, otherwise the tile."""
    if isinstance(decision.strategy, Vmap):
        return decision.spec.cardinality
    return decision.batch_size


def _abstract_signature(arg: Any) -> tuple[Any, ...]:
    """Hashable shape/dtype identity for one lowered_memory_estimate input."""
    shape = getattr(arg, "shape", None)
    dtype = getattr(arg, "dtype", None)
    if shape is not None or dtype is not None:
        shape_key = tuple(shape) if shape is not None else None
        return ("shaped", shape_key, str(dtype))
    return ("repr", repr(arg))


def _bytes_per_element(
    estimate: int | Callable[..., Any],
    abstract_args: tuple[Any, ...],
    memo: MutableMapping[Any, int] | None,
) -> int:
    """Resolve ``plan_axis``'s estimate to bytes for one live element."""
    if isinstance(estimate, bool) or isinstance(estimate, int):
        if isinstance(estimate, bool) or estimate < 0:
            raise ValueError(
                f"plan_axis: int estimate is bytes per element and must be >= 0, got {estimate!r}"
            )
        if abstract_args:
            raise TypeError(
                "plan_axis: abstract args are only valid when estimate is a callable "
                "passed to lowered_memory_estimate"
            )
        return estimate
    if not callable(estimate):
        raise TypeError(
            "plan_axis: estimate must be an int (bytes per element) or a callable "
            f"to measure with lowered_memory_estimate, got {type(estimate).__name__}"
        )
    if not abstract_args:
        raise TypeError(
            "plan_axis: a callable estimate needs abstract args "
            "(ShapeDtypeStruct or concrete arrays) for lowered_memory_estimate"
        )
    # Identity of the callable, not id(): a recycled id can serve another
    # function's entry after the original is collected. Holding the function
    # keeps that entry from being reused.
    key = (estimate, tuple(_abstract_signature(arg) for arg in abstract_args))
    if memo is not None and key in memo:
        return memo[key]
    measured = lowered_memory_estimate(estimate, *abstract_args)
    if memo is not None:
        memo[key] = measured
    return measured


def plan_axis(
    spec: AxisSpec,
    *abstract_args: Any,
    estimate: int | Callable[..., Any],
    budget: int,
    memo: MutableMapping[Any, int] | None = None,
) -> AxisStrategy:
    """Choose one axis's strategy with ``BatchPlanner`` joint-budget mode.

    Thin wrapper: builds a one-axis ``MemoryBudget`` and returns the strategy
    ``BatchPlanner.plan`` selected. Not a second planner.

    ``estimate`` is either:

    - an ``int`` — bytes per live element. Vmap keeps ``cardinality`` elements
      live; every other strategy keeps ``batch_size`` live. The joint estimate
      is that extent times ``estimate``.
    - a callable plus ``abstract_args`` — measured once with
      ``lowered_memory_estimate`` and then scaled the same way on every greedy
      step. Pass abstract args that describe one element. ``memo``, if given,
      caches that lowered byte count by function identity and abstract
      shape/dtype so repeated ``plan_axis`` calls do not recompile.

    Args:
        spec: Axis to plan.
        *abstract_args: Abstract inputs for a callable ``estimate``. Must be
            empty when ``estimate`` is an int.
        estimate: Bytes per live element, or a JAX-traceable callable.
        budget: Joint memory budget in bytes (``MemoryBudget.bytes``).
        memo: Optional cache of lowered per-element byte counts.

    Returns:
        The ``AxisStrategy`` for ``spec``.

    Raises:
        TypeError: If ``estimate`` is neither an int nor a callable, or if
            abstract args are paired with the wrong form.
        ValueError: If an int ``estimate`` is negative. ``budget`` is
            validated by ``MemoryBudget``.
        BudgetInfeasibleError: If ChunkedMap still exceeds ``budget``.
    """
    per_element = _bytes_per_element(estimate, abstract_args, memo)

    def joint_estimate(decisions: Sequence[AxisDecision]) -> int:
        live = 1
        for decision in decisions:
            live *= _live_extent(decision)
        return int(per_element * live)

    plan = BatchPlanner(budget=MemoryBudget(bytes=budget, estimate=joint_estimate)).plan([spec])
    return plan.decisions[0].strategy

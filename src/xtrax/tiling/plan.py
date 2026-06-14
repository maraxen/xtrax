"""BatchPlan and BatchPlanner for composable axis tiling strategy selection."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from xtrax.tiling.strategy import (
    AxisStrategy,
    Bucket,
    DedupGather,
    SafeMap,
    Scan,
    Vmap,
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
        batch_size: Default batch size threshold and chunk size for SafeMap.
        granularity: Alignment granularity (default 1, no constraint).
        heterogeneous: Whether elements have different sizes (default False).
        dedup_eligible: Whether this axis is eligible for deduplication (default False).
        bucket_boundaries: Optional sorted, strictly-ascending bucket sizes. When
            provided, the planner selects the Bucket strategy (length-padding to the
            nearest boundary) instead of the cardinality-based rules. None disables
            bucketing (default).
    """

    name: str
    cardinality: int
    batch_size: int
    granularity: int = 1
    heterogeneous: bool = False
    dedup_eligible: bool = False
    bucket_boundaries: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate bucket_boundaries when provided."""
        if self.bucket_boundaries is None:
            return
        boundaries = tuple(self.bucket_boundaries)
        # Coerce to tuple so the frozen dataclass stays hashable even if a list
        # was passed for ergonomics.
        object.__setattr__(self, "bucket_boundaries", boundaries)
        if len(boundaries) == 0:
            raise ValueError(
                f"AxisSpec(name={self.name!r}): bucket_boundaries must be non-empty."
            )
        if any(b <= 0 for b in boundaries):
            raise ValueError(
                f"AxisSpec(name={self.name!r}): bucket_boundaries must be positive, "
                f"got {boundaries}."
            )
        if list(boundaries) != sorted(boundaries) or len(set(boundaries)) != len(
            boundaries
        ):
            raise ValueError(
                f"AxisSpec(name={self.name!r}): bucket_boundaries must be strictly "
                f"ascending, got {boundaries}."
            )


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
    """

    def __init__(
        self,
        memory_estimator: Callable[[AxisSpec], int] | None = None,
        carry_specs: list[CarrySpec] | None = None,
        dedup_specs: list[DedupSpec] | None = None,
    ) -> None:
        """Initialize the planner.

        Args:
            memory_estimator: Optional function that estimates Vmap memory (bytes)
                for a given AxisSpec. If provided and estimate exceeds device limit,
                SafeMap is preferred over Vmap. If the estimator raises an exception,
                falls back to default rules silently.
            carry_specs: Optional list of CarrySpec objects declaring which axes
                should use Scan strategy (Phase 0 pre-demotion).
            dedup_specs: Optional list of DedupSpec objects declaring which axes
                should use DedupGather strategy (Phase 0b pre-demotion).
        """
        self.memory_estimator = memory_estimator
        self.carry_specs = carry_specs or []
        self.dedup_specs = dedup_specs or []

    def plan(self, specs: Sequence[AxisSpec]) -> BatchPlan:
        """Generate a tiling plan for the given specs.

        Phase 0: Pre-demote axes with declared CarrySpec to Scan.
        Phase 0b: Pre-demote axes with declared DedupSpec to DedupGather.
        Phases 1+: Apply standard strategy selection rules to remaining axes.

        Args:
            specs: Sequence of AxisSpec objects to plan.

        Returns:
            BatchPlan with decisions for each spec.
        """
        specs_by_name = {spec.name: spec for spec in specs}
        decisions = []

        # Phase 0: Pre-demote axes with CarrySpec to Scan
        carry_by_name = {cs.axis_name: cs for cs in self.carry_specs}
        phase0_names = set()
        for spec in specs:
            if spec.name in carry_by_name:
                cs = carry_by_name[spec.name]
                scan_strategy = Scan(
                    init=cs.init,
                    transition=cs.transition,
                    ordered_sinks=cs.ordered_sinks,
                )
                decisions.append(
                    AxisDecision(
                        spec=spec,
                        batch_size=1,
                        reasoning=f"carry-bearing scan (CarrySpec declared for '{spec.name}')",
                        strategy=scan_strategy,
                    ),
                )
                phase0_names.add(spec.name)

        # Phase 0b: Pre-demote axes with DedupSpec to DedupGather
        dedup_by_name = {ds.axis_name: ds for ds in self.dedup_specs}
        phase0b_names = set()
        for spec in specs:
            if spec.name in dedup_by_name and spec.name not in phase0_names:
                ds = dedup_by_name[spec.name]
                dg_strategy = ds.to_dedup_gather()
                decisions.append(
                    AxisDecision(
                        spec=spec,
                        batch_size=ds.k,
                        reasoning=f"dedup-gather (DedupSpec declared for '{spec.name}', k={ds.k}, k_bucket={dg_strategy.k_bucket})",
                        strategy=dg_strategy,
                    ),
                )
                phase0b_names.add(spec.name)

        # Apply standard rules to remaining axes
        remaining_names = set(specs_by_name.keys()) - phase0_names - phase0b_names
        for spec in specs:
            if spec.name in remaining_names:
                decision = self._decide_strategy(spec)
                decisions.append(decision)

        return BatchPlan(decisions=tuple(decisions))

    def _decide_strategy(self, spec: AxisSpec) -> AxisDecision:
        """Decide strategy for a single AxisSpec following selection rules."""

        # Rule 1: explicit bucket_boundaries → Bucket (host-side length-padding).
        # This is the strongest, most explicit signal and wins over dedup/cardinality
        # rules: the caller has declared the variable-length axis and its buckets.
        # Bucket is a host plan descriptor — padding happens before the JIT boundary
        # via select_bucket()/bucketize(), not in make_axis_dispatch.
        if spec.bucket_boundaries is not None:
            boundaries = spec.bucket_boundaries
            strategy = Bucket(boundaries=boundaries)
            return AxisDecision(
                spec=spec,
                batch_size=spec.batch_size,
                reasoning=(
                    f"bucket_boundaries={boundaries} → Bucket (host-side padding)"
                ),
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
                    device_limit = (
                        jax.devices()[0].memory_stats().get("bytes_limit", 4 * (2**30))
                    )
                except Exception:
                    device_limit = 4 * (2**30)

                if estimated_bytes > device_limit:
                    should_prefer_safemap_for_memory = True
            except Exception:
                # Fall back silently to default rules
                pass

        # Rule 3: cardinality <= batch_size → Vmap (unless memory override)
        if spec.cardinality <= spec.batch_size:
            if should_prefer_safemap_for_memory:
                # Memory estimator overrides: use SafeMap
                strategy = SafeMap(batch_size=spec.batch_size)
                reasoning = (
                    "cardinality <= batch_size but memory_estimator override → SafeMap"
                )
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            else:
                strategy = Vmap()
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.batch_size,
                    reasoning="cardinality <= batch_size → Vmap",
                    strategy=strategy,
                )

        # Rule 4 & 5: cardinality > batch_size
        # Check divisibility
        if spec.cardinality % spec.batch_size == 0:
            # Rule 4: divisible → SafeMap (unless memory estimator allows Vmap)
            if should_prefer_safemap_for_memory:
                # Memory estimate exceeds limit: use SafeMap
                strategy = SafeMap(batch_size=spec.batch_size)
                reasoning = (
                    "cardinality > batch_size and divisible but "
                    "memory_estimator override → SafeMap"
                )
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            elif self.memory_estimator is not None:
                # Memory estimator is provided and under limit: prefer Vmap
                strategy = Vmap()
                reasoning = (
                    "cardinality > batch_size and divisible but memory safe → Vmap"
                )
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.batch_size,
                    reasoning=reasoning,
                    strategy=strategy,
                )
            else:
                # No memory estimator: use default SafeMap
                strategy = SafeMap(batch_size=spec.batch_size)
                return AxisDecision(
                    spec=spec,
                    batch_size=spec.batch_size,
                    reasoning="cardinality > batch_size and divisible → SafeMap",
                    strategy=strategy,
                )
        else:
            # Rule 5: non-divisible → SafeMap + warning (deferred-failure contract)
            warnings.warn(
                f"AxisSpec(name={spec.name!r}): cardinality={spec.cardinality} "
                f"is not divisible by batch_size={spec.batch_size}. "
                f"This plan will raise ValueError at make_axis_dispatch time.",
                RuntimeWarning,
                stacklevel=3,
            )
            strategy = SafeMap(batch_size=spec.batch_size)
            reasoning = (
                "cardinality > batch_size but not divisible → "
                "SafeMap (deferred failure)"
            )
            return AxisDecision(
                spec=spec,
                batch_size=spec.batch_size,
                reasoning=reasoning,
                strategy=strategy,
            )

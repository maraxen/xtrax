"""BatchPlan and BatchPlanner for composable axis tiling strategy selection."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax

from xtrax.tiling.strategy import (
    AxisStrategy,
    DedupGather,
    SafeMap,
    Vmap,
)


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
    """

    name: str
    cardinality: int
    batch_size: int
    granularity: int = 1
    heterogeneous: bool = False
    dedup_eligible: bool = False


@dataclass(frozen=True)
class AxisDecision:
    """Decision for how to tile a single axis.

    Attributes:
        spec: The AxisSpec that was analyzed.
        batch_size: Final batch size used (from spec).
        reasoning: Human-readable explanation of the decision.
        strategy: Selected AxisStrategy (Vmap, SafeMap, or DedupGather).
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
    1. dedup_eligible=True → DedupGather
    2. cardinality <= batch_size → Vmap
    3. cardinality > batch_size AND divisible → SafeMap
    4. non-divisible → SafeMap with warning (deferred-failure contract)

    When memory_estimator is provided, it overrides rule 2/3 decisions
    to prefer SafeMap if estimated Vmap memory exceeds device limit.
    """

    def __init__(
        self,
        memory_estimator: Callable[[AxisSpec], int] | None = None,
    ) -> None:
        """Initialize the planner.

        Args:
            memory_estimator: Optional function that estimates Vmap memory (bytes)
                for a given AxisSpec. If provided and estimate exceeds device limit,
                SafeMap is preferred over Vmap. If the estimator raises an exception,
                falls back to default rules silently.
        """
        self.memory_estimator = memory_estimator

    def plan(self, specs: Sequence[AxisSpec]) -> BatchPlan:
        """Generate a tiling plan for the given specs.

        Pure Python — no JAX tracing. Scan is never returned.

        Args:
            specs: Sequence of AxisSpec objects to plan.

        Returns:
            BatchPlan with decisions for each spec.
        """
        decisions = []

        for spec in specs:
            decision = self._decide_strategy(spec)
            decisions.append(decision)

        return BatchPlan(decisions=tuple(decisions))

    def _decide_strategy(self, spec: AxisSpec) -> AxisDecision:
        """Decide strategy for a single AxisSpec following selection rules."""

        # Rule 1: dedup_eligible → DedupGather (with placeholder fns)
        if spec.dedup_eligible:
            strategy = DedupGather(
                dedup_fn=lambda xs: (xs, None),
                gather_fn=lambda ys, indices: ys,
                k_bucket=256,  # Placeholder
            )
            return AxisDecision(
                spec=spec,
                batch_size=spec.batch_size,
                reasoning="dedup_eligible=True → DedupGather",
                strategy=strategy,
            )

        # Check memory estimate before deciding between Vmap and SafeMap
        should_prefer_safemap_for_memory = False
        if self.memory_estimator is not None:
            try:
                estimated_bytes = self.memory_estimator(spec)
                # Get device memory limit (default 4 GiB)
                try:
                    device_limit = jax.devices()[0].memory_stats().get(
                        "bytes_limit", 4 * (2**30)
                    )
                except Exception:
                    device_limit = 4 * (2**30)

                if estimated_bytes > device_limit:
                    should_prefer_safemap_for_memory = True
            except Exception:
                # Fall back silently to default rules
                pass

        # Rule 2: cardinality <= batch_size → Vmap (unless memory override)
        if spec.cardinality <= spec.batch_size:
            if should_prefer_safemap_for_memory:
                # Memory estimator overrides: use SafeMap
                strategy = SafeMap(batch_size=spec.batch_size)
                reasoning = (
                    "cardinality <= batch_size but "
                    "memory_estimator override → SafeMap"
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

        # Rule 3 & 4: cardinality > batch_size
        # Check divisibility
        if spec.cardinality % spec.batch_size == 0:
            # Rule 3: divisible → SafeMap (unless memory estimator allows Vmap)
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
                    "cardinality > batch_size and divisible but "
                    "memory safe → Vmap"
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
            # Rule 4: non-divisible → SafeMap + warning (deferred-failure contract)
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

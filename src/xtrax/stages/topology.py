"""Plan topology validation — catches structurally-impossible plan/boundary pairings.

AxisBoundary's own docstring (boundaries.py) documents two topology rules that
must hold before any JAX trace:

1. Scan strategy is invalid on a heterogeneous axis (jax.lax.scan requires a
   static carry shape; heterogeneous axes have variable-shape elements).
2. An ordered=True Tap or Sink on a Vmap axis has no step-ordering guarantee
   (vmap does not preserve step order; ordered io_callback needs SafeMap or
   Scan).

This module implements that promised validator. It is structural/duck-typed
(matches by `type(strategy).__name__`, not isinstance against xtrax's own
Vmap/SafeMap/Scan classes) so it works correctly on ANY library's plan
objects with matching field names -- including a parallel BatchPlanner
reimplementation (e.g. aminx.tiling) whose strategy instances are distinct
classes from xtrax's. Nominal isinstance checks here would silently never
fire for such a caller, which is worse than no validator at all (false
confidence). See xtrax.eda.types for the same pattern applied to plan
statistics extraction.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from xtrax.stages.boundaries import AxisBoundary


class PlanTopologyError(Exception):
    """Raised when a tiling plan's strategy choices conflict with its boundary ops.

    Fires at plan-construction time, before any JAX compilation -- topology
    errors are caught here, not at trace time or runtime.
    """


@runtime_checkable
class AxisSpecLike(Protocol):
    """Minimal structural shape validate_plan_topology reads from an axis spec."""

    @property
    def name(self) -> str: ...

    @property
    def heterogeneous(self) -> bool: ...


@runtime_checkable
class AxisDecisionLike(Protocol):
    """Minimal structural shape validate_plan_topology reads from an axis decision.

    `strategy` is typed `object`, not `Any`: this function only ever reads
    `type(strategy).__name__`, never calls strategy methods.
    """

    @property
    def spec(self) -> AxisSpecLike: ...

    @property
    def strategy(self) -> object: ...


def validate_plan_topology(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
) -> None:
    """Validate plan topology against AxisBoundary's documented rules.

    Args:
        decisions: Axis decisions from a BatchPlan (xtrax's own, or any
            structurally-compatible plan from another library).
        axis_boundaries: Map of axis name -> AxisBoundary, as wired into the
            pipeline's stage set.

    Raises:
        PlanTopologyError: on the first violation found.

    Example:
        >>> validate_plan_topology(plan.decisions, stage_set.axis_boundaries)
    """
    for decision in decisions:
        strategy_name = type(decision.strategy).__name__

        # Rule 1: Scan on heterogeneous axis is structurally impossible.
        if decision.spec.heterogeneous and strategy_name == "Scan":
            msg = (
                f"PlanTopologyError: axis '{decision.spec.name}' is heterogeneous "
                f"(element shapes vary) but has a Scan strategy. "
                f"jax.lax.scan requires static carry shape -- heterogeneous axes "
                f"must use SafeMap. Use CarrySpec only on homogeneous axes."
            )
            raise PlanTopologyError(msg)

        # Rule 2: ordered boundary op on Vmap axis has no step-ordering guarantee.
        if strategy_name == "Vmap":
            boundary = axis_boundaries.get(decision.spec.name)
            if boundary is not None:
                if boundary.tap is not None and getattr(boundary.tap, "ordered", False):
                    msg = (
                        f"PlanTopologyError: axis '{decision.spec.name}' has an "
                        f"ordered=True Tap but uses Vmap strategy. vmap does not "
                        f"preserve step order. Use SafeMap or Scan on axes with "
                        f"ordered boundary ops."
                    )
                    raise PlanTopologyError(msg)
                if boundary.sink is not None and getattr(boundary.sink, "ordered", False):
                    msg = (
                        f"PlanTopologyError: axis '{decision.spec.name}' has an "
                        f"ordered=True Sink but uses Vmap strategy. vmap does not "
                        f"preserve step order. Use SafeMap or Scan on axes with "
                        f"ordered boundary ops."
                    )
                    raise PlanTopologyError(msg)


__all__ = [
    "PlanTopologyError",
    "AxisSpecLike",
    "AxisDecisionLike",
    "validate_plan_topology",
]

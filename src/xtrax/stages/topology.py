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


def axis_boundaries_by_name(
    axes: Sequence[AxisSpecLike],
    boundaries: Sequence[AxisBoundary] | None,
) -> Mapping[str, AxisBoundary]:
    """Adapt RunSpec's positional axes/boundaries lists into a name-keyed Mapping.

    Fork-9 resolution (T1-02): `RunSpec.boundaries` stays a plain
    `list[AxisBoundary] | None` -- aminx subclasses construct it positionally,
    one entry per axis in `RunSpec.axes` order -- rather than breaking the
    field to `Mapping[str, AxisBoundary]` directly. This adapter is the seam
    where axis identity gets attached, at the executor entry, matching what
    `validate_plan_topology` already consumes.

    Args:
        axes: Axis specs in the same order as `boundaries` (e.g. `RunSpec.axes`).
        boundaries: One entry per axis, or None if no axis has any boundary op.

    Returns:
        Mapping of axis name -> AxisBoundary. Empty if `boundaries` is None.

    Raises:
        PlanTopologyError: if `boundaries` is not None and its length differs
            from `axes` (a mismatched positional pairing), or if two axes
            share the same name -- a plain dict comprehension would silently
            keep only the last boundary for a duplicate name, masking a real
            keying bug rather than rejecting it (PM4).

    Example:
        >>> axis_boundaries_by_name(run_spec.axes, run_spec.boundaries)
    """
    if boundaries is None:
        return {}
    if len(boundaries) != len(axes):
        msg = (
            f"PlanTopologyError: boundaries has {len(boundaries)} entries but "
            f"axes has {len(axes)}; RunSpec.boundaries must have exactly one "
            f"entry per axis (or be None)."
        )
        raise PlanTopologyError(msg)

    result: dict[str, AxisBoundary] = {}
    for axis, boundary in zip(axes, boundaries, strict=True):
        if axis.name in result:
            msg = (
                f"PlanTopologyError: duplicate axis name {axis.name!r} in axes; "
                f"cannot key boundaries unambiguously by name."
            )
            raise PlanTopologyError(msg)
        result[axis.name] = boundary
    return result


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
    "axis_boundaries_by_name",
    "validate_plan_topology",
]

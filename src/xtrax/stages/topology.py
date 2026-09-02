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


class MaterializeFuseConflictError(PlanTopologyError):
    """materialize=True and a non-None fuse on the same axis.

    fuse collapses the per-step stacked array that materialize needs to expose as
    the axis's own output, so the two cannot both apply to one axis's export.

    This is a consequence of keeping the executor unchanged, not a structural
    impossibility: execute_map_axis/execute_scan_axis always call _apply_fuse on
    the stacked ys and return only the fused result. Exposing both the pre-fuse
    per-step stream and the fused value would require the executor to return a
    second value.

    A caller who wants both should drop `fuse` from the AxisBoundary, keep
    materialize=True, and perform the reduction outside the exported function on
    the materialized array export_pipeline returns.
    """


class MaterializeWithoutSinkError(PlanTopologyError):
    """materialize=True with sink=None -- nothing to materialize.

    A caller error rather than a silent no-op. Fires from validate_plan_topology
    at export-time plan validation, before any jit/compile call. It says nothing
    about an eager run of the same AxisBoundary, where the combination is never
    reached: materialize is read only by xtrax.export.
    """


class MultipleMaterializeAxesError(PlanTopologyError):
    """More than one axis in the same plan has materialize=True.

    Two independently-materialized axes have no defined output shape here, so
    this is rejected by name rather than left to produce an unspecified second
    output slot.
    """


@runtime_checkable
class AxisSpecLike(Protocol):
    """Minimal structural shape validate_plan_topology reads from an axis spec."""

    @property
    def name(self) -> str: ...

    @property
    def heterogeneous(self) -> bool: ...


@runtime_checkable
class AxisBoundaryLike(Protocol):
    """Minimal structural shape validate_plan_topology reads from a boundary.

    Widened from the concrete AxisBoundary so this module's structural-
    compatibility promise covers boundaries as well as decisions. Annotating the
    concrete class made the promise false in practice: the runtime type-check
    hook rejects a foreign plan's boundary object before any duck-typed read
    happens. Note that `materialize` is deliberately absent -- a boundary
    predating that field must still satisfy this protocol, which is why Rule 3
    reads it with getattr.
    """

    @property
    def fuse(self) -> object | None: ...

    @property
    def tap(self) -> object | None: ...

    @property
    def sink(self) -> object | None: ...


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


_EXPORTABLE_STRATEGIES = ("Vmap", "SafeMap", "Scan", "DedupGather")


def _check_export_boundary(decision: AxisDecisionLike, boundary: AxisBoundaryLike | None) -> bool:
    """Apply the per-axis export boundary rules; report whether it materializes.

    Args:
        decision: The axis decision being validated.
        boundary: That axis's boundary, or None.

    Returns:
        True if this axis passed with materialize=True, else False.

    Raises:
        PlanTopologyError: tap present, or a sink not declared materializing.
        MaterializeWithoutSinkError: materialize=True with no sink.
        MaterializeFuseConflictError: materialize=True alongside a fuse.
    """
    if boundary is None:
        return False

    axis = decision.spec.name
    # Duck-typed like Rule 2's `.ordered` reads: a foreign plan's boundary object
    # predating this field must be treated as materialize=False, not raise.
    materialize = getattr(boundary, "materialize", False)

    # A Tap is T -> T and feeds downstream, so it is never droppable.
    if boundary.tap is not None:
        msg = (
            f"PlanTopologyError: axis '{axis}' has a Tap, which pierces the "
            f"export boundary. A Tap is T -> T and participates in dataflow, so "
            f"it cannot be stripped for export on any target. Remove the tap, or "
            f"fold its transform into the axis's own step function."
        )
        raise PlanTopologyError(msg)

    if materialize and boundary.sink is None:
        msg = (
            f"PlanTopologyError: axis '{axis}' declares materialize=True but has "
            f"no sink, so there is nothing to materialize. Set a sink, or drop "
            f"materialize."
        )
        raise MaterializeWithoutSinkError(msg)

    if boundary.sink is not None and not materialize:
        msg = (
            f"PlanTopologyError: axis '{axis}' has a Sink, which pierces the "
            f"export boundary. A sink runs host code the exported program cannot "
            f"call. If the sink only records the per-step values, declare it with "
            f"AxisBoundary(..., materialize=True) and read those values off the "
            f"exported output instead."
        )
        raise PlanTopologyError(msg)

    if boundary.sink is not None and materialize and boundary.fuse is not None:
        msg = (
            f"PlanTopologyError: axis '{axis}' declares materialize=True and also "
            f"has a fuse. fuse collapses the per-step stacked array that "
            f"materialize needs to expose as this axis's output. Drop the fuse and "
            f"reduce outside the exported function."
        )
        raise MaterializeFuseConflictError(msg)

    return bool(materialize)


def validate_plan_topology(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundaryLike],
    *,
    export_safe: bool = False,
) -> None:
    """Validate plan topology against AxisBoundary's documented rules.

    Args:
        decisions: Axis decisions from a BatchPlan (xtrax's own, or any
            structurally-compatible plan from another library).
        axis_boundaries: Map of axis name -> AxisBoundary, as wired into the
            pipeline's stage set.
        export_safe: Also apply the rules that only matter when the plan is
            about to cross an export boundary (rules 3 and 4). Defaults to
            False, leaving existing callers' behavior unchanged.

    Raises:
        PlanTopologyError: on the first violation found.

    Example:
        >>> validate_plan_topology(plan.decisions, stage_set.axis_boundaries)
    """
    materializing_axes: list[str] = []

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

        if not export_safe:
            continue

        # Rule 3: boundary ops that would pierce the export boundary. Kind-based:
        # a Tap always rejects; a Sink rejects unless declared materializing.
        if _check_export_boundary(decision, axis_boundaries.get(decision.spec.name)):
            materializing_axes.append(decision.spec.name)

        # Rule 4: only strategies this package can fold into a traceable callable.
        if strategy_name not in _EXPORTABLE_STRATEGIES:
            supported = "/".join(_EXPORTABLE_STRATEGIES)
            msg = (
                f"PlanTopologyError: axis '{decision.spec.name}' uses strategy "
                f"{strategy_name!r}, which cannot cross an export boundary. "
                f"Supported strategies are {supported}. Bucket is host-tier (pad "
                f"with bucketize() before the boundary); WhileCarry has an "
                f"unbounded trip count (convert to Scan with a static length)."
            )
            raise PlanTopologyError(msg)

    # Rule 3, whole-plan tier: two independently materialized axes have no
    # defined output shape, so reject rather than emit an unspecified second slot.
    if len(materializing_axes) > 1:
        named = ", ".join(repr(a) for a in materializing_axes)
        msg = (
            f"PlanTopologyError: {len(materializing_axes)} axes declare "
            f"materialize=True ({named}), but only one materialized axis per plan "
            f"has a defined exported output shape. Materialize one axis and "
            f"handle the others outside the exported function."
        )
        raise MultipleMaterializeAxesError(msg)


__all__ = [
    "PlanTopologyError",
    "MaterializeFuseConflictError",
    "MaterializeWithoutSinkError",
    "MultipleMaterializeAxesError",
    "AxisSpecLike",
    "AxisBoundaryLike",
    "AxisDecisionLike",
    "axis_boundaries_by_name",
    "validate_plan_topology",
]

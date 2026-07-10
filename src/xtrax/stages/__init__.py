"""Composable stage protocols and bundles."""

from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap
from xtrax.stages.executor import ExecutorError, execute_map_axis, execute_scan_axis
from xtrax.stages.protocols import RollingFn, TransformFn
from xtrax.stages.topology import (
    PlanTopologyError,
    axis_boundaries_by_name,
    validate_plan_topology,
)

__all__ = [
    "TransformFn",
    "RollingFn",
    "Fuse",
    "Tap",
    "Sink",
    "AxisBoundary",
    "PlanTopologyError",
    "axis_boundaries_by_name",
    "validate_plan_topology",
    "ExecutorError",
    "execute_map_axis",
    "execute_scan_axis",
]


def __getattr__(name: str):
    """Fallback for deprecated FuseFn."""
    if name == "FuseFn":
        import warnings

        warnings.warn(
            "FuseFn is deprecated; import Fuse from xtrax.stages.boundaries instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Fuse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

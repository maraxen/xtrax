"""Composable stage protocols and bundles."""

from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap
from xtrax.stages.protocols import RollingFn, TransformFn

__all__ = ["TransformFn", "RollingFn", "Fuse", "Tap", "Sink", "AxisBoundary"]


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

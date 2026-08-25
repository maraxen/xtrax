"""Error types and sentinel values for signature inference.

AmbiguousAxisError and AxisRole are defined in xtrax.tiling.roles (a
pure-stdlib leaf) and re-exported here for backward compatibility so that
``from xtrax.inference.errors import AmbiguousAxisError, AxisRole`` and
``from xtrax.inference import AmbiguousAxisError, AxisRole`` continue to work.
"""

from __future__ import annotations

# Re-exports from the leaf module — tiling.roles has zero xtrax imports.
from xtrax.tiling.roles import AmbiguousAxisError, AxisRole

__all__ = [
    "AmbiguousAxisError",
    "AxisRole",
    "CseTraceError",
    "StructureMismatchError",
    "XtraxInferenceError",
]


class XtraxInferenceError(Exception):
    """Base exception for xtrax.inference module."""

    pass


class StructureMismatchError(XtraxInferenceError):
    """Raised when eval_shape structure diverges from concrete batch.

    This error indicates that a traced computation structure does not match
    the structure inferred from a concrete batch. This typically occurs when
    control flow or branching introduces different tree structures depending
    on runtime values.

    Example:
        If eval_shape produces a tree with different branches than the actual
        concrete execution, this error is raised to signal the structural mismatch.
    """

    pass


class CseTraceError(XtraxInferenceError):
    """Raised when analyze_cse cannot trace the target function.

    Wraps tracing failures (unsupported control flow, wrong argument count,
    non-traceable operations) so callers can distinguish analysis failures
    from computation failures.
    """

    pass

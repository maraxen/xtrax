"""Error types and sentinel values for signature inference."""

from __future__ import annotations

import enum


class XtraxInferenceError(Exception):
    """Base exception for xtrax.inference module."""

    pass


class AmbiguousAxisError(XtraxInferenceError):
    """Raised when an axis role is UNKNOWN at planning time.

    This error indicates that the BatchPlanner encountered an axis whose role
    could not be determined during the planning phase. The error message will
    include the axis name and guidance on how to resolve the ambiguity (e.g.,
    by providing explicit role annotations or constraints).

    Example:
        If an axis's role cannot be inferred from context and no explicit
        annotation is provided, this error is raised during plan construction.
    """

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


class AxisRole(enum.Enum):
    """Enumeration of axis role sentinels for MVP.

    In the MVP (v1), only two roles are defined:
    - KNOWN: The axis role is determined and planner proceeds normally.
    - UNKNOWN: The axis role could not be determined; signals fail-loud guard.

    Tier-2 will extend this with concrete role members (e.g., BATCH, SEQUENCE).
    All future concrete roles are treated as non-fail-loud.
    """

    KNOWN = "known"
    UNKNOWN = "unknown"

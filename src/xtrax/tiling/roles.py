"""Axis role sentinels and ambiguous-axis error for xtrax.tiling.

This module is a pure-stdlib leaf: it imports nothing from xtrax so that
tiling can use AxisRole and AmbiguousAxisError without depending on
xtrax.inference.  xtrax.inference.errors re-exports both symbols for
backward compatibility.
"""

from __future__ import annotations

import enum


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


class AmbiguousAxisError(Exception):
    """Raised when an axis role is UNKNOWN at planning time.

    This error indicates that the BatchPlanner encountered an axis whose role
    could not be determined during the planning phase. The error message will
    include the axis name and guidance on how to resolve the ambiguity (e.g.,
    by providing explicit role annotations or constraints).

    Example:
        If an axis's role cannot be inferred from context and no explicit
        annotation is provided, this error is raised during plan construction.

    Note:
        This class is defined here (xtrax.tiling.roles) so that tiling does
        not depend on xtrax.inference.  It is re-exported by
        xtrax.inference.errors for backward compatibility.
    """

    pass

"""Smoke and contract tests for xtrax.inference error types."""

from __future__ import annotations

import xtrax.inference
import xtrax.inference.errors
import xtrax.tiling.roles


def test_import_works() -> None:
    """Test that xtrax.inference imports successfully."""
    assert xtrax.inference is not None


def test_ambiguous_axis_error_importable() -> None:
    """Test that AmbiguousAxisError is importable from xtrax.inference.

    AmbiguousAxisError is now defined in xtrax.tiling.roles (a pure-stdlib leaf)
    and re-exported by xtrax.inference.errors.  It subclasses Exception directly
    (not XtraxInferenceError) to keep tiling free of inference imports.
    """
    assert hasattr(xtrax.inference, "AmbiguousAxisError")
    AmbiguousAxisError = xtrax.inference.AmbiguousAxisError
    # Importable from canonical home and from inference re-export
    assert AmbiguousAxisError is xtrax.tiling.roles.AmbiguousAxisError
    assert AmbiguousAxisError is xtrax.inference.errors.AmbiguousAxisError
    # Hierarchy: direct Exception subclass (NOT XtraxInferenceError — see roles.py)
    assert issubclass(AmbiguousAxisError, Exception)
    assert not issubclass(AmbiguousAxisError, xtrax.inference.errors.XtraxInferenceError)


def test_structure_mismatch_error_importable() -> None:
    """Test that StructureMismatchError is importable from xtrax.inference."""
    assert hasattr(xtrax.inference, "StructureMismatchError")
    StructureMismatchError = xtrax.inference.StructureMismatchError
    assert issubclass(StructureMismatchError, xtrax.inference.errors.XtraxInferenceError)
    assert issubclass(StructureMismatchError, Exception)


def test_axis_role_importable() -> None:
    """Test that AxisRole is importable from xtrax.inference and xtrax.tiling.roles."""
    assert hasattr(xtrax.inference, "AxisRole")
    AxisRole = xtrax.inference.AxisRole
    assert hasattr(AxisRole, "KNOWN")
    assert hasattr(AxisRole, "UNKNOWN")
    # Same object re-exported from canonical home
    assert AxisRole is xtrax.tiling.roles.AxisRole
    assert AxisRole is xtrax.inference.errors.AxisRole


def test_axis_role_members_distinct() -> None:
    """Test that AxisRole.KNOWN and UNKNOWN are distinct and comparable."""
    AxisRole = xtrax.inference.AxisRole
    assert AxisRole.KNOWN != AxisRole.UNKNOWN


def test_axis_role_hashable() -> None:
    """Test that AxisRole members are hashable (can be used in sets/dicts)."""
    AxisRole = xtrax.inference.AxisRole
    role_set = {AxisRole.KNOWN, AxisRole.UNKNOWN}
    assert len(role_set) == 2

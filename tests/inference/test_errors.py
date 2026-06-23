"""Smoke and contract tests for xtrax.inference error types."""

from __future__ import annotations

import xtrax.inference


def test_import_works() -> None:
    """Test that xtrax.inference imports successfully."""
    assert xtrax.inference is not None


def test_ambiguous_axis_error_importable() -> None:
    """Test that AmbiguousAxisError is importable from xtrax.inference."""
    assert hasattr(xtrax.inference, "AmbiguousAxisError")
    AmbiguousAxisError = xtrax.inference.AmbiguousAxisError
    assert issubclass(AmbiguousAxisError, xtrax.inference.errors.XtraxInferenceError)
    assert issubclass(AmbiguousAxisError, Exception)


def test_structure_mismatch_error_importable() -> None:
    """Test that StructureMismatchError is importable from xtrax.inference."""
    assert hasattr(xtrax.inference, "StructureMismatchError")
    StructureMismatchError = xtrax.inference.StructureMismatchError
    assert issubclass(StructureMismatchError, xtrax.inference.errors.XtraxInferenceError)
    assert issubclass(StructureMismatchError, Exception)


def test_axis_role_importable() -> None:
    """Test that AxisRole is importable from xtrax.inference."""
    assert hasattr(xtrax.inference, "AxisRole")
    AxisRole = xtrax.inference.AxisRole
    assert hasattr(AxisRole, "KNOWN")
    assert hasattr(AxisRole, "UNKNOWN")


def test_axis_role_members_distinct() -> None:
    """Test that AxisRole.KNOWN and UNKNOWN are distinct and comparable."""
    AxisRole = xtrax.inference.AxisRole
    assert AxisRole.KNOWN != AxisRole.UNKNOWN


def test_axis_role_hashable() -> None:
    """Test that AxisRole members are hashable (can be used in sets/dicts)."""
    AxisRole = xtrax.inference.AxisRole
    role_set = {AxisRole.KNOWN, AxisRole.UNKNOWN}
    assert len(role_set) == 2

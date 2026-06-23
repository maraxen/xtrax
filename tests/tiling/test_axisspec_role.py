"""Tests for AxisSpec.role field (E1.3a) — regression + new unit tests."""

from __future__ import annotations

import warnings

import pytest

from xtrax.tiling.roles import AxisRole
from xtrax.tiling.plan import AxisSpec


class TestAxisSpecRoleField:
    """AxisSpec.role field — default KNOWN, frozen, hashable."""

    def test_existing_construction_unchanged(self):
        """AxisSpec('batch', 64, 128) constructs with role == KNOWN (not UNKNOWN)."""
        spec = AxisSpec("batch", 64, 128)
        assert spec.name == "batch"
        assert spec.cardinality == 64
        assert spec.default_batch_size == 128
        # D1: default MUST be KNOWN so all existing specs are unaffected
        assert spec.role == AxisRole.KNOWN

    def test_role_default_is_known_not_unknown(self):
        """Explicit check: default role is KNOWN, never UNKNOWN."""
        spec = AxisSpec(name="seq", cardinality=256, default_batch_size=64)
        assert spec.role is AxisRole.KNOWN
        assert spec.role is not AxisRole.UNKNOWN

    def test_role_can_be_set_to_unknown(self):
        """role=AxisRole.UNKNOWN can be passed explicitly."""
        spec = AxisSpec(
            name="inferred", cardinality=32, default_batch_size=32, role=AxisRole.UNKNOWN
        )
        assert spec.role == AxisRole.UNKNOWN

    def test_role_can_be_set_to_known(self):
        """role=AxisRole.KNOWN can be passed explicitly (same as default)."""
        spec = AxisSpec(name="batch", cardinality=8, default_batch_size=8, role=AxisRole.KNOWN)
        assert spec.role == AxisRole.KNOWN

    def test_hashable_set_works(self):
        """Two AxisSpec instances are hashable — {spec_a, spec_b} works."""
        spec_a = AxisSpec("batch", 64, 128)
        spec_b = AxisSpec("seq", 256, 64)
        result = {spec_a, spec_b}
        assert len(result) == 2

    def test_equal_specs_hash_equal(self):
        """Equal AxisSpec instances produce equal hashes."""
        spec_a = AxisSpec("batch", 64, 128)
        spec_b = AxisSpec("batch", 64, 128)
        assert spec_a == spec_b
        assert hash(spec_a) == hash(spec_b)

    def test_equal_specs_with_role_hash_equal(self):
        """Equal AxisSpec with role field produce equal hashes."""
        spec_a = AxisSpec("batch", 64, 128, role=AxisRole.UNKNOWN)
        spec_b = AxisSpec("batch", 64, 128, role=AxisRole.UNKNOWN)
        assert spec_a == spec_b
        assert hash(spec_a) == hash(spec_b)

    def test_different_roles_not_equal(self):
        """AxisSpec with different roles are not equal."""
        spec_known = AxisSpec("batch", 64, 128, role=AxisRole.KNOWN)
        spec_unknown = AxisSpec("batch", 64, 128, role=AxisRole.UNKNOWN)
        assert spec_known != spec_unknown

    def test_frozen_role_cannot_be_mutated(self):
        """AxisSpec is frozen — assigning to role raises FrozenInstanceError."""
        spec = AxisSpec("batch", 64, 128)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError or AttributeError
            spec.role = AxisRole.UNKNOWN  # type: ignore[misc]

    def test_batch_size_deprecation_shim_still_warns(self):
        """The batch_size deprecation shim still raises DeprecationWarning."""
        spec = AxisSpec("batch", 64, 128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = spec.batch_size
            assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
                "Expected DeprecationWarning for spec.batch_size access"
            )

    def test_granularity_deprecation_shim_still_warns(self):
        """The granularity deprecation shim still raises DeprecationWarning."""
        spec = AxisSpec("batch", 64, 128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = spec.granularity
            assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
                "Expected DeprecationWarning for spec.granularity access"
            )

    def test_role_field_is_last_after_bucket_boundaries(self):
        """role field comes after bucket_boundaries in AxisSpec field order."""
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(AxisSpec)]
        assert "bucket_boundaries" in field_names
        assert "role" in field_names
        bb_idx = field_names.index("bucket_boundaries")
        role_idx = field_names.index("role")
        assert role_idx > bb_idx, (
            f"role (idx={role_idx}) must come after bucket_boundaries (idx={bb_idx})"
        )

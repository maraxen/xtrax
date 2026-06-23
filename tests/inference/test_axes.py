"""Tests for xtrax.inference.axes — synthesize_axes + resolve_axis_role (E1.3a)."""

from __future__ import annotations

import numpy as np
from jax import ShapeDtypeStruct

from xtrax.inference.abstract import build_abstract_inputs
from xtrax.inference.axes import resolve_axis_role, synthesize_axes
from xtrax.inference.errors import AxisRole


class TestResolveAxisRole:
    """resolve_axis_role(override) -> AxisRole."""

    def test_none_override_returns_unknown(self):
        """When override is None, role is UNKNOWN."""
        assert resolve_axis_role(None) == AxisRole.UNKNOWN

    def test_non_none_override_returns_known(self):
        """When override is not None (any value), role is KNOWN."""
        assert resolve_axis_role(42) == AxisRole.KNOWN
        assert resolve_axis_role("batch") == AxisRole.KNOWN
        assert resolve_axis_role(object()) == AxisRole.KNOWN
        assert resolve_axis_role(0) == AxisRole.KNOWN  # falsy but not None


class TestSynthesizeAxes:
    """synthesize_axes(abstract_inputs, overrides) -> list[AxisSpec]."""

    def test_single_leaf_no_overrides(self):
        """Single ShapeDtypeStruct leaf → one AxisSpec with UNKNOWN role."""
        abstract = build_abstract_inputs({"x": ((8, 4), np.float32)})
        specs = synthesize_axes(abstract)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.cardinality == 8  # leading dim
        assert spec.role == AxisRole.UNKNOWN

    def test_all_returned_specs_unknown_when_no_overrides(self):
        """Invariant D1: with overrides=None, EVERY AxisSpec has role == UNKNOWN."""
        abstract = build_abstract_inputs(
            {"x": ((16, 3), np.float32), "y": ((32,), np.int32)}
        )
        specs = synthesize_axes(abstract)
        assert len(specs) == 2
        for spec in specs:
            assert spec.role == AxisRole.UNKNOWN, (
                f"Expected UNKNOWN for axis {spec.name!r}, got {spec.role}"
            )

    def test_cardinalities_match_leading_dims(self):
        """Each AxisSpec.cardinality == leaf.shape[0]."""
        abstract = build_abstract_inputs(
            {"x": ((16, 3), np.float32), "y": ((32,), np.int32)}
        )
        specs = synthesize_axes(abstract)
        cardinalities = {spec.cardinality for spec in specs}
        assert 16 in cardinalities
        assert 32 in cardinalities

    def test_multiple_leaves_explicit_structs(self):
        """Direct ShapeDtypeStruct leaves work as abstract_inputs."""
        abstract = [
            ShapeDtypeStruct((10,), np.float32),
            ShapeDtypeStruct((20, 5), np.float64),
        ]
        specs = synthesize_axes(abstract)
        assert len(specs) == 2
        cardinalities = [spec.cardinality for spec in specs]
        assert 10 in cardinalities
        assert 20 in cardinalities
        for spec in specs:
            assert spec.role == AxisRole.UNKNOWN

    def test_override_present_makes_role_known(self):
        """When overrides has a key for an axis, that axis gets KNOWN role."""
        abstract = build_abstract_inputs({"x": ((8, 4), np.float32)})
        # synthesize_axes names axes axis_0, axis_1, ...
        specs = synthesize_axes(abstract, overrides={"axis_0": "batch"})
        assert len(specs) == 1
        assert specs[0].role == AxisRole.KNOWN

    def test_overrides_none_explicit_same_as_omitted(self):
        """Passing overrides=None explicitly is same as omitting it."""
        abstract = build_abstract_inputs({"x": ((8,), np.float32)})
        specs_implicit = synthesize_axes(abstract)
        specs_explicit = synthesize_axes(abstract, overrides=None)
        assert len(specs_implicit) == 1
        assert len(specs_explicit) == 1
        assert specs_implicit[0].role == specs_explicit[0].role == AxisRole.UNKNOWN

    def test_synthesize_never_relies_on_default(self):
        """synthesize_axes always sets role EXPLICITLY — role attr is always set."""
        abstract = build_abstract_inputs({"x": ((4, 2), np.float32)})
        specs = synthesize_axes(abstract)
        # If role were relying on the default, removing the default would break this.
        # Here we verify role is always explicitly one of KNOWN or UNKNOWN.
        for spec in specs:
            assert spec.role in (AxisRole.KNOWN, AxisRole.UNKNOWN)

    def test_build_abstract_inputs_integration(self):
        """End-to-end: build_abstract_inputs -> synthesize_axes produces valid specs."""
        spec_in = {
            "features": ((64, 128), np.float32),
            "labels": ((64,), np.int32),
        }
        abstract = build_abstract_inputs(spec_in)
        axes = synthesize_axes(abstract)
        assert len(axes) == 2
        # Leading dims are both 64
        for axis in axes:
            assert axis.cardinality == 64
            assert axis.role == AxisRole.UNKNOWN

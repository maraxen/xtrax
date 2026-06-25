"""End-to-end tests for xtrax.inference.api.infer_bundle (E1.9 / AC2)."""

from __future__ import annotations

import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.inference import (
    AxisOverride,
    AxisRole,
    BundleSchema,
    StructureMismatchError,
    axis_config,
    infer_bundle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_fn(x):
    """Return a dict with one field so BundleSchema.fields is inspectable."""
    return {"out": x * 2}


def _two_input_fn(x, y):
    """Two-input fn — used for AC2 zero-config test."""
    return {"out_x": x, "out_y": y}


# ---------------------------------------------------------------------------
# 1. Happy path: basic function + abstract inputs
# ---------------------------------------------------------------------------


class TestInferBundleHappyPath:
    def test_returns_bundle_schema_and_axis_specs(self):
        abstract_inputs = [ShapeDtypeStruct((4, 8), np.float32)]
        schema, axes = infer_bundle(_simple_fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert isinstance(axes, list)
        assert len(axes) >= 1

    def test_bundle_schema_has_expected_fields(self):
        abstract_inputs = [ShapeDtypeStruct((4, 8), np.float32)]
        schema, _ = infer_bundle(_simple_fn, abstract_inputs)

        assert "out" in schema.fields
        leaf = schema.fields["out"]
        assert leaf.shape == (4, 8)
        assert leaf.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. AC2: zero-config end-to-end — no decorator, 2 inputs
#    EVERY axis must be AxisRole.UNKNOWN; count covers all input axes.
# ---------------------------------------------------------------------------


class TestAC2ZeroConfigEndToEnd:
    """AC2: zero-config, no @axis_config — every axis role is UNKNOWN."""

    def test_returns_valid_bundle_schema(self):
        abstract_inputs = [
            ShapeDtypeStruct((8, 16), np.float32),
            ShapeDtypeStruct((8, 32), np.float32),
        ]
        schema, axes = infer_bundle(_two_input_fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert len(schema.fields) > 0

    def test_returns_axis_specs_covering_all_input_axes(self):
        abstract_inputs = [
            ShapeDtypeStruct((8, 16), np.float32),
            ShapeDtypeStruct((8, 32), np.float32),
        ]
        _, axes = infer_bundle(_two_input_fn, abstract_inputs)

        # Two qualifying leaves (ndim >= 1) -> 2 AxisSpecs
        assert len(axes) == 2

    def test_every_axis_is_unknown(self):
        """D1 invariant: no-decorator => every AxisSpec has role UNKNOWN."""
        abstract_inputs = [
            ShapeDtypeStruct((8, 16), np.float32),
            ShapeDtypeStruct((8, 32), np.float32),
        ]
        _, axes = infer_bundle(_two_input_fn, abstract_inputs)

        for spec in axes:
            assert spec.role == AxisRole.UNKNOWN, (
                f"Expected AxisRole.UNKNOWN, got {spec.role!r} for axis {spec.name!r}"
            )

    def test_no_decorator_means_no_overrides(self):
        """Confirm get_axis_config returns None for a bare fn (regression guard)."""
        from xtrax.inference.config import get_axis_config

        assert get_axis_config(_two_input_fn) is None


# ---------------------------------------------------------------------------
# 3. @axis_config path: decorated fn -> KNOWN axis with correct name
# ---------------------------------------------------------------------------


class TestAxisConfigDecorator:
    def test_decorated_fn_has_known_axis(self):
        @axis_config(AxisOverride(name="batch", default_batch_size=32))
        def _decorated(x):
            return {"out": x}

        abstract_inputs = [ShapeDtypeStruct((32, 16), np.float32)]
        _, axes = infer_bundle(_decorated, abstract_inputs)

        assert len(axes) == 1
        spec = axes[0]
        assert spec.role == AxisRole.KNOWN
        assert spec.name == "batch"

    def test_decorated_fn_batch_size_propagated(self):
        @axis_config(AxisOverride(name="batch", default_batch_size=32))
        def _decorated(x):
            return {"out": x}

        abstract_inputs = [ShapeDtypeStruct((32, 16), np.float32)]
        _, axes = infer_bundle(_decorated, abstract_inputs)

        assert axes[0].default_batch_size == 32


# ---------------------------------------------------------------------------
# 4. verify_against: consistent concrete passes; divergent raises
# ---------------------------------------------------------------------------


class TestVerifyAgainst:
    def test_consistent_concrete_passes(self):
        import jax.numpy as jnp

        abstract_inputs = [ShapeDtypeStruct((4, 8), np.float32)]
        concrete_inputs = [jnp.ones((4, 8), dtype=np.float32)]

        # Should not raise
        schema, axes = infer_bundle(_simple_fn, abstract_inputs, verify_against=concrete_inputs)
        assert isinstance(schema, BundleSchema)

    def test_divergent_concrete_raises_structure_mismatch(self):
        import jax.numpy as jnp

        def _data_dependent(x):
            # Returns different structure depending on shape — forces mismatch
            if x.shape[0] > 2:
                return {"a": x, "b": x}
            return {"a": x}

        abstract_inputs = [ShapeDtypeStruct((1, 4), np.float32)]
        # concrete has shape (4, 4) -> x.shape[0] > 2 -> different output structure
        concrete_inputs = [jnp.ones((4, 4), dtype=np.float32)]

        with pytest.raises(StructureMismatchError):
            infer_bundle(_data_dependent, abstract_inputs, verify_against=concrete_inputs)


# ---------------------------------------------------------------------------
# 5. Public import surface (AC import smoke test)
# ---------------------------------------------------------------------------


class TestPublicImportSurface:
    def test_infer_bundle_importable(self):
        from xtrax.inference import infer_bundle as _ib  # noqa: F401

        assert callable(_ib)

    def test_axis_config_importable(self):
        from xtrax.inference import axis_config as _ac  # noqa: F401

        assert callable(_ac)

    def test_axis_override_importable(self):
        from xtrax.inference import AxisOverride as _AO  # noqa: F401

        assert _AO is not None

    def test_bundle_schema_importable(self):
        from xtrax.inference import BundleSchema as _BS  # noqa: F401

        assert _BS is not None

    def test_synthesize_axes_importable(self):
        from xtrax.inference import synthesize_axes as _sa  # noqa: F401

        assert callable(_sa)

    def test_error_types_importable(self):
        from xtrax.inference import AmbiguousAxisError  # noqa: F401
        from xtrax.inference import AxisRole as _AR  # noqa: F401
        from xtrax.inference import StructureMismatchError as _SME  # noqa: F401

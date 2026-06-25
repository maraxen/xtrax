"""Tests for xtrax.inference.config — AxisOverride + axis_config decorator (E1.4 AC4)."""

from __future__ import annotations

import numpy as np
import pytest

from xtrax.inference.abstract import build_abstract_inputs
from xtrax.inference.axes import synthesize_axes
from xtrax.inference.config import AxisOverride, axis_config, get_axis_config
from xtrax.inference.errors import AxisRole


class TestAxisOverride:
    """AxisOverride frozen dataclass construction and defaults."""

    def test_required_fields_name_and_default_batch_size(self):
        """AxisOverride(name=..., default_batch_size=...) constructs without error."""
        ov = AxisOverride(name="batch", default_batch_size=32)
        assert ov.name == "batch"
        assert ov.default_batch_size == 32

    def test_optional_defaults(self):
        """Unset optional fields have their structural defaults."""
        ov = AxisOverride(name="seq", default_batch_size=8)
        assert ov.cardinality is None
        assert ov.tile_granularity == 1
        assert ov.heterogeneous is False
        assert ov.dedup_eligible is False
        assert ov.bucket_boundaries is None

    def test_all_fields_settable(self):
        """All optional fields can be overridden."""
        ov = AxisOverride(
            name="seq",
            default_batch_size=16,
            cardinality=128,
            tile_granularity=4,
            heterogeneous=True,
            dedup_eligible=True,
            bucket_boundaries=(8, 16, 32),
        )
        assert ov.cardinality == 128
        assert ov.tile_granularity == 4
        assert ov.heterogeneous is True
        assert ov.dedup_eligible is True
        assert ov.bucket_boundaries == (8, 16, 32)

    def test_a3_missing_default_batch_size_raises_type_error(self):
        """A3: AxisOverride without default_batch_size raises TypeError (required field)."""
        with pytest.raises(TypeError):
            AxisOverride(name="batch")  # type: ignore[call-arg]

    def test_a3_missing_name_raises_type_error(self):
        """A3: AxisOverride without name raises TypeError (required field)."""
        with pytest.raises(TypeError):
            AxisOverride(default_batch_size=32)  # type: ignore[call-arg]

    def test_frozen_immutable(self):
        """AxisOverride is frozen — attribute assignment raises FrozenInstanceError."""
        ov = AxisOverride(name="batch", default_batch_size=32)
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
            ov.name = "other"  # type: ignore[misc]


class TestAxisConfigDecorator:
    """axis_config(*overrides) decorator factory."""

    def test_attaches_overrides_to_fn(self):
        """@axis_config attaches the ordered tuple to fn.__xtrax_axis_config__."""
        ov = AxisOverride(name="batch", default_batch_size=32)

        @axis_config(ov)
        def my_fn(x):
            return x

        attached = get_axis_config(my_fn)
        assert attached is not None
        assert len(attached) == 1
        assert attached[0] is ov

    def test_multiple_overrides_ordered(self):
        """Multiple AxisOverrides are preserved in order."""
        ov0 = AxisOverride(name="batch", default_batch_size=32)
        ov1 = AxisOverride(name="seq", default_batch_size=8)

        @axis_config(ov0, ov1)
        def my_fn(x, y):
            return x, y

        attached = get_axis_config(my_fn)
        assert attached is not None
        assert len(attached) == 2
        assert attached[0] is ov0
        assert attached[1] is ov1

    def test_decorator_does_not_change_call_result(self):
        """The decorator must not alter fn's call behavior."""
        sentinel = object()

        @axis_config(AxisOverride(name="batch", default_batch_size=32))
        def my_fn():
            return sentinel

        assert my_fn() is sentinel

    def test_decorator_does_not_change_call_args(self):
        """fn still receives its arguments correctly after decoration."""

        @axis_config(AxisOverride(name="batch", default_batch_size=4))
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_get_axis_config_none_when_no_decorator(self):
        """get_axis_config returns None for a plain function."""

        def plain():
            pass

        assert get_axis_config(plain) is None

    def test_zero_overrides_attaches_empty_tuple(self):
        """@axis_config() with zero args attaches an empty tuple."""

        @axis_config()
        def my_fn():
            pass

        attached = get_axis_config(my_fn)
        assert attached == ()


class TestSynthesizeAxesWithAxisOverride:
    """synthesize_axes with Sequence[AxisOverride] positional overrides (E1.4)."""

    def test_single_override_makes_axis_known(self):
        """AC4: axis 0 with AxisOverride -> role == KNOWN, name == ov.name."""
        ov = AxisOverride(name="batch", default_batch_size=32)
        abstract = build_abstract_inputs({"x": ((8, 4), np.float32)})
        specs = synthesize_axes(abstract, overrides=[ov])
        assert len(specs) == 1
        spec = specs[0]
        assert spec.name == "batch"
        assert spec.default_batch_size == 32
        assert spec.role == AxisRole.KNOWN

    def test_override_cardinality_none_uses_leading_dim(self):
        """When AxisOverride.cardinality is None, cardinality is taken from leading dim."""
        ov = AxisOverride(name="batch", default_batch_size=16, cardinality=None)
        abstract = build_abstract_inputs({"x": ((24, 3), np.float32)})
        specs = synthesize_axes(abstract, overrides=[ov])
        assert specs[0].cardinality == 24  # from leaf.shape[0]

    def test_override_cardinality_explicit_uses_override(self):
        """When AxisOverride.cardinality is set, it overrides the leading dim."""
        ov = AxisOverride(name="batch", default_batch_size=16, cardinality=99)
        abstract = build_abstract_inputs({"x": ((24, 3), np.float32)})
        specs = synthesize_axes(abstract, overrides=[ov])
        assert specs[0].cardinality == 99

    def test_partial_override_other_fields_structural_defaults(self):
        """AC4 partial: fields not set on AxisOverride fall back to structural defaults."""
        ov = AxisOverride(name="batch", default_batch_size=32)
        abstract = build_abstract_inputs({"x": ((8,), np.float32)})
        specs = synthesize_axes(abstract, overrides=[ov])
        spec = specs[0]
        assert spec.tile_granularity == 1
        assert spec.heterogeneous is False
        assert spec.dedup_eligible is False
        assert spec.bucket_boundaries is None

    def test_override_fields_propagate(self):
        """Non-default fields in AxisOverride propagate to AxisSpec."""
        ov = AxisOverride(
            name="seq",
            default_batch_size=8,
            tile_granularity=4,
            heterogeneous=True,
            dedup_eligible=True,
            bucket_boundaries=(8, 16),
        )
        abstract = build_abstract_inputs({"x": ((64,), np.float32)})
        specs = synthesize_axes(abstract, overrides=[ov])
        spec = specs[0]
        assert spec.tile_granularity == 4
        assert spec.heterogeneous is True
        assert spec.dedup_eligible is True
        assert spec.bucket_boundaries == (8, 16)

    def test_partial_positional_override_second_axis_unknown(self):
        """Partial override: axis 0 -> KNOWN, axis 1 has no override -> UNKNOWN."""
        ov = AxisOverride(name="batch", default_batch_size=32)
        abstract = build_abstract_inputs({"x": ((16, 3), np.float32), "y": ((32,), np.int32)})
        specs = synthesize_axes(abstract, overrides=[ov])
        assert len(specs) == 2
        # axis 0 gets override
        known = [s for s in specs if s.role == AxisRole.KNOWN]
        unknown = [s for s in specs if s.role == AxisRole.UNKNOWN]
        assert len(known) == 1
        assert len(unknown) == 1
        assert known[0].name == "batch"

    def test_overrides_none_all_unknown_preserved(self):
        """Invariant D1: with overrides=None, every axis is still UNKNOWN."""
        abstract = build_abstract_inputs({"x": ((16, 3), np.float32), "y": ((32,), np.int32)})
        specs = synthesize_axes(abstract, overrides=None)
        for spec in specs:
            assert spec.role == AxisRole.UNKNOWN

    def test_get_axis_config_wires_into_synthesize_axes(self):
        """AC4 end-to-end: @axis_config on fn -> get_axis_config -> synthesize_axes."""

        @axis_config(AxisOverride(name="batch", default_batch_size=32))
        def my_model(x):
            return x

        abstract = build_abstract_inputs({"x": ((8, 4), np.float32)})
        cfg = get_axis_config(my_model)
        specs = synthesize_axes(abstract, overrides=cfg)
        assert specs[0].role == AxisRole.KNOWN
        assert specs[0].name == "batch"
        assert specs[0].default_batch_size == 32

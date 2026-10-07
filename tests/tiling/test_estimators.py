"""Tests for native-JAX-backed MemoryBudget estimator helpers.

Spec: .praxia/docs/specs/260706_joint-budget-batch-planner.md (AC12-AC13).
"""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling import (
    device_memory_budget,
    estimate_memory_theoretical,
    lowered_memory_estimate,
)


class _FakeDevice:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats

    def __repr__(self) -> str:
        return "FakeDevice"


class TestDeviceMemoryBudget:
    def test_reads_bytes_limit_with_fraction(self) -> None:
        device = _FakeDevice({"bytes_limit": 1_000})
        assert device_memory_budget(fraction=0.9, device=device) == 900

    def test_fractional_budget_truncates(self) -> None:
        device = _FakeDevice({"bytes_limit": 7})
        assert device_memory_budget(fraction=0.5, device=device) == 3

    def test_full_fraction(self) -> None:
        device = _FakeDevice({"bytes_limit": 1_000})
        assert device_memory_budget(fraction=1.0, device=device) == 1_000

    def test_invalid_fraction_rejected(self) -> None:
        device = _FakeDevice({"bytes_limit": 1_000})
        with pytest.raises(ValueError, match="fraction"):
            device_memory_budget(fraction=0.0, device=device)
        with pytest.raises(ValueError, match="fraction"):
            device_memory_budget(fraction=1.5, device=device)

    def test_missing_stats_fails_loud(self) -> None:
        with pytest.raises(RuntimeError, match="memory_stats"):
            device_memory_budget(device=_FakeDevice(None))
        with pytest.raises(RuntimeError, match="memory_stats"):
            device_memory_budget(device=_FakeDevice({}))


class TestLoweredMemoryEstimate:
    def test_returns_xla_buffer_assignment_bytes(self) -> None:
        # (64, 64) float32 matmul: argument 16 KiB; output + temps backend-set.
        estimate = lowered_memory_estimate(
            lambda x: (x @ x.T).sum(),
            jax.ShapeDtypeStruct((64, 64), jnp.float32),
        )
        assert isinstance(estimate, int)
        assert estimate >= 64 * 64 * 4  # at least the argument buffer

    def test_scales_with_input_shape(self) -> None:
        small = lowered_memory_estimate(
            lambda x: x + 1.0, jax.ShapeDtypeStruct((128,), jnp.float32)
        )
        large = lowered_memory_estimate(
            lambda x: x + 1.0, jax.ShapeDtypeStruct((128 * 1024,), jnp.float32)
        )
        assert large > small

    def test_accepts_multiple_abstract_args(self) -> None:
        estimate = lowered_memory_estimate(
            lambda x, y: x + y,
            jax.ShapeDtypeStruct((32,), jnp.float32),
            jax.ShapeDtypeStruct((32,), jnp.float32),
        )
        assert estimate > 0

    def test_no_memory_analysis_fails_loud(self, monkeypatch) -> None:
        class _FakeCompiled:
            def memory_analysis(self):
                return None

        class _FakeLowered:
            def compile(self):
                return _FakeCompiled()

        class _FakeJitted:
            def lower(self, *args):
                return _FakeLowered()

        class _FakeJax:
            def jit(self, fn):
                return _FakeJitted()

        import xtrax.tiling.estimators as estimators_module

        monkeypatch.setattr(estimators_module, "jax", _FakeJax())
        with pytest.raises(RuntimeError, match="memory_analysis"):
            lowered_memory_estimate(lambda x: x, object())

    def test_analysis_missing_size_attrs_fails_loud(self, monkeypatch) -> None:
        class _AttrlessAnalysis:
            pass

        class _FakeCompiled:
            def memory_analysis(self):
                return _AttrlessAnalysis()

        class _FakeLowered:
            def compile(self):
                return _FakeCompiled()

        class _FakeJitted:
            def lower(self, *args):
                return _FakeLowered()

        class _FakeJax:
            def jit(self, fn):
                return _FakeJitted()

        import xtrax.tiling.estimators as estimators_module

        monkeypatch.setattr(estimators_module, "jax", _FakeJax())
        with pytest.raises(RuntimeError, match="memory_analysis"):
            lowered_memory_estimate(lambda x: x, object())


class TestEstimateMemoryTheoretical:
    """Product-of-extents estimator (domain-free)."""

    def test_product_of_extents_times_bytes(self) -> None:
        assert estimate_memory_theoretical({"a": 4, "b": 8}, 10) == 320

    def test_dtype_bytes_scales_value_count(self) -> None:
        assert estimate_memory_theoretical({"n": 10, "d": 8}, 1, dtype_bytes=4) == 320

    def test_activation_multiplier(self) -> None:
        assert estimate_memory_theoretical({"n": 10}, 4, activation_multiplier=2.5) == 100

    def test_empty_extents_is_the_base_element(self) -> None:
        assert estimate_memory_theoretical({}, 16, dtype_bytes=4) == 64

    def test_extent_order_does_not_matter(self) -> None:
        left = estimate_memory_theoretical({"a": 3, "b": 5}, 2)
        right = estimate_memory_theoretical({"b": 5, "a": 3}, 2)
        assert left == right == 30

    def test_zero_extent_is_zero(self) -> None:
        assert estimate_memory_theoretical({"n": 0, "d": 8}, 4) == 0

    def test_negative_extent_rejected(self) -> None:
        with pytest.raises(ValueError, match="extent"):
            estimate_memory_theoretical({"n": -1}, 4)

    def test_bool_extent_rejected(self) -> None:
        with pytest.raises(TypeError, match="extent"):
            estimate_memory_theoretical({"n": True}, 4)

    def test_negative_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes_per_element"):
            estimate_memory_theoretical({"n": 2}, -1)

    def test_truncates_fractional_product_toward_zero(self) -> None:
        # 3 * 0.5 = 1.5. int() yields 1; round() would yield 2.
        assert estimate_memory_theoretical({"n": 3}, 0.5) == 1

    def test_negative_activation_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="activation_multiplier"):
            estimate_memory_theoretical({"n": 3}, 4, activation_multiplier=-1)

    def test_bool_and_str_bytes_per_element_rejected(self) -> None:
        with pytest.raises(TypeError, match="bytes_per_element"):
            estimate_memory_theoretical({"n": 3}, True)
        with pytest.raises(TypeError, match="bytes_per_element"):
            estimate_memory_theoretical({"n": 3}, "4")

    def test_non_int_dtype_bytes_rejected(self) -> None:
        with pytest.raises(TypeError, match="dtype_bytes"):
            estimate_memory_theoretical({"n": 3}, 4, dtype_bytes=1.5)
        with pytest.raises(TypeError, match="dtype_bytes"):
            estimate_memory_theoretical({"n": 3}, 4, dtype_bytes=True)
        with pytest.raises(TypeError, match="dtype_bytes"):
            estimate_memory_theoretical({"n": 3}, 4, dtype_bytes="4")

    def test_non_positive_dtype_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="dtype_bytes"):
            estimate_memory_theoretical({"n": 2}, 4, dtype_bytes=0)

    def test_importable_from_tiling(self) -> None:
        import xtrax.tiling as tiling

        assert tiling.estimate_memory_theoretical is estimate_memory_theoretical

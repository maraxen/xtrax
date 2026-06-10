"""Tests for safe_scan transform."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.transforms import safe_scan


class TestSafeScanBasic:
    """Test basic safe_scan functionality matching jax.lax.scan."""

    def test_normal_scan_matches_jax_scan(self):
        """Normal scan should match jax.lax.scan exactly."""

        def add_carry(carry, x):
            """Add x to carry, output carry."""
            return carry + x, carry + x

        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Our safe_scan
        final_carry, ys = safe_scan(add_carry, init, xs)

        # Reference jax.lax.scan
        ref_final_carry, ref_ys = jax.lax.scan(add_carry, init, xs)

        assert jnp.allclose(final_carry, ref_final_carry)
        assert jnp.allclose(ys, ref_ys)

    def test_scan_with_pytree_carry(self):
        """Scan with pytree carry should work."""

        def step(carry, x):
            """Update carry dict with new values."""
            new_carry = {
                "sum": carry["sum"] + x,
                "count": carry["count"] + 1,
            }
            return new_carry, new_carry["sum"]

        init = {"sum": jnp.array(0.0), "count": jnp.array(0)}
        xs = jnp.array([1.0, 2.0, 3.0])

        final_carry, ys = safe_scan(step, init, xs)

        assert jnp.allclose(final_carry["sum"], jnp.array(6.0))
        assert jnp.allclose(final_carry["count"], jnp.array(3))
        assert jnp.allclose(ys, jnp.array([1.0, 3.0, 6.0]))

    def test_scan_with_none_carry_is_legal(self):
        """init=None should be a legal carry (not rejected by safe_scan)."""

        def step(carry, x):
            """No-op carry, just accumulate outputs."""
            return None, x * 2

        init = None
        xs = jnp.array([1.0, 2.0, 3.0])

        # Should not raise
        final_carry, ys = safe_scan(step, init, xs)

        assert final_carry is None
        assert jnp.allclose(ys, jnp.array([2.0, 4.0, 6.0]))


class TestSafeScanEmptyInput:
    """Test safe_scan with empty xs."""

    def test_empty_xs_inferred_length_raises_valueerror(self):
        """Inferred length 0 should raise ValueError before tracing."""

        def dummy_step(carry, x):
            return carry, x

        init = jnp.array(0.0)
        xs = jnp.array([], dtype=jnp.float32)  # shape (0,)

        with pytest.raises(ValueError, match="safe_scan.*length 0"):
            safe_scan(dummy_step, init, xs)

    def test_explicit_length_zero_raises_valueerror(self):
        """Explicit length=0 should raise ValueError before tracing."""

        def dummy_step(carry, x):
            return carry, x

        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0])  # actual data, but we override length

        with pytest.raises(ValueError, match="safe_scan.*length 0"):
            safe_scan(dummy_step, init, xs, length=0)


class TestSafeScanReverse:
    """Test safe_scan with reverse=True."""

    def test_reverse_true_matches_jax(self):
        """reverse=True should match jax.lax.scan behavior."""

        def step(carry, x):
            return carry + x, x

        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Our safe_scan
        final_carry, ys = safe_scan(step, init, xs, reverse=True)

        # Reference
        ref_final_carry, ref_ys = jax.lax.scan(step, init, xs, reverse=True)

        assert jnp.allclose(final_carry, ref_final_carry)
        assert jnp.allclose(ys, ref_ys)


class TestSafeScanUnroll:
    """Test safe_scan with unroll parameter."""

    def test_unroll_1_default_matches_no_unroll(self):
        """unroll=1 (default) should match unroll behavior."""

        def step(carry, x):
            return carry * x, carry * x

        init = jnp.array(1.0)
        xs = jnp.array([2.0, 3.0, 4.0])

        result_no_unroll = safe_scan(step, init, xs, unroll=1)
        result_default = safe_scan(step, init, xs)

        final_carry_no, ys_no = result_no_unroll
        final_carry_def, ys_def = result_default

        assert jnp.allclose(final_carry_no, final_carry_def)
        assert jnp.allclose(ys_no, ys_def)

    def test_unroll_greater_than_1_matches_jax(self):
        """unroll>1 should match jax behavior."""

        def step(carry, x):
            return carry + x, carry + x

        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Our safe_scan with unroll=2
        final_carry, ys = safe_scan(step, init, xs, unroll=2)

        # Reference with unroll=2
        ref_final_carry, ref_ys = jax.lax.scan(step, init, xs, unroll=2)

        assert jnp.allclose(final_carry, ref_final_carry)
        assert jnp.allclose(ys, ref_ys)


class TestSafeScanPytree:
    """Test safe_scan with pytree inputs."""

    def test_pytree_xs_structure_preserved(self):
        """Pytree xs structure should be preserved in output."""

        def step(carry, x):
            """x is a dict, output is same structure."""
            new_carry = carry + x["value"]
            return new_carry, {"doubled": x["value"] * 2, "orig": x["value"]}

        init = jnp.array(0.0)
        xs = {
            "value": jnp.array([1.0, 2.0, 3.0]),
        }

        final_carry, ys = safe_scan(step, init, xs)

        assert final_carry.shape == ()
        assert "doubled" in ys
        assert "orig" in ys
        assert jnp.allclose(ys["doubled"], jnp.array([2.0, 4.0, 6.0]))
        assert jnp.allclose(ys["orig"], jnp.array([1.0, 2.0, 3.0]))

    def test_nested_pytree_xs(self):
        """Nested pytree xs should work."""

        def step(carry, x):
            val = x["nested"]["inner"] + x["value"]
            return carry + val, val

        init = jnp.array(0.0)
        xs = {
            "value": jnp.array([1.0, 2.0]),
            "nested": {
                "inner": jnp.array([10.0, 20.0]),
            },
        }

        final_carry, ys = safe_scan(step, init, xs)

        expected_ys = jnp.array([11.0, 22.0])
        assert jnp.allclose(ys, expected_ys)

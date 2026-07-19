"""Tests for tiling iterators: VmapIterator, SafeMapIterator, BucketIterator."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling.iterator import (
    BucketIterator,
    SafeMapIterator,
    VmapIterator,
    WhileLoopIterator,
)


class TestVmapIterator:
    """Test VmapIterator shape and behavior."""

    def test_vmap_iterator_yields_items(self):
        """VmapIterator should yield individual items from vmapped result."""

        def fn(x):
            return x * 2

        xs = jnp.arange(4)  # shape (4,)

        iterator = VmapIterator()
        results = iterator(fn, xs)

        # Results should have shape (4,) after vmap reduces leading dim
        assert results.shape == (4,), f"Expected shape (4,), got {results.shape}"

        # Check values
        expected = jax.vmap(fn)(xs)
        assert jnp.allclose(results, expected)

    def test_vmap_iterator_pytree_shape(self):
        """VmapIterator should handle pytree inputs."""

        def fn(x):
            return {"y": x["y"] * 2, "z": x["z"] + 1}

        xs = {"y": jnp.arange(3).reshape(3, 1), "z": jnp.ones((3, 2))}

        iterator = VmapIterator()
        results = iterator(fn, xs)

        # Results should be a dict with same structure
        assert isinstance(results, dict)
        assert "y" in results and "z" in results
        assert results["y"].shape == (3, 1)
        assert results["z"].shape == (3, 2)


class TestSafeMapIterator:
    """Test SafeMapIterator shape, divisibility, and equivalence."""

    def test_safe_map_iterator_equals_vmap_when_batch_size_gte_n(self):
        """SafeMapIterator with tile >= n should equal VmapIterator."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)

        vmap_iter = VmapIterator()
        vmap_result = vmap_iter(fn, xs)

        safe_iter = SafeMapIterator(tile=20)
        safe_result = safe_iter(fn, xs)

        assert jnp.allclose(vmap_result, safe_result)

    def test_safe_map_iterator_batch_size_equals_n(self):
        """SafeMapIterator with tile == n should equal VmapIterator."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)

        vmap_iter = VmapIterator()
        vmap_result = vmap_iter(fn, xs)

        safe_iter = SafeMapIterator(tile=10)
        safe_result = safe_iter(fn, xs)

        assert jnp.allclose(vmap_result, safe_result)

    def test_safe_map_iterator_non_divisible_raises_valueerror(self):
        """SafeMapIterator should propagate ValueError for non-divisible n."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)  # n=10

        safe_iter = SafeMapIterator(tile=3)  # 10 % 3 != 0

        with pytest.raises(ValueError, match="not divisible"):
            safe_iter(fn, xs)

    def test_safe_map_iterator_divisible_batch(self):
        """SafeMapIterator should work with divisible batch sizes."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)

        safe_iter = SafeMapIterator(tile=5)
        result = safe_iter(fn, xs)

        expected = jax.vmap(fn)(xs)
        assert jnp.allclose(result, expected)


class TestBucketIterator:
    """Test BucketIterator construction and ValueError."""

    def test_bucket_iterator_construction_validation(self):
        """
        BucketIterator raises ValueError if len(batch_sizes) !=
        len(boundaries) + 1.
        """

        def fn(x):
            return x

        xs = jnp.arange(100)

        # boundaries has 2 elements, so batch_sizes should have 3
        with pytest.raises(ValueError):
            BucketIterator(
                boundaries=[10, 20],
                batch_sizes=[8, 16],  # Only 2 elements, should be 3
                fn=fn,
                xs=xs,
            )

    def test_bucket_iterator_valid_construction(self):
        """BucketIterator constructs with correct batch_sizes length."""

        def fn(x):
            return x

        xs = jnp.arange(100)

        # boundaries has 2 elements, batch_sizes has 3
        bucket_iter = BucketIterator(
            boundaries=[10, 20],
            batch_sizes=[8, 16, 32],
            fn=fn,
            xs=xs,
        )

        assert bucket_iter is not None

    def test_bucket_iterator_single_boundary(self):
        """BucketIterator with single boundary and two batch sizes."""

        def fn(x):
            return x

        xs = jnp.arange(100)

        bucket_iter = BucketIterator(
            boundaries=[50],
            batch_sizes=[8, 16],
            fn=fn,
            xs=xs,
        )

        assert bucket_iter is not None

    def test_bucket_iterator_empty_boundaries(self):
        """BucketIterator with no boundaries and one batch size."""

        def fn(x):
            return x

        xs = jnp.arange(100)

        bucket_iter = BucketIterator(
            boundaries=[],
            batch_sizes=[8],
            fn=fn,
            xs=xs,
        )

        assert bucket_iter is not None

    def test_bucket_iterator_pads_to_boundary(self):
        """BucketIterator pads input to smallest bucket >= seq_len."""

        def fn(x):
            return x * 2

        input_data = jnp.arange(100)  # seq_len = 100
        bucket_iter = BucketIterator(
            boundaries=[128, 256, 512],
            batch_sizes=[8, 4, 2, 1],
            fn=fn,
            xs=input_data,
        )

        # Collect all yields
        results = list(bucket_iter)
        assert len(results) == 1, f"Expected 1 yield, got {len(results)}"

        result, original_length_mask = results[0]

        # Should pad to 128 (smallest boundary >= 100)
        assert result.shape[0] == 128, f"Expected padded shape (128,), got {result.shape}"

        # Check mask: first 100 should be True, rest False
        assert jnp.sum(original_length_mask) == 100
        assert jnp.all(original_length_mask[:100])
        assert jnp.all(~original_length_mask[100:])

    def test_bucket_iterator_exact_boundary(self):
        """BucketIterator accepts exact boundary (pad_amount=0)."""

        def fn(x):
            return x * 2

        input_data = jnp.arange(128)  # seq_len = 128
        bucket_iter = BucketIterator(
            boundaries=[128, 256, 512],
            batch_sizes=[8, 4, 2, 1],
            fn=fn,
            xs=input_data,
        )

        results = list(bucket_iter)
        assert len(results) == 1

        result, original_length_mask = results[0]

        # Should not pad (exact match)
        assert result.shape[0] == 128
        assert jnp.all(original_length_mask)  # All True

    def test_bucket_iterator_exact_max_boundary(self):
        """BucketIterator accepts input at max boundary without error."""

        def fn(x):
            return x * 2

        input_data = jnp.arange(256)  # seq_len = 256
        bucket_iter = BucketIterator(
            boundaries=[128, 256, 512],
            batch_sizes=[8, 4, 2, 1],
            fn=fn,
            xs=input_data,
        )

        results = list(bucket_iter)
        assert len(results) == 1

        result, original_length_mask = results[0]

        # Should pad to 256 (exact match, no padding)
        assert result.shape[0] == 256
        assert jnp.all(original_length_mask)

    def test_bucket_iterator_exceeds_max_raises(self):
        """BucketIterator raises ValueError when input exceeds max boundary."""

        def fn(x):
            return x

        input_data = jnp.arange(600)  # seq_len = 600, max boundary = 512
        bucket_iter = BucketIterator(
            boundaries=[128, 256, 512],
            batch_sizes=[4, 2, 1, 1],
            fn=fn,
            xs=input_data,
        )

        # Should raise ValueError AND emit UserWarning
        with pytest.warns(UserWarning, match="exceeds maximum bucket size"):
            with pytest.raises(ValueError, match="exceeds maximum bucket size"):
                list(bucket_iter)

    def test_bucket_iterator_empty_xs_yields_nothing(self):
        """BucketIterator with empty pytree yields nothing."""

        def fn(x):
            return x

        empty_xs = {}
        bucket_iter = BucketIterator(
            boundaries=[128, 256, 512],
            batch_sizes=[8, 4, 2, 1],
            fn=fn,
            xs=empty_xs,
        )

        results = list(bucket_iter)
        assert len(results) == 0


class TestWhileLoopIterator:
    """Test WhileLoopIterator: carry-bearing, no output collection."""

    def test_while_loop_iterator_counts_to_n(self):
        """WhileLoopIterator runs body until cond is False, returning only final carry."""

        def cond(carry):
            return carry < 5

        def body(carry):
            return carry + 1

        iterator = WhileLoopIterator()
        result = iterator(cond, body, jnp.array(0))

        assert result == 5

    def test_while_loop_iterator_matches_jax_lax_while_loop(self):
        """WhileLoopIterator(cond, body, init) is equivalent to jax.lax.while_loop directly."""

        def cond(carry):
            return carry < 10

        def body(carry):
            return carry * 2 + 1

        iterator = WhileLoopIterator()
        result = iterator(cond, body, jnp.array(0))
        expected = jax.lax.while_loop(cond, body, jnp.array(0))

        assert result == expected

    def test_while_loop_iterator_returns_only_final_carry_no_ys(self):
        """Unlike JaxScanIterator, the return value is a single carry, not (carry, ys)."""

        def cond(carry):
            return carry < 3

        def body(carry):
            return carry + 1

        iterator = WhileLoopIterator()
        result = iterator(cond, body, jnp.array(0))

        # A bare scalar array, not a 2-tuple (final_carry, ys).
        assert not isinstance(result, tuple)
        assert result.shape == ()

    def test_while_loop_iterator_pytree_carry(self):
        """WhileLoopIterator handles a pytree carry, mirroring prolix's (step_i, state) shape."""

        def cond(carry):
            step_i, _ = carry
            return step_i < 4

        def body(carry):
            step_i, total = carry
            return (step_i + 1, total + step_i)

        iterator = WhileLoopIterator()
        final_step_i, final_total = iterator(cond, body, (jnp.array(0), jnp.array(0)))

        assert final_step_i == 4
        assert final_total == 0 + 1 + 2 + 3

    def test_while_loop_iterator_zero_iterations(self):
        """WhileLoopIterator with a cond that is immediately False returns init unchanged."""

        def cond(carry):
            return carry < 0

        def body(carry):
            return carry + 100  # would change the result if ever actually executed

        iterator = WhileLoopIterator()
        result = iterator(cond, body, jnp.array(0))

        assert result == 0

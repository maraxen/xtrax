"""Tests for tiling iterators: VmapIterator, SafeMapIterator, BucketIterator."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling.iterator import BucketIterator, SafeMapIterator, VmapIterator


class TestVmapIterator:
    """Test VmapIterator shape and behavior."""

    def test_vmap_iterator_yields_items(self):
        """VmapIterator should yield individual items from vmapped result."""
        fn = lambda x: x * 2
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
        fn = lambda x: {"y": x["y"] * 2, "z": x["z"] + 1}
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
        """SafeMapIterator with batch_size >= n should equal VmapIterator."""
        fn = lambda x: x * 2
        xs = jnp.arange(10)

        vmap_iter = VmapIterator()
        vmap_result = vmap_iter(fn, xs)

        safe_iter = SafeMapIterator(batch_size=20)
        safe_result = safe_iter(fn, xs)

        assert jnp.allclose(vmap_result, safe_result)

    def test_safe_map_iterator_batch_size_equals_n(self):
        """SafeMapIterator with batch_size == n should equal VmapIterator."""
        fn = lambda x: x * 2
        xs = jnp.arange(10)

        vmap_iter = VmapIterator()
        vmap_result = vmap_iter(fn, xs)

        safe_iter = SafeMapIterator(batch_size=10)
        safe_result = safe_iter(fn, xs)

        assert jnp.allclose(vmap_result, safe_result)

    def test_safe_map_iterator_non_divisible_raises_valueerror(self):
        """SafeMapIterator should propagate ValueError for non-divisible n."""
        fn = lambda x: x * 2
        xs = jnp.arange(10)  # n=10

        safe_iter = SafeMapIterator(batch_size=3)  # 10 % 3 != 0

        with pytest.raises(ValueError, match="not divisible"):
            safe_iter(fn, xs)

    def test_safe_map_iterator_divisible_batch(self):
        """SafeMapIterator should work with divisible batch sizes."""
        fn = lambda x: x * 2
        xs = jnp.arange(10)

        safe_iter = SafeMapIterator(batch_size=5)
        result = safe_iter(fn, xs)

        expected = jax.vmap(fn)(xs)
        assert jnp.allclose(result, expected)


class TestBucketIterator:
    """Test BucketIterator construction and ValueError."""

    def test_bucket_iterator_construction_validation(self):
        """BucketIterator should raise ValueError if len(batch_sizes) != len(boundaries) + 1."""
        fn = lambda x: x
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
        """BucketIterator should construct successfully with correct batch_sizes length."""
        fn = lambda x: x
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
        fn = lambda x: x
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
        fn = lambda x: x
        xs = jnp.arange(100)

        bucket_iter = BucketIterator(
            boundaries=[],
            batch_sizes=[8],
            fn=fn,
            xs=xs,
        )

        assert bucket_iter is not None

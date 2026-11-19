"""Tests for BucketDispatcher host-side batch coordinator (#3194)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.bucket import BucketDispatcher


class TestBucketDispatcher:
    """Positive and negative verification controls for BucketDispatcher."""

    def test_dispatch_heterogeneous_lengths_reassembled_exact(self):
        """Dispatch items with variable lengths; results must match 1:1 in original order."""

        # Function: double array and return both doubled array and its sum
        def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            return x * 2, jnp.sum(x)

        dispatcher = BucketDispatcher(fn=fn, boundaries=(16, 32, 64), axis=0)

        # Create 4 items with distinct lengths (5, 20, 12, 30)
        items = [
            jnp.ones((5, 4), dtype=jnp.float32) * 1.0,
            jnp.ones((20, 4), dtype=jnp.float32) * 2.0,
            jnp.ones((12, 4), dtype=jnp.float32) * 3.0,
            jnp.ones((30, 4), dtype=jnp.float32) * 4.0,
        ]

        results = dispatcher.dispatch_batch(items)
        assert len(results) == len(items)

        for orig, (out_arr, out_sum) in zip(items, results, strict=True):
            # Trimmed output array must match original shape
            assert out_arr.shape == orig.shape
            np.testing.assert_allclose(out_arr, orig * 2)
            # Scalar output must match sum of unpadded elements
            expected_sum = float(jnp.sum(orig))
            np.testing.assert_allclose(float(out_sum), expected_sum, rtol=1e-5)

    def test_runner_is_cached(self):
        """_get_runner caches compiled JIT executables per bucket size (Finding C3)."""

        def fn(x: jax.Array) -> jax.Array:
            return x + 1.0

        dispatcher = BucketDispatcher(fn=fn, boundaries=(8, 16))
        runner_8_first = dispatcher._get_runner(8)
        runner_8_second = dispatcher._get_runner(8)
        assert runner_8_first is runner_8_second

        runner_16 = dispatcher._get_runner(16)
        assert runner_16 is not runner_8_first
        assert 8 in dispatcher._cached_runners
        assert 16 in dispatcher._cached_runners

    def test_empty_batch_returns_empty_list(self):
        """Dispatching an empty sequence returns an empty list without error."""
        dispatcher = BucketDispatcher(fn=lambda x: x)
        assert dispatcher.dispatch_batch([]) == []

    def test_non_zero_axis_padding_and_trimming(self):
        """BucketDispatcher handles axis != 0 without dimension mismatch (Finding C4)."""

        def fn(x: jax.Array) -> jax.Array:
            return x * 3.0

        dispatcher = BucketDispatcher(fn=fn, boundaries=(10, 20), axis=1)

        items = [
            jnp.ones((4, 7), dtype=jnp.float32),
            jnp.ones((4, 15), dtype=jnp.float32),
        ]

        results = dispatcher.dispatch_batch(items)
        assert len(results) == 2
        assert results[0].shape == (4, 7)
        assert results[1].shape == (4, 15)
        np.testing.assert_allclose(results[0], items[0] * 3.0)
        np.testing.assert_allclose(results[1], items[1] * 3.0)

    def test_exceeding_max_boundary_raises_value_error(self):
        """Input length exceeding largest bucket boundary raises ValueError."""
        dispatcher = BucketDispatcher(fn=lambda x: x, boundaries=(10, 20))
        oversized = [jnp.ones((25, 4))]
        with pytest.raises(ValueError, match="exceeds the largest bucket boundary"):
            dispatcher.dispatch_batch(oversized)

    def test_empty_or_non_ascending_boundaries_raise_value_error(self):
        """Empty boundaries or non-strictly ascending boundaries raise ValueError."""
        with pytest.raises(ValueError, match="boundaries must not be empty"):
            BucketDispatcher(fn=lambda x: x, boundaries=())

        with pytest.raises(ValueError, match="strictly ascending"):
            BucketDispatcher(fn=lambda x: x, boundaries=(20, 10))

        with pytest.raises(ValueError, match="strictly ascending"):
            BucketDispatcher(fn=lambda x: x, boundaries=(10, 10))

    def test_item_with_no_array_leaves_raises_value_error(self):
        """Item containing no array leaves raises ValueError."""
        dispatcher = BucketDispatcher(fn=lambda x: x, boundaries=(10, 20))
        with pytest.raises(ValueError, match="contains no array leaves"):
            dispatcher.dispatch_batch([{}])

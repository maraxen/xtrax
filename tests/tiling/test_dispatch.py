"""Tests for make_axis_dispatch and all strategy variants."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap


class TestVmapDispatch:
    """Test Vmap strategy dispatch."""

    def test_vmap_dispatch_simple(self):
        """Vmap dispatches to jax.vmap(fn)."""

        def fn(x):
            return x * 2

        xs = jnp.arange(5)  # shape: (5,)
        strategy = Vmap()
        result = make_axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_vmap_dispatch_pytree(self):
        """Vmap works with nested pytrees."""

        def fn(carry):
            a, b = carry
            return (a * 2, b + 1)

        xs = (jnp.arange(5), jnp.ones(5))  # tuple pytree
        strategy = Vmap()
        result = make_axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result[0], expected[0])
        assert jnp.allclose(result[1], expected[1])


class TestSafeMapDispatch:
    """Test SafeMap strategy dispatch."""

    def test_safemap_no_chunking_when_small(self):
        """SafeMap with batch_size >= n uses vmap."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)
        strategy = SafeMap(batch_size=50)
        result = make_axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_safemap_chunking_divisible(self):
        """SafeMap with divisible chunking matches vmap."""

        def fn(x):
            return x * 3

        xs = jnp.arange(100)
        strategy = SafeMap(batch_size=25)
        result = make_axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_safemap_non_divisible_raises(self):
        """SafeMap with non-divisible batch raises ValueError."""

        def fn(x):
            return x * 2

        xs = jnp.arange(100)
        strategy = SafeMap(batch_size=30)

        with pytest.raises(ValueError, match="safe_map.*not divisible"):
            make_axis_dispatch(strategy, fn, xs)


class TestScanDispatch:
    """Test Scan strategy dispatch."""

    def test_scan_init_field_accessible(self):
        """Scan.init field should be accessible (backwards compat)."""
        init_carry = {"counter": 0}
        strategy = Scan(transition=None, init=init_carry)
        assert strategy.init == init_carry

    def test_scan_construction_without_args(self):
        """Scan() with no arguments should work (backwards compat)."""
        strategy = Scan()
        assert strategy.transition is None
        assert strategy.init is None

    def test_scan_dispatch_with_init(self):
        """Scan dispatch uses strategy.transition and init."""

        def transition(carry, x):
            return carry + x, x * 2

        init = jnp.array(0.0)
        xs = jnp.arange(5, dtype=jnp.float32)
        strategy = Scan(transition=transition)

        # fn is ignored for Scan; we pass None as placeholder
        result_carry, result_ys = make_axis_dispatch(strategy, None, xs, init=init)

        # Verify the result is correct: final carry should be sum of xs
        expected_carry = jnp.sum(xs)
        assert jnp.allclose(result_carry, expected_carry)
        assert jnp.allclose(result_ys, xs * 2)

    def test_scan_dispatch_requires_transition(self):
        """Scan dispatch raises ValueError if transition is None."""
        xs = jnp.arange(5, dtype=jnp.float32)
        init = jnp.array(0.0)
        strategy = Scan(transition=None, init=init)

        with pytest.raises(ValueError, match="Scan strategy requires.*transition"):
            make_axis_dispatch(strategy, None, xs, init=None)

    def test_scan_dispatch_requires_init(self):
        """Scan dispatch raises ValueError if init is None."""

        def transition(carry, x):
            return carry + x, x * 2

        xs = jnp.arange(5, dtype=jnp.float32)
        strategy = Scan(transition=transition)

        with pytest.raises(ValueError, match="Scan strategy requires.*init"):
            make_axis_dispatch(strategy, None, xs, init=None)

    def test_scan_dispatch_ignores_fn(self):
        """Scan dispatch ignores the fn parameter."""

        def transition(carry, x):
            return carry + x, x * 2

        def ignored_fn(carry, x):
            # This should never be called
            raise RuntimeError("fn should be ignored for Scan")

        init = jnp.array(0.0)
        xs = jnp.arange(5, dtype=jnp.float32)
        strategy = Scan(transition=transition)

        # Should work fine; fn is not called
        result_carry, result_ys = make_axis_dispatch(
            strategy, ignored_fn, xs, init=init
        )
        assert jnp.allclose(result_carry, jnp.sum(xs))


class TestDedupGatherDispatch:
    """Test DedupGather strategy dispatch."""

    def test_dedupgather_dispatch(self):
        """DedupGather dispatches through dedup -> map -> gather."""

        # Simple dedup: group by unique values
        def dedup_fn(xs):
            """Returns (unique_xs, indices_to_gather)."""
            unique, indices = jnp.unique(xs, return_inverse=True)
            return unique, indices

        def gather_fn(ys, indices):
            """Scatter back to original shape using indices."""
            return ys[indices]

        def fn(x):
            return x * 2

        xs = jnp.array([1, 2, 1, 3, 2, 1])  # has duplicates
        strategy = DedupGather(dedup_fn=dedup_fn, gather_fn=gather_fn, k_bucket=8)

        result = make_axis_dispatch(strategy, fn, xs)

        # Expected: dedup to [1, 2, 3], map to [2, 4, 6],
        # gather back to [2, 4, 2, 6, 4, 2]
        expected = jnp.array([2, 4, 2, 6, 4, 2])
        assert jnp.allclose(result, expected)

    def test_dedupgather_unpacking(self):
        """DedupGather correctly unpacks dedup_fn output."""

        def dedup_fn(xs):
            # Return exactly two values
            unique = jnp.unique(xs)
            indices = jnp.searchsorted(unique, xs)
            return unique, indices

        def gather_fn(ys, indices):
            return ys[indices]

        def fn(x):
            return x + 10

        xs = jnp.array([5, 5, 3, 3, 7])
        strategy = DedupGather(dedup_fn=dedup_fn, gather_fn=gather_fn, k_bucket=8)

        result = make_axis_dispatch(strategy, fn, xs)
        # unique: [3, 5, 7], indices: [1, 1, 0, 0, 2]
        # mapped: [13, 15, 17]
        # gathered: [15, 15, 13, 13, 17]
        expected = jnp.array([15, 15, 13, 13, 17])
        assert jnp.allclose(result, expected)


class TestBucketDispatch:
    """Bucket is host-side and must not execute in the device-tier dispatch."""

    def test_bucket_dispatch_raises_with_guidance(self):
        """make_axis_dispatch(Bucket, ...) raises TypeError pointing to host helpers."""

        def fn(x):
            return x * 2

        xs = jnp.arange(5)
        strategy = Bucket(boundaries=(8, 16))

        with pytest.raises(TypeError, match="host-side"):
            make_axis_dispatch(strategy, fn, xs)

    def test_bucket_dispatch_error_mentions_bucketize(self):
        """The error directs callers to select_bucket/bucketize."""

        def fn(x):
            return x

        xs = jnp.arange(3)
        strategy = Bucket(boundaries=(4,))

        with pytest.raises(TypeError, match="bucketize"):
            make_axis_dispatch(strategy, fn, xs)


class TestDispatchExhaustiveness:
    """Test that all strategy types are handled."""

    def test_vmap_isinstance(self):
        """Vmap is recognized."""
        strategy = Vmap()
        assert isinstance(strategy, Vmap)

    def test_safemap_isinstance(self):
        """SafeMap is recognized."""
        strategy = SafeMap(batch_size=32)
        assert isinstance(strategy, SafeMap)

    def test_scan_isinstance(self):
        """Scan is recognized."""

        def transition(carry, x):
            return carry, x

        strategy = Scan(transition=transition)
        assert isinstance(strategy, Scan)

    def test_dedupgather_isinstance(self):
        """DedupGather is recognized."""

        def dedup_fn(xs):
            return xs, jnp.arange(len(xs))

        def gather_fn(ys, indices):
            return ys

        strategy = DedupGather(dedup_fn=dedup_fn, gather_fn=gather_fn, k_bucket=8)
        assert isinstance(strategy, DedupGather)


class TestDispatchIntegration:
    """Integration tests combining strategies."""

    def test_dispatch_with_matrix_multiplication(self):
        """Test dispatch with a real JAX operation (matmul)."""
        W = jnp.ones((10, 10))
        xs = jnp.ones((5, 10))

        def fn(x):
            return jnp.matmul(x, W)

        strategy = Vmap()
        result = make_axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_dispatch_safemap_with_neural_net_like(self):
        """Test SafeMap with neural net-like operation."""

        def fn(x):
            return jax.nn.relu(x)

        xs = jnp.linspace(-1, 1, 100)
        strategy = SafeMap(batch_size=25)
        result = make_axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

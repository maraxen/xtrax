"""Tests for make_axis_dispatch (factory) and axis_dispatch (backward-compat eager API)."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling.dispatch import DispatchRejected, axis_dispatch, make_axis_dispatch
from xtrax.tiling.iterator import (
    JaxScanIterator,
    SafeMapIterator,
    VmapIterator,
    WhileLoopIterator,
)
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap, WhileCarry


class TestVmapDispatch:
    """Test Vmap strategy dispatch."""

    def test_vmap_factory_returns_iterator(self):
        """make_axis_dispatch(Vmap) returns VmapIterator."""
        strategy = Vmap()
        iterator = make_axis_dispatch(strategy)
        assert isinstance(iterator, VmapIterator)

    def test_vmap_dispatch_simple(self):
        """Vmap dispatches to jax.vmap(fn) via axis_dispatch."""

        def fn(x):
            return x * 2

        xs = jnp.arange(5)  # shape: (5,)
        strategy = Vmap()
        result = axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_vmap_dispatch_pytree(self):
        """Vmap works with nested pytrees via axis_dispatch."""

        def fn(carry):
            a, b = carry
            return (a * 2, b + 1)

        xs = (jnp.arange(5), jnp.ones(5))  # tuple pytree
        strategy = Vmap()
        result = axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result[0], expected[0])
        assert jnp.allclose(result[1], expected[1])

    def test_vmap_iterator_direct_call(self):
        """VmapIterator can be called directly as iterator(fn, xs)."""

        def fn(x):
            return x * 2

        xs = jnp.arange(5)
        iterator = VmapIterator()
        result = iterator(fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)


class TestSafeMapDispatch:
    """Test SafeMap strategy dispatch."""

    def test_safemap_factory_returns_iterator(self):
        """make_axis_dispatch(SafeMap) returns SafeMapIterator."""
        strategy = SafeMap(batch_size=32)
        iterator = make_axis_dispatch(strategy)
        assert isinstance(iterator, SafeMapIterator)
        assert iterator.tile == 32

    def test_safemap_no_chunking_when_small(self):
        """SafeMap with batch_size >= n uses vmap via axis_dispatch."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)
        strategy = SafeMap(batch_size=50)
        result = axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_safemap_chunking_divisible(self):
        """SafeMap with divisible chunking matches vmap via axis_dispatch."""

        def fn(x):
            return x * 3

        xs = jnp.arange(100)
        strategy = SafeMap(batch_size=25)
        result = axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_safemap_non_divisible_raises(self):
        """SafeMap with non-divisible batch raises ValueError via axis_dispatch."""

        def fn(x):
            return x * 2

        xs = jnp.arange(100)
        strategy = SafeMap(batch_size=30)

        with pytest.raises(ValueError, match="safe_map.*not divisible"):
            axis_dispatch(strategy, fn, xs)


class TestScanDispatch:
    """Test Scan strategy dispatch."""

    def test_scan_factory_returns_iterator(self):
        """make_axis_dispatch(Scan) returns JaxScanIterator."""
        strategy = Scan()
        iterator = make_axis_dispatch(strategy)
        assert isinstance(iterator, JaxScanIterator)

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
        """Scan dispatch uses strategy.transition and init via axis_dispatch."""

        def transition(carry, x):
            return carry + x, x * 2

        init = jnp.array(0.0)
        xs = jnp.arange(5, dtype=jnp.float32)
        strategy = Scan(transition=transition)

        # fn is ignored for Scan; we pass None as placeholder
        result_carry, result_ys = axis_dispatch(strategy, None, xs, init=init)

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
            axis_dispatch(strategy, None, xs, init=None)

    def test_scan_dispatch_requires_init(self):
        """Scan dispatch raises ValueError if init is None."""

        def transition(carry, x):
            return carry + x, x * 2

        xs = jnp.arange(5, dtype=jnp.float32)
        strategy = Scan(transition=transition)

        with pytest.raises(ValueError, match="Scan strategy requires.*init"):
            axis_dispatch(strategy, None, xs, init=None)

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
        result_carry, result_ys = axis_dispatch(strategy, ignored_fn, xs, init=init)
        assert jnp.allclose(result_carry, jnp.sum(xs))

    def test_scan_factory_rejects_heterogeneous_axis(self):
        """make_axis_dispatch(Scan) raises DispatchRejected on a heterogeneous axis."""
        strategy = Scan(transition=lambda carry, x: (carry, x))

        with pytest.raises(DispatchRejected, match="heterogeneous"):
            make_axis_dispatch(strategy, axis="state", heterogeneous_axes={"state"})


class TestWhileCarryDispatch:
    """Test WhileCarry strategy dispatch."""

    def test_whilecarry_factory_returns_iterator(self):
        """make_axis_dispatch(WhileCarry) returns WhileLoopIterator."""
        strategy = WhileCarry()
        iterator = make_axis_dispatch(strategy)
        assert isinstance(iterator, WhileLoopIterator)

    def test_whilecarry_factory_rejects_heterogeneous_axis(self):
        """make_axis_dispatch(WhileCarry) raises DispatchRejected on a heterogeneous axis."""
        strategy = WhileCarry(body=lambda carry: carry, cond=lambda carry: carry < 5, init=0)

        with pytest.raises(DispatchRejected, match="heterogeneous"):
            make_axis_dispatch(strategy, axis="state", heterogeneous_axes={"state"})

    def test_whilecarry_factory_allows_non_heterogeneous_axis(self):
        """make_axis_dispatch(WhileCarry) succeeds when axis is not in heterogeneous_axes."""
        strategy = WhileCarry(body=lambda carry: carry, cond=lambda carry: carry < 5, init=0)

        iterator = make_axis_dispatch(strategy, axis="step", heterogeneous_axes={"other_axis"})
        assert isinstance(iterator, WhileLoopIterator)

    def test_whilecarry_dispatch_with_init(self):
        """WhileCarry dispatch uses strategy.cond/.body/.init via axis_dispatch."""

        def cond(carry):
            return carry < 5

        def body(carry):
            return carry + 1

        strategy = WhileCarry(body=body, cond=cond, init=jnp.array(0))

        # fn/xs are ignored for WhileCarry; pass placeholders.
        result = axis_dispatch(strategy, None, None)

        assert result == 5

    def test_whilecarry_dispatch_requires_body(self):
        """WhileCarry dispatch raises ValueError if body is None."""
        strategy = WhileCarry(body=None, cond=lambda carry: carry < 5, init=0)

        with pytest.raises(ValueError, match="WhileCarry strategy requires.*body"):
            axis_dispatch(strategy, None, None)

    def test_whilecarry_dispatch_requires_cond(self):
        """WhileCarry dispatch raises ValueError if cond is None."""
        strategy = WhileCarry(body=lambda carry: carry, cond=None, init=0)

        with pytest.raises(ValueError, match="WhileCarry strategy requires.*cond"):
            axis_dispatch(strategy, None, None)

    def test_whilecarry_dispatch_requires_init(self):
        """WhileCarry dispatch raises ValueError if init is None (and no init parameter given)."""
        strategy = WhileCarry(body=lambda carry: carry, cond=lambda carry: carry < 5, init=None)

        with pytest.raises(ValueError, match="WhileCarry strategy requires.*init"):
            axis_dispatch(strategy, None, None, init=None)

    def test_whilecarry_dispatch_falls_back_to_init_parameter(self):
        """WhileCarry dispatch uses the init= parameter when strategy.init is None."""

        def cond(carry):
            return carry < 3

        def body(carry):
            return carry + 1

        strategy = WhileCarry(body=body, cond=cond, init=None)

        result = axis_dispatch(strategy, None, None, init=jnp.array(0))
        assert result == 3

    def test_whilecarry_dispatch_ignores_fn(self):
        """WhileCarry dispatch ignores the fn parameter."""

        def cond(carry):
            return carry < 3

        def body(carry):
            return carry + 1

        def ignored_fn(carry):
            raise RuntimeError("fn should be ignored for WhileCarry")

        strategy = WhileCarry(body=body, cond=cond, init=jnp.array(0))

        result = axis_dispatch(strategy, ignored_fn, None)
        assert result == 3


class TestDedupGatherDispatch:
    """Test DedupGather strategy dispatch."""

    def test_dedupgather_factory_raises_dispatch_rejected(self):
        """make_axis_dispatch(DedupGather) raises DispatchRejected."""
        import numpy as np

        def dedup_fn(xs, unique_indices):
            return xs[unique_indices]

        def gather_fn(ys, index_map):
            return ys[index_map]

        unique_indices = np.array([0, 1], dtype=np.int32)
        index_map = np.array([0, 1], dtype=np.int32)

        strategy = DedupGather(
            unique_indices=unique_indices,
            index_map=index_map,
            k=2,
            k_bucket=2,
            dedup_fn=dedup_fn,
            gather_fn=gather_fn,
        )

        with pytest.raises(DispatchRejected):
            make_axis_dispatch(strategy)

    def test_dedupgather_dispatch(self):
        """DedupGather dispatches through dedup -> map -> gather via axis_dispatch."""
        import numpy as np

        # Simple dedup: group by unique values
        def dedup_fn(xs, unique_indices):
            """Select unique elements from xs by index."""
            return xs[unique_indices]

        def gather_fn(ys, index_map):
            """Scatter unique results back to original positions."""
            return ys[index_map]

        def fn(x):
            return x * 2

        xs = jnp.array([1, 2, 1, 3, 2, 1])  # has duplicates [1, 2, 3]
        # Unique elements at indices [0, 1, 3]
        unique_indices = np.array([0, 1, 3], dtype=np.int32)
        # Map original positions to unique positions [0, 1, 0, 2, 1, 0]
        index_map = np.array([0, 1, 0, 2, 1, 0], dtype=np.int32)

        strategy = DedupGather(
            unique_indices=unique_indices,
            index_map=index_map,
            k=3,
            k_bucket=4,
            dedup_fn=dedup_fn,
            gather_fn=gather_fn,
        )

        result = axis_dispatch(strategy, fn, xs)

        # Expected: dedup to [1, 2, 3], map to [2, 4, 6],
        # gather back to [2, 4, 2, 6, 4, 2]
        expected = jnp.array([2, 4, 2, 6, 4, 2])
        assert jnp.allclose(result, expected)

    def test_dedupgather_unpacking(self):
        """DedupGather correctly unpacks dedup_fn output via axis_dispatch."""
        import numpy as np

        def dedup_fn(xs, unique_indices):
            return xs[unique_indices]

        def gather_fn(ys, index_map):
            return ys[index_map]

        def fn(x):
            return x + 10

        xs = jnp.array([5, 5, 3, 3, 7])
        # Unique values are [3, 5, 7] at indices [2, 0, 4]
        unique_indices = np.array([2, 0, 4], dtype=np.int32)
        # Map back: [5, 5, 3, 3, 7] -> [1, 1, 0, 0, 2]
        index_map = np.array([1, 1, 0, 0, 2], dtype=np.int32)

        strategy = DedupGather(
            unique_indices=unique_indices,
            index_map=index_map,
            k=3,
            k_bucket=4,
            dedup_fn=dedup_fn,
            gather_fn=gather_fn,
        )

        result = axis_dispatch(strategy, fn, xs)
        # unique: [3, 5, 7], indices: [1, 1, 0, 0, 2]
        # mapped: [13, 15, 17]
        # gathered: [15, 15, 13, 13, 17]
        expected = jnp.array([15, 15, 13, 13, 17])
        assert jnp.allclose(result, expected)


class TestBucketDispatch:
    """Bucket is host-side and must not execute in the device-tier dispatch."""

    def test_bucket_dispatch_raises_with_guidance(self):
        """axis_dispatch(Bucket, ...) raises TypeError pointing to host helpers."""

        def fn(x):
            return x * 2

        xs = jnp.arange(5)
        strategy = Bucket(boundaries=(8, 16))

        with pytest.raises(TypeError, match="host-side"):
            axis_dispatch(strategy, fn, xs)

    def test_bucket_dispatch_error_mentions_bucketize(self):
        """The error directs callers to select_bucket/bucketize via axis_dispatch."""

        def fn(x):
            return x

        xs = jnp.arange(3)
        strategy = Bucket(boundaries=(4,))

        with pytest.raises(TypeError, match="Bucket.*host-side"):
            axis_dispatch(strategy, fn, xs)


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

    def test_whilecarry_isinstance(self):
        """WhileCarry is recognized."""
        strategy = WhileCarry(body=lambda carry: carry, cond=lambda carry: carry < 5)
        assert isinstance(strategy, WhileCarry)

    def test_dedupgather_isinstance(self):
        """DedupGather is recognized."""
        import numpy as np

        def dedup_fn(xs, unique_indices):
            return xs[unique_indices]

        def gather_fn(ys, index_map):
            return ys[index_map]

        unique_indices = np.array([0, 1], dtype=np.int32)
        index_map = np.array([0, 1], dtype=np.int32)

        strategy = DedupGather(
            unique_indices=unique_indices,
            index_map=index_map,
            k=2,
            k_bucket=2,
            dedup_fn=dedup_fn,
            gather_fn=gather_fn,
        )
        assert isinstance(strategy, DedupGather)


class TestDispatchIntegration:
    """Integration tests combining strategies."""

    def test_dispatch_with_matrix_multiplication(self):
        """Test dispatch with a real JAX operation (matmul) via axis_dispatch."""
        W = jnp.ones((10, 10))
        xs = jnp.ones((5, 10))

        def fn(x):
            return jnp.matmul(x, W)

        strategy = Vmap()
        result = axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

    def test_dispatch_safemap_with_neural_net_like(self):
        """Test SafeMap with neural net-like operation via axis_dispatch."""

        def fn(x):
            return jax.nn.relu(x)

        xs = jnp.linspace(-1, 1, 100)
        strategy = SafeMap(batch_size=25)
        result = axis_dispatch(strategy, fn, xs)
        expected = jax.vmap(fn)(xs)

        assert jnp.allclose(result, expected)

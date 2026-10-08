"""Tests for tiling iterators: VmapIterator, ChunkedMapIterator, BucketIterator."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling import WhileLoopWithYsIterator
from xtrax.tiling.iterator import (
    BucketIterator,
    ChunkedMapIterator,
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
    """Test ChunkedMapIterator shape, divisibility, and equivalence."""

    def test_safe_map_iterator_equals_vmap_when_batch_size_gte_n(self):
        """ChunkedMapIterator with tile >= n should equal VmapIterator."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)

        vmap_iter = VmapIterator()
        vmap_result = vmap_iter(fn, xs)

        safe_iter = ChunkedMapIterator(tile=20)
        safe_result = safe_iter(fn, xs)

        assert jnp.allclose(vmap_result, safe_result)

    def test_safe_map_iterator_batch_size_equals_n(self):
        """ChunkedMapIterator with tile == n should equal VmapIterator."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)

        vmap_iter = VmapIterator()
        vmap_result = vmap_iter(fn, xs)

        safe_iter = ChunkedMapIterator(tile=10)
        safe_result = safe_iter(fn, xs)

        assert jnp.allclose(vmap_result, safe_result)

    def test_chunked_map_iterator_non_divisible_runs_a_ragged_final_chunk(self):
        """ChunkedMapIterator should propagate ValueError for non-divisible n."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)  # n=10

        safe_iter = ChunkedMapIterator(tile=3)  # 10 % 3 != 0: ragged final chunk (#5565)
        assert jnp.array_equal(safe_iter(fn, xs), jax.vmap(fn)(xs))

    def test_safe_map_iterator_divisible_batch(self):
        """ChunkedMapIterator should work with divisible batch sizes."""

        def fn(x):
            return x * 2

        xs = jnp.arange(10)

        safe_iter = ChunkedMapIterator(tile=5)
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


def _python_while_ys(cond, body, init, max_steps):
    """Host reference: same stop rule as WhileLoopWithYsIterator, ys as a list."""
    carry = init
    ys = []
    step = 0
    while bool(cond(carry)) and step < max_steps:
        carry, y = body(carry)
        ys.append(y)
        step += 1
    return carry, ys, step


def _stack_ys(ys):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *ys)


def _assert_tree_equal(actual, expected):
    jax.tree_util.tree_map(
        lambda left, right: assert_array_equal(left, right),
        actual,
        expected,
    )


def assert_array_equal(left, right):
    assert jnp.array_equal(left, right), f"{left} != {right}"


def _primitive_names(jaxpr):
    names = []
    for eqn in jaxpr.eqns:
        names.append(eqn.primitive.name)
        for param in eqn.params.values():
            names.extend(_names_in_param(param))
    return names


def _names_in_param(param):
    sub = getattr(param, "jaxpr", None)
    if sub is not None:
        return _primitive_names(sub)
    if isinstance(param, tuple | list):
        names = []
        for item in param:
            names.extend(_names_in_param(item))
        return names
    return []


class TestWhileLoopWithYsIterator:
    """while_loop that stores each step's y in a preallocated buffer."""

    def test_pytree_ys_match_python_reference(self):
        """ys[:length] matches a host loop, including a pytree of outputs."""

        def cond(carry):
            return carry < 3

        def body(carry):
            nxt = carry + 1
            return nxt, {"token": nxt, "feat": jnp.array([nxt, nxt * 2])}

        init = jnp.int32(0)
        prototype = {"token": jnp.int32(0), "feat": jnp.zeros((2,), dtype=jnp.int32)}
        max_steps = 6
        iterator = WhileLoopWithYsIterator(max_steps=max_steps)
        final, ys, length = iterator(cond, body, init, prototype)
        ref_carry, ref_ys, ref_len = _python_while_ys(cond, body, init, max_steps)

        assert length == ref_len
        assert final == ref_carry
        _assert_tree_equal(
            jax.tree_util.tree_map(lambda buf: buf[: int(length)], ys),
            _stack_ys(ref_ys),
        )
        assert ys["token"].shape == (max_steps,)
        assert ys["feat"].shape == (max_steps, 2)

    def test_early_stop_fill_region_is_zero(self):
        """Indices at and after length are the zero fill, not body outputs."""

        def cond(carry):
            return carry < jnp.float32(3)

        def body(carry):
            nxt = carry + jnp.float32(1)
            return nxt, nxt * jnp.float32(10)

        init = jnp.float32(0)
        # Nonzero prototype: the buffer fill is 0, not a copy of this value.
        prototype = jnp.float32(7)
        max_steps = 5
        iterator = WhileLoopWithYsIterator(max_steps=max_steps)
        _final, ys, length = iterator(cond, body, init, prototype)

        assert length == 3
        assert jnp.array_equal(ys[: int(length)], jnp.array([10.0, 20.0, 30.0]))
        assert jnp.all(ys[int(length) :] == 0)

    def test_max_steps_with_cond_still_true_returns_full_buffer(self):
        """Hitting the cap with cond still true yields length == max_steps."""

        def cond(carry):
            return carry < jnp.int32(100)

        def body(carry):
            nxt = carry + jnp.int32(1)
            return nxt, nxt

        init = jnp.int32(0)
        max_steps = 4
        iterator = WhileLoopWithYsIterator(max_steps=max_steps)
        final, ys, length = iterator(cond, body, init, jnp.int32(0))
        ref_carry, ref_ys, ref_len = _python_while_ys(cond, body, init, max_steps)

        assert length == max_steps
        assert length == ref_len
        assert bool(cond(final))
        assert final == ref_carry
        assert jnp.array_equal(ys, _stack_ys(ref_ys))

    def test_cond_false_at_cap_also_fills_the_buffer(self):
        """A full buffer with cond false means the predicate ended the loop."""

        def cond(carry):
            return carry < jnp.int32(4)

        def body(carry):
            nxt = carry + jnp.int32(1)
            return nxt, nxt

        init = jnp.int32(0)
        max_steps = 4
        iterator = WhileLoopWithYsIterator(max_steps=max_steps)
        final, ys, length = iterator(cond, body, init, jnp.int32(0))

        assert length == max_steps
        assert not bool(cond(final))
        assert final == 4
        assert jnp.array_equal(ys, jnp.array([1, 2, 3, 4], dtype=jnp.int32))

    def test_zero_iterations_keeps_init_and_zero_fill(self):
        """A cond that is false at the start writes nothing."""

        def cond(carry):
            return carry < jnp.int32(0)

        def body(carry):
            return carry + jnp.int32(100), carry + jnp.int32(100)

        init = jnp.int32(0)
        iterator = WhileLoopWithYsIterator(max_steps=5)
        final, ys, length = iterator(cond, body, init, jnp.int32(9))

        assert length == 0
        assert final == init
        assert ys.shape == (5,)
        assert jnp.all(ys == 0)

    def test_jit_matches_python_reference(self):
        """The same buffer and length are produced under jax.jit."""

        def cond(carry):
            return carry < jnp.int32(3)

        def body(carry):
            nxt = carry + jnp.int32(1)
            return nxt, {"n": nxt, "pair": jnp.array([nxt, -nxt])}

        init = jnp.int32(0)
        prototype = {"n": jnp.int32(0), "pair": jnp.zeros((2,), dtype=jnp.int32)}
        max_steps = 8
        iterator = WhileLoopWithYsIterator(max_steps=max_steps)

        @jax.jit
        def run(carry, proto):
            return iterator(cond, body, carry, proto)

        final, ys, length = run(init, prototype)
        ref_carry, ref_ys, ref_len = _python_while_ys(cond, body, init, max_steps)

        assert length == ref_len
        assert final == ref_carry
        _assert_tree_equal(
            jax.tree_util.tree_map(lambda buf: buf[: int(length)], ys),
            _stack_ys(ref_ys),
        )
        assert jnp.all(ys["n"][int(length) :] == 0)
        assert jnp.all(ys["pair"][int(length) :] == 0)

    def test_jaxpr_lowers_to_while_without_scan(self):
        """The collected loop is a lax.while_loop."""

        def cond(carry):
            return carry < jnp.int32(3)

        def body(carry):
            nxt = carry + jnp.int32(1)
            return nxt, nxt

        iterator = WhileLoopWithYsIterator(max_steps=5)

        def run(carry, proto):
            return iterator(cond, body, carry, proto)

        closed = jax.make_jaxpr(run)(jnp.int32(0), jnp.int32(0))
        names = _primitive_names(closed.jaxpr)
        assert "while" in names
        assert "scan" not in names

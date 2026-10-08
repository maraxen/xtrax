"""Iterator protocols and concrete implementations for axis iteration.

Five iterator strategies control how a mapped axis is iterated:
- VmapIterator: jax.vmap — fully parallel, stateless.
- ChunkedMapIterator: chunked_map with tiling — memory-bounded, stateless.
- JaxScanIterator: jax.lax.scan — carry-bearing, sequential.
- WhileLoopIterator: jax.lax.while_loop — carry-bearing, no output collection.
- WhileLoopWithYsIterator: jax.lax.while_loop — carry-bearing, buffered ys.

MapIterator and ScanIterator are runtime_checkable Protocols defining the
two fundamental iteration patterns: stateless (MapIterator) and carry-bearing
(ScanIterator).

Pattern 5 note: Concrete iterators are eqx.Module instances, NOT marked
@runtime_checkable. The protocols (MapIterator, ScanIterator) are the types
that are @runtime_checkable; users check isinstance(concrete, Protocol).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp

from xtrax.transforms.map import _apply_size1, _is_size1_axis, chunked_map


@runtime_checkable
class MapIterator(Protocol):
    """Stateless axis iteration protocol.

    Maps a function over an axis without carrying state. Signature:
        fn: Callable — function to apply per-element
        xs: Any — pytree of arrays; first axis will be iterated
        in_axes: Any — specifies which axes to iterate over (default: 0)

    Returns: ys where tree_structure(ys) == tree_structure(xs) but with
    the mapped axis consumed.
    """

    def __call__(
        self,
        fn: Any,
        xs: Any,
        *,
        in_axes: Any = 0,
    ) -> Any:
        """Apply fn over the first (or specified) axis of xs.

        Args:
            fn: Callable to apply per-element.
            xs: Input pytree; iteration happens over axis 0 (or in_axes).
            in_axes: Axis specification (default 0).

        Returns:
            Output pytree with iterated axis consumed.

        """
        ...


@runtime_checkable
class ScanIterator(Protocol):
    """Carry-bearing axis iteration protocol.

    Scans over an axis, threading a carry value through iterations. Signature:
        fn: Callable — (carry, x) -> (carry, y)
        init: Any — initial carry value
        xs: Any — pytree to scan over

    Returns: (final_carry, ys) where final_carry is the final carry value
    after all iterations, and ys contains all outputs.
    """

    def __call__(self, fn: Any, init: Any, xs: Any) -> tuple[Any, Any]:
        """Scan a function over the first axis of xs with carry.

        Args:
            fn: Callable(carry, x) -> (carry, y).
            init: Initial carry value.
            xs: Input pytree; scan happens over axis 0.

        Returns:
            (final_carry, ys): Final carry and stacked outputs.

        """
        ...


class VmapIterator(eqx.Module):
    """Iterate via jax.vmap — fully parallel.

    All elements are materialized and computed simultaneously. Use when
    memory budget allows and elements are independent (no cross-talk).
    A mapped axis of length 1 is a direct call, not a vmap (#2520).
    """

    def __call__(
        self,
        fn: Any,
        xs: Any,
        *,
        in_axes: Any = 0,
    ) -> Any:
        """Apply fn using jax.vmap.

        Args:
            fn: Callable to apply per-element.
            xs: Input pytree.
            in_axes: Axis specification for vmap (default 0).

        Returns:
            Output after vmapping over the specified axis.

        """
        # Same length-1 guard as chunked_map: never emit a vmap-of-1 (#2520).
        if _is_size1_axis(xs, in_axes):
            return _apply_size1(fn, xs, in_axes)
        return jax.vmap(fn, in_axes=in_axes)(xs)


class ChunkedMapIterator(eqx.Module):
    """Iterate via chunked_map with tile chunking — memory-bounded, stateless.

    Elements are processed in tiles to avoid memory exhaustion and XLA
    loop construct issues. No carry state; elements are independent.
    """

    tile: int = eqx.field(static=True)

    def __call__(
        self,
        fn: Any,
        xs: Any,
        *,
        in_axes: Any = 0,
    ) -> Any:
        """Apply fn using chunked_map with tiling.

        Args:
            fn: Callable to apply per-element.
            xs: Input pytree.
            in_axes: Axis specification (default 0; chunked_map always uses axis 0).

        Returns:
            Output after chunked_map over the first axis.

        """
        # Note: chunked_map always iterates over axis 0; in_axes parameter is
        # accepted for protocol compatibility.
        if in_axes != 0:
            msg = "ChunkedMapIterator currently only supports in_axes=0"
            raise NotImplementedError(msg)
        return chunked_map(fn, xs, batch_size=self.tile)


class JaxScanIterator(eqx.Module):
    """Iterate via jax.lax.scan — carry-bearing, sequential.

    Elements are processed sequentially with a carry value threading through.
    Use when elements have dependencies or when state must be accumulated.
    """

    def __call__(self, fn: Any, init: Any, xs: Any) -> tuple[Any, Any]:
        """Apply fn using jax.lax.scan.

        Args:
            fn: Callable(carry, x) -> (carry, y).
            init: Initial carry value.
            xs: Input pytree to scan over.

        Returns:
            (final_carry, ys): Final carry and stacked outputs.

        """
        return jax.lax.scan(fn, init, xs, unroll=1)


class WhileLoopIterator(eqx.Module):
    """Iterate via jax.lax.while_loop -- carry-bearing, no output collection.

    Unlike JaxScanIterator, returns only the final carry (no `ys`) --
    genuinely nothing to stack, since there's no per-step output.
    """

    def __call__(self, cond: Any, body: Any, init: Any) -> Any:
        """Apply fn using jax.lax.while_loop.

        Args:
            cond: Callable(carry) -> bool (traced scalar continuation predicate).
            body: Callable(carry) -> new_carry.
            init: Initial carry value.

        Returns:
            final_carry: The carry after the loop's condition first fails.

        """
        return jax.lax.while_loop(cond, body, init)


class WhileLoopWithYsIterator(eqx.Module):
    """``lax.while_loop`` that records each step's ``y`` in a fixed buffer.

    ``WhileLoopIterator`` returns only the final carry. This iterator's
    ``body(carry)`` returns ``(new_carry, y)``, and the iterator returns
    ``(final_carry, ys_buffer, length)``.

    ``max_steps`` is a static Python int: the leading dimension of every
    buffer leaf. ``y_prototype`` is a pytree of arrays. Each leaf is allocated
    with ``jnp.zeros((max_steps, *leaf.shape), dtype=leaf.dtype)``. The fill
    value is ``0`` of that leaf's dtype. Indices ``>= length`` hold fill;
    the valid outputs are the prefix ``ys[:length]``. Values in
    ``y_prototype`` supply shape and dtype only.

    The loop runs while ``cond(carry)`` is true and ``step < max_steps``.
    ``length`` is that step count, an int32 scalar. A return with
    ``length == max_steps`` and ``cond(final_carry)`` still true means the
    cap stopped the loop and every buffer index holds a body output.
    ``cond(final_carry)`` distinguishes that full buffer from one whose
    predicate became false on the last step.

    Lowers to ``jax.lax.while_loop``.
    """

    max_steps: int = eqx.field(static=True)

    def __call__(
        self,
        cond: Any,
        body: Any,
        init: Any,
        y_prototype: Any,
    ) -> tuple[Any, Any, Any]:
        """Run ``body`` until ``cond`` fails or ``max_steps`` is reached.

        Args:
            cond: Callable(carry) -> scalar bool. True means continue.
            body: Callable(carry) -> (new_carry, y). ``y`` matches
                ``y_prototype`` leaf for leaf.
            init: Initial carry.
            y_prototype: Pytree of arrays giving each ``y`` leaf's shape and
                dtype. The stored values are ignored.

        Returns:
            ``(final_carry, ys_buffer, length)``. ``ys_buffer`` has leading
            dimension ``max_steps``. ``length`` counts body calls. The fill
            value in ``ys_buffer[length:]`` is ``0``.

        """
        ys0 = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros((self.max_steps, *leaf.shape), dtype=leaf.dtype),
            y_prototype,
        )
        step0 = jnp.zeros((), dtype=jnp.int32)

        def loop_cond(state: tuple[Any, Any, Any]) -> Any:
            step, carry, _ys = state
            return jnp.logical_and(step < self.max_steps, cond(carry))

        def loop_body(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            step, carry, ys = state
            new_carry, y = body(carry)
            ys = jax.tree_util.tree_map(lambda buf, val: buf.at[step].set(val), ys, y)
            return step + jnp.int32(1), new_carry, ys

        step, final_carry, ys = jax.lax.while_loop(
            loop_cond,
            loop_body,
            (step0, init, ys0),
        )
        return final_carry, ys, step


class BucketIterator:
    """Iterator that buckets data by boundaries with different batch sizes each.

    Note: BucketIterator is xtrax-specific (not in aminx). It provides
    host-side bucketing with padded computation.
    """

    def __init__(
        self,
        boundaries: list[int],
        batch_sizes: list[int],
        fn: Any,
        xs: Any,
    ) -> None:
        """Initialize BucketIterator.

        Args:
            boundaries: Sorted list of N boundary values for bucketing.
            batch_sizes: List of N+1 batch sizes, one per bucket.
            fn: Function to apply to each batch.
            xs: Input data to bucket and process.

        Raises:
            ValueError: If len(batch_sizes) != len(boundaries) + 1.
        """

        if len(batch_sizes) != len(boundaries) + 1:
            raise ValueError(
                f"len(batch_sizes)={len(batch_sizes)} must equal "
                f"len(boundaries) + 1 = {len(boundaries) + 1}"
            )
        self.boundaries = boundaries
        self.batch_sizes = batch_sizes
        self.fn = fn
        self.xs = xs

    def __iter__(self):
        """Iterate over bucketed batches.

        Yields:
            Tuples of (result, original_length_mask) where result is the output
            of fn applied to the padded input, and original_length_mask is a
            boolean array indicating which elements of the padded batch are
            from the original (non-padded) input.
        """
        import bisect
        import warnings

        import jax.numpy as jnp

        leaves = jax.tree_util.tree_leaves(self.xs)
        if not leaves:
            return

        seq_len = leaves[0].shape[0]
        # bisect_left finds the first boundary >= seq_len
        # (exact match is valid, pad_amount=0)
        bucket_idx = bisect.bisect_left(self.boundaries, seq_len)

        if bucket_idx >= len(self.boundaries):
            max_bucket = self.boundaries[-1]
            warnings.warn(
                f"BucketIterator: input length {seq_len} exceeds maximum bucket "
                f"size {max_bucket}; this input cannot be processed without "
                "recompilation risk",
                stacklevel=2,
            )
            raise ValueError(
                f"Input length {seq_len} exceeds maximum bucket size {max_bucket}. "
                "Add a larger boundary or use a different iterator."
            )

        bucket_size = self.boundaries[bucket_idx]
        pad_amount = bucket_size - seq_len

        padded_xs = jax.tree_util.tree_map(
            lambda x: jnp.pad(x, [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1)),
            self.xs,
        )

        original_length_mask = jnp.arange(bucket_size) < seq_len
        result = self.fn(padded_xs)
        yield (result, original_length_mask)


__all__ = [
    "JaxScanIterator",
    "MapIterator",
    "ChunkedMapIterator",
    "ScanIterator",
    "VmapIterator",
    "BucketIterator",
    "WhileLoopIterator",
    "WhileLoopWithYsIterator",
]

"""Iterator protocols and concrete implementations for axis iteration.

Three iterator strategies control how a mapped axis is iterated:
- VmapIterator: jax.vmap — fully parallel, stateless.
- SafeMapIterator: safe_map with tiling — memory-bounded, stateless.
- JaxScanIterator: jax.lax.scan — carry-bearing, sequential.

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

from xtrax.transforms.map import safe_map


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
        return jax.vmap(fn, in_axes=in_axes)(xs)


class SafeMapIterator(eqx.Module):
    """Iterate via safe_map with tile chunking — memory-bounded, stateless.

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
        """Apply fn using safe_map with tiling.

        Args:
            fn: Callable to apply per-element.
            xs: Input pytree.
            in_axes: Axis specification (default 0; safe_map always uses axis 0).

        Returns:
            Output after safe_map over the first axis.

        """
        # Note: safe_map always iterates over axis 0; in_axes parameter is
        # accepted for protocol compatibility.
        if in_axes != 0:
            msg = "SafeMapIterator currently only supports in_axes=0"
            raise NotImplementedError(msg)
        return safe_map(fn, xs, batch_size=self.tile)


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
    "SafeMapIterator",
    "ScanIterator",
    "VmapIterator",
    "BucketIterator",
]

"""Iterators for tiling strategies: VmapIterator, SafeMapIterator, BucketIterator."""

from __future__ import annotations

import bisect
import warnings
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.transforms.map import safe_map


@runtime_checkable
class MapIterator(Protocol):
    """Protocol for iterators that apply a function to a pytree."""

    def __call__(self, fn: Callable, xs: Any) -> Any: ...


class VmapIterator(eqx.Module):
    """Iterator that applies jax.vmap to a function and pytree."""

    def __call__(self, fn: Callable, xs: Any) -> Any:
        """Apply vmap to fn over xs.

        Args:
            fn: Function to vmap over the leading axis.
            xs: Input pytree with leading batch dimension.

        Returns:
            Result of jax.vmap(fn)(xs).
        """
        return jax.vmap(fn)(xs)


class SafeMapIterator(eqx.Module):
    """Iterator that applies safe_map with chunking for memory efficiency."""

    batch_size: int

    def __init__(self, batch_size: int):
        """Initialize SafeMapIterator with batch size.

        Args:
            batch_size: Batch size threshold and chunk size for safe_map.
        """
        self.batch_size = batch_size

    def __call__(self, fn: Callable, xs: Any) -> Any:
        """Apply safe_map to fn over xs with the configured batch size.

        Args:
            fn: Function to apply.
            xs: Input pytree with leading batch dimension.

        Returns:
            Result of safe_map(fn, xs, batch_size=self.batch_size).

        Raises:
            ValueError: If leading axis is not divisible by batch_size.
        """
        return safe_map(fn, xs, batch_size=self.batch_size)


@runtime_checkable
class ScanIterator(Protocol):
    """Protocol for iterators that perform scan operations."""

    def __call__(self, fn: Callable, init: Any, xs: Any) -> tuple[Any, Any]: ...


class JaxScanIterator(eqx.Module):
    """Iterator that applies jax.lax.scan (carry-bearing sequential iteration)."""

    def __call__(self, fn: Callable, init: Any, xs: Any) -> tuple[Any, Any]:
        """Apply scan operation.

        Args:
            fn: Scan transition function (carry, x) -> (new_carry, output).
            init: Initial carry state.
            xs: Input pytree sequence.

        Returns:
            Tuple of (final_carry, outputs).
        """
        return jax.lax.scan(fn, init, xs)


class BucketIterator:
    """Iterator that buckets data by boundaries with different batch sizes each."""

    def __init__(
        self,
        boundaries: list[int],
        batch_sizes: list[int],
        fn: Callable,
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

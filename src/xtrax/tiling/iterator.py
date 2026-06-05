"""Iterators for tiling strategies: VmapIterator, SafeMapIterator, BucketIterator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax

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
            Batches processed with batch sizes appropriate for each bucket.
        """
        # This is a placeholder implementation.
        # In a real implementation, this would:
        # 1. Sort/bucket elements by their size using length_fn
        # 2. Apply fn to each bucket with the appropriate batch_size
        # 3. Yield results
        yield from []

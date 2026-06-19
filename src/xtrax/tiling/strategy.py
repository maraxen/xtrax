"""AxisStrategy sealed union and variants for composable tiling strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax


@runtime_checkable
class ScanTransition(Protocol):
    """Scan transition function: (carry, x) -> (new_carry, output)."""

    def __call__(self, carry: Any, x: Any) -> tuple[Any, Any]: ...


@runtime_checkable
class DedupFn(Protocol):
    """Deduplication function: (xs, unique_indices) -> deduped_xs."""

    def __call__(self, xs: Any, unique_indices: Any) -> Any: ...


@runtime_checkable
class GatherFn(Protocol):
    """Gather function: (ys, gather_indices) -> gathered_ys."""

    def __call__(self, ys: Any, gather_indices: Any) -> Any: ...


def _default_dedup_fn(xs: Any, unique_indices: Any) -> Any:
    """Default deduplication function: select unique elements by index."""
    return jax.tree.map(lambda x: x[unique_indices], xs)


def _default_gather_fn(ys: Any, gather_indices: Any) -> Any:
    """Default gather function: scatter unique results back to original positions."""
    return jax.tree.map(lambda y: y[gather_indices], ys)


@dataclass(frozen=True)
class Vmap:
    """Vmap strategy: vectorize over the leading axis."""

    pass


@dataclass(frozen=True)
class SafeMap:
    """Safe chunked map strategy: vmap with explicit chunking via lax.map."""

    batch_size: int


@dataclass(frozen=True)
class Scan:
    """Scan strategy: carry-bearing sequential iteration (no batch_size field)."""

    transition: ScanTransition | None = None
    init: Any | None = None
    ordered_sinks: bool = True


@dataclass(frozen=True)
class DedupGather:
    """Deduplication + gather strategy for handling repeated elements.

    Attributes:
        unique_indices: (K_bucket,) int32 array — padded indices into the N-batch
            selecting unique entries. Computed once at plan-build; never a JAX dynamic value.
        index_map: (N,) int32 array — inverse map; index_map[i]=k means position i
            takes result from unique slot k. NOT padded.
        k: int — raw unique count (pre-padding).
        k_bucket: int — padded static bucket size (>= k). XLA compiles on k_bucket, not k.
        dedup_fn: JIT-compatible in-trace gather. Default: _default_dedup_fn.
        gather_fn: JIT-compatible in-trace scatter. Default: _default_gather_fn.
    """

    unique_indices: Any  # np.ndarray
    index_map: Any  # np.ndarray
    k: int
    k_bucket: int
    dedup_fn: DedupFn
    gather_fn: GatherFn


@dataclass(frozen=True)
class Bucket:
    """Host-side length-bucketing strategy for variable-length axes.

    A *plan descriptor*, not a device transform. Bucketing pads a variable-length
    leading axis up to the smallest enclosing boundary so XLA observes a bounded
    set of shapes (one cached executable per bucket) instead of recompiling per
    distinct length. Because JAX has no dynamic shapes inside ``jit``, the bucket
    decision and the padding both happen on the host *before* the JIT boundary —
    see ``select_bucket`` and ``bucketize``. ``make_axis_dispatch`` does not
    execute this strategy.

    Attributes:
        boundaries: Sorted, strictly-ascending bucket sizes. An input of length L
            is padded up to the smallest boundary >= L; L exceeding the largest
            boundary is a runtime error (recompilation-risk contract).
    """

    boundaries: tuple[int, ...]


AxisStrategy = Vmap | SafeMap | Scan | DedupGather | Bucket

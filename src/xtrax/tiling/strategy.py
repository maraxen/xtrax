"""AxisStrategy sealed union and variants for composable tiling strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScanTransition(Protocol):
    """Scan transition function: (carry, x) -> (new_carry, output)."""

    def __call__(self, carry: Any, x: Any) -> tuple[Any, Any]: ...


@runtime_checkable
class DedupFn(Protocol):
    """Deduplication function: (xs) -> (deduped_xs, gather_indices)."""

    def __call__(self, xs: Any) -> tuple[Any, Any]: ...


@runtime_checkable
class GatherFn(Protocol):
    """Gather function: (ys, gather_indices) -> gathered_ys."""

    def __call__(self, ys: Any, gather_indices: Any) -> Any: ...


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

    transition: ScanTransition


@dataclass(frozen=True)
class DedupGather:
    """Deduplication + gather strategy for handling repeated elements."""

    dedup_fn: DedupFn
    gather_fn: GatherFn
    k_bucket: int  # metadata: max unique elements bucket size (power-of-2 padded)


AxisStrategy = Vmap | SafeMap | Scan | DedupGather

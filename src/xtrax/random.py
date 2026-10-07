"""Chunk- and resume-invariant PRNG key stream.

Keys depend only on ``(base key, global element index)``. Chunk size and
which prefix of the stream has already been consumed do not enter the
derivation, so a run can resume at any completed element and reproduce
the same keys.

The derivation is byte-identical to ``aminx.host.plan.compute_sample_keys``:

    sample_indices = np.arange(count, dtype=np.int32) + int(start)
    keys = jax.vmap(lambda idx: jax.random.fold_in(base_key, idx))(sample_indices)

``start`` and ``count`` are host integers. The index vector is ``int32``;
that dtype is part of the contract, because ``fold_in`` hashes the index
bits. An integer seed is turned into a base key with ``jax.random.key``,
the same constructor aminx uses for ``RunSpec`` sampling seeds, so a later
wiring can pass ``RunSpec.seed`` through unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import jax
import numpy as np

# Index dtype hashed by fold_in. Matches aminx.host.plan.compute_sample_keys.
INDEX_DTYPE = np.dtype(np.int32)


def _host_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer, got {type(value).__name__}"
        raise TypeError(msg)
    return int(value)


def _resolve_base_key(seed_or_key: int | np.integer | jax.Array) -> jax.Array:
    """Return a PRNG key. Integers become ``jax.random.key(seed)``."""
    if isinstance(seed_or_key, (bool, np.bool_)):
        msg = f"seed must be an int or a PRNG key, got {type(seed_or_key).__name__}"
        raise TypeError(msg)
    if isinstance(seed_or_key, (int, np.integer)):
        seed = int(seed_or_key)
        if seed < 0:
            msg = f"seed must be a non-negative integer, got {seed}"
            raise ValueError(msg)
        return jax.random.key(seed)
    return seed_or_key


def element_keys(
    base_key: int | np.integer | jax.Array,
    start: int,
    count: int,
) -> jax.Array:
    """PRNG keys for global indices ``[start, start + count)``.

    Args:
        base_key: Integer seed or an existing PRNG key. An integer is
            converted with ``jax.random.key`` and is not folded further.
        start: Global index of the first element. Host integer, ``>= 0``.
        count: Number of keys. Host integer, ``>= 0``.

    Returns:
        Key array of shape ``(count,)``. ``keys[i]`` is
        ``fold_in(base_key, int32(start + i))``.

    Raises:
        TypeError: ``start`` or ``count`` is not an integer.
        ValueError: ``start`` or ``count`` is negative, the seed is
            negative, or ``start`` does not fit in ``int32`` (numpy would
            raise ``OverflowError``; that input is outside the aminx
            derivation). Sums that overflow ``int32`` wrap, matching
            numpy's ``int32`` addition.
    """
    start_i = _host_int("start", start)
    count_i = _host_int("count", count)
    if start_i < 0:
        msg = f"start must be >= 0, got {start_i}"
        raise ValueError(msg)
    if count_i < 0:
        msg = f"count must be >= 0, got {count_i}"
        raise ValueError(msg)
    key = _resolve_base_key(base_key)
    # Byte-identical to aminx.host.plan.compute_sample_keys.
    try:
        sample_indices = np.arange(count_i, dtype=np.int32) + int(start_i)
    except OverflowError as exc:
        msg = f"start={start_i} does not fit in int32"
        raise ValueError(msg) from exc
    if sample_indices.dtype != INDEX_DTYPE:
        msg = (
            "element indices must stay int32; "
            f"start={start_i} count={count_i} promoted to {sample_indices.dtype}"
        )
        raise ValueError(msg)
    return jax.vmap(lambda idx: jax.random.fold_in(key, idx))(sample_indices)


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """One slice of a key stream, addressed by global index.

    Attributes:
        start: Global index of the first element in the slice.
        count: Number of elements in the slice.
    """

    start: int
    count: int


def make_chunk_plan(
    count: int,
    chunk_size: int,
    *,
    start: int = 0,
) -> tuple[ChunkSpan, ...]:
    """Split global indices ``[start, start + count)`` into contiguous chunks.

    The last chunk is shorter when ``count`` is not a multiple of
    ``chunk_size``. ``start`` is the resume point: a plan that begins at
    any completed element covers the same global indices as the suffix of
    a plan that began at 0.

    Args:
        count: Number of elements still to cover. Host integer, ``>= 0``.
        chunk_size: Maximum elements per chunk. Host integer, ``>= 1``.
        start: Global index of the first element. Host integer, ``>= 0``.

    Returns:
        Chunk spans in order. Empty when ``count`` is 0.
    """
    count_i = _host_int("count", count)
    chunk_i = _host_int("chunk_size", chunk_size)
    start_i = _host_int("start", start)
    if count_i < 0:
        msg = f"count must be >= 0, got {count_i}"
        raise ValueError(msg)
    if chunk_i < 1:
        msg = f"chunk_size must be >= 1, got {chunk_i}"
        raise ValueError(msg)
    if start_i < 0:
        msg = f"start must be >= 0, got {start_i}"
        raise ValueError(msg)
    spans: list[ChunkSpan] = []
    offset = 0
    while offset < count_i:
        n = min(chunk_i, count_i - offset)
        spans.append(ChunkSpan(start=start_i + offset, count=n))
        offset += n
    return tuple(spans)


def iter_chunk_keys(
    base_key: int | np.integer | jax.Array,
    plan: Sequence[ChunkSpan],
) -> Iterator[tuple[ChunkSpan, jax.Array]]:
    """Yield ``(span, keys)`` for each span of a chunk plan.

    ``keys`` has shape ``(span.count,)`` and is ``element_keys`` for that
    span's global indices. Two plans that mention the same global index
    produce the same key there, whether or not their chunk boundaries match.
    """
    for span in plan:
        yield span, element_keys(base_key, span.start, span.count)


__all__ = [
    "INDEX_DTYPE",
    "ChunkSpan",
    "element_keys",
    "iter_chunk_keys",
    "make_chunk_plan",
]

"""Host-side length-bucketing: bucket selection and pre-JIT padding.

Bucketing bounds XLA recompilation by padding each variable-length input up to one
of a small, fixed set of bucket sizes, so the device only ever sees a handful of
shapes (one cached executable per bucket) instead of one per distinct length.

Both the bucket *decision* (``select_bucket``) and the *padding* (``bucketize``)
run on the host, **before** the JIT boundary. This is deliberate: JAX has no
dynamic shapes inside ``jit``, and padding the variable-length input on-device
(e.g. ``jnp.pad`` of a length-L array) is itself shape-specialized and would
recompile per length — defeating the purpose. Padding host-side with NumPy keeps
the device executable keyed only on the bucket size.

Typical use::

    bucket = select_bucket(len(seq), boundaries=(128, 256, 512))
    padded, mask = bucketize(seq, bucket)          # host (NumPy)
    out = jitted_step(padded)                       # device; cache hit per bucket
    out = out[mask]                                 # drop padding
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Any

import numpy as np


def select_bucket(length: int, boundaries: Sequence[int]) -> int:
    """Return the smallest bucket boundary >= ``length``.

    Pure host function (no JAX). Mirrors the recompilation-risk contract: a length
    exceeding the largest boundary is an error rather than a silent new shape.

    Args:
        length: Non-negative leading-axis length of a single input.
        boundaries: Sorted, strictly-ascending bucket sizes (non-empty).

    Returns:
        The smallest boundary value that is >= ``length``.

    Raises:
        ValueError: If ``length`` is negative, ``boundaries`` is empty, or
            ``length`` exceeds the largest boundary.
    """
    if length < 0:
        raise ValueError(f"select_bucket: length must be >= 0, got {length}.")
    if len(boundaries) == 0:
        raise ValueError("select_bucket: boundaries must be non-empty.")

    # bisect_left finds the first boundary >= length (exact match is valid).
    idx = bisect.bisect_left(boundaries, length)
    if idx >= len(boundaries):
        raise ValueError(
            f"select_bucket: length {length} exceeds the largest bucket boundary "
            f"{boundaries[-1]}. Add a larger boundary or shard the input."
        )
    return boundaries[idx]


def bucketize(xs: Any, bucket_size: int) -> tuple[Any, np.ndarray]:
    """Pad the leading axis of every leaf of ``xs`` up to ``bucket_size`` (host).

    Pads with NumPy so the device only sees ``bucket_size``-shaped arrays. Returns
    the padded pytree alongside a boolean mask marking the original (non-padded)
    positions so callers can drop the padding after the device step.

    Args:
        xs: A pytree of array-likes sharing the same leading-axis length L.
        bucket_size: Target leading-axis length; must be >= L.

    Returns:
        A tuple ``(padded_xs, original_length_mask)`` where ``padded_xs`` mirrors
        the structure of ``xs`` with each leaf's leading axis padded to
        ``bucket_size``, and ``original_length_mask`` is a boolean ``np.ndarray`` of
        shape ``(bucket_size,)`` that is True over the first L positions.

    Raises:
        ValueError: If ``xs`` has no array leaves, leaves disagree on leading-axis
            length, or ``bucket_size`` is smaller than the leading-axis length.
    """
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(xs)
    if not leaves:
        raise ValueError("bucketize: xs has no array leaves to pad.")

    arrays = [np.asarray(leaf) for leaf in leaves]
    seq_len = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != seq_len:
            raise ValueError(
                "bucketize: all leaves must share the same leading-axis length; "
                f"got {seq_len} and {arr.shape[0]}."
            )
    if bucket_size < seq_len:
        raise ValueError(
            f"bucketize: bucket_size {bucket_size} is smaller than the input "
            f"leading-axis length {seq_len}."
        )

    pad_amount = bucket_size - seq_len
    padded_leaves = [
        np.pad(arr, [(0, pad_amount)] + [(0, 0)] * (arr.ndim - 1))
        for arr in arrays
    ]
    padded_xs = jax.tree_util.tree_unflatten(treedef, padded_leaves)
    mask = np.arange(bucket_size) < seq_len
    return padded_xs, mask

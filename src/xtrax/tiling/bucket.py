"""Host-side length-bucketing: bucket selection and pre-JIT padding.

Bucketing bounds XLA recompilation by padding each variable-length input up to one
of a small, fixed set of bucket sizes, so the device only ever sees a handful of
shapes (one cached executable per bucket) instead of one per distinct length.

``BUCKET_LADDER`` is the shared length ladder (re-exported by
``xtrax.export.rings``). ``valid_span``, ``select_rung``, ``trim_axis``, and
``pad_axis`` are the domain-free span, rung, and axis-resize helpers.

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

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int

# Fixed length buckets. Owned here so runtime tiling does not import the export
# stack. ``xtrax.export.rings`` re-exports this same object.
BUCKET_LADDER: tuple[int, ...] = (64, 128, 256, 512, 1024, 1536, 2048)


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


def valid_span(mask: Any) -> int:
    """Return how many leading positions the valid entries of ``mask`` occupy.

    Host-side. The length axis is the last axis. Leading axes are rows: the
    result is the maximum, over rows, of (index of the last nonzero entry + 1).
    That is the span a bucket rung has to cover, including holes between valid
    positions. An all-masked row contributes 0. A 1-D mask is a single row.

    Args:
        mask: Boolean or numeric array-like of shape ``(L,)`` or ``(batch, L)``.
            Nonzero entries are valid. Extra leading axes are flattened into the
            batch.

    Returns:
        The span. ``0`` when every entry is masked out, the array is empty, or
        the length axis has size 0.
    """
    mask_np = np.asarray(mask)
    if mask_np.ndim == 0 or mask_np.size == 0 or mask_np.shape[-1] == 0:
        return 0
    length = int(mask_np.shape[-1])
    valid = mask_np.reshape(-1, length) > 0
    if not bool(valid.any()):
        return 0
    # span per row = index of the last valid position + 1 (0 if the row is empty)
    last_from_end = np.argmax(valid[:, ::-1], axis=-1)
    span = np.where(valid.any(axis=-1), length - last_from_end, 0)
    return int(span.max())


def select_rung(span: int, cap: int, ladder: Sequence[int]) -> int:
    """Return the smallest ladder rung >= ``span``, capped at ``cap``.

    ``cap`` is the caller's current length along the padded axis. The selected
    rung is never larger than ``cap``, so a short input is not grown to a bigger
    bucket. ``cap`` itself does not have to be on the ladder.

    Args:
        span: Valid span, typically from ``valid_span``. Must be >= 0.
        cap: Upper bound on the returned length. Must be >= 0.
        ladder: Sorted, strictly-ascending rung sizes (non-empty).

    Returns:
        ``min(select_bucket(span, ladder), cap)``.

    Raises:
        ValueError: If ``span`` or ``cap`` is negative, ``ladder`` is empty, or
            ``span`` exceeds the largest rung. A span past the ladder is an
            error even when ``cap`` is larger — there is no legal rung that
            covers it.
    """
    if cap < 0:
        raise ValueError(f"select_rung: cap must be >= 0, got {cap}.")
    return min(select_bucket(span, ladder), cap)


def trim_axis(arr: Any, n: int, axis: int) -> Any:
    """Slice ``arr`` to length ``n`` along ``axis``. ``None`` passes through.

    Args:
        arr: Array to trim. ``None`` is returned unchanged.
        n: Exclusive end index on ``axis``. Must be >= 0.
        axis: Axis to trim. Negative axes count from the end.

    Returns:
        ``arr`` sliced to ``n`` on ``axis``, or ``None``.

    Raises:
        ValueError: If ``n`` is negative.
        IndexError: If ``axis`` is out of range.
    """
    if arr is None:
        return None
    if n < 0:
        raise ValueError(f"trim_axis: n must be >= 0, got {n}.")
    axis_norm = _normalize_axis(axis, arr.ndim, op="trim_axis")
    slices: list[slice] = [slice(None)] * arr.ndim
    slices[axis_norm] = slice(0, n)
    return arr[tuple(slices)]


def pad_axis(arr: Any, n: int, axis: int, fill: Any = 0) -> Any:
    """Pad ``arr`` with ``fill`` along ``axis`` out to length ``n``.

    ``None`` passes through. NumPy arrays stay NumPy; other arrays are padded
    with ``jax.numpy.pad``.

    Args:
        arr: Array to pad. ``None`` is returned unchanged.
        n: Length of ``axis`` after padding. Must be >= the current length.
        axis: Axis to pad. Negative axes count from the end.
        fill: Pad value. Default 0.

    Returns:
        ``arr`` when that axis is already length ``n``, otherwise a copy padded
        with ``fill`` at the end of ``axis``.

    Raises:
        ValueError: If ``n`` is negative, or shorter than the current length.
        IndexError: If ``axis`` is out of range.
    """
    if arr is None:
        return None
    if n < 0:
        raise ValueError(f"pad_axis: target length must be >= 0, got {n}.")
    axis_norm = _normalize_axis(axis, arr.ndim, op="pad_axis")
    current = int(arr.shape[axis_norm])
    if current == n:
        return arr
    if current > n:
        raise ValueError(f"pad_axis: cannot pad axis {axis} from {current} to {n}.")
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis_norm] = (0, n - current)
    if isinstance(arr, np.ndarray):
        return np.pad(arr, pad_width, mode="constant", constant_values=fill)
    import jax.numpy as jnp

    return jnp.pad(arr, pad_width, constant_values=fill)


def _normalize_axis(axis: int, ndim: int, *, op: str) -> int:
    axis_norm = axis if axis >= 0 else ndim + axis
    if axis_norm < 0 or axis_norm >= ndim:
        raise IndexError(f"{op}: axis {axis} is out of range for ndim {ndim}.")
    return axis_norm


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
    padded_leaves = [np.pad(arr, [(0, pad_amount)] + [(0, 0)] * (arr.ndim - 1)) for arr in arrays]
    padded_xs = jax.tree_util.tree_unflatten(treedef, padded_leaves)
    mask = np.arange(bucket_size) < seq_len
    return padded_xs, mask


class BucketDispatcher:
    """Host-side batch coordinator for variable-length inputs across fixed bucket rungs.

    Attributes:
        boundaries: Strictly ascending sequence of bucket capacity thresholds.
        fn: Core device computation to apply to each bucketed batch.
        axis: The variable-length axis to bucket (default: 0).
        pad_value: Value used to pad unused slots (default: 0).
    """

    boundaries: tuple[int, ...]
    fn: Callable[[Any], Any]
    axis: int
    pad_value: Any
    _cached_runners: dict[int, Callable[[Any], Any]]

    def __init__(
        self,
        fn: Callable[[Any], Any],
        boundaries: Sequence[int] = BUCKET_LADDER,
        axis: int = 0,
        pad_value: Any = 0,
    ) -> None:
        if not boundaries:
            raise ValueError("BucketDispatcher: boundaries must not be empty.")
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError(
                f"BucketDispatcher: boundaries must be strictly ascending, got {boundaries}"
            )
        self.boundaries = tuple(boundaries)
        self.fn = fn
        self.axis = axis
        self.pad_value = pad_value
        self._cached_runners = {}

    def _get_runner(self, bucket_size: int) -> Callable[[Any], Any]:
        """Retrieve or compile the JIT runner for a specific bucket size."""
        if bucket_size not in self._cached_runners:
            self._cached_runners[bucket_size] = jax.jit(jax.vmap(self.fn))
        return self._cached_runners[bucket_size]

    def dispatch_batch(self, items: Sequence[Any]) -> list[Any]:
        """Dispatch a heterogeneous sequence of PyTree items through bucketed device execution.

        Args:
            items: Sequence of PyTree inputs, each sharing leaf structure but differing in length.

        Returns:
            List of unpadded PyTree results in exact 1:1 correspondence with `items`.

        Raises:
            ValueError: If any item exceeds the largest bucket boundary or has no array leaves.
        """
        if not items:
            return []

        item_lengths = []
        bucket_assignments: dict[int, list[int]] = {b: [] for b in self.boundaries}

        for idx, item in enumerate(items):
            leaves = jax.tree_util.tree_leaves(item)
            if not leaves:
                raise ValueError(f"BucketDispatcher: item {idx} contains no array leaves.")
            length = int(leaves[0].shape[self.axis])
            item_lengths.append(length)

            bucket = select_bucket(length, self.boundaries)
            bucket_assignments[bucket].append(idx)

        results: list[Any | None] = [None] * len(items)

        for bucket, indices in bucket_assignments.items():
            if not indices:
                continue

            runner = self._get_runner(bucket)

            bucket_items = [items[i] for i in indices]
            padded_batch = []

            for item in bucket_items:
                padded = jax.tree_util.tree_map(
                    lambda leaf: pad_axis(
                        np.asarray(leaf), bucket, axis=self.axis, fill=self.pad_value
                    ),
                    item,
                )
                padded_batch.append(padded)

            stacked_batch = jax.tree_util.tree_map(
                lambda *leaves: np.stack(leaves, axis=0),
                *padded_batch,
            )

            device_out = runner(stacked_batch)

            for local_idx, global_idx in enumerate(indices):
                orig_len = item_lengths[global_idx]
                single_out = jax.tree_util.tree_map(
                    lambda leaf, l_idx=local_idx: leaf[l_idx],
                    device_out,
                )

                def _trim_leaf(leaf: Any, trim_len: int = orig_len) -> Any:
                    if (
                        hasattr(leaf, "ndim")
                        and leaf.ndim > self.axis
                        and leaf.shape[self.axis] == bucket
                    ):
                        return trim_axis(leaf, trim_len, axis=self.axis)
                    return leaf

                trimmed_out = jax.tree_util.tree_map(_trim_leaf, single_out)
                results[global_idx] = trimmed_out

        return [r for r in results if r is not None]


@dataclass(frozen=True)
class BatchedCapacityState:
    """Immutable state tracking capacity and occupancy for batched memory structures.

    Attributes:
        allocated_capacity: Static integer capacity currently allocated per batch lane.
        active_counts: JAX array of shape (B,) with integer active element counts.
        headroom_threshold: Minimum headroom before reallocation is required.
    """

    allocated_capacity: int
    active_counts: Int[Array, "..."]
    headroom_threshold: int = 0

    def __post_init__(self) -> None:
        if self.allocated_capacity < 0:
            raise ValueError(
                "BatchedCapacityState: allocated_capacity must be >= 0, "
                f"got {self.allocated_capacity}"
            )

    @property
    def headroom(self) -> Int[Array, "..."]:
        """Remaining capacity per batch item: (allocated_capacity - active_counts)."""
        return self.allocated_capacity - self.active_counts

    @property
    def needs_reallocation(self) -> Bool[Array, "..."]:
        """Scalar boolean indicating if any batch item has headroom <= threshold."""
        return jnp.any(self.headroom <= self.headroom_threshold)

    def suggest_next_rung(self, ladder: Sequence[int] = BUCKET_LADDER) -> int:
        """Compute the next bucket rung from a ladder that accommodates max occupancy.

        Host-Side Allocation Coordinator:
            This method concretizes active occupancy via `int(jnp.max(self.active_counts))`
            to select the next discrete capacity rung from the ladder. Must be called
            outside the `@jax.jit` boundary.
        """
        max_active = int(jnp.max(self.active_counts))
        return select_bucket(max_active + 1, ladder)


jax.tree_util.register_pytree_node(
    BatchedCapacityState,
    lambda s: ((s.active_counts,), (s.allocated_capacity, s.headroom_threshold)),
    lambda aux, children: BatchedCapacityState(
        allocated_capacity=aux[0],
        active_counts=children[0],
        headroom_threshold=aux[1],
    ),
)

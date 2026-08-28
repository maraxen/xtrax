"""synthesize_dedup_spec — exact-on-fire DedupSpec construction (spec §4.3).

Synthesizes DedupSpec instances from batch data via two-stage algorithm:
  1. Sample stage (cheap gate): uniform-stride sample over [0, N), estimates
     duplication ratio. Below threshold → no spec, zero O(N) transfer.
  2. Exact stage (only on fire): transfer all N rows, compute exact unique_indices
     and index_map covering every position (len(index_map) == N). If k > max_unique_k,
     returns with stage="k_over_limit" and the O(N) transfer cost recorded.

Component C covers dedup-spec synthesis and collision semantics (spec §4.3).
Two distinct collision errors (OBJ-R1-03, C4): DedupSynthesisCollisionError
(synthesize path, when caller already declared the axis) and DedupSpecCollisionError
(merge-helper path, generic multi-spec collision).

Design invariants (F3, F4):
  - unique_indices = ascending FIRST-OCCURRENCE POSITIONS of distinct rows
  - index_map[i] ∈ [0, k) selects which canonical row position i uses
  - to_dedup_gather() edge-pads unique_indices by repeating last; no index_map
    entry must select ≥k (verified by DedupSpec.__post_init__)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.tiling.dedup import DedupSpec

__all__ = [
    "DedupSpecCollisionError",
    "DedupSynthesisCollisionError",
    "DedupSynthesisResult",
    "DedupSynthesisUnsupportedError",
    "merge_dedup_specs",
    "synthesize_dedup_spec",
]


class DedupSynthesisUnsupportedError(Exception):
    """Raised when synthesize_dedup_spec encounters unsupported input structure.

    E.g., heterogeneous axes (different element widths) in v1.
    """


class DedupSynthesisCollisionError(Exception):
    """Raised when existing_specs already declares the target axis_name.

    Caller-declared intent always wins; collision indicates conflicting
    dedup specifications for the same axis.
    """


class DedupSpecCollisionError(Exception):
    """Raised by merge_dedup_specs when multiple specs target the same axis_name.

    Generic merge-helper error (used when caller-vs-synthesized or caller-vs-caller
    specs collide during merge operations).
    """


@dataclass(frozen=True)
class DedupSynthesisResult:
    """Result of dedup-spec synthesis (spec §4.3, OBJ-R2-07).

    Attributes:
        spec: DedupSpec instance, or None if synthesis did not produce a spec.
        stage: String describing the outcome. One of:
            "no_duplication" (sampled_ratio == 0.0; below threshold at 0)
            "below_threshold" (0 < sampled_ratio < threshold; did not justify exact stage)
            "synthesized" (exact stage succeeded; spec produced)
            "k_over_limit" (exact stage found k > max_unique_k; rejected)
        sampled_ratio: Estimated duplication ratio from the sample stage, [0, 1].
        transfer_bytes_spent: Total device→host bytes transferred across all stages.
            Sample stage contributes sampled rows; exact stage adds all N rows if fired.
        k_bucket_bytes: Padded working-set bytes when synthesized (k_bucket * element_width);
            0 if no spec produced. Budget-mode advisory input.
    """

    spec: DedupSpec | None
    stage: str
    sampled_ratio: float
    transfer_bytes_spent: int
    k_bucket_bytes: int


def synthesize_dedup_spec(
    batch_leaves: Sequence[Any],
    *,
    axis: int = 0,
    threshold: float = 0.5,
    max_sample_rows: int = 4096,
    max_unique_k: int = 256,
    existing_specs: Mapping[str, DedupSpec] | None = None,
) -> DedupSynthesisResult:
    """Auto-synthesize exact DedupSpec from batch evidence via two-stage algorithm.

    Two-stage construction (OBJ-R1-01):
      1. **Sample stage (cheap gate)**: rows sampled by UNIFORM STRIDE over [0, N).
         Rows selected: idx = linspace(0, N-1, min(N, max_sample_rows)).astype(int);
         deduplicated and sorted. Estimates duplication ratio. Below threshold →
         result with stage="no_duplication" (if ratio==0.0) or "below_threshold",
         zero O(N) spend. Sampling NEVER produces a spec.
      2. **Exact stage (only on fire)**: transfer ALL N rows once; compute exact
         unique_indices and index_map covering every one of the N positions. If
         exact k > max_unique_k → result with stage="k_over_limit", carrying O(N)
         bytes actually spent (OBJ-R2-07). Otherwise constructs DedupSpec with
         stage="synthesized".

    Args:
        batch_leaves: Sequence of array-like objects (numpy.ndarray or jax.Array)
            representing batch dimensions. All arrays must have same length along `axis`.
            Stacked along `axis` for deduplication analysis.
        axis: Batch axis for deduplication (default 0).
        threshold: Duplication-ratio threshold; below triggers early exit (default 0.5).
        max_sample_rows: Maximum rows to sample in stage 1 (default 4096).
        max_unique_k: Maximum acceptable k; exceeding triggers k_over_limit stage.
        existing_specs: Mapping of axis_name → DedupSpec for collision policy (OBJ-R1-02).
            If the target axis_name ("batch") is already present, raises
            DedupSynthesisCollisionError; caller-declared intent always wins.

    Returns:
        DedupSynthesisResult with spec=None if synthesis did not fire, or the
        constructed spec if stage="synthesized".

    Raises:
        DedupSynthesisUnsupportedError: For unsupported input structure (e.g.,
            heterogeneous axes in v1).
        DedupSynthesisCollisionError: If existing_specs already declares axis_name.
        ValueError: If batch_leaves is empty or shapes are inconsistent.

    Note:
        Residual false-negative direction (admitted): duplication confined
        between stride points at low density may not be detected (depends on
        sampled coverage). Profitable envelope (OBJ-R1-16): contiguous-row axes,
        high duplication ratio, k ≤ ~256, N ≫ k.
    """
    if not batch_leaves:
        raise ValueError("batch_leaves cannot be empty")

    axis_name = "batch"

    # Collision check: caller-declared intent wins (OBJ-R1-02)
    if existing_specs is not None and axis_name in existing_specs:
        raise DedupSynthesisCollisionError(
            f"existing_specs already declares axis_name={axis_name!r}; "
            "caller-declared intent always wins"
        )

    # Stack batch_leaves along axis for analysis; ensure consistent shapes.
    stacked = _stack_batch_leaves(batch_leaves, axis=axis)
    N = stacked.shape[axis]

    if N == 0:
        raise ValueError(f"batch axis {axis} has length 0")

    # Stage 1: Sample-gate
    sampled_ratio, _, sample_transfer_bytes = _sample_stage(
        stacked, axis=axis, max_sample_rows=max_sample_rows, N=N
    )

    if sampled_ratio < threshold:
        # Below threshold: no exact stage, zero O(N) spend.
        stage_label = "no_duplication" if sampled_ratio == 0.0 else "below_threshold"
        return DedupSynthesisResult(
            spec=None,
            stage=stage_label,
            sampled_ratio=sampled_ratio,
            transfer_bytes_spent=sample_transfer_bytes,
            k_bucket_bytes=0,
        )

    # Stage 2: Exact stage (only on fire)
    unique_indices, index_map, exact_n_unique, exact_transfer_bytes = _exact_stage(
        stacked, axis=axis, N=N
    )

    # Bounds-check unique_indices (F3b: indices must be in [0, N))
    if unique_indices.size > 0:
        if unique_indices.min() < 0 or unique_indices.max() >= N:
            raise ValueError(
                f"unique_indices out of bounds [0, {N}): "
                f"got range [{unique_indices.min()}, {unique_indices.max()}]"
            )

    k = exact_n_unique
    if k > max_unique_k:
        # k exceeded: return with stage="k_over_limit", carrying O(N) cost.
        return DedupSynthesisResult(
            spec=None,
            stage="k_over_limit",
            sampled_ratio=sampled_ratio,
            transfer_bytes_spent=sample_transfer_bytes + exact_transfer_bytes,
            k_bucket_bytes=0,
        )

    # Self-assert len(index_map) == N (synthesizer responsibility, spec §4.3).
    # DedupSpec.__post_init__ verifies k and index_map range [0, k), but not length.
    if len(index_map) != N:
        raise ValueError(
            f"index_map length {len(index_map)} != N ({N}); "
            "synthesizer must produce exactly one index_map entry per row"
        )

    # Construct DedupSpec (spec will self-assert k == len(unique_indices), bounds checks)
    spec = DedupSpec(
        axis_name=axis_name,
        unique_indices=unique_indices,
        index_map=index_map,
        k=k,
    )

    # Calculate k_bucket working-set bytes
    from xtrax.tiling.dedup import get_k_bucket

    k_bucket = get_k_bucket(k)
    element_width = _element_width_bytes(stacked)
    k_bucket_bytes = k_bucket * element_width

    return DedupSynthesisResult(
        spec=spec,
        stage="synthesized",
        sampled_ratio=sampled_ratio,
        transfer_bytes_spent=sample_transfer_bytes + exact_transfer_bytes,
        k_bucket_bytes=k_bucket_bytes,
    )


def merge_dedup_specs(
    *spec_mappings: Mapping[str, DedupSpec],
) -> dict[str, DedupSpec]:
    """Merge multiple DedupSpec mappings, detecting axis_name collisions.

    Utility for merging caller-declared and synthesized specs. Raises
    DedupSpecCollisionError on ANY duplicate axis_name regardless of entry
    route (one loud failure semantic for the whole subsystem, completing
    OBJ-R1-03).

    Args:
        *spec_mappings: Variable number of Mapping[str, DedupSpec] to merge.

    Returns:
        Merged dict of axis_name → DedupSpec.

    Raises:
        DedupSpecCollisionError: If any axis_name appears in more than one mapping.
    """
    seen_axes = {}
    result = {}

    for mapping in spec_mappings:
        if mapping is None:
            continue
        for axis_name, spec in mapping.items():
            if axis_name in seen_axes:
                raise DedupSpecCollisionError(
                    f"axis_name={axis_name!r} appears in multiple spec mappings; "
                    "dedup axis names must be unique"
                )
            seen_axes[axis_name] = True
            result[axis_name] = spec

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _stack_batch_leaves(batch_leaves: Sequence[Any], axis: int) -> jax.Array:
    """Stack batch_leaves along axis and validate shape consistency.

    Raises DedupSynthesisUnsupportedError if any leaf is heterogeneous/ragged
    (spec §4.3: v1 requires all batch leaves to be proper rectangular arrays).
    """
    if not batch_leaves:
        raise ValueError("batch_leaves is empty")

    # Normalize to list and check for heterogeneous/ragged leaves early.
    leaves_list = list(batch_leaves)

    # Check for heterogeneous/ragged leaves (spec §4.3, OBJ-R1-10).
    # A leaf is heterogeneous if it cannot be represented as a single rectangular
    # numeric array. Only genuinely untyped inputs (raw Python list/tuple) require
    # conversion to detect raggedness. jax.Array and np.ndarray are always dense/
    # rectangular with real numeric dtype by construction, so no check/conversion needed.
    for i, leaf in enumerate(leaves_list):
        if isinstance(leaf, jax.Array):
            # jax.Array is always dense/rectangular with a real numeric dtype;
            # ragged/object-dtype data cannot be represented as a jax.Array,
            # so no check is needed and no device->host transfer should occur here.
            continue
        if isinstance(leaf, np.ndarray):
            # Already a real ndarray -- checking .dtype is metadata-only, no data copy.
            if leaf.dtype == np.object_:
                raise DedupSynthesisUnsupportedError(
                    f"batch_leaves[{i}] is heterogeneous/ragged (dtype=object): "
                    "spec §4.3 v1 does not support heterogeneous batch axes; "
                    "all leaves must be proper rectangular numpy/jax arrays"
                )
            continue
        # Only genuinely untyped inputs (e.g. raw Python list/tuple) need conversion
        # to detect raggedness -- these are already host-side, so np.asarray here is cheap.
        try:
            leaf_arr = np.asarray(leaf)
        except ValueError as e:
            # numpy 2.5+ raises ValueError for inhomogeneous/ragged arrays
            if "inhomogeneous" in str(e):
                raise DedupSynthesisUnsupportedError(
                    f"batch_leaves[{i}] is heterogeneous/ragged: "
                    "spec §4.3 v1 does not support heterogeneous batch axes; "
                    "all leaves must be proper rectangular numpy/jax arrays"
                ) from e
            raise
        if leaf_arr.dtype == np.object_:
            raise DedupSynthesisUnsupportedError(
                f"batch_leaves[{i}] is heterogeneous/ragged (dtype=object): "
                "spec §4.3 v1 does not support heterogeneous batch axes; "
                "all leaves must be proper rectangular numpy/jax arrays"
            )

    first = leaves_list[0]

    if axis < 0 or axis >= first.ndim:
        raise ValueError(f"axis={axis} out of range for array with ndim={first.ndim}")

    # Move axis to position 0 for analysis.
    leaves_moved = [jnp.moveaxis(leaf, axis, 0) for leaf in leaves_list]
    N = leaves_moved[0].shape[0]

    if N == 0:
        raise ValueError(f"batch axis {axis} has length 0")

    # Verify all have same batch dimension N.
    for i, leaf in enumerate(leaves_moved):
        if leaf.shape[0] != N:
            raise ValueError(
                f"batch_leaves[{i}] has batch dimension {leaf.shape[0]} but expected {N}"
            )

    # Concatenate along feature dimension (after axis 0).
    return jnp.concatenate([leaf.reshape(N, -1) for leaf in leaves_moved], axis=1)


def _sample_stage(
    stacked: jax.Array, axis: int, max_sample_rows: int, N: int
) -> tuple[float, int, int]:
    """Sample-stage: estimate duplication ratio via uniform-stride sampling.

    Returns: (sampled_ratio, n_unique_sampled, transfer_bytes).
    """
    # Uniform-stride sampling: idx = linspace(0, N-1, min(N, max_sample_rows))
    sample_count = min(N, max_sample_rows)
    idx = np.linspace(0, N - 1, sample_count, dtype=np.int32)
    # Deduplicate and sort to get unique sample indices.
    idx = np.unique(idx)

    # Transfer sampled rows to host for deduplication analysis.
    sampled_rows = np.asarray(stacked[idx, :])
    sampled_ratio = _estimate_duplication_ratio(sampled_rows)

    # Calculate bytes transferred in sample stage.
    element_width = _element_width_bytes(stacked)
    transfer_bytes = len(idx) * element_width

    # Estimated number of unique rows from sampling.
    n_unique_sampled = int(np.round(len(idx) * (1.0 - sampled_ratio)))

    return sampled_ratio, n_unique_sampled, transfer_bytes


def _exact_stage(stacked: jax.Array, axis: int, N: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Exact-stage: compute unique_indices and index_map for all N rows.

    Returns: (unique_indices, index_map, k, transfer_bytes).
    """
    # Transfer all N rows to host for exact deduplication.
    all_rows = np.asarray(stacked)
    element_width = _element_width_bytes(stacked)
    transfer_bytes = N * element_width

    # Compute unique rows by identity (row equality).
    # unique_indices = ascending FIRST-OCCURRENCE POSITIONS (F3).
    # index_map[i] selects which canonical row position i uses.
    unique_rows, index_map_raw = np.unique(all_rows, axis=0, return_inverse=True)
    # Defensive reshape: numpy 2.5.1's return_inverse+axis semantics return flat (N,).
    # Confirmed empirically; reshape below handles any future numpy versions gracefully.
    index_map_raw = np.asarray(index_map_raw).reshape(-1)
    n_unique = len(unique_rows)

    # Convert to first-occurrence positions.
    # index_map_raw[i] ∈ [0, n_unique) — which unique row row i equals.
    # We need unique_indices = first position where each unique row appears.
    unique_indices_list = []
    for unique_idx in range(n_unique):
        first_pos = np.where(index_map_raw == unique_idx)[0][0]
        unique_indices_list.append(first_pos)

    unique_indices = np.array(unique_indices_list, dtype=np.int32)
    # Sort by first-occurrence position to maintain ascending order (F3).
    sort_order = np.argsort(unique_indices)
    unique_indices = unique_indices[sort_order]

    # Rebuild index_map to reflect the sorted unique_indices order.
    # Map old unique_idx (from np.unique return_inverse) to new position in sorted order.
    old_to_new = np.empty(n_unique, dtype=np.int32)
    old_to_new[sort_order] = np.arange(n_unique, dtype=np.int32)
    index_map = old_to_new[index_map_raw]

    return unique_indices, index_map, n_unique, transfer_bytes


def _estimate_duplication_ratio(rows: np.ndarray) -> float:
    """Estimate duplication ratio (1 - unique_count / total_count) from sample."""
    n_rows = len(rows)
    n_unique = len(np.unique(rows, axis=0))
    return 1.0 - (n_unique / n_rows)


def _element_width_bytes(stacked: jax.Array) -> int:
    """Calculate bytes per row in the stacked array."""
    nbytes = stacked.dtype.itemsize
    # nbytes is bytes per element; multiply by feature dimension.
    feature_dim = stacked.shape[1]
    return nbytes * feature_dim

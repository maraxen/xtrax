"""synthesize_dedup_spec / verify_dedup_spec — dedup-spec synthesis and verification.

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

Row identity (spec 260922 §5, T4): rows are compared by **exact native bytes per
leaf**, not by a promoted, concatenated float view. Every leaf is converted to an
(N, B_l) uint8 block — the device path bitcasts in native dtype (bool/complex/
sub-byte integers get their own recipe; sub-byte floats are refused), the numpy
host path views native bytes directly — and blocks are concatenated along the
feature axis. This makes `±0.0` rows distinct, deduplicates bitwise-identical NaN
rows, and stops x64-off truncation of numpy int64/float64 leaves (260922 P2-P4).

`verify_dedup_spec` (spec 260922 §4) checks a DedupSpec's row-identity claim
against the same byte definition, so the two functions cannot drift: every row
`i` must be byte-identical, per leaf, to its canonical row
`unique_indices[index_map[i]]`. It checks structure only (index_type, index_dtype,
k_mismatch, index_map_length, unique_indices_bounds, index_map_bounds) before any
device→host transfer, then compares row bytes with exactly one transfer call.
Compute-equivalence (claim ii, "does re-running fn on deduped rows reproduce the
full output") is not checked here; that is follow-up #5217.

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
from jax import lax

from xtrax.tiling.dedup import DedupSpec

__all__ = [
    "DedupSpecCollisionError",
    "DedupSpecVerificationError",
    "DedupSynthesisCollisionError",
    "DedupSynthesisResult",
    "DedupSynthesisUnsupportedError",
    "DedupVerificationResult",
    "merge_dedup_specs",
    "synthesize_dedup_spec",
    "verify_dedup_spec",
]


class DedupSynthesisUnsupportedError(Exception):
    """Raised when synthesize_dedup_spec or verify_dedup_spec encounters
    unsupported input structure.

    E.g., heterogeneous axes (different element widths), typed PRNG key
    leaves, sub-byte float dtypes, or dtypes with no defined byte recipe.
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


class DedupSpecVerificationError(ValueError):
    """Raised by verify_dedup_spec when a DedupSpec's claim does not hold
    (spec 260922 §4.2).

    Attributes:
        check: which check failed. One of "index_type", "index_dtype",
            "k_mismatch", "index_map_length", "unique_indices_bounds",
            "index_map_bounds", "row_mismatch".
        first_bad_row: for "row_mismatch", the smallest row index that is bad
            in any leaf (union semantics). None otherwise.
        n_bad: for "row_mismatch", the count of rows that are bad in any leaf
            (union semantics). 0 otherwise.
        leaf_index: for "row_mismatch", the lowest leaf index mismatching at
            first_bad_row. None otherwise.
    """

    def __init__(
        self,
        check: str,
        message: str,
        *,
        first_bad_row: int | None = None,
        n_bad: int = 0,
        leaf_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.check = check
        self.first_bad_row = first_bad_row
        self.n_bad = n_bad
        self.leaf_index = leaf_index


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


@dataclass(frozen=True)
class DedupVerificationResult:
    """Result of verify_dedup_spec (spec 260922 §4.2).

    Attributes:
        n_rows: N, the batch length verified.
        k: spec.k, the number of canonical rows the spec claims.
        transfer_bytes_spent: device→host bytes moved via `_to_host` (the
            (N, L) mismatch mask; N*L bytes regardless of leaf dtype).
    """

    n_rows: int
    k: int
    transfer_bytes_spent: int


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
         Rows selected: idx = round(linspace(0, N-1, min(N, max_sample_rows)));
         deduplicated and sorted. Estimates duplication ratio. Below threshold →
         result with stage="no_duplication" (if ratio==0.0) or "below_threshold",
         zero O(N) spend. Sampling NEVER produces a spec.
      2. **Exact stage (only on fire)**: transfer ALL N rows once; compute exact
         unique_indices and index_map covering every one of the N positions. If
         exact k > max_unique_k → result with stage="k_over_limit", carrying O(N)
         bytes actually spent (OBJ-R2-07). Otherwise constructs DedupSpec with
         stage="synthesized".

    Row identity is **exact native bytes per leaf** (spec 260922 §5): every leaf
    is converted to an (N, B_l) uint8 block in its own dtype's native byte
    layout, and blocks are concatenated along the feature axis before
    `np.unique(..., axis=0)`. Behaviour changes from the promoted-float-concat
    predecessor (see CHANGELOG):
      - `±0.0` rows are now distinct (k may rise, possibly past max_unique_k).
      - Bitwise-identical NaN rows now dedup (k may fall).
      - Wide numpy int64/float64 leaves are no longer truncated under x64-off
        (k may rise).
      - Mixed-dtype leaves are no longer promoted to a common dtype for
        `transfer_bytes_spent`/`k_bucket_bytes` accounting (true byte widths).
      - `axis != 0` now works (the batch length was previously misread).
      - Typed PRNG key leaves are refused with a typed error.
      - Sub-byte integer leaves are compared exactly after widening; sub-byte
        float leaves are refused.

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
            heterogeneous axes, typed PRNG keys, sub-byte float dtypes).
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

    # Stack batch_leaves into a batch-first (N, ΣB_l) uint8 byte-rows array.
    stacked = _stack_batch_leaves(batch_leaves, axis=axis)
    N = stacked.shape[0]

    # Stage 1: Sample-gate
    sampled_ratio, _, sample_transfer_bytes = _sample_stage(
        stacked, N=N, max_sample_rows=max_sample_rows
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
    unique_indices, index_map, exact_n_unique, exact_transfer_bytes = _exact_stage(stacked, N=N)

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


def verify_dedup_spec(
    spec: DedupSpec, leaves: Sequence[Any], *, axis: int = 0
) -> DedupVerificationResult:
    """Verify that a DedupSpec's row-identity claim holds (spec 260922 §4).

    Checks claim (i), row-equality: every row `i` of `leaves` is byte-identical,
    per leaf, to its canonical row `spec.unique_indices[spec.index_map[i]]`. The
    comparison is bitwise, using the same native-byte recipe `synthesize_dedup_spec`
    uses, so the two functions cannot drift (spec 260922 §5.1). It does **not**
    check claim (ii), compute-equivalence ("does re-running fn on the deduped
    rows reproduce fn's full output") — that needs `fn` and is follow-up #5217.

    Structural checks (index_type, index_dtype, k_mismatch, index_map_length,
    unique_indices_bounds, index_map_bounds) run first, entirely on host, before
    any device→host transfer. The row check then moves exactly one (N, L) bool
    mask from device to host via `_to_host`.

    Args:
        spec: The DedupSpec to verify.
        leaves: The same batch_leaves synthesize_dedup_spec would have been
            called with. Validated identically (via the shared
            `_validate_batch_leaves`), so both functions accept the same
            inputs and raise the same errors for unsupported ones.
        axis: Batch axis (default 0), matching the axis leaves were declared on.

    Returns:
        DedupVerificationResult on success.

    Raises:
        DedupSpecVerificationError: If any structural check or the row check
            fails. `.check` names which one.
        DedupSynthesisUnsupportedError: For unsupported leaf dtypes/structure
            (identical to synthesize_dedup_spec).
        ValueError: For N mismatch or a bad axis (identical to
            synthesize_dedup_spec).
    """
    validated_leaves, N = _validate_batch_leaves(leaves, axis)
    _check_spec_structure(spec, N)

    canon = spec.unique_indices[spec.index_map]
    blocks = [_leaf_row_bytes(leaf, axis, N) for leaf in validated_leaves]
    mask = _to_host(_mismatch_mask_device(blocks, canon))

    union = mask.any(axis=1)
    n_bad = union.sum().item()
    if n_bad:
        first_bad_row = union.argmax().item()
        leaf_index = mask[first_bad_row].argmax().item()
        raise DedupSpecVerificationError(
            "row_mismatch",
            f"{n_bad} row(s) are not byte-identical to their canonical row; "
            f"first mismatch at row {first_bad_row}, leaf {leaf_index}",
            first_bad_row=first_bad_row,
            n_bad=n_bad,
            leaf_index=leaf_index,
        )

    return DedupVerificationResult(
        n_rows=N, k=spec.k, transfer_bytes_spent=N * len(validated_leaves)
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
# Private helpers — host-side glue (unrestricted control flow)
# ---------------------------------------------------------------------------


def _validate_batch_leaves(
    batch_leaves: Sequence[Any], axis: int
) -> tuple[list[np.ndarray | jax.Array], int]:
    """Validate batch_leaves and return (typed leaves, N) (spec 260922 §5.1).

    Shared by synthesize_dedup_spec and verify_dedup_spec so both accept
    identical inputs and raise identical errors.

    Raises DedupSynthesisUnsupportedError if any leaf is heterogeneous/ragged,
    object-dtype, or a typed PRNG key (spec §4.3, P7). Raises ValueError for a
    bad axis or mismatched batch length.
    """
    if not batch_leaves:
        raise ValueError("batch_leaves is empty")

    leaves_list: list[np.ndarray | jax.Array] = []

    # Check for heterogeneous/ragged leaves (spec §4.3, OBJ-R1-10).
    # A leaf is heterogeneous if it cannot be represented as a single rectangular
    # numeric array. Only genuinely untyped inputs (raw Python list/tuple) require
    # conversion to detect raggedness. jax.Array and np.ndarray are always dense/
    # rectangular with real numeric dtype by construction, so no check/conversion needed.
    for i, leaf in enumerate(batch_leaves):
        if isinstance(leaf, jax.Array):
            # jax.Array is always dense/rectangular with a real numeric dtype;
            # ragged/object-dtype data cannot be represented as a jax.Array,
            # so no check is needed and no device->host transfer should occur here.
            arr: np.ndarray | jax.Array = leaf
        elif isinstance(leaf, np.ndarray):
            # Already a real ndarray -- checking .dtype is metadata-only, no data copy.
            if leaf.dtype == np.object_:
                raise DedupSynthesisUnsupportedError(
                    f"batch_leaves[{i}] is heterogeneous/ragged (dtype=object): "
                    "spec §4.3 v1 does not support heterogeneous batch axes; "
                    "all leaves must be proper rectangular numpy/jax arrays"
                )
            arr = leaf
        else:
            # Only genuinely untyped inputs (e.g. raw Python list/tuple) need
            # conversion to detect raggedness -- these are already host-side, so
            # np.asarray here is cheap.
            try:
                arr = np.asarray(leaf)
            except ValueError as e:
                # numpy 2.5+ raises ValueError for inhomogeneous/ragged arrays
                if "inhomogeneous" in str(e):
                    raise DedupSynthesisUnsupportedError(
                        f"batch_leaves[{i}] is heterogeneous/ragged: "
                        "spec §4.3 v1 does not support heterogeneous batch axes; "
                        "all leaves must be proper rectangular numpy/jax arrays"
                    ) from e
                raise
            if arr.dtype == np.object_:
                raise DedupSynthesisUnsupportedError(
                    f"batch_leaves[{i}] is heterogeneous/ragged (dtype=object): "
                    "spec §4.3 v1 does not support heterogeneous batch axes; "
                    "all leaves must be proper rectangular numpy/jax arrays"
                )

        # Typed-key rejection (P7): a leaf whose dtype is a typed PRNG key
        # cannot be reduced to bytes meaningfully; the remedy is key_data().
        if jax.dtypes.issubdtype(arr.dtype, jax.dtypes.prng_key):
            raise DedupSynthesisUnsupportedError(
                f"batch_leaves[{i}] has a typed PRNG key dtype ({arr.dtype}); "
                "pass jax.random.key_data(keys) instead"
            )

        leaves_list.append(arr)

    first = leaves_list[0]

    if axis < 0 or axis >= first.ndim:
        raise ValueError(f"axis={axis} out of range for array with ndim={first.ndim}")

    N = first.shape[axis]

    if N == 0:
        raise ValueError(f"batch axis {axis} has length 0")

    # Verify all have same batch dimension N.
    for i, leaf in enumerate(leaves_list):
        if axis < 0 or axis >= leaf.ndim:
            raise ValueError(f"axis={axis} out of range for array with ndim={leaf.ndim}")
        if leaf.shape[axis] != N:
            raise ValueError(
                f"batch_leaves[{i}] has batch dimension {leaf.shape[axis]} but expected {N}"
            )

    return leaves_list, N


def _row_byte_kind(dtype: Any) -> str:
    """Classify a leaf's dtype for row-byte conversion (spec 260922 §5.1).

    Order matters: bool and complex are checked before the generic integer/
    floating branches so they never fall through to sub-byte widening. Uses
    only `jnp.issubdtype`/`jax.dtypes.itemsize_bits` — adds no `np.*` call site.
    """
    if jnp.issubdtype(dtype, np.bool_):
        return "bool"
    if jnp.issubdtype(dtype, np.complexfloating):
        return "complex"
    if jnp.issubdtype(dtype, np.integer):
        if jax.dtypes.itemsize_bits(dtype) < 8:
            return (
                "subbyte_unsigned"
                if jnp.issubdtype(dtype, np.unsignedinteger)
                else "subbyte_signed"
            )
        return "plain"
    if jnp.issubdtype(dtype, np.floating):
        if jax.dtypes.itemsize_bits(dtype) < 8:
            raise DedupSynthesisUnsupportedError(
                f"dedup row identity does not support sub-byte float dtype {dtype}: "
                "widening would be a numeric, not a lossless, conversion"
            )
        return "plain"
    raise DedupSynthesisUnsupportedError(f"dedup row identity does not support dtype {dtype}")


def _host_leaf_row_bytes(a: np.ndarray, axis: int, N: int, kind: str) -> np.ndarray:
    """Host-only: view a numpy leaf's rows as native uint8 bytes (spec §5.1).

    Uses `np.moveaxis` (not `jnp.moveaxis`) to avoid x64-off truncation of
    numpy int64/float64 leaves (P4). `ascontiguousarray` makes transposed and
    strided leaves viewable. Complex and bool leaves view natively; only the
    sub-byte integer kinds need a widening `astype` first (P11).
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"_host_leaf_row_bytes expects a np.ndarray, got {type(a).__name__}")
    if kind == "subbyte_signed":
        a = a.astype(np.int8)
    elif kind == "subbyte_unsigned":
        a = a.astype(np.uint8)
    return np.ascontiguousarray(np.moveaxis(a, axis, 0)).reshape(N, -1).view(np.uint8)


def _leaf_row_bytes(leaf: np.ndarray | jax.Array, axis: int, N: int) -> jax.Array:
    """Return a leaf's rows as an (N, B_l) uint8 array on device (spec §5.1).

    NumPy leaves take the host byte-view path; jax.Array leaves take the
    device bitcast path, dispatched by `_row_byte_kind`. Unsupported dtypes
    (from either the kind classification or the device dispatch itself) are
    normalized to `DedupSynthesisUnsupportedError`.
    """
    try:
        kind = _row_byte_kind(leaf.dtype)
        if isinstance(leaf, np.ndarray):
            return jnp.asarray(_host_leaf_row_bytes(leaf, axis, N, kind))
        m = _device_rows(leaf, axis, N)
        if kind == "bool":
            return _device_bytes_bool(m)
        if kind == "complex":
            return _device_bytes_complex(m)
        if kind == "subbyte_signed":
            return _device_bytes_subbyte_signed(m)
        if kind == "subbyte_unsigned":
            return _device_bytes_subbyte_unsigned(m)
        return _device_bytes_plain(m)
    except (TypeError, ValueError) as e:
        raise DedupSynthesisUnsupportedError(
            f"dedup row identity cannot convert dtype {leaf.dtype} to native bytes"
        ) from e


def _stack_batch_leaves(batch_leaves: Sequence[Any], axis: int) -> jax.Array:
    """Validate batch_leaves and concatenate their native-byte rows (spec §5.1).

    Returns an (N, ΣB_l) uint8 array. Concatenating uint8 blocks cannot promote
    dtypes (P3), unlike the predecessor's promoted-float concatenation.
    """
    leaves, N = _validate_batch_leaves(batch_leaves, axis)
    return _concat_row_bytes([_leaf_row_bytes(leaf, axis, N) for leaf in leaves])


def _check_spec_structure(spec: DedupSpec, N: int) -> None:
    """Structural checks on a DedupSpec, run before any device→host transfer
    (spec 260922 §4.2). The first failure raises.
    """
    if not isinstance(spec.unique_indices, np.ndarray) or not isinstance(
        spec.index_map, np.ndarray
    ):
        raise DedupSpecVerificationError(
            "index_type",
            "DedupSpec.unique_indices and index_map must be np.ndarray, got "
            f"{type(spec.unique_indices).__name__}/{type(spec.index_map).__name__}",
        )
    if (
        spec.unique_indices.ndim != 1
        or not np.issubdtype(spec.unique_indices.dtype, np.integer)
        or spec.index_map.ndim != 1
        or not np.issubdtype(spec.index_map.dtype, np.integer)
    ):
        raise DedupSpecVerificationError(
            "index_dtype",
            "DedupSpec.unique_indices and index_map must be 1-D integer arrays "
            f"(got dtypes {spec.unique_indices.dtype}, {spec.index_map.dtype} and "
            f"ndims {spec.unique_indices.ndim}, {spec.index_map.ndim})",
        )
    if len(spec.unique_indices) != spec.k:
        raise DedupSpecVerificationError(
            "k_mismatch",
            f"DedupSpec.k={spec.k} but len(unique_indices)={len(spec.unique_indices)}",
        )
    if len(spec.index_map) != N:
        raise DedupSpecVerificationError(
            "index_map_length",
            f"DedupSpec.index_map has length {len(spec.index_map)}, expected N={N}",
        )
    if spec.unique_indices.size > 0 and (
        spec.unique_indices.min() < 0 or spec.unique_indices.max() >= N
    ):
        raise DedupSpecVerificationError(
            "unique_indices_bounds",
            f"DedupSpec.unique_indices out of bounds [0, {N}): got range "
            f"[{spec.unique_indices.min()}, {spec.unique_indices.max()}]",
        )
    if spec.index_map.size > 0 and (spec.index_map.min() < 0 or spec.index_map.max() >= spec.k):
        raise DedupSpecVerificationError(
            "index_map_bounds",
            f"DedupSpec.index_map out of bounds [0, {spec.k}): got range "
            f"[{spec.index_map.min()}, {spec.index_map.max()}]",
        )


def _sample_stage(stacked: jax.Array, N: int, max_sample_rows: int) -> tuple[float, int, int]:
    """Sample-stage: estimate duplication ratio via uniform-stride sampling.

    The gather happens on device (`_take_rows_device`) before the one
    `_to_host` transfer, so only the sampled rows ever leave the device.

    Returns: (sampled_ratio, n_unique_sampled, transfer_bytes).
    """
    # Uniform-stride sampling: idx = round(linspace(0, N-1, min(N, max_sample_rows)))
    sample_count = min(N, max_sample_rows)
    idx = np.round(np.linspace(0, N - 1, sample_count)).astype(np.int64)
    # Deduplicate and sort to get unique sample indices.
    idx = np.unique(idx)

    sampled_rows = _to_host(_take_rows_device(stacked, idx))
    sampled_ratio = _estimate_duplication_ratio(sampled_rows)

    # Calculate bytes transferred in sample stage.
    element_width = _element_width_bytes(stacked)
    transfer_bytes = len(idx) * element_width

    # Estimated number of unique rows from sampling.
    n_unique_sampled = round(len(idx) * (1.0 - sampled_ratio))

    return sampled_ratio, n_unique_sampled, transfer_bytes


def _exact_stage(stacked: jax.Array, N: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Exact-stage: compute unique_indices and index_map for all N rows.

    Returns: (unique_indices, index_map, k, transfer_bytes).
    """
    # Transfer all N rows to host for exact deduplication.
    all_rows = _to_host(stacked)
    element_width = _element_width_bytes(stacked)
    transfer_bytes = N * element_width

    # Compute unique rows by exact byte identity (row equality).
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
    """Calculate bytes per row in the stacked array (Σ B_l across leaves)."""
    nbytes = stacked.dtype.itemsize
    # nbytes is bytes per element; multiply by feature dimension.
    feature_dim = stacked.shape[1]
    return nbytes * feature_dim


def _to_host(x: jax.Array) -> np.ndarray:
    """The only device→host transfer route in this module (spec §7.1).

    Both verify_dedup_spec and synthesize_dedup_spec's stages route every
    transfer through this function, so the §7.1 structural oracle can assert
    on it as the single chokepoint.
    """
    if not isinstance(x, jax.Array):
        raise TypeError(f"_to_host expects a jax.Array, got {type(x).__name__}")
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Private helpers — device-only helper set D (spec §7.1; AST-allowlisted)
# ---------------------------------------------------------------------------


def _device_rows(leaf: jax.Array, axis: int, N: int) -> jax.Array:
    """Move `axis` to the front and flatten the remaining dims (on device)."""
    return jnp.moveaxis(leaf, axis, 0).reshape(N, -1)


def _device_bytes_plain(m: jax.Array) -> jax.Array:
    """Bitcast a "plain" (non-bool, non-complex, non-sub-byte) row block to uint8."""
    return lax.bitcast_convert_type(m, jnp.uint8).reshape(m.shape[0], -1)


def _device_bytes_bool(m: jax.Array) -> jax.Array:
    """bool rows: astype to uint8 (0/1), one byte per element."""
    return m.astype(jnp.uint8)


def _device_bytes_complex(m: jax.Array) -> jax.Array:
    """complex rows: split into real/imag halves, each bitcast, then concatenated.

    Injective (P9): the split preserves ±0 in the imaginary part. Device
    bitcast of complex itself is unsupported (P6), which is why this splits
    rather than bitcasting directly.
    """
    return jnp.concatenate(
        [_device_bytes_plain(jnp.real(m)), _device_bytes_plain(jnp.imag(m))], axis=1
    )


def _device_bytes_subbyte_signed(m: jax.Array) -> jax.Array:
    """Signed sub-byte integer rows: widen to int8 (exact, P11), then bitcast."""
    return _device_bytes_plain(m.astype(jnp.int8))


def _device_bytes_subbyte_unsigned(m: jax.Array) -> jax.Array:
    """Unsigned sub-byte integer rows: widen to uint8 (exact, P11)."""
    return m.astype(jnp.uint8)


def _concat_row_bytes(blocks: Sequence[jax.Array]) -> jax.Array:
    """Concatenate per-leaf uint8 row-byte blocks along the feature axis."""
    return jnp.concatenate(blocks, axis=1)


def _take_rows_device(stacked: jax.Array, idx: np.ndarray) -> jax.Array:
    """Gather rows on device, before any transfer. All indices are proven to
    lie in [0, N) on host before use, so `jnp.take`'s out-of-bounds mode never
    applies."""
    return jnp.take(stacked, jnp.asarray(idx), axis=0)


def _mismatch_mask_device(blocks: Sequence[jax.Array], canon: np.ndarray) -> jax.Array:
    """Build the (N, L) per-leaf mismatch mask on device (spec §4.2 row check)."""
    return jnp.stack(
        [jnp.any(jnp.not_equal(b, jnp.take(b, canon, axis=0)), axis=1) for b in blocks],
        axis=1,
    )

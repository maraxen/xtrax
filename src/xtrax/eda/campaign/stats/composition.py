"""Library composition and diversity analytics (Question 1).

Implements Q1 analytics per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Includes boundary guards for N <= 1 and |M| = 0 (BLK-04), safe positive-support Shannon
entropy (MAJ-07), recovery margin vs shuffled native, and cross-knob collision table (MIN-01).
Fully vectorized with zero Python loops over positions (L) or sequence pairs (N*(N-1)/2).
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np

logger = logging.getLogger("xtrax.eda.campaign.stats.composition")

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

# Precomputed 256-element ASCII to token index translation table
_AA_LUT = np.full(256, 20, dtype=np.int64)
for _i, _aa in enumerate(STANDARD_AA):
    _AA_LUT[ord(_aa)] = _i


def _seqs_to_bytes_and_tokens(sequences: Sequence[str], L: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert sequence strings to (N, L) uint8 and int64 token arrays with zero loops over L."""
    N = len(sequences)
    if N == 0 or L == 0:
        return np.empty((N, L), dtype=np.uint8), np.empty((N, L), dtype=np.int64)

    joined = "".join(s[:L] if len(s) >= L else s.ljust(L) for s in sequences)
    b_arr = np.frombuffer(joined.encode("ascii", errors="replace"), dtype=np.uint8).reshape(N, L)

    tokens = _AA_LUT[b_arr]
    return b_arr, tokens


def safe_shannon_entropy(probs: np.ndarray) -> float:
    """Calculate safe positive-support Shannon entropy in bits (MAJ-07).

    H = - sum_{p > 0} p * log2(p) >= 0.0
    Avoids negative bits and divide-by-zero warnings without artificial epsilon offsets.
    """
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    h = -float(np.sum(p * np.log2(p)))
    return max(0.0, h)


def calculate_pairwise_diversity(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> float:
    """Calculate mean pairwise diversity (1.0 - PID) with strict boundary guards (BLK-04).

    When N <= 1: returns 0.0.
    When |M| == 0: returns 0.0.
    """
    N = len(sequences)
    if N <= 1:
        return 0.0

    L = len(sequences[0])
    if L == 0:
        return 0.0

    if designable_mask is not None:
        mask = np.asarray(designable_mask, dtype=bool)
        if len(mask) != L:
            m = np.ones(L, dtype=bool)
            m[: min(len(mask), L)] = mask[: min(len(mask), L)]
            mask = m
    else:
        mask = np.ones(L, dtype=bool)

    m_count = int(np.sum(mask))
    if m_count == 0:
        return 0.0

    _, tokens = _seqs_to_bytes_and_tokens(sequences, L)

    # Broadcast match matrix
    match_matrix = (tokens[:, None, :] == tokens[None, :, :]) & mask[None, None, :]
    pid_matrix = np.sum(match_matrix, axis=2) / float(m_count)

    triu_i, triu_j = np.triu_indices(N, k=1)
    mean_pid = float(np.mean(pid_matrix[triu_i, triu_j]))
    return float(max(0.0, 1.0 - mean_pid))


def compute_positional_entropy(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute per-position and mean positional Shannon entropy in bits (vectorized).

    Zero Python loops over positions (L). Uses 2D flat-index bincount histogram
    flat_idx = col * 21 + token, running in O(N * L) time and O(L * 21) memory.
    """
    N = len(sequences)
    if N == 0:
        return 0.0, np.array([], dtype=np.float64)

    L = len(sequences[0])
    if L == 0:
        return 0.0, np.array([], dtype=np.float64)

    if designable_mask is not None:
        mask = np.asarray(designable_mask, dtype=bool)
        if len(mask) != L:
            m = np.ones(L, dtype=bool)
            m[: min(len(mask), L)] = mask[: min(len(mask), L)]
            mask = m
    else:
        mask = np.ones(L, dtype=bool)

    _, tokens = _seqs_to_bytes_and_tokens(sequences, L)
    k_classes = 21

    flat_idx = (np.arange(L, dtype=np.int64)[None, :] * k_classes + tokens).ravel()
    counts_2d = np.bincount(flat_idx, minlength=L * k_classes).reshape(L, k_classes)

    p = counts_2d.astype(np.float64) / float(N)
    log_p = np.zeros_like(p)
    np.log2(p, out=log_p, where=p > 0)
    pos_entropies = -np.sum(p * log_p, axis=1)
    pos_entropies = np.maximum(pos_entropies, 0.0)

    # Zero non-designable positions
    pos_entropies[~mask] = 0.0

    masked_h = pos_entropies[mask] if np.any(mask) else np.array([0.0])
    mean_h = float(np.mean(masked_h))
    return mean_h, pos_entropies


def compute_composition_entropy(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> float:
    """Compute global amino acid composition Shannon entropy in bits (vectorized).

    Zero Python loops over positions or sequences. Flattens designable positions
    across all sequences and evaluates empirical probability distribution via bincount.
    """
    if not sequences:
        return 0.0

    L = len(sequences[0])
    if L == 0:
        return 0.0

    if designable_mask is not None:
        mask = np.asarray(designable_mask, dtype=bool)
        if len(mask) != L:
            m = np.ones(L, dtype=bool)
            m[: min(len(mask), L)] = mask[: min(len(mask), L)]
            mask = m
    else:
        mask = np.ones(L, dtype=bool)

    m_count = int(np.sum(mask))
    if m_count == 0:
        return 0.0

    b_arr, _ = _seqs_to_bytes_and_tokens(sequences, L)
    des_bytes = b_arr[:, mask].ravel()
    counts = np.bincount(des_bytes, minlength=256)
    pos_counts = counts[counts > 0]
    probs = pos_counts.astype(np.float64) / float(len(des_bytes))
    return safe_shannon_entropy(probs)


def compute_target_recovery(
    sequences: Sequence[str],
    native_seq: str,
    designable_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute native recovery, shuffled recovery baseline, and recovery margin (vectorized).

    Zero Python loops over sequences or positions.
    Evaluates native and shuffled baseline matches via array broadcasting.

    Returns:
        (native_recovery, shuffled_recovery, recovery_margin)
    """
    if not sequences or not native_seq:
        return 0.0, 0.0, 0.0

    N = len(sequences)
    L = min(len(sequences[0]), len(native_seq))
    if L == 0:
        return 0.0, 0.0, 0.0

    if designable_mask is not None:
        mask = np.asarray(designable_mask, dtype=bool)
        if len(mask) != L:
            m = np.ones(L, dtype=bool)
            m[: min(len(mask), L)] = mask[: min(len(mask), L)]
            mask = m
    else:
        mask = np.ones(L, dtype=bool)

    m_count = int(np.sum(mask[:L]))
    if m_count == 0:
        return 0.0, 0.0, 0.0

    b_arr, _ = _seqs_to_bytes_and_tokens(sequences, L)
    native_bytes = np.frombuffer(native_seq[:L].encode("ascii", errors="replace"), dtype=np.uint8)

    # Vectorized native recovery
    native_matches = (b_arr == native_bytes[None, :]) & mask[None, :]
    mean_native_rec = float(np.sum(native_matches) / float(N * m_count))

    # Vectorized shuffled baseline with fixed seed 42
    rng = np.random.default_rng(42)
    des_idx = np.flatnonzero(mask[:L])
    perm = rng.permutation(des_idx)
    shuf_bytes = native_bytes.copy()
    shuf_bytes[des_idx] = native_bytes[perm]

    shuf_matches = (b_arr == shuf_bytes[None, :]) & mask[None, :]
    mean_shuf_rec = float(np.sum(shuf_matches) / float(N * m_count))

    margin = float(mean_native_rec - mean_shuf_rec)
    return mean_native_rec, mean_shuf_rec, margin


def compute_duplicate_rates(sequences: Sequence[str]) -> tuple[float, float]:
    """Compute within-condition exact duplicate rate and near-duplicate rate (vectorized).

    Zero Python loops over sequence pairs.
    Uses vectorized 2D array broadcasting to compute pairwise Hamming distances.

    Returns:
        (exact_duplicate_rate, near_duplicate_rate)
    """
    N = len(sequences)
    if N <= 1:
        return 0.0, 0.0

    exact_dup_rate = float(1.0 - len(set(sequences)) / float(N))

    L = len(sequences[0])
    if L == 0:
        return exact_dup_rate, 1.0

    b_arr, _ = _seqs_to_bytes_and_tokens(sequences, L)

    # Vectorized 2D pairwise Hamming distance without pair loops
    diffs = np.sum(b_arr[:, None, :] != b_arr[None, :, :], axis=2)
    triu_i, triu_j = np.triu_indices(N, k=1)
    pair_diffs = diffs[triu_i, triu_j]
    near_dups = int(np.count_nonzero(pair_diffs <= 1))
    total_pairs = len(triu_i)
    near_dup_rate = float(near_dups / float(total_pairs)) if total_pairs > 0 else 0.0

    return exact_dup_rate, near_dup_rate


def compute_cross_knob_collisions(
    condition_sequences: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str], float]:
    """Compute symmetric cross-knob sequence collision matrix (MIN-01).

    Collision(Ka, Kb) = |S(Ka) intersect S(Kb)| / min(|S(Ka)|, |S(Kb)|)
    """
    keys = list(condition_sequences.keys())
    collisions: dict[tuple[str, str], float] = {}

    for i in range(len(keys)):
        k_a = keys[i]
        set_a = set(condition_sequences[k_a])
        collisions[(k_a, k_a)] = 1.0 if len(set_a) > 0 else 0.0

        for j in range(i + 1, len(keys)):
            k_b = keys[j]
            set_b = set(condition_sequences[k_b])
            min_size = min(len(set_a), len(set_b))
            if min_size == 0:
                rate = 0.0
            else:
                overlap = len(set_a & set_b)
                rate = float(overlap / float(min_size))
            collisions[(k_a, k_b)] = rate
            collisions[(k_b, k_a)] = rate

    return collisions

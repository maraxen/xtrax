"""Library composition and diversity analytics (Question 1).

Implements Q1 analytics per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Includes boundary guards for N <= 1 and |M| = 0 (BLK-04), safe positive-support Shannon
entropy (MAJ-07), recovery margin vs shuffled native, and cross-knob collision table (MIN-01).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np

logger = logging.getLogger("xtrax.eda.campaign.stats.composition")

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}


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
            mask = np.ones(L, dtype=bool)
    else:
        mask = np.ones(L, dtype=bool)

    M_count = int(np.sum(mask))
    if M_count == 0:
        return 0.0

    # Vectorized token conversion
    tokens = np.zeros((N, L), dtype=np.int64)
    for i, s in enumerate(sequences):
        for j, c in enumerate(s[:L]):
            tokens[i, j] = AA_TO_IDX.get(c, 20)

    # Broadcast match matrix
    match_matrix = (tokens[:, None, :] == tokens[None, :, :]) & mask[None, None, :]
    pid_matrix = np.sum(match_matrix, axis=2) / float(M_count)

    triu_i, triu_j = np.triu_indices(N, k=1)
    if len(triu_i) == 0:
        return 0.0

    mean_pid = float(np.mean(pid_matrix[triu_i, triu_j]))
    return float(max(0.0, 1.0 - mean_pid))


def compute_positional_entropy(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute per-position and mean positional Shannon entropy in bits."""
    N = len(sequences)
    if N == 0:
        return 0.0, np.array([], dtype=np.float64)

    L = len(sequences[0])
    mask = (
        np.asarray(designable_mask, dtype=bool)
        if designable_mask is not None
        else np.ones(L, dtype=bool)
    )
    pos_entropies = np.zeros(L, dtype=np.float64)

    for pos in range(L):
        if pos < len(mask) and mask[pos]:
            col_aas = [s[pos] for s in sequences if pos < len(s)]
            if col_aas:
                counts = [col_aas.count(aa) for aa in set(col_aas)]
                probs = np.array(counts, dtype=np.float64) / len(col_aas)
                pos_entropies[pos] = safe_shannon_entropy(probs)

    masked_h = pos_entropies[mask] if np.any(mask) else np.array([0.0])
    mean_h = float(np.mean(masked_h))
    return mean_h, pos_entropies


def compute_composition_entropy(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> float:
    """Compute global amino acid composition Shannon entropy in bits."""
    if not sequences:
        return 0.0

    L = len(sequences[0])
    mask = (
        np.asarray(designable_mask, dtype=bool)
        if designable_mask is not None
        else np.ones(L, dtype=bool)
    )

    designable_chars: list[str] = []
    for s in sequences:
        for idx, char in enumerate(s[:L]):
            if idx < len(mask) and mask[idx]:
                designable_chars.append(char)

    if not designable_chars:
        return 0.0

    unique_chars = set(designable_chars)
    counts = [designable_chars.count(c) for c in unique_chars]
    probs = np.array(counts, dtype=np.float64) / len(designable_chars)
    return safe_shannon_entropy(probs)


def compute_target_recovery(
    sequences: Sequence[str],
    native_seq: str,
    designable_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute native recovery, shuffled recovery baseline, and recovery margin.

    Returns:
        (native_recovery, shuffled_recovery, recovery_margin)
    """
    if not sequences or not native_seq:
        return 0.0, 0.0, 0.0

    L = min(len(sequences[0]), len(native_seq))
    mask = (
        np.asarray(designable_mask, dtype=bool)
        if designable_mask is not None
        else np.ones(L, dtype=bool)
    )
    M_count = int(np.sum(mask[:L]))
    if M_count == 0:
        return 0.0, 0.0, 0.0

    # Native recovery
    recs = []
    for s in sequences:
        matches = sum(1 for idx in range(L) if mask[idx] and s[idx] == native_seq[idx])
        recs.append(matches / float(M_count))
    mean_native_rec = float(np.mean(recs))

    # Shuffled baseline
    rng = np.random.default_rng(42)
    native_chars = list(native_seq[:L])
    designable_indices = [idx for idx in range(L) if mask[idx]]
    perm = rng.permutation(designable_indices)
    shuf_chars = list(native_chars)
    for orig_idx, perm_idx in zip(designable_indices, perm):
        shuf_chars[orig_idx] = native_chars[perm_idx]
    shuf_seq = "".join(shuf_chars)

    shuf_recs = []
    for s in sequences:
        matches = sum(1 for idx in range(L) if mask[idx] and s[idx] == shuf_seq[idx])
        shuf_recs.append(matches / float(M_count))
    mean_shuf_rec = float(np.mean(shuf_recs))

    margin = float(mean_native_rec - mean_shuf_rec)
    return mean_native_rec, mean_shuf_rec, margin


def compute_duplicate_rates(sequences: Sequence[str]) -> tuple[float, float]:
    """Compute within-condition exact duplicate rate and near-duplicate rate.

    Returns:
        (exact_duplicate_rate, near_duplicate_rate)
    """
    N = len(sequences)
    if N <= 1:
        return 0.0, 0.0

    exact_dup_rate = float(1.0 - len(set(sequences)) / float(N))

    # Near duplicates: hamming distance <= 1
    pairs = 0
    near_dups = 0
    for i in range(N):
        for j in range(i + 1, N):
            diffs = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
            if diffs <= 1:
                near_dups += 1
            pairs += 1

    near_dup_rate = float(near_dups / float(pairs)) if pairs > 0 else 0.0
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

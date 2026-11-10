"""Comprehensive vectorization parity and property-based test suite.

Validates Gap 2 acceptance criteria:
1. batched == looped equality for all analytical statistics:
   - pairwise diversity
   - positional entropy
   - composition entropy
   - target recovery
   - duplicate rates
   - best-of-N (from scores.py)
   - Pareto frontier (from scores.py)
   at tolerance < 1e-10 (or exact for discrete counts).
2. Mathematical invariants and property-based tests:
   - Entropy bounds in [0, log2(20)]
   - Duplicate rates and recovery rates in [0.0, 1.0]
   - Pairwise diversity in [0.0, 1.0] and symmetry
   - Invariance to sequence permutation
   - Strict edge cases (N=0, 1, all-identical, all-orthogonal, uniform 20 AA).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from xtrax.eda.campaign.stats.composition import (
    AA_TO_IDX,
    STANDARD_AA,
    _seqs_to_bytes_and_tokens,
    calculate_pairwise_diversity,
    compute_composition_entropy,
    compute_cross_knob_collisions,
    compute_duplicate_rates,
    compute_positional_entropy,
    compute_target_recovery,
)
from xtrax.eda.campaign.stats.scores import (
    compute_best_of_n,
    compute_pareto_frontier_2d,
)

# -----------------------------------------------------------------------------
# Explicit Un-vectorized Reference Implementations (batched == looped baseline)
# -----------------------------------------------------------------------------


def looped_pairwise_diversity(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> float:
    """Explicit un-vectorized reference loop for pairwise diversity."""
    N = len(sequences)
    if N <= 1:
        return 0.0

    L = len(sequences[0])
    if L == 0:
        return 0.0

    if designable_mask is not None:
        mask = [bool(m) for m in designable_mask[:L]]
        if len(mask) < L:
            mask.extend([True] * (L - len(mask)))
    else:
        mask = [True] * L

    m_count = sum(1 for m in mask if m)
    if m_count == 0:
        return 0.0

    pids = []
    for i in range(N):
        for j in range(i + 1, N):
            matches = 0
            for pos in range(L):
                if mask[pos]:
                    c_i = sequences[i][pos] if pos < len(sequences[i]) else " "
                    c_j = sequences[j][pos] if pos < len(sequences[j]) else " "
                    tok_i = AA_TO_IDX.get(c_i, 20)
                    tok_j = AA_TO_IDX.get(c_j, 20)
                    if tok_i == tok_j:
                        matches += 1
            pids.append(matches / float(m_count))

    if not pids:
        return 0.0
    mean_pid = sum(pids) / len(pids)
    return max(0.0, 1.0 - mean_pid)


def looped_positional_entropy(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Explicit un-vectorized reference loop for positional Shannon entropy."""
    N = len(sequences)
    if N == 0:
        return 0.0, np.array([], dtype=np.float64)

    L = len(sequences[0])
    if L == 0:
        return 0.0, np.array([], dtype=np.float64)

    if designable_mask is not None:
        mask = [bool(m) for m in designable_mask[:L]]
        if len(mask) < L:
            mask.extend([True] * (L - len(mask)))
    else:
        mask = [True] * L

    pos_entropies = np.zeros(L, dtype=np.float64)
    for pos in range(L):
        if mask[pos]:
            col_tokens = []
            for s in sequences:
                c = s[pos] if pos < len(s) else " "
                col_tokens.append(AA_TO_IDX.get(c, 20))
            unique_tokens = set(col_tokens)
            h = 0.0
            for t in unique_tokens:
                cnt = sum(1 for tok in col_tokens if tok == t)
                p = cnt / float(len(col_tokens))
                if p > 0.0:
                    h -= p * math.log2(p)
            pos_entropies[pos] = max(0.0, h)

    masked_vals = [pos_entropies[pos] for pos in range(L) if mask[pos]]
    mean_h = sum(masked_vals) / len(masked_vals) if masked_vals else 0.0
    return float(mean_h), pos_entropies


def looped_composition_entropy(
    sequences: Sequence[str],
    designable_mask: np.ndarray | None = None,
) -> float:
    """Explicit un-vectorized reference loop for composition Shannon entropy."""
    if not sequences:
        return 0.0

    L = len(sequences[0])
    if L == 0:
        return 0.0

    if designable_mask is not None:
        mask = [bool(m) for m in designable_mask[:L]]
        if len(mask) < L:
            mask.extend([True] * (L - len(mask)))
    else:
        mask = [True] * L

    if sum(1 for m in mask if m) == 0:
        return 0.0

    chars: list[int] = []
    for s in sequences:
        for pos in range(min(L, len(s))):
            if mask[pos]:
                chars.append(ord(s[pos]))

    if not chars:
        return 0.0

    unique_chars = set(chars)
    total = len(chars)
    h = 0.0
    for c in unique_chars:
        cnt = sum(1 for x in chars if x == c)
        p = cnt / float(total)
        if p > 0.0:
            h -= p * math.log2(p)
    return float(max(0.0, h))


def looped_target_recovery(
    sequences: Sequence[str],
    native_seq: str,
    designable_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Explicit un-vectorized reference loop for target recovery and margin."""
    if not sequences or not native_seq:
        return 0.0, 0.0, 0.0

    L = min(len(sequences[0]), len(native_seq))
    if L == 0:
        return 0.0, 0.0, 0.0

    if designable_mask is not None:
        mask = [bool(m) for m in designable_mask[:L]]
        if len(mask) < L:
            mask.extend([True] * (L - len(mask)))
    else:
        mask = [True] * L

    m_count = sum(1 for m in mask if m)
    if m_count == 0:
        return 0.0, 0.0, 0.0

    # Native recovery
    native_recs = []
    for s in sequences:
        matches = 0
        for pos in range(L):
            if mask[pos] and pos < len(s) and s[pos] == native_seq[pos]:
                matches += 1
        native_recs.append(matches / float(m_count))
    mean_native = sum(native_recs) / len(native_recs)

    # Shuffled recovery with seed 42
    rng = np.random.default_rng(42)
    des_idx = [idx for idx in range(L) if mask[idx]]
    perm = rng.permutation(des_idx)
    shuf_chars = list(native_seq[:L])
    for orig_idx, perm_idx in zip(des_idx, perm):
        shuf_chars[orig_idx] = native_seq[perm_idx]
    shuf_seq = "".join(shuf_chars)

    shuf_recs = []
    for s in sequences:
        matches = 0
        for pos in range(L):
            if mask[pos] and pos < len(s) and s[pos] == shuf_seq[pos]:
                matches += 1
        shuf_recs.append(matches / float(m_count))
    mean_shuf = sum(shuf_recs) / len(shuf_recs)

    margin = mean_native - mean_shuf
    return float(mean_native), float(mean_shuf), float(margin)


def looped_duplicate_rates(sequences: Sequence[str]) -> tuple[float, float]:
    """Explicit un-vectorized reference loop for exact and near duplicate rates."""
    N = len(sequences)
    if N <= 1:
        return 0.0, 0.0

    exact_dup_rate = 1.0 - len(set(sequences)) / float(N)

    L = len(sequences[0])
    if L == 0:
        return float(exact_dup_rate), 1.0

    pairs = 0
    near_dups = 0
    for i in range(N):
        for j in range(i + 1, N):
            diffs = sum(1 for pos in range(L) if sequences[i][pos] != sequences[j][pos])
            if diffs <= 1:
                near_dups += 1
            pairs += 1

    near_dup_rate = near_dups / float(pairs) if pairs > 0 else 0.0
    return float(exact_dup_rate), float(near_dup_rate)


def looped_best_of_n(
    scores: Sequence[float],
    candidate_grid: Sequence[int] | None = None,
) -> dict[int, float]:
    """Explicit un-vectorized reference loop for Best-of-N using exact combinatorics."""
    s_list = sorted(list(scores))
    N = len(s_list)
    if N == 0:
        return {}

    if candidate_grid is None:
        grid = []
        k = 1
        while k <= N:
            grid.append(k)
            k *= 2
        if N not in grid:
            grid.append(N)
    else:
        grid = sorted(list({int(n) for n in candidate_grid if 0 < n <= N}))

    results = {}
    for n in grid:
        total_comb = math.comb(N, n)
        expected_min = 0.0
        for j in range(1, N - n + 2):
            w = math.comb(N - j, n - 1) / total_comb
            expected_min += w * s_list[j - 1]
        results[n] = float(expected_min)
    return results


def looped_pareto_frontier_2d(
    points: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Explicit un-vectorized reference loop for 2D Pareto non-dominated frontier."""
    if not points:
        return []

    frontier = []
    for i, pt_a in enumerate(points):
        s_a, pid_a, _ = pt_a
        dominated = False
        for j, pt_b in enumerate(points):
            if i != j:
                s_b, pid_b, _ = pt_b
                if (s_b <= s_a and pid_b <= pid_a) and (s_b < s_a or pid_b < pid_a):
                    dominated = True
                    break
        if not dominated:
            frontier.append(pt_a)

    frontier.sort(key=lambda x: (x[0], x[1]))
    return frontier


# -----------------------------------------------------------------------------
# Test Fixtures & Generators
# -----------------------------------------------------------------------------


def generate_synthetic_library(N: int, L: int, seed: int = 42) -> tuple[list[str], str, np.ndarray]:
    """Generate reproducible synthetic sequence libraries and native wild type."""
    rng = np.random.default_rng(seed)
    aa_list = list(STANDARD_AA)
    native_chars = [rng.choice(aa_list) for _ in range(L)]
    native_seq = "".join(native_chars)

    sequences: list[str] = []
    for _ in range(N):
        # 70% native conservation, 30% random mutations
        chars = []
        for l_idx in range(L):
            if rng.random() < 0.7:
                chars.append(native_chars[l_idx])
            else:
                chars.append(rng.choice(aa_list))
        sequences.append("".join(chars))

    mask = rng.random(L) > 0.25  # ~75% designable positions
    return sequences, native_seq, mask


# -----------------------------------------------------------------------------
# Part 1: Comprehensive batched == looped Equality Tests (< 1e-10)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("N,L,seed", [(5, 10, 101), (20, 25, 202), (40, 50, 303)])
def test_batched_equals_looped_pairwise_diversity(N: int, L: int, seed: int):
    """Verify batched pairwise diversity matches looped reference (< 1e-10)."""
    sequences, _, mask = generate_synthetic_library(N, L, seed=seed)

    batched_val = calculate_pairwise_diversity(sequences, designable_mask=mask)
    looped_val = looped_pairwise_diversity(sequences, designable_mask=mask)

    assert abs(batched_val - looped_val) < 1e-10, (
        f"Diversity mismatch: batched={batched_val}, looped={looped_val}"
    )


@pytest.mark.parametrize("N,L,seed", [(6, 12, 102), (25, 30, 203), (50, 45, 304)])
def test_batched_equals_looped_positional_entropy(N: int, L: int, seed: int):
    """Verify batched positional entropy matches looped reference (< 1e-10)."""
    sequences, _, mask = generate_synthetic_library(N, L, seed=seed)

    batched_mean, batched_arr = compute_positional_entropy(sequences, designable_mask=mask)
    looped_mean, looped_arr = looped_positional_entropy(sequences, designable_mask=mask)

    assert abs(batched_mean - looped_mean) < 1e-10, (
        f"Mean positional entropy mismatch: {batched_mean} vs {looped_mean}"
    )
    assert np.allclose(batched_arr, looped_arr, atol=1e-10), "Per-position entropy array mismatch"


@pytest.mark.parametrize("N,L,seed", [(8, 15, 103), (30, 35, 204), (45, 40, 305)])
def test_batched_equals_looped_composition_entropy(N: int, L: int, seed: int):
    """Verify batched composition entropy matches looped reference (< 1e-10)."""
    sequences, _, mask = generate_synthetic_library(N, L, seed=seed)

    batched_val = compute_composition_entropy(sequences, designable_mask=mask)
    looped_val = looped_composition_entropy(sequences, designable_mask=mask)

    assert abs(batched_val - looped_val) < 1e-10, (
        f"Composition entropy mismatch: {batched_val} vs {looped_val}"
    )


@pytest.mark.parametrize("N,L,seed", [(7, 14, 104), (22, 28, 205), (35, 42, 306)])
def test_batched_equals_looped_target_recovery(N: int, L: int, seed: int):
    """Verify batched target recovery matches looped reference (< 1e-10)."""
    sequences, native_seq, mask = generate_synthetic_library(N, L, seed=seed)

    b_nat, b_shuf, b_marg = compute_target_recovery(
        sequences, native_seq=native_seq, designable_mask=mask
    )
    l_nat, l_shuf, l_marg = looped_target_recovery(
        sequences, native_seq=native_seq, designable_mask=mask
    )

    assert abs(b_nat - l_nat) < 1e-10, f"Native recovery mismatch: {b_nat} vs {l_nat}"
    assert abs(b_shuf - l_shuf) < 1e-10, f"Shuffled recovery mismatch: {b_shuf} vs {l_shuf}"
    assert abs(b_marg - l_marg) < 1e-10, f"Margin mismatch: {b_marg} vs {l_marg}"


@pytest.mark.parametrize("N,L,seed", [(10, 15, 105), (30, 20, 206), (50, 30, 307)])
def test_batched_equals_looped_duplicate_rates(N: int, L: int, seed: int):
    """Verify batched duplicate rates match looped reference (< 1e-10)."""
    sequences, _, _ = generate_synthetic_library(N, L, seed=seed)
    # Inject exact and near duplicates to stress the rates
    sequences.append(sequences[0])  # Exact duplicate
    near_mut = list(sequences[1])
    near_mut[0] = "W" if near_mut[0] != "W" else "Y"
    sequences.append("".join(near_mut))  # 1-Hamming near duplicate

    b_exact, b_near = compute_duplicate_rates(sequences)
    l_exact, l_near = looped_duplicate_rates(sequences)

    assert abs(b_exact - l_exact) < 1e-10, f"Exact duplicate rate mismatch: {b_exact} vs {l_exact}"
    assert abs(b_near - l_near) < 1e-10, f"Near duplicate rate mismatch: {b_near} vs {l_near}"


@pytest.mark.parametrize("N,seed", [(10, 401), (25, 402), (60, 403)])
def test_batched_equals_looped_best_of_n(N: int, seed: int):
    """Verify Best-of-N ratio recurrence matches exact combinatorial loop (< 1e-10)."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(loc=0.0, scale=1.0, size=N)
    candidate_grid = [1, 2, 4, 8, 16, 32, N]

    batched_res = compute_best_of_n(scores, candidate_grid=candidate_grid)
    looped_res = looped_best_of_n(scores, candidate_grid=candidate_grid)

    assert set(batched_res.keys()) == set(looped_res.keys())
    for k in batched_res:
        assert abs(batched_res[k] - looped_res[k]) < 1e-10, (
            f"Best-of-{k} mismatch: {batched_res[k]} vs {looped_res[k]}"
        )


def test_batched_equals_looped_pareto_frontier():
    """Verify 2D Pareto frontier matches explicit pairwise dominance loop."""
    rng = np.random.default_rng(505)
    P = 40
    scores = rng.uniform(0.5, 3.0, size=P)
    pids = rng.uniform(0.2, 0.9, size=P)
    labels = [f"cand_{i}" for i in range(P)]
    points = [(float(scores[i]), float(pids[i]), labels[i]) for i in range(P)]

    batched_frontier = compute_pareto_frontier_2d(points)
    looped_frontier = looped_pareto_frontier_2d(points)

    assert len(batched_frontier) == len(looped_frontier)
    for b_pt, l_pt in zip(batched_frontier, looped_frontier):
        assert abs(b_pt[0] - l_pt[0]) < 1e-10
        assert abs(b_pt[1] - l_pt[1]) < 1e-10
        assert b_pt[2] == l_pt[2]


# -----------------------------------------------------------------------------
# Part 2: Mathematical Invariants & Property-Based Tests
# -----------------------------------------------------------------------------


def test_property_entropy_bounds():
    """Property: All entropies lie strictly within [0, log2(20)] bits."""
    max_h = math.log2(20.0)

    for seed in [11, 22, 33]:
        seqs, _, mask = generate_synthetic_library(N=30, L=25, seed=seed)
        mean_pos, pos_arr = compute_positional_entropy(seqs, designable_mask=mask)
        comp_h = compute_composition_entropy(seqs, designable_mask=mask)

        # Non-negative and bounded by alphabet size log2(20)
        assert 0.0 <= mean_pos <= max_h + 1e-12
        assert 0.0 <= comp_h <= max_h + 1e-12
        for pos, val in enumerate(pos_arr):
            if mask[pos]:
                assert 0.0 <= val <= max_h + 1e-12
            else:
                assert val == 0.0, "Non-designable position must have 0.0 entropy"


def test_property_rate_bounds():
    """Property: Duplicate rates and recovery rates lie strictly within [0.0, 1.0]."""
    for seed in [44, 55, 66]:
        seqs, nat, mask = generate_synthetic_library(N=20, L=20, seed=seed)
        exact_dup, near_dup = compute_duplicate_rates(seqs)
        nat_rec, shuf_rec, margin = compute_target_recovery(seqs, nat, designable_mask=mask)

        assert 0.0 <= exact_dup <= 1.0
        assert 0.0 <= near_dup <= 1.0
        assert 0.0 <= nat_rec <= 1.0
        assert 0.0 <= shuf_rec <= 1.0
        assert -1.0 <= margin <= 1.0


def test_property_pairwise_diversity_bounds_and_symmetry():
    """Property: Pairwise diversity lies in [0, 1] and collision matrices are symmetric."""
    for seed in [77, 88]:
        seqs, _, mask = generate_synthetic_library(N=25, L=18, seed=seed)
        div = calculate_pairwise_diversity(seqs, designable_mask=mask)
        assert 0.0 <= div <= 1.0

    # Cross-knob collision symmetry
    conds = {
        "k1": ["ACDEF", "ACDGY", "WWWWW"],
        "k2": ["ACDEF", "KKKKK", "LLLLL"],
        "k3": ["WWWWW", "ACDGY"],
    }
    collisions = compute_cross_knob_collisions(conds)
    for (k_a, k_b), rate in collisions.items():
        assert collisions[(k_b, k_a)] == rate
        assert 0.0 <= rate <= 1.0


def test_property_sequence_permutation_invariance():
    """Property: All analytical metrics are invariant to sequence ordering."""
    seqs, nat, mask = generate_synthetic_library(N=30, L=25, seed=123)

    rng = np.random.default_rng(999)
    perm_indices = rng.permutation(len(seqs))
    perm_seqs = [seqs[i] for i in perm_indices]

    # Diversity
    d_orig = calculate_pairwise_diversity(seqs, mask)
    d_perm = calculate_pairwise_diversity(perm_seqs, mask)
    assert abs(d_orig - d_perm) < 1e-12

    # Positional entropy
    h_orig_mean, h_orig_arr = compute_positional_entropy(seqs, mask)
    h_perm_mean, h_perm_arr = compute_positional_entropy(perm_seqs, mask)
    assert abs(h_orig_mean - h_perm_mean) < 1e-12
    assert np.allclose(h_orig_arr, h_perm_arr, atol=1e-12)

    # Composition entropy
    c_orig = compute_composition_entropy(seqs, mask)
    c_perm = compute_composition_entropy(perm_seqs, mask)
    assert abs(c_orig - c_perm) < 1e-12

    # Target recovery
    r_orig = compute_target_recovery(seqs, nat, mask)
    r_perm = compute_target_recovery(perm_seqs, nat, mask)
    assert abs(r_orig[0] - r_perm[0]) < 1e-12
    assert abs(r_orig[1] - r_perm[1]) < 1e-12
    assert abs(r_orig[2] - r_perm[2]) < 1e-12

    # Duplicate rates
    dup_orig = compute_duplicate_rates(seqs)
    dup_perm = compute_duplicate_rates(perm_seqs)
    assert abs(dup_orig[0] - dup_perm[0]) < 1e-12
    assert abs(dup_orig[1] - dup_perm[1]) < 1e-12


def test_property_best_of_n_monotonicity_and_bounds():
    """Property: Best-of-N is monotonically non-increasing as N grows."""
    scores = np.array([3.2, 1.1, 4.5, -0.2, 2.3, 0.5, 1.8, -1.0, 0.9, 2.7])
    N = len(scores)
    grid = list(range(1, N + 1))
    res = compute_best_of_n(scores, grid)

    # Boundary n=1: expected minimum is sample mean
    assert np.isclose(res[1], np.mean(scores), atol=1e-10)
    # Boundary n=N: expected minimum is exact minimum
    assert np.isclose(res[N], np.min(scores), atol=1e-10)

    # Monotonicity
    for i in range(1, N):
        assert res[i] >= res[i + 1] - 1e-12


def test_property_pareto_frontier_non_domination():
    """Property: Every point on Pareto frontier is mutually non-dominated."""
    points = [
        (1.0, 0.5, "p1"),
        (0.8, 0.7, "p2"),
        (0.6, 0.9, "p3"),
        (1.2, 0.6, "dom_by_p1"),
        (0.9, 0.8, "dom_by_p2"),
    ]
    frontier = compute_pareto_frontier_2d(points)
    frontier_labels = {p[2] for p in frontier}

    assert frontier_labels == {"p1", "p2", "p3"}

    # Verify no mutual dominance within the frontier
    for i, pt_a in enumerate(frontier):
        for j, pt_b in enumerate(frontier):
            if i != j:
                s_a, pid_a, _ = pt_a
                s_b, pid_b, _ = pt_b
                assert not ((s_b <= s_a and pid_b <= pid_a) and (s_b < s_a or pid_b < pid_a))


# -----------------------------------------------------------------------------
# Part 3: Extreme Edge Cases & Boundary Conditions
# -----------------------------------------------------------------------------


def test_edge_case_n_zero():
    """Verify N=0 empty sequence libraries return zero values gracefully."""
    assert calculate_pairwise_diversity([]) == 0.0
    mean_h, arr_h = compute_positional_entropy([])
    assert mean_h == 0.0 and len(arr_h) == 0
    assert compute_composition_entropy([]) == 0.0
    assert compute_target_recovery([], "ACDEF") == (0.0, 0.0, 0.0)
    assert compute_duplicate_rates([]) == (0.0, 0.0)
    assert compute_best_of_n([]) == {}
    assert compute_pareto_frontier_2d([]) == []


def test_edge_case_n_one():
    """Verify N=1 returns 0.0 diversity and zero duplicate rates."""
    seqs = ["MQIFVKTLTG"]
    assert calculate_pairwise_diversity(seqs) == 0.0
    exact_dup, near_dup = compute_duplicate_rates(seqs)
    assert exact_dup == 0.0 and near_dup == 0.0

    mean_h, arr_h = compute_positional_entropy(seqs)
    assert mean_h == 0.0 and np.all(arr_h == 0.0)


def test_edge_case_all_identical():
    """Verify all-identical library has entropy=0, diversity=0, duplicate rate ~1."""
    N = 10
    seqs = ["MQIFVKTLTG"] * N
    assert calculate_pairwise_diversity(seqs) == 0.0
    mean_h, arr_h = compute_positional_entropy(seqs)
    assert mean_h == 0.0 and np.all(arr_h == 0.0)

    exact_dup, near_dup = compute_duplicate_rates(seqs)
    assert np.isclose(exact_dup, (N - 1) / float(N))
    assert near_dup == 1.0


def test_edge_case_all_orthogonal():
    """Verify completely orthogonal sequences produce diversity=1.0, duplicates=0."""
    # 4 sequences with completely distinct amino acids at each position
    seqs = [
        "AAAA",
        "CCCC",
        "DDDD",
        "EEEE",
    ]
    div = calculate_pairwise_diversity(seqs)
    assert np.isclose(div, 1.0)

    exact_dup, near_dup = compute_duplicate_rates(seqs)
    assert exact_dup == 0.0
    assert near_dup == 0.0


def test_edge_case_uniform_distribution_matches_log2_20():
    """Verify uniform 20 amino acid distribution yields exactly log2(20) bits."""
    # 20 sequences, each with one unique amino acid at position 0
    seqs = [aa * 5 for aa in STANDARD_AA]
    mean_h, arr_h = compute_positional_entropy(seqs)

    expected_h = math.log2(20.0)
    assert np.isclose(mean_h, expected_h, atol=1e-10)
    assert np.allclose(arr_h, expected_h, atol=1e-10)


def test_edge_case_empty_string_and_zero_designable():
    """Verify empty sequence strings and zero designable positions."""
    assert calculate_pairwise_diversity(["", ""]) == 0.0
    mean_h, arr_h = compute_positional_entropy(["", ""])
    assert mean_h == 0.0 and len(arr_h) == 0
    assert compute_composition_entropy(["", ""]) == 0.0

    # Zero designable positions (|M| = 0)
    seqs = ["ACDEF", "ACDGY"]
    zero_mask = np.zeros(5, dtype=bool)
    assert calculate_pairwise_diversity(seqs, designable_mask=zero_mask) == 0.0
    mean_h, arr_h = compute_positional_entropy(seqs, designable_mask=zero_mask)
    assert mean_h == 0.0 and np.all(arr_h == 0.0)
    assert compute_composition_entropy(seqs, designable_mask=zero_mask) == 0.0
    assert compute_target_recovery(seqs, "ACDEF", designable_mask=zero_mask) == (0.0, 0.0, 0.0)


def test_edge_case_invalid_characters_and_ragged_lengths():
    """Verify invalid / non-standard characters and ragged lengths are handled safely."""
    # 'X', '-', '1', '?' are non-standard tokens
    seqs = ["ACD-F", "ACXEF", "AC1?F"]
    div = calculate_pairwise_diversity(seqs)
    assert 0.0 <= div <= 1.0

    mean_h, arr_h = compute_positional_entropy(seqs)
    assert mean_h >= 0.0 and len(arr_h) == 5

    # Ragged lengths: padded safely without exception
    ragged_seqs = ["ACDEF", "ACD", "ACDEFGH"]
    div_ragged = calculate_pairwise_diversity(ragged_seqs)
    assert 0.0 <= div_ragged <= 1.0


def test_coverage_edge_branches():
    """Verify mismatched masks, None masks, and boundary conditions across all functions."""
    seqs = ["ACDEFGHIKL", "ACDEFGHKLM"]  # L = 10
    short_mask = np.array([True, False, True])  # len = 3 != 10
    long_mask = np.ones(15, dtype=bool)  # len = 15 != 10

    # 1. Mismatched mask lengths
    for m in [short_mask, long_mask]:
        div = calculate_pairwise_diversity(seqs, designable_mask=m)
        assert 0.0 <= div <= 1.0

        mean_h, arr_h = compute_positional_entropy(seqs, designable_mask=m)
        assert mean_h >= 0.0 and len(arr_h) == 10

        comp_h = compute_composition_entropy(seqs, designable_mask=m)
        assert comp_h >= 0.0

        nat_rec, shuf_rec, marg = compute_target_recovery(seqs, "ACDEFGHIKL", designable_mask=m)
        assert 0.0 <= nat_rec <= 1.0

    # 2. None mask for composition entropy and target recovery
    comp_h_none = compute_composition_entropy(seqs, designable_mask=None)
    assert comp_h_none > 0.0

    nat_rec_none, shuf_rec_none, marg_none = compute_target_recovery(
        seqs, "ACDEFGHIKL", designable_mask=None
    )
    assert nat_rec_none > 0.0

    # 3. L = 0 for target recovery and duplicate rates
    assert compute_target_recovery(["", ""], "ACDEF") == (0.0, 0.0, 0.0)
    assert compute_target_recovery(["ACDEF", "ACDEF"], "") == (0.0, 0.0, 0.0)
    exact_dup, near_dup = compute_duplicate_rates(["", ""])
    assert exact_dup == 0.5 and near_dup == 1.0

    # 4. Cross knob collision with empty sets
    conds_empty = {"empty": [], "normal": ["ACDEF"]}
    collisions = compute_cross_knob_collisions(conds_empty)
    assert collisions[("empty", "normal")] == 0.0
    assert collisions[("empty", "empty")] == 0.0

    # 5. Empty inputs to _seqs_to_bytes_and_tokens and zero probabilities
    from xtrax.eda.campaign.stats.composition import safe_shannon_entropy

    assert safe_shannon_entropy(np.array([0.0, 0.0])) == 0.0

    b_arr, tokens = _seqs_to_bytes_and_tokens([], 0)
    assert b_arr.size == 0 and tokens.size == 0

    b_arr, tokens = _seqs_to_bytes_and_tokens(["ABC"], 0)
    assert b_arr.size == 0 and tokens.size == 0

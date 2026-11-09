"""Unit tests for Campaign EDA vectorized statistics engine (Q1, Q2, Q3)."""

import numpy as np

from xtrax.eda.campaign.stats import (
    calculate_pairwise_diversity,
    compute_best_of_n,
    compute_cross_knob_collisions,
    compute_pareto_frontier_2d,
    evaluate_cluster_bootstrap_condition,
    evaluate_contrast,
    safe_shannon_entropy,
    vectorized_cluster_ratio_bootstrap,
)


def test_pairwise_diversity_boundary_guards():
    """Verify boundary guards for N <= 1 and |M| = 0 (BLK-04)."""
    # N = 0
    assert calculate_pairwise_diversity([]) == 0.0

    # N = 1 (single sample)
    assert calculate_pairwise_diversity(["MQIFVKTLTG"]) == 0.0

    # |M| = 0 (zero designable positions)
    assert (
        calculate_pairwise_diversity(
            ["MQIFVKTLTG", "MQIFVKTLTA"],
            designable_mask=np.zeros(10, dtype=bool),
        )
        == 0.0
    )

    # Normal pairwise diversity
    seqs = ["AAAA", "AAAT", "AATT", "TTTT"]
    div = calculate_pairwise_diversity(seqs)
    assert 0.0 < div < 1.0


def test_safe_shannon_entropy_positive_support():
    """Verify safe positive-support Shannon entropy (MAJ-07)."""
    # Empty
    assert safe_shannon_entropy(np.array([])) == 0.0

    # Single probability = 1.0 -> 0 bits
    assert safe_shannon_entropy(np.array([1.0, 0.0, 0.0])) == 0.0

    # Uniform over 2 tokens -> 1.0 bit
    assert np.isclose(safe_shannon_entropy(np.array([0.5, 0.5, 0.0])), 1.0)

    # Uniform over 4 tokens -> 2.0 bits
    assert np.isclose(safe_shannon_entropy(np.array([0.25, 0.25, 0.25, 0.25])), 2.0)


def test_best_of_n_bounded_grid_and_recurrence():
    """Verify Best-of-N candidate grid bounding and iterative recurrence (BLK-03)."""
    scores = np.array([2.5, 1.2, 3.4, 0.8, 1.9, 2.1, 1.5, 0.9])
    N = len(scores)

    # Request grid with n > N -> must bound strictly to n <= N
    grid = [1, 2, 4, 8, 16, 32]
    res = compute_best_of_n(scores, grid)

    # Values for n > N must not be present
    assert 16 not in res
    assert 32 not in res

    # Monotonically non-increasing as n grows
    assert res[1] >= res[2] >= res[4] >= res[8]
    # For n = 1, expected min is the mean
    assert np.isclose(res[1], np.mean(scores))
    # For n = N, expected min is the exact minimum
    assert np.isclose(res[N], np.min(scores))


def test_pareto_frontier_non_dominated_sorting():
    """Verify 2D Pareto sorting minimizing score and pairwise identity (MIN-02)."""
    # Points: (NLL, PID, label)
    # Point A: NLL=1.0, PID=0.5 (good energy, good diversity)
    # Point B: NLL=1.5, PID=0.8 (worse energy, worse diversity -> dominated by A)
    # Point C: NLL=0.8, PID=0.7 (better energy, worse diversity -> non-dominated with A)
    points = [
        (1.0, 0.5, "A"),
        (1.5, 0.8, "B"),
        (0.8, 0.7, "C"),
    ]
    frontier = compute_pareto_frontier_2d(points)
    labels = [p[2] for p in frontier]
    assert "A" in labels
    assert "C" in labels
    assert "B" not in labels  # B is dominated


def test_cross_knob_collision_table():
    """Verify cross-knob collision matrix (MIN-01)."""
    cond_seqs = {
        "cond_A": ["SEQ1", "SEQ2", "SEQ3"],
        "cond_B": ["SEQ2", "SEQ3", "SEQ4"],
        "cond_C": ["SEQ5", "SEQ6"],
    }
    col = compute_cross_knob_collisions(cond_seqs)

    # Self-collision is 1.0
    assert col[("cond_A", "cond_A")] == 1.0
    assert col[("cond_B", "cond_B")] == 1.0

    # A and B share 2 out of 3: overlap = 2/3
    assert np.isclose(col[("cond_A", "cond_B")], 2.0 / 3.0)
    assert np.isclose(col[("cond_B", "cond_A")], 2.0 / 3.0)

    # A and C share 0: overlap = 0.0
    assert col[("cond_A", "cond_C")] == 0.0


def test_bootstrap_zero_sample_guard():
    """Verify zero-count units do not trigger ZeroDivisionError or NaN leaks (F-XTX-04)."""
    # 1. Total count is zero
    unit_sums = np.array([[0.0], [0.0], [0.0], [0.0]])
    unit_counts = np.array([[0], [0], [0], [0]])
    estimates = evaluate_cluster_bootstrap_condition(
        unit_sums=unit_sums,
        unit_counts=unit_counts,
        metric_names=["score"],
        B=50,
    )
    assert "score" in estimates
    est = estimates["score"]
    assert est.descriptive_only is True
    assert est.point_estimate == 0.0
    assert est.ci_95 is None

    # 2. Resampled zero units in vectorized_cluster_ratio_bootstrap
    weights = np.array([[1, 0, 0], [0, 1, 0]])
    # All counts zero
    res = vectorized_cluster_ratio_bootstrap(
        unit_sums=np.zeros((3, 1)),
        unit_counts=np.zeros(3),
        weights=weights,
    )
    # Must not crash, safe_denominator makes it NaN without ZeroDivisionError
    assert np.isnan(res).all()

    # 3. evaluate_contrast with zero counts
    contrast = evaluate_contrast(
        unit_sums_a=np.zeros((3, 1)),
        unit_counts_a=np.zeros(3),
        unit_sums_b=np.zeros((3, 1)),
        unit_counts_b=np.zeros(3),
        metric_name="score",
        B=50,
    )
    assert contrast.descriptive_only is True
    assert contrast.ci_95 is None

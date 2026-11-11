"""Scores, Best-of-N order statistics, and Pareto frontier analytics (Question 2).

Implements Q2 analytics per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Includes numerically stable Best-of-N order statistics ratio recurrence avoiding float64
overflow (BLK-03), bounded candidate grid (n <= N), and 2D Pareto non-dominated sorting (MIN-02).
"""

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger("xtrax.eda.campaign.stats.scores")


def compute_best_of_n(
    scores: np.ndarray | Sequence[float],
    candidate_grid: Sequence[int] | None = None,
) -> dict[int, float]:
    """Calculate expected Best-of-N minimum scores using iterative ratio recurrence (BLK-03).

    Binds candidate grid n to n <= N:
    n in {2^k <= N} union {N}

    Evaluates weights via numerically stable recurrence avoiding comb() overflow:
    omega_1 = n / N
    omega_j = omega_{j-1} * (N - n - j + 2) / (N - j + 1)
    """
    s_arr = np.asarray(scores, dtype=np.float64)
    N = len(s_arr)
    if N == 0:
        return {}

    sorted_scores = np.sort(s_arr)

    # Determine candidate grid if not specified
    if candidate_grid is None:
        grid: list[int] = []
        k = 1
        while k <= N:
            grid.append(k)
            k *= 2
        if N not in grid:
            grid.append(N)
    else:
        # Bounded candidate grid (BLK-03)
        grid = sorted(list({int(n) for n in candidate_grid if 0 < n <= N}))

    results: dict[int, float] = {}

    for n in grid:
        expected_min = 0.0
        w = float(n) / float(N)
        for j in range(1, N - n + 2):
            score_val = sorted_scores[j - 1]
            expected_min += w * score_val
            if j < N - n + 1:
                w *= float(N - n - j + 1) / float(N - j)
        results[n] = float(expected_min)

    return results


def compute_pareto_frontier_2d(
    points: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Compute 2D non-dominated Pareto frontier minimizing score and pairwise identity (MIN-02).

    Simultaneously minimizes:
    - S (Negative Log-Likelihood / energy)
    - PID (Pairwise Identity, maximizing library diversity)

    A point A dominates B if S_A <= S_B and PID_A <= PID_B with at least one strict inequality.
    """
    if not points:
        return []

    non_dominated: list[tuple[float, float, str]] = []

    for i, pt_a in enumerate(points):
        s_a, pid_a, _ = pt_a
        dominated = False
        for j, pt_b in enumerate(points):
            if i != j:
                s_b, pid_b, _ = pt_b
                if s_b <= s_a and pid_b <= pid_a:
                    if s_b < s_a or pid_b < pid_a:
                        dominated = True
                        break
        if not dominated:
            non_dominated.append(pt_a)

    # Sort along first objective (S)
    non_dominated.sort(key=lambda x: (x[0], x[1]))
    return non_dominated

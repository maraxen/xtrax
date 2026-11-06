"""Vectorized cluster ratio bootstrap and honest statistical uncertainty engine (Question 3).

Implements Q3 analytics per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Includes vectorized cluster ratio estimator (BLK-01), Small-Sample Guard for U < 3 (BLK-02),
and Minimum Detectable Effect (MDE_0.80) power checks under global MD-replicas rule.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from xtrax.eda.campaign.types import BootstrapEstimate, ContrastEstimate

logger = logging.getLogger("xtrax.eda.campaign.stats.bootstrap")


def generate_multinomial_cluster_weights(
    U: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate multinomial cluster resampling weight matrix W in N^{B x U}.

    W[b, :] ~ Multinomial(U, [1/U, 1/U, ..., 1/U])
    """
    if U <= 0:
        raise ValueError(f"Replicate count U must be positive, got {U}")
    pvals = [1.0 / float(U)] * U
    return rng.multinomial(n=U, pvals=pvals, size=B)


def vectorized_cluster_ratio_bootstrap(
    unit_sums: np.ndarray,  # shape (U, D)
    unit_counts: np.ndarray,  # shape (U,) or (U, 1)
    weights: np.ndarray,  # shape (B, U)
) -> np.ndarray:  # shape (B, D)
    """Evaluate B cluster ratio bootstrap replications via BLAS matrix algebra (BLK-01).

    Theta* = (W @ T) / (W @ n)[:, None]
    where T is the (U, D) matrix of unit sums and n is the (U, 1) column of sample counts.

    Mathematically unbiased across unequal cluster sizes, matching unweighted bootstrap
    on equal cluster sizes.
    """
    n_vec = np.asarray(unit_counts, dtype=np.float64).reshape(-1, 1)
    t_mat = np.asarray(unit_sums, dtype=np.float64)
    w_mat = np.asarray(weights, dtype=np.float64)

    numerator = w_mat @ t_mat  # shape (B, D)
    denominator = w_mat @ n_vec  # shape (B, 1)
    return numerator / denominator  # shape (B, D)


def looped_cluster_ratio_bootstrap(
    unit_sums: np.ndarray,
    unit_counts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Iterative Python loop implementation of cluster ratio bootstrap for parity testing."""
    B, U = weights.shape
    D = unit_sums.shape[1] if unit_sums.ndim > 1 else 1
    t_mat = unit_sums.reshape(U, D)
    n_vec = np.asarray(unit_counts, dtype=np.float64).reshape(-1)

    theta_loop = np.zeros((B, D), dtype=np.float64)
    for b in range(B):
        num_b = np.zeros(D, dtype=np.float64)
        den_b = 0.0
        for u in range(U):
            w = weights[b, u]
            num_b += w * t_mat[u]
            den_b += w * n_vec[u]
        theta_loop[b] = num_b / den_b
    return theta_loop


def compute_mde_080(se: float) -> float:
    """Calculate Minimum Detectable Effect at 80% power and alpha=0.05.

    MDE_0.80 = (z_0.975 + z_0.80) * SE = (1.95996 + 0.84162) * SE ≈ 2.80158 * SE
    """
    return 2.80158 * float(se)


def evaluate_cluster_bootstrap_condition(
    unit_sums: np.ndarray,
    unit_counts: np.ndarray,
    metric_names: Sequence[str],
    B: int = 2000,
    seed: int = 42,
) -> dict[str, BootstrapEstimate]:
    """Evaluate cluster ratio bootstrap estimates with Small-Sample Guard for U < 3 (BLK-02).

    When replicate count U < 3:
        Sets descriptive_only = True, ci_95 = None, and emits warning.
    When U >= 3:
        Computes 95% empirical percentile CIs via vectorized cluster ratio estimator.
    """
    U = len(unit_counts)
    D = len(metric_names)
    t_mat = np.asarray(unit_sums, dtype=np.float64).reshape(U, D)
    n_vec = np.asarray(unit_counts, dtype=np.float64).reshape(U, 1)

    # Point estimates
    point_estimates = np.sum(t_mat, axis=0) / float(np.sum(n_vec))

    estimates: dict[str, BootstrapEstimate] = {}

    # Small-sample guard (spec §5.3.3)
    if U < 3:
        warning_msg = "Warning: Replicate count U < 3; reporting descriptive point estimates only."
        for d, name in enumerate(metric_names):
            estimates[name] = BootstrapEstimate(
                metric_name=name,
                point_estimate=float(point_estimates[d]),
                ci_95=None,
                se=0.0,
                mde_080=0.0,
                descriptive_only=True,
                warning=warning_msg,
            )
        return estimates

    rng = np.random.default_rng(seed)
    weights = generate_multinomial_cluster_weights(U, B, rng)
    boot_thetas = vectorized_cluster_ratio_bootstrap(t_mat, n_vec, weights)  # (B, D)

    for d, name in enumerate(metric_names):
        col_thetas = boot_thetas[:, d]
        ci_lower = float(np.percentile(col_thetas, 2.5))
        ci_upper = float(np.percentile(col_thetas, 97.5))
        se = float(np.std(col_thetas, ddof=1))
        mde = compute_mde_080(se)

        estimates[name] = BootstrapEstimate(
            metric_name=name,
            point_estimate=float(point_estimates[d]),
            ci_95=(ci_lower, ci_upper),
            se=se,
            mde_080=mde,
            descriptive_only=False,
            warning=None,
        )

    return estimates


def evaluate_contrast(
    unit_sums_a: np.ndarray,
    unit_counts_a: np.ndarray,
    unit_sums_b: np.ndarray,
    unit_counts_b: np.ndarray,
    metric_name: str,
    B: int = 2000,
    seed: int = 42,
) -> ContrastEstimate:
    """Evaluate contrast Delta_hat = B - A with honest uncertainty and MDE power checks."""
    u_a = len(unit_counts_a)
    u_b = len(unit_counts_b)

    mean_a = float(np.sum(unit_sums_a) / np.sum(unit_counts_a))
    mean_b = float(np.sum(unit_sums_b) / np.sum(unit_counts_b))
    delta_hat = mean_b - mean_a

    if u_a < 3 or u_b < 3:
        return ContrastEstimate(
            contrast_name="contrast",
            metric_name=metric_name,
            delta_hat=delta_hat,
            ci_95=None,
            se=0.0,
            mde_080=0.0,
            detected_and_powered=False,
            descriptive_only=True,
            warning="Warning: Contrast with U < 3; reporting descriptive delta only.",
        )

    rng = np.random.default_rng(seed)
    w_a = generate_multinomial_cluster_weights(u_a, B, rng)
    w_b = generate_multinomial_cluster_weights(u_b, B, rng)

    t_a = np.asarray(unit_sums_a, dtype=np.float64).reshape(u_a, 1)
    n_a = np.asarray(unit_counts_a, dtype=np.float64).reshape(u_a, 1)
    t_b = np.asarray(unit_sums_b, dtype=np.float64).reshape(u_b, 1)
    n_b = np.asarray(unit_counts_b, dtype=np.float64).reshape(u_b, 1)

    boot_a = vectorized_cluster_ratio_bootstrap(t_a, n_a, w_a).flatten()
    boot_b = vectorized_cluster_ratio_bootstrap(t_b, n_b, w_b).flatten()
    boot_delta = boot_b - boot_a

    ci_lower = float(np.percentile(boot_delta, 2.5))
    ci_upper = float(np.percentile(boot_delta, 97.5))
    se = float(np.std(boot_delta, ddof=1))
    mde = compute_mde_080(se)

    # Power discipline check
    detected = ci_lower > 0.0 or ci_upper < 0.0
    powered = abs(delta_hat) >= mde
    detected_and_powered = detected and powered

    warning = None
    if not detected_and_powered and (ci_lower <= 0.0 <= ci_upper):
        warning = "Indeterminate: contrast underpowered (observed effect < MDE)"

    return ContrastEstimate(
        contrast_name="contrast",
        metric_name=metric_name,
        delta_hat=delta_hat,
        ci_95=(ci_lower, ci_upper),
        se=se,
        mde_080=mde,
        detected_and_powered=detected_and_powered,
        descriptive_only=False,
        warning=warning,
    )

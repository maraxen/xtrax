"""Vectorized analytical statistics engine for xtrax Campaign EDA."""

from xtrax.eda.campaign.stats.bootstrap import (
    compute_mde_080,
    evaluate_cluster_bootstrap_condition,
    evaluate_contrast,
    generate_multinomial_cluster_weights,
    looped_cluster_ratio_bootstrap,
    vectorized_cluster_ratio_bootstrap,
)
from xtrax.eda.campaign.stats.composition import (
    calculate_pairwise_diversity,
    compute_composition_entropy,
    compute_cross_knob_collisions,
    compute_duplicate_rates,
    compute_positional_entropy,
    compute_target_recovery,
    safe_shannon_entropy,
)
from xtrax.eda.campaign.stats.scores import (
    compute_best_of_n,
    compute_pareto_frontier_2d,
)

__all__ = [
    "calculate_pairwise_diversity",
    "compute_best_of_n",
    "compute_composition_entropy",
    "compute_cross_knob_collisions",
    "compute_duplicate_rates",
    "compute_mde_080",
    "compute_pareto_frontier_2d",
    "compute_positional_entropy",
    "compute_target_recovery",
    "evaluate_cluster_bootstrap_condition",
    "evaluate_contrast",
    "generate_multinomial_cluster_weights",
    "looped_cluster_ratio_bootstrap",
    "safe_shannon_entropy",
    "vectorized_cluster_ratio_bootstrap",
]

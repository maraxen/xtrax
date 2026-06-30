"""Plan explanation — ensuring all reasoning is non-empty.

This module provides explain_plan, a thin wrapper around extract_plan_stats that
guarantees all reasoning fields are non-empty strings.
"""

from __future__ import annotations

from xtrax.eda.stats import extract_plan_stats
from xtrax.eda.types import BatchPlanLike, PlanStatsDict


def explain_plan(plan: BatchPlanLike) -> PlanStatsDict:
    """Extract plan statistics with guaranteed non-empty reasoning.

    Calls extract_plan_stats to get the full statistics dict, then ensures
    every axis entry has non-empty reasoning. If any entry has empty reasoning,
    substitutes a default message.

    Args:
        plan: A BatchPlan (or any BatchPlanLike-compatible plan) containing
            axis decisions.

    Returns:
        PlanStatsDict with all reasoning fields guaranteed non-empty.
        If any entry had empty reasoning, it is replaced with:
        f"No reasoning recorded for axis '{entry['name']}'"

    Example:
        >>> plan = BatchPlan(decisions=(decision1, decision2))
        >>> stats = explain_plan(plan)
        >>> # All stats["axes"][i]["reasoning"] are non-empty strings
    """
    stats = extract_plan_stats(plan)

    # Ensure all reasoning fields are non-empty
    for entry in stats["axes"]:
        if not entry["reasoning"]:
            entry["reasoning"] = f"No reasoning recorded for axis '{entry['name']}'"

    return stats


__all__ = [
    "explain_plan",
]

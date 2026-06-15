"""Statistics extraction from BatchPlan — stdlib + numpy only.

This module extracts structured statistics from tiling plans without any
visualization dependencies. It can be imported from xtrax core without the
optional eda extras.
"""

from __future__ import annotations

from xtrax.tiling.plan import AxisDecision, BatchPlan
from xtrax.tiling.strategy import Bucket, DedupGather

from .types import (
    AxisStatsEntry,
    BucketStatsEntry,
    DedupStatsEntry,
    PlanStatsDict,
)


def extract_plan_stats(plan: BatchPlan) -> PlanStatsDict:
    """Extract structured statistics from a BatchPlan.

    Walks through all axis decisions in the plan and accumulates:
    - Per-axis statistics (name, strategy type, cardinality, batch_size, reasoning).
    - Strategy type frequency counts.
    - Deduplication statistics (if DedupGather strategy detected).
    - Bucketing statistics (if Bucket strategy detected).
    - Memory warnings (if any estimates exceed thresholds).

    Args:
        plan: A BatchPlan containing axis decisions.

    Returns:
        PlanStatsDict with all fields populated. Empty plan returns empty lists
        and total_axes=0.

    Example:
        >>> plan = BatchPlan(decisions=(decision1, decision2))
        >>> stats = extract_plan_stats(plan)
        >>> stats["total_axes"]
        2
        >>> stats["strategy_counts"]
        {"Vmap": 1, "DedupGather": 1}
    """
    axes: list[AxisStatsEntry] = []
    strategy_counts: dict[str, int] = {}
    memory_warnings: list[str] = []
    dedup_stats: list[DedupStatsEntry] = []
    bucket_stats: list[BucketStatsEntry] = []

    for decision in plan.decisions:
        # Build per-axis entry
        strategy_name = type(decision.strategy).__name__
        axis_entry: AxisStatsEntry = {
            "name": decision.spec.name,
            "strategy": strategy_name,
            "cardinality": decision.spec.cardinality,
            "batch_size": decision.batch_size,
            "reasoning": decision.reasoning,
            "memory_estimate_bytes": None,
        }
        axes.append(axis_entry)

        # Count strategy type
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1

        # Analyze dedup if present
        if isinstance(decision.strategy, DedupGather):
            dedup_entry = analyze_dedup(decision)
            dedup_stats.append(dedup_entry)

        # Analyze bucket if present
        if isinstance(decision.strategy, Bucket):
            bucket_entry = analyze_bucket(decision)
            bucket_stats.append(bucket_entry)

    return {
        "axes": axes,
        "strategy_counts": strategy_counts,
        "total_axes": len(plan.decisions),
        "memory_warnings": memory_warnings,
        "dedup_stats": dedup_stats,
        "bucket_stats": bucket_stats,
    }


def analyze_dedup(decision: AxisDecision) -> DedupStatsEntry:
    """Analyze a DedupGather strategy decision.

    Computes deduplication ratio, padding waste, and other statistics specific
    to deduplication strategies.

    Args:
        decision: An AxisDecision with a DedupGather strategy.

    Returns:
        DedupStatsEntry with deduplication statistics.

    Raises:
        TypeError: If decision.strategy is not a DedupGather instance.

    Example:
        >>> decision = AxisDecision(
        ...     spec=...,
        ...     batch_size=128,
        ...     reasoning="...",
        ...     strategy=DedupGather(..., k=50, k_bucket=64)
        ... )
        >>> stats = analyze_dedup(decision)
        >>> stats["unique_count"]
        50
    """
    if not isinstance(decision.strategy, DedupGather):
        raise TypeError(
            f"analyze_dedup requires DedupGather strategy; "
            f"got {type(decision.strategy).__name__}"
        )

    strategy = decision.strategy
    total_count = decision.spec.cardinality
    unique_count = strategy.k
    padded_count = strategy.k_bucket
    padding_waste = padded_count - unique_count
    dedup_ratio = 1.0 - (unique_count / total_count) if total_count > 0 else 0.0

    return {
        "axis_name": decision.spec.name,
        "dedup_ratio": dedup_ratio,
        "unique_count": unique_count,
        "padded_count": padded_count,
        "total_count": total_count,
        "padding_waste": padding_waste,
    }


def analyze_bucket(decision: AxisDecision) -> BucketStatsEntry:
    """Analyze a Bucket strategy decision.

    Extracts bucket boundary information.

    Args:
        decision: An AxisDecision with a Bucket strategy.

    Returns:
        BucketStatsEntry with bucketing statistics.

    Raises:
        TypeError: If decision.strategy is not a Bucket instance.

    Example:
        >>> decision = AxisDecision(
        ...     spec=...,
        ...     batch_size=...,
        ...     reasoning="...",
        ...     strategy=Bucket(boundaries=(32, 64, 128))
        ... )
        >>> stats = analyze_bucket(decision)
        >>> stats["bucket_boundaries"]
        [32, 64, 128]
    """
    if not isinstance(decision.strategy, Bucket):
        raise TypeError(
            f"analyze_bucket requires Bucket strategy; "
            f"got {type(decision.strategy).__name__}"
        )

    strategy = decision.strategy
    return {
        "axis_name": decision.spec.name,
        "bucket_count": len(strategy.boundaries),
        "bucket_boundaries": list(strategy.boundaries),
    }


__all__ = [
    "extract_plan_stats",
    "analyze_dedup",
    "analyze_bucket",
]

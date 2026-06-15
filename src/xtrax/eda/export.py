"""Pandas DataFrame export for plan statistics.

This module provides conversion from PlanStatsDict to pandas DataFrame format
for advanced analysis and visualization. Pandas is optional and must be installed
via the eda extras.

Requires eda extras: pip install xtrax[eda]
"""

from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "xtrax.eda.export requires visualization extras. "
        "Install with: pip install xtrax[eda]"
    ) from exc


from xtrax.eda.types import PlanStatsDict


def plan_to_dataframe(stats: PlanStatsDict) -> 'pd.DataFrame':
    """Convert PlanStatsDict to a pandas DataFrame.

    Transforms extracted plan statistics into a tabular format with one row per axis.
    All AxisStatsEntry fields are included as columns. Dedup and bucket-specific
    columns (dedup_ratio, unique_count, padded_count, padding_waste, bucket_count,
    bucket_boundaries) are filled with NaN for axes that don't use those strategies.

    Args:
        stats: A PlanStatsDict containing extracted statistics from
            extract_plan_stats().

    Returns:
        pandas.DataFrame with one row per axis. Columns include:
            - name: Axis name (from AxisStatsEntry.name)
            - strategy: Strategy type as string (from AxisStatsEntry.strategy)
            - cardinality: Axis cardinality (from AxisStatsEntry.cardinality)
            - batch_size: Batch size (from AxisStatsEntry.batch_size)
            - reasoning: Decision reasoning (from AxisStatsEntry.reasoning)
            - memory_estimate_bytes: Memory estimate or NaN (from
              AxisStatsEntry.memory_estimate_bytes)
            - dedup_ratio: Dedup ratio (from DedupStatsEntry, NaN if not DedupGather)
            - unique_count: Unique element count (from DedupStatsEntry,
              NaN if not DedupGather)
            - padded_count: Padded bucket size (from DedupStatsEntry,
              NaN if not DedupGather)
            - padding_waste: Padding overhead (from DedupStatsEntry,
              NaN if not DedupGather)
            - bucket_count: Number of buckets (from BucketStatsEntry, NaN if not Bucket)
            - bucket_boundaries: List of bucket boundaries (from
              BucketStatsEntry, NaN if not Bucket)

    Raises:
        ImportError: If pandas is not installed. Install with: pip install xtrax[eda]

    Example:
        >>> from xtrax.eda import extract_plan_stats
        >>> from xtrax.eda.export import plan_to_dataframe
        >>> stats = extract_plan_stats(my_plan)
        >>> df = plan_to_dataframe(stats)
        >>> print(df[["name", "strategy", "cardinality"]])
    """
    # Build a list of rows, one per axis
    rows = []
    for axis_entry in stats["axes"]:
        row: dict[str, Any] = {
            "name": axis_entry["name"],
            "strategy": axis_entry["strategy"],
            "cardinality": axis_entry["cardinality"],
            "batch_size": axis_entry["batch_size"],
            "reasoning": axis_entry["reasoning"],
            "memory_estimate_bytes": axis_entry.get("memory_estimate_bytes"),
            "dedup_ratio": pd.NA,
            "unique_count": pd.NA,
            "padded_count": pd.NA,
            "padding_waste": pd.NA,
            "bucket_count": pd.NA,
            "bucket_boundaries": pd.NA,
        }
        rows.append(row)

    # Create base DataFrame from axes
    if not rows:
        # Return empty DataFrame with all expected columns
        return pd.DataFrame(columns=[
            "name",
            "strategy",
            "cardinality",
            "batch_size",
            "reasoning",
            "memory_estimate_bytes",
            "dedup_ratio",
            "unique_count",
            "padded_count",
            "padding_waste",
            "bucket_count",
            "bucket_boundaries",
        ])

    # Fill in dedup statistics where applicable
    dedup_by_name = {entry["axis_name"]: entry for entry in stats["dedup_stats"]}
    for row in rows:
        axis_name = row["name"]
        if axis_name in dedup_by_name:
            dedup_entry = dedup_by_name[axis_name]
            row["dedup_ratio"] = dedup_entry["dedup_ratio"]
            row["unique_count"] = dedup_entry["unique_count"]
            row["padded_count"] = dedup_entry["padded_count"]
            row["padding_waste"] = dedup_entry["padding_waste"]

    # Fill in bucket statistics where applicable
    bucket_by_name = {entry["axis_name"]: entry for entry in stats["bucket_stats"]}
    for row in rows:
        axis_name = row["name"]
        if axis_name in bucket_by_name:
            bucket_entry = bucket_by_name[axis_name]
            row["bucket_count"] = bucket_entry["bucket_count"]
            row["bucket_boundaries"] = bucket_entry["bucket_boundaries"]

    df = pd.DataFrame(rows)
    return df


__all__ = ["plan_to_dataframe"]

"""xtrax.eda — EDA visualization subpackage.

Core stats extraction (stdlib + numpy only):
    extract_plan_stats: Extract structured statistics from a BatchPlan.
    analyze_dedup: Analyze DedupGather strategy decisions.
    analyze_bucket: Analyze Bucket strategy decisions.

Visualization APIs (requires optional eda extras):
    render: Render plan visualization to PNG/SVG/HTML.
    plot_plan_dashboard: Render fixed-layout dashboard.
"""

from xtrax.eda.stats import analyze_bucket, analyze_dedup, extract_plan_stats
from xtrax.eda.types import (
    AxisStatsEntry,
    BucketStatsEntry,
    DedupStatsEntry,
    PanelName,
    PlanLogger,
    PlanStatsDict,
    _VALID_PANELS,
)

__all__ = [
    # Stats extraction (stdlib + numpy only)
    "extract_plan_stats",
    "analyze_dedup",
    "analyze_bucket",
    # Types
    "AxisStatsEntry",
    "DedupStatsEntry",
    "BucketStatsEntry",
    "PlanStatsDict",
    "PanelName",
    "PlanLogger",
    "_VALID_PANELS",
]

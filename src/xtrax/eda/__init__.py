"""xtrax.eda — EDA visualization subpackage.

Core stats extraction (stdlib + numpy only):
    extract_plan_stats: Extract structured statistics from a BatchPlan.
    analyze_dedup: Analyze DedupGather strategy decisions.
    analyze_bucket: Analyze Bucket strategy decisions.

Visualization APIs (requires optional eda extras):
    render: Render plan visualization to PNG/SVG/HTML.
    plan_to_dataframe: Convert PlanStatsDict to pandas DataFrame.
"""

from xtrax.eda.stats import (
    analyze_bucket,
    analyze_dedup,
    extract_plan_stats,
)
from xtrax.eda.types import (
    _VALID_PANELS,
    AxisStatsEntry,
    BucketStatsEntry,
    DedupStatsEntry,
    PanelName,
    PlanLogger,
    PlanStatsDict,
)


# Lazy wrapper for render — provides error message if matplotlib/seaborn not available
def render(*args, **kwargs):
    """Render a BatchPlan to PNG, SVG, or HTML format.

    Requires eda extras: pip install xtrax[eda]
    
    See xtrax.eda.viz.render for full documentation.
    """
    from xtrax.eda.viz import render as _render
    return _render(*args, **kwargs)


# Lazy wrapper for plan_to_dataframe — provides error message if pandas not available
def plan_to_dataframe(stats: PlanStatsDict):
    """Convert PlanStatsDict to a pandas DataFrame.

    Requires eda extras: pip install xtrax[eda]
    """
    from xtrax.eda.export import plan_to_dataframe as _plan_to_dataframe
    return _plan_to_dataframe(stats)


__all__ = [
    # Stats extraction (stdlib + numpy only)
    "extract_plan_stats",
    "analyze_dedup",
    "analyze_bucket",
    # Visualization (requires eda extras)
    "render",
    "plan_to_dataframe",
    # Types
    "AxisStatsEntry",
    "DedupStatsEntry",
    "BucketStatsEntry",
    "PlanStatsDict",
    "PanelName",
    "PlanLogger",
    "_VALID_PANELS",
]

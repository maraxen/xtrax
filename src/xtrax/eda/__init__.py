"""xtrax.eda — EDA visualization subpackage.

Core stats extraction (stdlib + numpy only):
    extract_plan_stats: Extract structured statistics from a BatchPlan.
    explain_plan: Extract plan statistics with guaranteed non-empty reasoning.
    analyze_dedup: Analyze DedupGather strategy decisions.
    analyze_bucket: Analyze Bucket strategy decisions.

Visualization APIs (requires optional eda extras):
    render: Render plan visualization to PNG/SVG/HTML.
    plan_to_dataframe: Convert PlanStatsDict to pandas DataFrame.
"""

from xtrax.eda.explain import explain_plan
from xtrax.eda.export import (
    export_axes_csv,
    export_canonical_json,
    export_plan_artifacts,
)
from xtrax.eda.plugins import get_quarantined_plugins, reload_plugins
from xtrax.eda.spec import generate_vega_spec
from xtrax.eda.stats import (
    analyze_bucket,
    analyze_dedup,
    extract_plan_stats,
)
from xtrax.eda.types import (
    PanelName,
    PlanLogger,
    PlanStatsDict,
)


def render(*args, **kwargs):
    """Render a BatchPlan to PNG, SVG, or HTML format.

    Lazy import: requires pip install xtrax[eda] or xtrax[eda-vega]
    """
    from xtrax.eda.viz import render as _render

    return _render(*args, **kwargs)


def plan_to_dataframe(*args, **kwargs):
    """Convert PlanStatsDict to DataFrame.

    Lazy import: requires pip install xtrax[eda]
    """
    from xtrax.eda.export import plan_to_dataframe as _ptdf

    return _ptdf(*args, **kwargs)


__all__ = [
    "extract_plan_stats",
    "analyze_dedup",
    "analyze_bucket",
    "explain_plan",
    "render",
    "plan_to_dataframe",
    "generate_vega_spec",
    "export_canonical_json",
    "export_axes_csv",
    "export_plan_artifacts",
    "reload_plugins",
    "get_quarantined_plugins",
    "PlanStatsDict",
    "PlanLogger",
    "PanelName",
]

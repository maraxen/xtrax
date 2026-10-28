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
    backend = kwargs.get("backend", "auto")
    if backend == "vega":
        from pathlib import Path

        from xtrax.eda.backends.vega import VegaBackend
        from xtrax.eda.spec import generate_vega_spec
        from xtrax.eda.stats import extract_plan_stats

        plan = args[0] if args else kwargs["plan"]
        fmt = kwargs.get("fmt", "png")
        panels = kwargs.get("panels")
        max_axes = kwargs.get("max_axes", 100)
        overflow = kwargs.get("overflow", "aggregate")
        offline = kwargs.get("offline", False)
        bundle_js = kwargs.get("bundle_js", False)
        path = kwargs.get("path")
        logger = kwargs.get("logger")
        step = kwargs.get("step")

        stats = extract_plan_stats(plan)
        vl_spec = generate_vega_spec(
            stats,
            panels=panels,
            max_axes=max_axes,
            overflow=overflow,
        )
        backend_inst = VegaBackend()
        result = backend_inst.render_spec(
            vl_spec,
            fmt=fmt,
            stats=stats,
            options={"offline": offline, "bundle_js": bundle_js},
        )
        if path is not None:
            p = Path(path)
            if isinstance(result, bytes):
                p.write_bytes(result)
            else:
                p.write_text(result, encoding="utf-8")
        if logger is not None:
            logger.log_figure(figure=result, fmt=fmt, step=step)
        return None if path is not None else result

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

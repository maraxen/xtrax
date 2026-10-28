"""Declarative Vega-Lite specification generator for xtrax compilation plans.

Compiles PlanStatsDict into valid, portable Vega-Lite v5 JSON specifications
without requiring any graphics libraries (zero-dependency on matplotlib/seaborn).
"""

import math
from typing import Any, Literal

from xtrax.eda.plugins import execute_panel_plugin, get_registered_panels
from xtrax.eda.types import PlanStatsDict

_STANDARD_PANELS: frozenset[str] = frozenset(
    ("strategy", "cardinality", "dedup", "bucket", "memory", "reasoning")
)


def _sanitize_float(val: float | None) -> float | None:
    """Sanitize float to RFC 8259 compliance (NaN and Inf to None)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError, OverflowError):
        return None


def _build_zero_axis_spec() -> dict[str, Any]:
    """Emit single valid Vega-Lite root chart for empty plans (never empty vconcat)."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Empty compilation plan",
        "width": 600,
        "height": 180,
        "padding": 20,
        "data": {"values": [{}]},
        "mark": {
            "type": "text",
            "text": "No active execution axes in compilation plan",
            "fontSize": 15,
            "fontStyle": "italic",
            "color": "#718096",
            "align": "center",
            "baseline": "middle",
        },
    }


def _build_strategy_panel(stats: PlanStatsDict) -> dict[str, Any]:
    """Donut chart showing proportion of execution strategies."""
    counts = stats.get("strategy_counts", {})
    values = [{"strategy": k, "count": v} for k, v in counts.items()]

    return {
        "title": "Strategy Distribution",
        "width": 500,
        "height": 200,
        "data": {"values": values},
        "mark": {"type": "arc", "innerRadius": 50, "tooltip": True},
        "encoding": {
            "theta": {"field": "count", "type": "quantitative"},
            "color": {"field": "strategy", "type": "nominal", "title": "Strategy"},
        },
    }


def _build_cardinality_panel(
    axes_data: list[dict[str, Any]],
    overflow_count: int,
    overflow_mode: str = "aggregate",
) -> dict[str, Any]:
    """Bar chart of axis cardinalities."""
    title = "Cardinality by Axis"
    if overflow_count > 0:
        if overflow_mode == "aggregate":
            title += (
                f" (Top {len(axes_data)}, +{overflow_count} remaining axes aggregated"
                " (see axes.csv for full streaming tabular record))"
            )
        else:
            title += f" (Top {len(axes_data)}, +{overflow_count} remaining truncated)"

    return {
        "title": title,
        "width": 500,
        "height": 200,
        "data": {"values": axes_data},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "name", "type": "nominal", "title": "Axis Name", "sort": None},
            "y": {"field": "cardinality", "type": "quantitative", "title": "Cardinality"},
            "color": {"field": "strategy", "type": "nominal", "title": "Strategy"},
        },
    }


def _build_dedup_panel(stats: PlanStatsDict) -> dict[str, Any]:
    """Bar chart of deduplication reduction efficiency."""
    dedup_list = stats.get("dedup_stats", [])
    if not dedup_list:
        return {
            "title": "Deduplication Efficiency",
            "width": 500,
            "height": 100,
            "data": {"values": [{}]},
            "mark": {"type": "text", "text": "No dedup data", "color": "#a0aec0"},
        }

    values = []
    for d in dedup_list:
        ratio = _sanitize_float(d.get("dedup_ratio"))
        values.append(
            {
                "axis_name": d["axis_name"],
                "dedup_ratio": ratio,
                "unique_count": d.get("unique_count", 0),
                "total_count": d.get("total_count", 0),
                "padding_waste": d.get("padding_waste", 0),
            }
        )

    return {
        "title": "Deduplication Efficiency",
        "width": 500,
        "height": 200,
        "data": {"values": values},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "axis_name", "type": "nominal", "title": "Axis Name"},
            "y": {
                "field": "dedup_ratio",
                "type": "quantitative",
                "title": "Dedup Reduction Ratio (1 - unique/total)",
                "scale": {"domain": [0, 1]},
            },
            "color": {"field": "axis_name", "type": "nominal", "legend": None},
        },
    }


def _build_bucket_panel(stats: PlanStatsDict) -> dict[str, Any]:
    """Overview of bucket distributions and boundaries."""
    bucket_list = stats.get("bucket_stats", [])
    if not bucket_list:
        return {
            "title": "Bucket Strategy Overview",
            "width": 500,
            "height": 100,
            "data": {"values": [{}]},
            "mark": {"type": "text", "text": "No bucket data", "color": "#a0aec0"},
        }

    values = []
    for b in bucket_list:
        values.append(
            {
                "axis_name": b["axis_name"],
                "bucket_count": b.get("bucket_count", 0),
                "boundaries": ", ".join(str(x) for x in b.get("bucket_boundaries", [])),
            }
        )

    return {
        "title": "Bucket Strategy Overview",
        "width": 500,
        "height": 200,
        "data": {"values": values},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "axis_name", "type": "nominal", "title": "Axis Name"},
            "y": {"field": "bucket_count", "type": "quantitative", "title": "Bucket Count"},
            "color": {"field": "axis_name", "type": "nominal", "legend": None},
        },
    }


def _build_memory_panel(axes_data: list[dict[str, Any]], stats: PlanStatsDict) -> dict[str, Any]:
    """Memory consumption estimates and warnings."""
    warnings = stats.get("memory_warnings", [])
    mem_values = []
    for a in axes_data:
        raw_mem = a.get("memory_estimate_bytes")
        sanitized = _sanitize_float(raw_mem)
        mem_values.append(
            {
                "name": a["name"],
                "memory_bytes": sanitized,
                "status": "Estimated" if sanitized is not None else "Unbounded / Dynamic",
            }
        )

    title = "Estimated Memory Consumption"
    if warnings:
        title += f" ({len(warnings)} warning{'s' if len(warnings) > 1 else ''})"

    return {
        "title": title,
        "width": 500,
        "height": 200,
        "data": {"values": mem_values},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "name", "type": "nominal", "title": "Axis Name"},
            "y": {"field": "memory_bytes", "type": "quantitative", "title": "Estimated Bytes"},
            "color": {"field": "status", "type": "nominal", "title": "Status"},
        },
    }


def _build_reasoning_panel(axes_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Text-based panel displaying compiler reasoning per axis."""
    return {
        "title": "Compiler Reasoning by Axis",
        "width": 500,
        "height": max(150, len(axes_data) * 25),
        "data": {"values": axes_data},
        "mark": {"type": "text", "align": "left", "baseline": "middle"},
        "encoding": {
            "y": {"field": "name", "type": "nominal", "title": "Axis", "sort": None},
            "text": {"field": "reasoning", "type": "nominal"},
            "tooltip": {"field": "reasoning", "type": "nominal"},
        },
    }


def generate_vega_spec(
    stats: PlanStatsDict,
    panels: set[str] | None = None,
    max_axes: int = 100,
    overflow: Literal["aggregate", "truncate"] = "aggregate",
) -> dict[str, Any]:
    """Generate a valid Vega-Lite v5 specification dictionary from PlanStatsDict.

    Args:
        stats: Intermediate compiler plan statistics.
        panels: Optional subset of panels to include. Defaults to all standard panels.
        max_axes: Maximum axes to display individually before overflow handling.
        overflow: Strategy when axes exceed max_axes ("aggregate" or "truncate").

    Returns:
        Vega-Lite v5 specification dictionary.
    """
    total_axes = stats.get("total_axes", 0)
    if total_axes == 0:
        return _build_zero_axis_spec()

    registered = get_registered_panels()
    all_valid = _STANDARD_PANELS | set(registered.keys())

    active_panels = _STANDARD_PANELS if panels is None else set(panels)
    invalid_panels = active_panels - all_valid
    if invalid_panels:
        raise ValueError(f"Invalid panel names: {invalid_panels}. Valid panels: {set(all_valid)}")

    raw_axes = stats.get("axes", [])
    overflow_count = max(0, len(raw_axes) - max_axes)
    if overflow_count > 0:
        sorted_axes = sorted(raw_axes, key=lambda a: a.get("cardinality", 0), reverse=True)
        display_axes = sorted_axes[:max_axes]
    else:
        display_axes = raw_axes

    # Prepare sanitized axis records
    sanitized_axes: list[dict[str, Any]] = []
    for a in display_axes:
        entry = dict(a)
        entry["cardinality"] = a.get("cardinality", 0)
        entry["memory_estimate_bytes"] = _sanitize_float(a.get("memory_estimate_bytes"))
        sanitized_axes.append(entry)

    vconcat_specs: list[dict[str, Any]] = []

    if "strategy" in active_panels:
        vconcat_specs.append(_build_strategy_panel(stats))
    if "cardinality" in active_panels:
        vconcat_specs.append(_build_cardinality_panel(sanitized_axes, overflow_count, overflow))
    if "dedup" in active_panels:
        vconcat_specs.append(_build_dedup_panel(stats))
    if "bucket" in active_panels:
        vconcat_specs.append(_build_bucket_panel(stats))
    if "memory" in active_panels:
        vconcat_specs.append(_build_memory_panel(sanitized_axes, stats))
    if "reasoning" in active_panels:
        vconcat_specs.append(_build_reasoning_panel(sanitized_axes))

    # Append custom registered plugin panels
    for custom_name in sorted(active_panels - _STANDARD_PANELS):
        plugin = registered[custom_name]
        snippet = execute_panel_plugin(plugin, stats)
        vconcat_specs.append(snippet)

    if not vconcat_specs:
        return _build_zero_axis_spec()

    if len(vconcat_specs) == 1:
        root_spec = vconcat_specs[0]
        root_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        return root_spec

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "xtrax execution plan dashboard",
        "vconcat": vconcat_specs,
        "resolve": {"scale": {"color": "independent"}},
    }

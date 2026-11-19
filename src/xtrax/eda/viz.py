"""Matplotlib drawing for BatchPlan visualizations (PNG/SVG/HTML).

MUST IMPORT matplotlib.use("Agg") FIRST, before any other matplotlib imports.
This module requires optional eda extras: pip install xtrax[eda]
"""

import io
from collections.abc import Callable

# CRITICAL: matplotlib backend must be set BEFORE importing pyplot
import matplotlib

matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise ImportError(
        "xtrax.eda.viz requires visualization extras. Install with: pip install xtrax[eda]"
    ) from exc

from xtrax.eda.render_core import render  # re-exported: viz.render is the shared pipeline
from xtrax.eda.types import _VALID_PANELS, PlanStatsDict


def _placeholder(ax, text: str, **text_kwargs) -> None:  # noqa: ANN001
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, **text_kwargs)
    ax.axis("off")


def _draw_strategy(ax, stats: PlanStatsDict) -> None:  # noqa: ANN001
    strategy_data = stats["strategy_counts"]
    if not strategy_data:
        _placeholder(ax, "No strategy data")
        return
    strategies = list(strategy_data.keys())
    counts = list(strategy_data.values())
    sns.barplot(x=strategies, y=counts, ax=ax, hue=strategies, legend=False, palette="Set2")
    ax.set_xlabel("Strategy Type")
    ax.set_ylabel("Count")
    ax.set_title("Strategy Distribution")


def _draw_cardinality(ax, stats: PlanStatsDict) -> None:  # noqa: ANN001
    axes_data = stats["axes"]
    if not axes_data:
        _placeholder(ax, "No cardinality data")
        return
    names = [a["name"] for a in axes_data]
    cardinalities = [a["cardinality"] for a in axes_data]
    sns.scatterplot(x=names, y=cardinalities, s=200, ax=ax, palette="husl", hue=names, legend=False)
    ax.set_ylabel("Cardinality")
    ax.set_xlabel("Axis Name")
    ax.set_title("Cardinality by Axis")


def _draw_dedup(ax, stats: PlanStatsDict) -> None:  # noqa: ANN001
    dedup_data = stats["dedup_stats"]
    if not dedup_data:
        _placeholder(ax, "No dedup data")
        return
    axis_names = [d["axis_name"] for d in dedup_data]
    ratios = [d["dedup_ratio"] for d in dedup_data]
    sns.barplot(x=axis_names, y=ratios, ax=ax, hue=axis_names, legend=False, palette="muted")
    ax.set_ylabel("Dedup Reduction Ratio (1 - unique/total)")
    ax.set_xlabel("Axis Name")
    ax.set_title("Deduplication Efficiency")
    ax.set_ylim([0, 1])


def _draw_bucket(ax, stats: PlanStatsDict) -> None:  # noqa: ANN001
    bucket_data = stats["bucket_stats"]
    if not bucket_data:
        _placeholder(ax, "No bucket data")
        return
    axis_names = [b["axis_name"] for b in bucket_data]
    bucket_counts = [b["bucket_count"] for b in bucket_data]
    sns.barplot(x=axis_names, y=bucket_counts, ax=ax, hue=axis_names, legend=False, palette="Set1")
    ax.set_ylabel("Number of Buckets")
    ax.set_xlabel("Axis Name")
    ax.set_title("Bucket Configuration")


def _draw_text_panel(ax, title: str, lines: list[str], fontsize: int) -> None:  # noqa: ANN001
    if not lines:
        return
    ax.text(
        0.05,
        0.95,
        f"{title}:\n" + "\n".join(lines),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        family="monospace",
    )
    ax.axis("off")


def _draw_memory(ax, stats: PlanStatsDict) -> None:  # noqa: ANN001
    lines = [f"• {w}" for w in stats["memory_warnings"]]
    _draw_text_panel(ax, "Memory Warnings", lines, fontsize=10)


def _draw_reasoning(ax, stats: PlanStatsDict) -> None:  # noqa: ANN001
    lines = [f"{a['name']}: {a['reasoning']}" for a in stats["axes"]]
    _draw_text_panel(ax, "Decision Reasoning", lines, fontsize=9)


# Panel order on the figure, each with the stats it needs to be drawn at all.
_PANELS: tuple[tuple[str, Callable[[PlanStatsDict], bool], Callable], ...] = (
    ("strategy", lambda _: True, _draw_strategy),
    ("cardinality", lambda s: s["total_axes"] > 0, _draw_cardinality),
    ("dedup", lambda s: bool(s["dedup_stats"]), _draw_dedup),
    ("bucket", lambda s: bool(s["bucket_stats"]), _draw_bucket),
    ("memory", lambda s: bool(s["memory_warnings"]), _draw_memory),
    ("reasoning", lambda _: True, _draw_reasoning),  # always last if present
)


def _draw_figure(stats: PlanStatsDict, active_panels: set[str]) -> list[str]:
    """Draw the dashboard on a new current figure; return the drawn panel names."""
    if stats["total_axes"] == 0:
        _, ax = plt.subplots(figsize=(8, 4))
        _placeholder(ax, "No axes in plan", fontsize=16)
        return []
    selected = [(name, draw) for name, has, draw in _PANELS if name in active_panels and has(stats)]
    rows = len(selected) or 2  # at least 2 rows (strategy + cardinality) when none apply
    _, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows), tight_layout=True)
    if rows == 1:
        axes = [axes]
    for ax, (_, draw) in zip(axes, selected, strict=False):
        draw(ax, stats)
    return [name for name, _ in selected]


def _encode_figure(fmt: str, panel_names: list[str]) -> bytes | str:
    buf = io.BytesIO()
    if fmt == "png":
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        return buf.getvalue()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    svg_str = buf.getvalue().decode("utf-8")
    if panel_names:
        svg_str = _inject_panel_attributes(svg_str, panel_names)
    if fmt == "svg":
        return svg_str.encode("utf-8")
    return (
        f"<!DOCTYPE html>\n"
        f"<html>\n"
        f"<head>\n"
        f'  <meta charset="utf-8">\n'
        f"  <title>Plan Visualization</title>\n"
        f"</head>\n"
        f"<body>\n"
        f"  {svg_str}\n"
        f"</body>\n"
        f"</html>"
    )


def _render_matplotlib(stats: PlanStatsDict, panels: set[str] | None, fmt: str) -> bytes | str:
    """Draw the dashboard with matplotlib and encode it; always closes the figures."""
    active_panels = panels if panels is not None else set(_VALID_PANELS)
    panel_names = _draw_figure(stats, active_panels)
    result = _encode_figure(fmt, panel_names)
    plt.close("all")
    return result


def _inject_panel_attributes(svg_str: str, panel_names: list[str]) -> str:
    """Post-process SVG to add data-panel attributes for each panel.

    Wraps each subplot's top-level <g> element with a data-panel attribute.
    Since matplotlib generates multiple <g> elements per subplot, we inject
    a marker comment before the first <g> of each panel's content.
    """
    # For each panel, insert a <!-- data-panel="name" --> marker
    # after the SVG declaration and metadata
    lines = svg_str.split("\n")

    # Find where to insert markers (after initial SVG tags but before content)
    result_lines = []
    in_metadata = False
    panel_idx = 0

    for i, line in enumerate(lines):
        result_lines.append(line)

        # Mark end of metadata section
        if "</metadata>" in line:
            in_metadata = False
        if "<metadata>" in line:
            in_metadata = True

        # After metadata and defs, before first <g> with actual content
        if not in_metadata and "</defs>" in line and panel_idx < len(panel_names):
            # Add panel markers after defs
            for panel_name in panel_names:
                result_lines.append(f'  <!-- data-panel="{panel_name}" -->')
            panel_idx = len(panel_names)  # Only inject once

    return "\n".join(result_lines)


__all__ = ["render"]

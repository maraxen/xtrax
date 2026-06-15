"""Visualization rendering for BatchPlan — PNG/SVG/HTML output.

MUST IMPORT matplotlib.use("Agg") FIRST, before any other matplotlib imports.
This module requires optional eda extras: pip install xtrax[eda]
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable, Literal

# CRITICAL: matplotlib backend must be set BEFORE importing pyplot
import matplotlib

matplotlib.use("Agg")

# Now safe to import pyplot and seaborn
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise ImportError(
        "xtrax.eda.viz requires visualization extras. "
        "Install with: pip install xtrax[eda]"
    ) from exc

from xtrax.tiling.plan import BatchPlan
from xtrax.eda.stats import extract_plan_stats
from xtrax.eda.types import PlanLogger, PlanStatsDict, _VALID_PANELS

# Required keys for a valid PlanStatsDict after transformation
_REQUIRED_STATS_KEYS = frozenset(
    {
        "axes",
        "strategy_counts",
        "total_axes",
        "memory_warnings",
        "dedup_stats",
        "bucket_stats",
    }
)


def render(
    plan: BatchPlan,
    view: str = "dashboard",
    fmt: Literal["png", "svg", "html"] = "png",
    path: str | Path | None = None,
    stats_transform: Callable[[PlanStatsDict], PlanStatsDict] | None = None,
    metadata: bool = False,
    logger: PlanLogger | None = None,
    step: int | None = None,
    panels: set[str] | None = None,
) -> bytes | str | None:
    """Render a BatchPlan to PNG, SVG, or HTML format.

    Extracts statistics from the plan, applies optional transformations, and
    renders a fixed-layout dashboard with strategy distribution, cardinality
    scatter, and other metrics.

    Args:
        plan: The BatchPlan to visualize.
        view: The view type (currently only "dashboard" is implemented). Default "dashboard".
        fmt: Output format — "png" (bytes), "svg" (bytes), or "html" (str). Default "png".
        path: Optional file path to write output. If None, returns bytes/str in-memory.
               Must be set if metadata=True.
        stats_transform: Optional function to transform the stats dict before rendering.
                        Receives and returns PlanStatsDict. If provided, result must
                        contain all required keys. Default None (no transformation).
        metadata: If True, writes a .json sidecar with the stats dict alongside the
                 output file. Requires path to be set. Default False.
        logger: Optional PlanLogger implementation for remote logging (tensorboard, wandb, etc.).
               Called with figure data after rendering. Default None.
        step: Optional iteration/epoch number passed to logger. Default None.
        panels: Optional set of panel names to render. Valid names are
               "strategy", "cardinality", "dedup", "bucket", "memory", "reasoning".
               If None, all available panels are rendered. Default None.

    Returns:
        bytes for PNG/SVG format with no path, str for HTML with no path, or None if
        output was written to path.

    Raises:
        ValueError: If metadata=True but path is None, or if panels contains unknown names.
        TypeError: If stats_transform returns a dict missing required keys.
    """
    # Validation
    if metadata and path is None:
        raise ValueError("metadata=True requires path to be set")

    if panels is not None:
        unknown = panels - _VALID_PANELS
        if unknown:
            raise ValueError(
                f"Unknown panel(s): {unknown!r}. Valid panels: {sorted(_VALID_PANELS)}"
            )

    # Extract stats
    stats = extract_plan_stats(plan)

    # Apply transform
    if stats_transform is not None:
        stats = stats_transform(stats)
        missing = _REQUIRED_STATS_KEYS - stats.keys()
        if missing:
            raise TypeError(
                f"stats_transform must return PlanStatsDict with all required keys; "
                f"missing: {missing!r}"
            )

    # Filter panels
    active_panels = panels if panels is not None else set(_VALID_PANELS)

    # Build figure
    # EMPTY PLAN GUARD: if stats["total_axes"] == 0, render a placeholder
    if stats["total_axes"] == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No axes in plan",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.axis("off")
        panel_names = []
    else:
        # Calculate number of rows based on active panels with data
        panel_names = []
        panel_count = 0
        if "strategy" in active_panels:
            panel_names.append("strategy")
            panel_count += 1
        if "cardinality" in active_panels and stats["total_axes"] > 0:
            panel_names.append("cardinality")
            panel_count += 1
        if "dedup" in active_panels and stats["dedup_stats"]:
            panel_names.append("dedup")
            panel_count += 1
        if "bucket" in active_panels and stats["bucket_stats"]:
            panel_names.append("bucket")
            panel_count += 1
        if "memory" in active_panels and stats["memory_warnings"]:
            panel_names.append("memory")
            panel_count += 1
        if "reasoning" in active_panels:
            panel_names.append("reasoning")
            panel_count += 1

        # Default to at least 2 rows (strategy + cardinality)
        if panel_count == 0:
            panel_count = 2

        fig, axes = plt.subplots(
            panel_count, 1, figsize=(10, 4 * panel_count), tight_layout=True
        )

        # Ensure axes is always a list
        if panel_count == 1:
            axes = [axes]

        ax_idx = 0

        # Strategy panel
        if "strategy" in panel_names:
            ax = axes[ax_idx]
            strategy_data = stats["strategy_counts"]
            if strategy_data:
                strategies = list(strategy_data.keys())
                counts = list(strategy_data.values())
                sns.barplot(x=strategies, y=counts, ax=ax, hue=strategies, legend=False, palette="Set2")
                ax.set_xlabel("Strategy Type")
                ax.set_ylabel("Count")
                ax.set_title("Strategy Distribution")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No strategy data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
            ax_idx += 1

        # Cardinality panel
        if "cardinality" in panel_names:
            ax = axes[ax_idx]
            axes_data = stats["axes"]
            if axes_data:
                names = [a["name"] for a in axes_data]
                cardinalities = [a["cardinality"] for a in axes_data]
                sns.scatterplot(
                    x=names,
                    y=cardinalities,
                    s=200,
                    ax=ax,
                    palette="husl",
                    hue=names,
                    legend=False,
                )
                ax.set_ylabel("Cardinality")
                ax.set_xlabel("Axis Name")
                ax.set_title("Cardinality by Axis")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No cardinality data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
            ax_idx += 1

        # Dedup panel
        if "dedup" in panel_names:
            ax = axes[ax_idx]
            dedup_data = stats["dedup_stats"]
            if dedup_data:
                axis_names = [d["axis_name"] for d in dedup_data]
                ratios = [d["dedup_ratio"] for d in dedup_data]
                sns.barplot(x=axis_names, y=ratios, ax=ax, hue=axis_names, legend=False, palette="muted")
                ax.set_ylabel("Dedup Ratio (unique / total)")
                ax.set_xlabel("Axis Name")
                ax.set_title("Deduplication Efficiency")
                ax.set_ylim([0, 1])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No dedup data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
            ax_idx += 1

        # Bucket panel
        if "bucket" in panel_names:
            ax = axes[ax_idx]
            bucket_data = stats["bucket_stats"]
            if bucket_data:
                axis_names = [b["axis_name"] for b in bucket_data]
                bucket_counts = [b["bucket_count"] for b in bucket_data]
                sns.barplot(x=axis_names, y=bucket_counts, ax=ax, hue=axis_names, legend=False, palette="Set1")
                ax.set_ylabel("Number of Buckets")
                ax.set_xlabel("Axis Name")
                ax.set_title("Bucket Configuration")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No bucket data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
            ax_idx += 1

        # Memory panel
        if "memory" in panel_names:
            ax = axes[ax_idx]
            warnings = stats["memory_warnings"]
            if warnings:
                ax.text(
                    0.05,
                    0.95,
                    "Memory Warnings:\n" + "\n".join(f"• {w}" for w in warnings),
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=10,
                    family="monospace",
                )
                ax.axis("off")
            ax_idx += 1

        # Reasoning panel (always last if present)
        if "reasoning" in panel_names:
            ax = axes[ax_idx]
            axes_data = stats["axes"]
            if axes_data:
                reasoning_text = "\n".join(
                    [f"{a['name']}: {a['reasoning']}" for a in axes_data]
                )
                ax.text(
                    0.05,
                    0.95,
                    "Decision Reasoning:\n" + reasoning_text,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                    family="monospace",
                )
                ax.axis("off")

    # Render to format
    buf = io.BytesIO()

    if fmt == "png":
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        result: bytes | str = buf.getvalue()
    elif fmt == "svg":
        plt.savefig(buf, format="svg", bbox_inches="tight")
        svg_bytes = buf.getvalue()
        svg_str = svg_bytes.decode("utf-8")
        
        # Post-process SVG to inject data-panel attributes
        # Add a comment with panel names for each subplot group
        if panel_names:
            svg_str = _inject_panel_attributes(svg_str, panel_names)
        
        result = svg_str.encode("utf-8")
    elif fmt == "html":
        plt.savefig(buf, format="svg", bbox_inches="tight")
        svg_bytes = buf.getvalue()
        svg_str = svg_bytes.decode("utf-8")

        # Post-process SVG to inject data-panel attributes
        if panel_names:
            svg_str = _inject_panel_attributes(svg_str, panel_names)

        # Wrap in minimal HTML
        html_content = (
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
        result = html_content

    plt.close("all")

    # Write to path or return
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "html":
            p.write_text(result if isinstance(result, str) else result.decode())
        else:
            p.write_bytes(result if isinstance(result, bytes) else result.encode())

        if metadata:
            metadata_path = p.with_suffix(".json")
            metadata_path.write_text(json.dumps(stats, default=str))

        if logger is not None:
            logger.log_figure(figure=result, fmt=fmt, step=step)

        return None
    else:
        if logger is not None:
            logger.log_figure(figure=result, fmt=fmt, step=step)
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

"""Backend-neutral render pipeline for BatchPlan visualizations.

Holds the single implementation behind every public ``render`` entry point.
It imports nothing that needs matplotlib, so the Vega-Lite path works with the
``eda-vega`` extra alone. The matplotlib drawing lives in ``xtrax.eda.viz`` and
is imported only when that backend is selected.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from xtrax.eda.backends.base import reject_bundle_js
from xtrax.eda.plugins import get_registered_backends
from xtrax.eda.spec import generate_vega_spec
from xtrax.eda.stats import extract_plan_stats
from xtrax.eda.types import _VALID_PANELS, PlanLogger, PlanStatsDict
from xtrax.tiling.plan import BatchPlan

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


def _write_output(
    p: Path, result: bytes | str, fmt: str, metadata_stats: PlanStatsDict | None
) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        p.write_text(result if isinstance(result, str) else result.decode())
    else:
        p.write_bytes(result if isinstance(result, bytes) else result.encode())
    if metadata_stats is not None:
        p.with_suffix(".json").write_text(json.dumps(metadata_stats, default=str))


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
    backend: Literal["auto", "vega", "matplotlib"] = "auto",
    max_axes: int = 100,
    overflow: Literal["aggregate", "truncate"] = "aggregate",
    offline: bool = False,
    bundle_js: bool = False,
) -> bytes | str | None:
    """Render a BatchPlan to PNG, SVG, or HTML format.

    Extracts statistics from the plan, applies optional transformations, and
    renders a fixed-layout dashboard with strategy distribution, cardinality
    scatter, and other metrics.

    Args:
        plan: The BatchPlan to visualize.
        view: The view type (currently only "dashboard" is implemented).
               Default "dashboard".
        fmt: Output format — "png" (bytes), "svg" (bytes), or "html"
               (str). Default "png".
        path: Optional file path to write output. If None, returns bytes/str in-memory.
               Must be set if metadata=True.
        stats_transform: Optional function to transform the stats dict before rendering.
                        Receives and returns PlanStatsDict. If provided, result must
                        contain all required keys. Default None (no transformation).
        metadata: If True, writes a .json sidecar with the stats dict alongside the
                 output file. Requires path to be set. Default False.
        logger: Optional PlanLogger implementation for remote logging
               (tensorboard, wandb, etc.).
               Called with figure data after rendering. Default None.
        step: Optional iteration/epoch number passed to logger. Default None.
        panels: Optional set of panel names to render. Valid names are
               "strategy", "cardinality", "dedup", "bucket", "memory", "reasoning".
               If None, all available panels are rendered. Default None.
        bundle_js: Not implemented. Passing True raises NotImplementedError,
               because vendored Vega JS is not shipped. Use offline=True for the
               static SVG fallback.

    Returns:
        bytes for PNG/SVG format with no path, str for HTML with no path, or None if
        output was written to path.

    Raises:
        ValueError: If fmt is not "png", "svg" or "html", if metadata=True but
            path is None, or if panels contains unknown names.
        TypeError: If stats_transform returns a dict missing required keys.
        NotImplementedError: If bundle_js=True.
    """
    # Validation
    if fmt not in ("png", "svg", "html"):
        raise ValueError(f"Unknown fmt {fmt!r}; expected 'png', 'svg' or 'html'")
    if metadata and path is None:
        raise ValueError("metadata=True requires path to be set")
    reject_bundle_js(bundle_js)

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

    valid_backends = {"auto", "vega", "matplotlib"}
    registered_backends = get_registered_backends()
    valid_backends.update(registered_backends.keys())

    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend!r}. Supported: {sorted(valid_backends)}")

    selected_backend = "matplotlib" if backend == "auto" else backend

    if selected_backend == "vega":
        from xtrax.eda.backends.vega import VegaBackend

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
            options={"offline": offline},
        )
    elif selected_backend == "matplotlib":
        from xtrax.eda.viz import _render_matplotlib

        result = _render_matplotlib(stats, panels, fmt)
    elif selected_backend in registered_backends:
        adapter = registered_backends[selected_backend]

        vl_spec = generate_vega_spec(
            stats,
            panels=panels,
            max_axes=max_axes,
            overflow=overflow,
        )
        result = adapter.render_spec(
            vl_spec,
            fmt=fmt,
            stats=stats,
            options={"offline": offline},
        )

    if path is not None:
        _write_output(Path(path), result, fmt, stats if metadata else None)
    if logger is not None:
        logger.log_figure(figure=result, fmt=fmt, step=step)
    return None if path is not None else result

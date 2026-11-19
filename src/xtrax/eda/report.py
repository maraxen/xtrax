"""Report rendering utilities for xtrax.eda (#3126)."""

from typing import Any

from xtrax.eda.render_core import render


def render_report(
    plan: Any,
    *,
    bundle_js: bool = True,
    **kwargs: Any,
) -> str | bytes | None:
    """Render an interactive HTML report for a plan with bundled JavaScript by default (#3126)."""
    return render(plan, fmt="html", backend="vega", bundle_js=bundle_js, **kwargs)


__all__ = ["render_report"]

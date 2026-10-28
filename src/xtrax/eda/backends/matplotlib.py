"""Legacy Matplotlib / Seaborn visualization backend for xtrax EDA.

Provides backward-compatible raster, vector, and SVG-in-HTML rendering.
Requires the optional xtrax[eda] or xtrax[eda-matplotlib] extra.
"""

from __future__ import annotations

from typing import Any, Literal

from xtrax.eda.backends.base import BaseBackend
from xtrax.eda.types import PlanStatsDict


class MatplotlibBackend(BaseBackend):
    """Rendering backend powered by Matplotlib and Seaborn."""

    name: str = "matplotlib"

    def render_spec(
        self,
        spec: dict[str, Any],
        fmt: Literal["png", "svg", "html"],
        stats: PlanStatsDict,
        options: dict[str, Any] | None = None,
    ) -> bytes | str:
        """Render using Matplotlib/Seaborn."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "MatplotlibBackend requires matplotlib and seaborn. "
                "Install them with `pip install xtrax[eda]` or `pip install xtrax[eda-matplotlib]`."
            ) from err

        from xtrax.eda.viz import _VALID_PANELS, _draw_figure, _encode_figure

        opts = options or {}
        panels = opts.get("panels")
        active_panels = _VALID_PANELS if panels is None else set(panels)

        plt.close("all")
        panel_names = _draw_figure(stats, active_panels)
        result = _encode_figure(fmt, panel_names)
        plt.close("all")
        return result

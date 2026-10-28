"""Vega-Lite rendering backend for xtrax EDA.

Provides headless SVG/PNG rasterization via vl-convert-python and
self-contained, airgap-safe progressive HTML dashboards.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from xtrax.eda.backends.base import BaseBackend
from xtrax.eda.types import PlanStatsDict


class VegaBackend(BaseBackend):
    """Canonical rendering backend using Vega-Lite v5 specifications."""

    name: str = "vega"

    def _render_fallback_svg(self, spec: dict[str, Any], stats: PlanStatsDict) -> str:
        """Generate static vector SVG snapshot for progressive airgapped fallback."""
        try:
            import vl_convert as vlc

            return vlc.vegalite_to_svg(spec)
        except Exception:
            # Zero-dependency SVG fallback for airgapped clusters without vl-convert
            total = stats.get("total_axes", 0)
            text = (
                "No active execution axes in compilation plan"
                if total == 0
                else f"xtrax Execution Plan Dashboard ({total} active axes)"
            )
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="180" '
                'viewBox="0 0 600 180">'
                '<rect width="600" height="180" fill="#f7fafc" stroke="#e2e8f0" rx="8"/>'
                '<text x="300" y="90" text-anchor="middle" dominant-baseline="middle" '
                'fill="#718096" font-family="system-ui, sans-serif" font-size="15" '
                'font-style="italic">'
                f"{text}</text></svg>"
            )

    def render_spec(
        self,
        spec: dict[str, Any],
        fmt: Literal["png", "svg", "html"],
        stats: PlanStatsDict,
        options: dict[str, Any] | None = None,
    ) -> bytes | str:
        """Render Vega-Lite specification to PNG, SVG, or progressive HTML."""
        opts = options or {}

        if fmt == "png":
            try:
                import vl_convert as vlc

                scale = opts.get("scale", 2.0)
                return vlc.vegalite_to_png(spec, scale=scale)
            except ImportError as err:
                raise ImportError(
                    "VegaBackend PNG rendering requires vl-convert-python. "
                    "Install it with `pip install xtrax[eda-vega]`."
                ) from err

        if fmt == "svg":
            try:
                import vl_convert as vlc

                svg_str = vlc.vegalite_to_svg(spec)
                return svg_str.encode("utf-8")
            except ImportError as err:
                raise ImportError(
                    "VegaBackend SVG rendering requires vl-convert-python. "
                    "Install it with `pip install xtrax[eda-vega]`."
                ) from err

        if fmt == "html":
            fallback_svg = self._render_fallback_svg(spec, stats)
            spec_json = json.dumps(spec).replace("</", "<\\/")
            offline = opts.get("offline", False)
            status_text = (
                "Static vector mode (offline)."
                if offline
                else "Loading interactive visualization..."
            )

            # Build self-contained HTML with progressive enhancement contract (AC-04, AC-16)
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                '  <meta charset="utf-8">\n'
                '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
                "  <title>xtrax Execution Plan Dashboard</title>\n"
                "  <style>\n"
                "    body { font-family: system-ui, sans-serif; margin: 20px; color: #2d3748; }\n"
                "    .dashboard-container { max-width: 900px; margin: 0 auto; }\n"
                "    .header { border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; }\n"
                "    .status-badge { font-size: 12px; color: #718096; margin-top: 4px; }\n"
                "    #fallback-svg { margin-top: 10px; }\n"
                "    #vega-vis { display: none; margin-top: 10px; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                '  <div class="dashboard-container">\n'
                '    <div class="header">\n'
                "      <h2>xtrax Execution Plan Inspection</h2>\n"
                f'      <div id="loader-status" class="status-badge">{status_text}</div>\n'
                "    </div>\n"
                '    <div id="fallback-svg">\n'
                f"      {fallback_svg}\n"
                "    </div>\n"
                '    <div id="vega-vis" style="display: none;"></div>\n'
                "  </div>\n"
                "  <script>\n"
                "    (function() {\n"
                f"      var isOffline = {str(offline).lower()};\n"
                "      if (isOffline) return;\n"
                "      var timeout = setTimeout(function() {\n"
                "        var status = document.getElementById('loader-status');\n"
                "        if (status) status.innerText = 'Airgap / offline fallback: static SVG.';\n"
                "      }, 2000);\n"
                "\n"
                "      function loadScript(src, cb) {\n"
                "        var s = document.createElement('script');\n"
                "        s.src = src;\n"
                "        s.onload = cb;\n"
                "        s.onerror = function() {\n"
                "          clearTimeout(timeout);\n"
                "          var status = document.getElementById('loader-status');\n"
                "          if (status) {\n"
                "            status.innerText = 'Airgap / offline fallback: static SVG.';\n"
                "          }\n"
                "        };\n"
                "        document.head.appendChild(s);\n"
                "      }\n"
                "\n"
                "      loadScript('https://cdn.jsdelivr.net/npm/vega@5', function() {\n"
                "        loadScript('https://cdn.jsdelivr.net/npm/vega-lite@5', function() {\n"
                "          loadScript('https://cdn.jsdelivr.net/npm/vega-embed@6', function() {\n"
                "            clearTimeout(timeout);\n"
                f"            var spec = {spec_json};\n"
                "            vegaEmbed('#vega-vis', spec, {actions: false}).then(function() {\n"
                "              var fb = document.getElementById('fallback-svg');\n"
                "              var vv = document.getElementById('vega-vis');\n"
                "              if (fb) fb.style.display = 'none';\n"
                "              if (vv) vv.style.display = 'block';\n"
                "              var status = document.getElementById('loader-status');\n"
                "              if (status) status.innerText = 'Interactive inspection active.';\n"
                "            }).catch(function(err) {\n"
                "              console.warn('Vega-Embed failed; static snapshot kept.', err);\n"
                "            });\n"
                "          });\n"
                "        });\n"
                "      });\n"
                "    })();\n"
                "  </script>\n"
                "</body>\n"
                "</html>\n"
            )

        raise ValueError(f"Unsupported format '{fmt}'. Valid formats: 'png', 'svg', 'html'.")

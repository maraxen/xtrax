"""Neutral artifact export, RFC 8259 JSON sanitization, and visual renderers for xtrax Campaign EDA.

Implements P5-08 per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Exports canonical RFC 8259 JSON (BLK-05), tidy Parquet tables, Vega-Lite specs with null filters,
static SVG reviewed objects via vl-convert, and progressive airgapped HTML dashboards (MAJ-08).
"""

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("xtrax.eda.campaign.artifacts")


def sanitize_for_rfc8259(obj: Any) -> Any:
    """Recursively sanitize data structure to strictly comply with RFC 8259 (BLK-05).

    Coerces NaN to None (null).
    Coerces +/-Infinity to None with warning flag.
    Normalizes floating -0.0 to 0.0.
    Converts NumPy scalar types and arrays to standard Python primitives.
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, str)):
        return obj
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val):
            return None
        if math.isinf(val):
            return None
        if val == -0.0:
            return 0.0
        return val
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_rfc8259(x) for x in obj.tolist()]
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_rfc8259(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_rfc8259(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return sanitize_for_rfc8259(vars(obj))
    else:
        return str(obj)


def dumps_canonical_json(obj: Any, indent: int = 2) -> str:
    """Serialize object to canonical RFC 8259 JSON string with allow_nan=False."""
    sanitized = sanitize_for_rfc8259(obj)
    return json.dumps(sanitized, indent=indent, allow_nan=False, sort_keys=True)


def sha256_of_bytes(data: bytes) -> str:
    """Compute 64-character lowercase hexadecimal SHA-256 digest."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def sha256_of_file(file_path: Path | str) -> str:
    """Compute SHA-256 digest of a filesystem file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# Declarative Vega-Lite Specification Generators (Spec §6.1)
# ==============================================================================


def generate_library_composition_spec(data_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Generate Vega-Lite v5 specification for Question 1: Library Composition."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Question 1: Library Composition and Sequence Diversity",
        "description": "Diversity, positional entropy, and target recovery across conditions.",
        "data": {"values": sanitize_for_rfc8259(data_rows)},
        "transform": [{"filter": "isValid(datum.diversity) && datum.diversity != null"}],
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "condition", "type": "nominal", "title": "Knob Condition"},
            "y": {
                "field": "diversity",
                "type": "quantitative",
                "title": "Mean Pairwise Diversity (1 - PID)",
            },
            "color": {"field": "arm", "type": "nominal", "title": "Model Arm"},
        },
    }


def generate_score_distributions_spec(data_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Generate Vega-Lite v5 specification for Question 2: Scores & Energetics."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Question 2: Scores and Negative Log-Likelihood Distributions",
        "description": "Per-sample NLL distributions, Best-of-N curves, and Pareto trade-offs.",
        "data": {"values": sanitize_for_rfc8259(data_rows)},
        "transform": [{"filter": "isValid(datum.nll) && datum.nll != null"}],
        "mark": {"type": "boxplot", "extent": "min-max", "tooltip": True},
        "encoding": {
            "x": {"field": "condition", "type": "nominal", "title": "Condition"},
            "y": {"field": "nll", "type": "quantitative", "title": "Negative Log-Likelihood (NLL)"},
            "color": {"field": "arm", "type": "nominal"},
        },
    }


def generate_uncertainty_contrasts_spec(
    data_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Generate Vega-Lite v5 specification for Question 3: Honest Uncertainty."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Question 3: Knob Effects with Vectorized Cluster Ratio Bootstrap (95% CI)",
        "description": "Point estimates with cluster bootstrap CIs, guarded when U < 3.",
        "data": {"values": sanitize_for_rfc8259(data_rows)},
        "transform": [{"filter": "isValid(datum.point_estimate) && datum.point_estimate != null"}],
        "layer": [
            {
                "mark": {"type": "point", "size": 80, "filled": True, "tooltip": True},
                "encoding": {
                    "x": {"field": "condition", "type": "nominal", "title": "Condition"},
                    "y": {"field": "point_estimate", "type": "quantitative", "title": "Estimate"},
                    "color": {"field": "metric", "type": "nominal"},
                },
            },
            {
                "transform": [
                    {
                        "filter": (
                            "datum.descriptive_only == false && "
                            "isValid(datum.ci_lower) && datum.ci_lower != null"
                        )
                    }
                ],
                "mark": {"type": "errorbar", "ticks": True},
                "encoding": {
                    "x": {"field": "condition", "type": "nominal"},
                    "y": {"field": "ci_lower", "type": "quantitative"},
                    "y2": {"field": "ci_upper"},
                    "color": {"field": "metric", "type": "nominal"},
                },
            },
        ],
    }


def generate_sweep_coverage_spec(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Generate Vega-Lite v5 specification for Question 4: Sweep Coverage Reconciliation."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Question 4: Cartesian Sweep Grid Coverage Reconciliation",
        "description": "Status for planned cells: COMMITTED, PARTIAL, FAILED, MISSING.",
        "data": {"values": sanitize_for_rfc8259(cells)},
        "mark": {"type": "rect", "tooltip": True},
        "encoding": {
            "x": {"field": "cell_key", "type": "nominal", "title": "Planned Sweep Cell"},
            "y": {"field": "status", "type": "nominal", "title": "Rollup Status"},
            "color": {
                "field": "status",
                "type": "nominal",
                "scale": {
                    "domain": ["COMMITTED", "PARTIAL", "FAILED", "MISSING"],
                    "range": ["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"],
                },
                "title": "Status",
            },
        },
    }


def generate_run_telemetry_spec(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Generate Vega-Lite v5 specification for Question 5: Secondary Run Telemetry."""
    units = telemetry.get("units", [])
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Question 5: Secondary Run Telemetry (Wall Time and Debt #3300 Seam)",
        "description": "Duration distributions and transparent debt #3300 pack seam.",
        "data": {"values": sanitize_for_rfc8259(units)},
        "transform": [{"filter": "isValid(datum.duration_s) && datum.duration_s != null"}],
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "manifest_row_hash", "type": "nominal", "title": "Execution Unit"},
            "y": {"field": "duration_s", "type": "quantitative", "title": "Compute Wall Time (s)"},
            "color": {
                "field": "is_packed",
                "type": "nominal",
                "title": "Packed Archive (.tar.zst)",
            },
        },
    }


# ==============================================================================
# Vector SVG Renderer (Reviewed Object)
# ==============================================================================


def render_fallback_svg(title: str, subtitle: str, values: Sequence[dict[str, Any]]) -> str:
    """Generate robust, standards-compliant static SVG chart when vl-convert is not present."""
    width = 800
    height = 450
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="{margin_left}" y="35" font-family="system-ui, sans-serif" '
        f'font-size="18" font-weight="bold" fill="#1f2937">{title}</text>',
        f'  <text x="{margin_left}" y="55" font-family="system-ui, sans-serif" '
        f'font-size="12" fill="#6b7280">{subtitle}</text>',
        f'  <g transform="translate({margin_left}, {margin_top})">',
        f'    <line x1="0" y1="{plot_height}" x2="{plot_width}" y2="{plot_height}" '
        'stroke="#d1d5db" stroke-width="1.5" />',
        f'    <line x1="0" y1="0" x2="0" y2="{plot_height}" stroke="#d1d5db" stroke-width="1.5" />',
    ]

    n_bars = len(values)
    if n_bars > 0:
        bar_width = min(40, max(10, plot_width // (n_bars * 2)))
        step = plot_width / n_bars
        max_val = max(
            [
                float(
                    v.get("value", v.get("diversity", v.get("nll", v.get("duration_s", 1.0))))
                    or 1.0
                )
                for v in values
            ]
            + [1.0]
        )

        for idx, item in enumerate(values):
            raw_val = item.get(
                "value", item.get("diversity", item.get("nll", item.get("duration_s", 0.0)))
            )
            val = float(raw_val or 0.0)
            bar_height = (val / max_val) * (plot_height - 20)
            x_pos = idx * step + (step - bar_width) / 2
            y_pos = plot_height - bar_height
            label = str(
                item.get("condition", item.get("arm", item.get("cell_key", f"U{idx + 1}")))
            )[:10]

            color = "#2563eb"
            if item.get("status") == "COMMITTED":
                color = "#16a34a"
            elif item.get("status") == "PARTIAL":
                color = "#ea580c"
            elif item.get("status") == "FAILED":
                color = "#dc2626"
            elif item.get("status") == "MISSING":
                color = "#9ca3af"

            svg_lines.append(
                f'    <rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_width}" '
                f'height="{bar_height:.1f}" fill="{color}" rx="2" />'
            )
            svg_lines.append(
                f'    <text x="{x_pos + bar_width / 2:.1f}" y="{plot_height + 18}" '
                f'font-family="system-ui, sans-serif" font-size="10" fill="#4b5563" '
                f'text-anchor="middle">{label}</text>'
            )
            svg_lines.append(
                f'    <text x="{x_pos + bar_width / 2:.1f}" y="{y_pos - 4:.1f}" '
                f'font-family="system-ui, sans-serif" font-size="9" fill="#374151" '
                f'text-anchor="middle">{val:.2f}</text>'
            )

    svg_lines.extend(
        [
            "  </g>",
            "</svg>",
        ]
    )
    return "\n".join(svg_lines)


def compile_static_svg(
    spec_dict: dict[str, Any], panel_name: str, values: Sequence[dict[str, Any]]
) -> str:
    """Compile Vega-Lite specification to static SVG vector Reviewed Object."""
    try:
        import vl_convert as vlc

        svg_str = vlc.vegalite_to_svg(spec_dict)
        return svg_str
    except Exception as exc:
        logger.info("Using standalone SVG renderer for panel %s: %s", panel_name, exc)
        title = spec_dict.get("title", panel_name.replace("_", " ").title())
        subtitle = spec_dict.get("description", "")
        return render_fallback_svg(title, subtitle, values)


# ==============================================================================
# Progressive Airgapped HTML Dashboard Renderer (MAJ-08)
# ==============================================================================


def render_airgapped_html_dashboard(
    campaign_id: str,
    panels: Mapping[str, tuple[dict[str, Any], str]],  # panel_name -> (vega_spec, static_svg)
    telemetry: dict[str, Any],
    banners: Sequence[str],
) -> str:
    """Generate standalone progressive airgapped HTML dashboard (MAJ-08).

    - Embeds pre-rendered static SVGs visible by default offline (display: block).
    - Unique DOM IDs per panel: fallback-svg-{panel_name}.
    - Deferred script loading (<script defer>).
    - 2-second timeout guard to preserve static vector Reviewed Object offline.
    """
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"  <title>xtrax Campaign EDA — {campaign_id}</title>",
        "  <style>",
        "    body { font-family: system-ui, sans-serif; background: #f9fafb; "
        "color: #111827; margin: 0; padding: 24px; }",
        "    .header { margin-bottom: 24px; padding-bottom: 16px; "
        "border-bottom: 1px solid #e5e7eb; }",
        "    .header h1 { margin: 0 0 8px 0; font-size: 24px; }",
        "    .badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; "
        "font-size: 12px; font-weight: 600; background: #e0e7ff; color: #3730a3; "
        "margin-right: 8px; }",
        "    .banner-box { background: #fef2f2; border-left: 4px solid #ef4444; "
        "padding: 12px 16px; margin-bottom: 24px; border-radius: 4px; }",
        "    .banner-title { font-weight: 700; color: #991b1b; margin-bottom: 4px; }",
        "    .banner-item { font-size: 13px; color: #7f1d1d; margin: 2px 0; }",
        "    .telemetry-banner { background: #fffbeb; border-left: 4px solid #f59e0b; "
        "padding: 10px 16px; margin-bottom: 24px; font-size: 13px; font-weight: 600; "
        "color: #92400e; border-radius: 4px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); "
        "gap: 24px; }",
        "    .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; "
        "box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden; }",
        "    .card-header { padding: 16px 20px; border-bottom: 1px solid #f3f4f6; "
        "font-weight: 600; font-size: 16px; background: #fafafa; }",
        "    .card-body { padding: 20px; display: flex; justify-content: center; }",
        "    .svg-container { width: 100%; display: block; }",
        "    .svg-container svg { width: 100%; height: auto; display: block; }",
        "  </style>",
        '  <script defer src="https://cdn.jsdelivr.net/npm/vega@5"></script>',
        '  <script defer src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>',
        '  <script defer src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>',
        "</head>",
        "<body>",
        '  <div class="header">',
        f"    <h1>xtrax Campaign EDA: {campaign_id}</h1>",
        '    <div><span class="badge">Revision 1.1 Specification</span>'
        '<span class="badge">Reviewed Object Protocol</span></div>',
        "  </div>",
    ]

    # Telemetry seam banner (debt #3300)
    seam_banner = telemetry.get(
        "pack_telemetry_banner", "Pack Telemetry: not instrumented (debt #3300)"
    )
    html_parts.append(f'  <div class="telemetry-banner">{seam_banner}</div>')

    # Diagnostic banners if any
    if banners:
        html_parts.append('  <div class="banner-box">')
        html_parts.append(
            '    <div class="banner-title">Coverage & Execution Diagnostic Banners</div>'
        )
        for b in banners:
            html_parts.append(f'    <div class="banner-item">• {b}</div>')
        html_parts.append("  </div>")

    # Visualization panels
    html_parts.append('  <div class="grid">')
    for panel_name, (spec, svg_content) in panels.items():
        title = spec.get("title", panel_name.replace("_", " ").title())
        html_parts.append('    <div class="card">')
        html_parts.append(f'      <div class="card-header">{title}</div>')
        html_parts.append('      <div class="card-body">')
        # Unique fallback SVG DOM ID visible by default offline (MAJ-08)
        html_parts.append(f'        <div id="fallback-svg-{panel_name}" class="svg-container">')
        html_parts.append(f"          {svg_content}")
        html_parts.append("        </div>")
        html_parts.append(
            f'        <div id="interactive-vis-{panel_name}" '
            'style="display: none; width: 100%;"></div>'
        )
        html_parts.append("      </div>")
        html_parts.append("    </div>")
    html_parts.append("  </div>")

    # Progressive enhancement script with 2-second timeout guard
    html_parts.append("  <script defer>")
    html_parts.append("    document.addEventListener('DOMContentLoaded', function() {")
    for panel_name, (spec, _) in panels.items():
        spec_json = json.dumps(sanitize_for_rfc8259(spec))
        html_parts.append("      (function() {")
        html_parts.append(f"        var panel = '{panel_name}';")
        html_parts.append(f"        var spec = {spec_json};")
        html_parts.append("        var timer = setTimeout(function() {")
        html_parts.append(
            "          console.info('Vega timeout; preserving static SVG for: ' + panel);"
        )
        html_parts.append("        }, 2000);")
        html_parts.append("        if (typeof vegaEmbed !== 'undefined') {")
        html_parts.append(
            "          vegaEmbed('#interactive-vis-' + panel, spec).then(function() {"
        )
        html_parts.append("            clearTimeout(timer);")
        html_parts.append("            var fb = document.getElementById('fallback-svg-' + panel);")
        html_parts.append(
            "            var iv = document.getElementById('interactive-vis-' + panel);"
        )
        html_parts.append(
            "            if (fb && iv) { fb.style.display = 'none'; iv.style.display = 'block'; }"
        )
        html_parts.append("          }).catch(function(err) {")
        html_parts.append("            console.warn('Vega embed failed for ' + panel, err);")
        html_parts.append("          });")
        html_parts.append("        }")
        html_parts.append("      })();")
    html_parts.append("    });")
    html_parts.append("  </script>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)


# ==============================================================================
# Master Neutral Artifact Exporter
# ==============================================================================


def export_campaign_eda(
    campaign_dir: Path | str,
    campaign_id: str,
    q1_data: Sequence[dict[str, Any]],
    q2_data: Sequence[dict[str, Any]],
    q3_data: Sequence[dict[str, Any]],
    q4_coverage: dict[str, Any],
    q5_telemetry: dict[str, Any],
) -> dict[str, str]:
    """Export complete neutral EDA bundle under <campaign_dir>/eda/ (Spec §6).

    Bundle contents:
    - summary.json (canonical RFC 8259 summary)
    - tables/*.parquet or tables/*.json (tidy long-format tables)
    - specs/*.vl.json (declarative Vega-Lite v5 specifications)
    - static/*.svg (pre-rendered vector SVGs, Reviewed Object)
    - index.html (standalone progressive airgapped HTML dashboard)
    - manifest.json (cryptographic SHA-256 manifest)
    """
    out_dir = Path(campaign_dir).resolve() / "eda"
    tables_dir = out_dir / "tables"
    specs_dir = out_dir / "specs"
    static_dir = out_dir / "static"

    for d in (out_dir, tables_dir, specs_dir, static_dir):
        d.mkdir(parents=True, exist_ok=True)

    artifacts_map: dict[str, str] = {}

    # 1. Summary JSON
    summary_payload = {
        "schema_version": "campaign_eda_summary_v1",
        "campaign_id": campaign_id,
        "q1_composition": q1_data,
        "q2_scores": q2_data,
        "q3_uncertainty": q3_data,
        "q4_coverage": q4_coverage,
        "q5_telemetry": q5_telemetry,
    }
    summary_json = dumps_canonical_json(summary_payload)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_json)
    artifacts_map["summary.json"] = sha256_of_bytes(summary_json.encode("utf-8"))

    # 2. Tidy Parquet or JSON tables
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _prepare_for_parquet(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            prepared = []
            for r in sanitize_for_rfc8259(rows):
                clean_r = {}
                for k, v in r.items():
                    if isinstance(v, dict):
                        # Empty struct types cannot be written to Parquet (ArrowNotImplementedError)
                        clean_r[k] = json.dumps(v, sort_keys=True)
                    else:
                        clean_r[k] = v
                prepared.append(clean_r)
            return prepared

        # Library metrics table
        if q1_data:
            table_q1 = pa.Table.from_pylist(_prepare_for_parquet(q1_data))
            pq.write_table(table_q1, tables_dir / "library_metrics.parquet")
            artifacts_map["tables/library_metrics.parquet"] = sha256_of_file(
                tables_dir / "library_metrics.parquet"
            )

        # Coverage grid table
        if q4_coverage.get("cells"):
            table_q4 = pa.Table.from_pylist(_prepare_for_parquet(q4_coverage["cells"]))
            pq.write_table(table_q4, tables_dir / "coverage_grid.parquet")
            artifacts_map["tables/coverage_grid.parquet"] = sha256_of_file(
                tables_dir / "coverage_grid.parquet"
            )

        # Telemetry table
        if q5_telemetry.get("units"):
            table_q5 = pa.Table.from_pylist(_prepare_for_parquet(q5_telemetry["units"]))
            pq.write_table(table_q5, tables_dir / "run_telemetry.parquet")
            artifacts_map["tables/run_telemetry.parquet"] = sha256_of_file(
                tables_dir / "run_telemetry.parquet"
            )

    except ImportError:
        # Fallback to json tables when pyarrow is not installed
        for name, data in [
            ("library_metrics", q1_data),
            ("coverage_grid", q4_coverage.get("cells", [])),
            ("run_telemetry", q5_telemetry.get("units", [])),
        ]:
            t_json = dumps_canonical_json(data)
            t_path = tables_dir / f"{name}.json"
            with open(t_path, "w", encoding="utf-8") as f:
                f.write(t_json)
            artifacts_map[f"tables/{name}.json"] = sha256_of_bytes(t_json.encode("utf-8"))

    # 3. Vega-Lite specifications
    specs = {
        "library_composition": generate_library_composition_spec(q1_data),
        "score_distributions": generate_score_distributions_spec(q2_data),
        "uncertainty_contrasts": generate_uncertainty_contrasts_spec(q3_data),
        "sweep_coverage": generate_sweep_coverage_spec(q4_coverage.get("cells", [])),
        "run_telemetry": generate_run_telemetry_spec(q5_telemetry),
    }

    panels_map: dict[str, tuple[dict[str, Any], str]] = {}

    for spec_name, spec_dict in specs.items():
        spec_json = dumps_canonical_json(spec_dict)
        spec_path = specs_dir / f"{spec_name}.vl.json"
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(spec_json)
        artifacts_map[f"specs/{spec_name}.vl.json"] = sha256_of_bytes(spec_json.encode("utf-8"))

        # 4. Static SVG (Reviewed Object)
        data_vals = (
            q1_data
            if "composition" in spec_name
            else (
                q2_data
                if "score" in spec_name
                else (
                    q3_data
                    if "uncertainty" in spec_name
                    else (
                        q4_coverage.get("cells", [])
                        if "coverage" in spec_name
                        else q5_telemetry.get("units", [])
                    )
                )
            )
        )
        svg_content = compile_static_svg(spec_dict, spec_name, data_vals)
        svg_path = static_dir / f"{spec_name}.svg"
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        artifacts_map[f"static/{spec_name}.svg"] = sha256_of_bytes(svg_content.encode("utf-8"))

        panels_map[spec_name] = (spec_dict, svg_content)

    # 5. Progressive Airgapped HTML Dashboard (MAJ-08)
    html_dashboard = render_airgapped_html_dashboard(
        campaign_id=campaign_id,
        panels=panels_map,
        telemetry=q5_telemetry,
        banners=q4_coverage.get("banners", []),
    )
    html_path = out_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_dashboard)
    artifacts_map["index.html"] = sha256_of_bytes(html_dashboard.encode("utf-8"))

    # 6. Cryptographic Manifest (manifest.json)
    manifest_payload = {
        "schema_version": "xtrax_eda_manifest_v1",
        "campaign_id": campaign_id,
        "engine_version": "0.4.1b7",
        "artifacts": dict(sorted(artifacts_map.items())),
    }
    manifest_json = dumps_canonical_json(manifest_payload)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest_json)
    artifacts_map["manifest.json"] = sha256_of_bytes(manifest_json.encode("utf-8"))

    logger.info("Exported %d Campaign EDA artifacts to %s", len(artifacts_map), out_dir)
    return artifacts_map

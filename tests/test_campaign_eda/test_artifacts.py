"""Unit tests for Campaign EDA artifact export and visual renderers."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from xtrax.eda.campaign import runner, schema
from xtrax.eda.campaign.artifacts import (
    compile_static_svg,
    dumps_canonical_json,
    export_campaign_eda,
    generate_library_composition_spec,
    render_airgapped_html_dashboard,
    render_fallback_svg,
    sanitize_for_rfc8259,
)
from xtrax.eda.campaign.telemetry import parse_reuse_log


def test_rfc8259_sanitization():
    """Verify recursive RFC 8259 sanitization pass (BLK-05)."""
    raw_dict = {
        "nan_val": float("nan"),
        "np_nan": np.nan,
        "inf_val": float("inf"),
        "neg_inf": float("-inf"),
        "neg_zero": -0.0,
        "valid_float": 1.25,
        "np_int": np.int64(42),
        "np_arr": np.array([1.0, np.nan, 3.0]),
        "nested": {"deep_nan": float("nan")},
    }

    sanitized = sanitize_for_rfc8259(raw_dict)

    # Must convert NaNs and Infs to None
    assert sanitized["nan_val"] is None
    assert sanitized["np_nan"] is None
    assert sanitized["inf_val"] is None
    assert sanitized["neg_inf"] is None
    assert sanitized["neg_zero"] == 0.0
    assert sanitized["valid_float"] == 1.25
    assert sanitized["np_int"] == 42
    assert sanitized["np_arr"] == [1.0, None, 3.0]
    assert sanitized["nested"]["deep_nan"] is None

    # Serialization with allow_nan=False must succeed without crashing
    serialized_str = dumps_canonical_json(raw_dict)
    parsed = json.loads(serialized_str)
    assert parsed["nan_val"] is None
    assert parsed["inf_val"] is None


def test_vega_spec_null_filters():
    """Verify explicit null filters in Vega-Lite transforms (BLK-05)."""
    data = [{"diversity": 0.45, "condition": "A", "arm": "mpnn"}]
    spec = generate_library_composition_spec(data)

    # Check transform contains null filter
    assert "transform" in spec
    filters = [t.get("filter", "") for t in spec["transform"]]
    assert any("!= null" in f for f in filters)


def test_airgapped_html_dashboard_dom_ids_and_script_defer():
    """Verify progressive airgapped HTML dashboard DOM IDs and deferred scripts (MAJ-08)."""
    panels = {
        "library_composition": (
            {"title": "Library Composition"},
            '<svg id="svg1"><rect/></svg>',
        ),
    }
    telemetry = {"pack_telemetry_banner": "Pack Telemetry: not instrumented (debt #3300)"}

    html = render_airgapped_html_dashboard(
        campaign_id="test_camp",
        panels=panels,
        telemetry=telemetry,
        banners=["Test Warning Banner"],
    )

    # Must contain unique DOM ID for panel
    assert 'id="fallback-svg-library_composition"' in html
    assert 'id="interactive-vis-library_composition"' in html

    # Must contain deferred script loading
    assert "<script defer" in html

    # Must contain 2-second timeout guard
    assert "2000" in html

    # Must contain debt #3300 seam banner
    assert "Pack Telemetry: not instrumented (debt #3300)" in html


def test_render_fallback_svg_and_compile_fallback():
    """Verify standalone SVG renderer across status colors and vl_convert fallback."""
    values = [
        {"condition": "A", "value": 1.2, "status": "COMMITTED"},
        {"arm": "B", "diversity": 0.5, "status": "PARTIAL"},
        {"cell_key": "C", "nll": 2.1, "status": "FAILED"},
        {"duration_s": 0.8, "status": "MISSING"},
        {"value": 0.3, "status": "OTHER"},
    ]
    svg = render_fallback_svg("Title", "Subtitle", values)
    assert "<svg" in svg
    assert "#16a34a" in svg
    assert "#ea580c" in svg
    assert "#dc2626" in svg
    assert "#9ca3af" in svg
    assert "#2563eb" in svg

    with patch.dict("sys.modules", {"vl_convert": None}):
        fallback = compile_static_svg({"title": "T", "description": "D"}, "panel", values)
        assert "<svg" in fallback


def test_sanitize_custom_object_and_reexports(tmp_path: Path):
    """Verify sanitize_for_rfc8259 on custom objects/scalars and re-export modules."""

    class Custom:
        def __init__(self) -> None:
            self.x = float("nan")

    assert sanitize_for_rfc8259(Custom()) == {"x": None}
    assert sanitize_for_rfc8259(complex(1, 2)) == "(1+2j)"
    assert callable(runner.run_campaign_eda)
    assert "BootstrapEstimate" in schema.__all__

    # parse_reuse_log with valid, invalid, and missing JSONL
    assert parse_reuse_log(None) == []
    assert parse_reuse_log(tmp_path / "missing.jsonl") == []
    log_file = tmp_path / "reuse.jsonl"
    log_file.write_text('{"unit": "u1"}\nnot-json\n\n{"unit": "u2"}\n', encoding="utf-8")
    assert parse_reuse_log(log_file) == [{"unit": "u1"}, {"unit": "u2"}]
    assert parse_reuse_log(tmp_path) == []


def test_export_campaign_eda_json_fallback(tmp_path: Path):
    """Verify JSON table fallback when pyarrow is unavailable."""
    with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
        artifacts = export_campaign_eda(
            campaign_dir=tmp_path / "eda_out",
            campaign_id="camp_fallback",
            q1_data=[{"condition": "A", "diversity": 0.5}],
            q2_data=[],
            q3_data=[],
            q4_coverage={"cells": [{"cell_key": "c1", "status": "COMMITTED"}], "banners": []},
            q5_telemetry={"units": [{"manifest_row_hash": "h1", "duration_s": 1.0}]},
        )
    assert "tables/library_metrics.json" in artifacts
    assert "tables/coverage_grid.json" in artifacts
    assert "tables/run_telemetry.json" in artifacts

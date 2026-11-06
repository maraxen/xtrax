"""Unit tests for Campaign EDA artifact export and visual renderers."""

import json

import numpy as np

from xtrax.eda.campaign.artifacts import (
    dumps_canonical_json,
    generate_library_composition_spec,
    render_airgapped_html_dashboard,
    sanitize_for_rfc8259,
)


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

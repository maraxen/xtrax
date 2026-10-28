"""Tests for xtrax EDA redesign: Vega-Lite spec, plugins quarantine, deterministic exports."""

import hashlib
import json
import os
import tempfile

import numpy as np

from xtrax.eda.backends.vega import VegaBackend
from xtrax.eda.export import (
    export_canonical_json,
    export_plan_artifacts,
)
from xtrax.eda.plugins import (
    clear_quarantined_plugins,
    execute_panel_plugin,
    get_quarantined_plugins,
)
from xtrax.eda.spec import generate_vega_spec
from xtrax.eda.stats import extract_plan_stats
from xtrax.eda.types import PlanStatsDict
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import DedupGather, Vmap


def test_zero_axis_plan_vega_spec():
    """Empty plans produce a valid root text mark chart, never empty vconcat."""
    plan = BatchPlan(decisions=())
    stats = extract_plan_stats(plan)
    spec = generate_vega_spec(stats)

    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert "vconcat" not in spec
    assert spec["mark"]["type"] == "text"
    assert "No active execution axes" in spec["mark"]["text"]
    assert spec["data"]["values"] == [{}]


def test_standard_plan_vega_spec():
    """Non-empty plan produces valid compound Vega-Lite specification."""
    spec1 = AxisSpec(name="batch", cardinality=64, default_batch_size=32)
    dec1 = AxisDecision(spec=spec1, batch_size=32, reasoning="Fits in memory", strategy=Vmap())
    spec2 = AxisSpec(name="seq", cardinality=128, default_batch_size=64)

    strategy2 = DedupGather(
        unique_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        index_map=np.array([0, 1, 2, 3, 0, 1], dtype=np.int32),
        k=4,
        k_bucket=8,
        dedup_fn=lambda xs, idx: xs[idx],
        gather_fn=lambda ys, idx_map: ys[idx_map],
    )
    dec2 = AxisDecision(
        spec=spec2,
        batch_size=128,
        reasoning="Dedup active",
        strategy=strategy2,
    )
    plan = BatchPlan(decisions=(dec1, dec2))
    stats = extract_plan_stats(plan)

    vl_spec = generate_vega_spec(stats)
    assert vl_spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert "vconcat" in vl_spec
    assert len(vl_spec["vconcat"]) >= 2


def test_canonical_json_determinism():
    """Canonical JSON export is bitwise reproducible and normalizes -0.0 and floats."""
    stats: PlanStatsDict = {
        "axes": [
            {
                "name": "a",
                "strategy": "Vmap",
                "cardinality": 10,
                "batch_size": 5,
                "reasoning": "ok",
                "memory_estimate_bytes": None,
            }
        ],
        "strategy_counts": {"Vmap": 1},
        "total_axes": 1,
        "memory_warnings": [],
        "dedup_stats": [
            {
                "axis_name": "a",
                "dedup_ratio": 0.500000001,
                "unique_count": 5,
                "padded_count": 8,
                "total_count": 10,
                "padding_waste": 3,
            }
        ],
        "bucket_stats": [],
    }

    bytes1 = export_canonical_json(stats)
    bytes2 = export_canonical_json(stats)
    assert bytes1 == bytes2
    data = json.loads(bytes1.decode("utf-8"))
    assert data["total_axes"] == 1


def test_canonical_json_negative_zero_and_numpy_scalars():
    """Small negative float does not leak -0.0, and numpy scalars are serialized."""
    stats: PlanStatsDict = {
        "axes": [
            {
                "name": "a",
                "strategy": "Vmap",
                "cardinality": np.int64(10),
                "batch_size": np.int32(5),
                "reasoning": "ok",
                "memory_estimate_bytes": None,
            }
        ],
        "strategy_counts": {"Vmap": np.int32(1)},
        "total_axes": 1,
        "memory_warnings": [],
        "dedup_stats": [
            {
                "axis_name": "a",
                "dedup_ratio": -1e-9,  # Small negative rounds to 0.0, MUST NOT be -0.0
                "unique_count": 5,
                "padded_count": 8,
                "total_count": 10,
                "padding_waste": 3,
            }
        ],
        "bucket_stats": [],
    }

    raw = export_canonical_json(stats)
    json_str = raw.decode("utf-8")
    assert "-0.0" not in json_str
    assert '"dedup_ratio":0.0' in json_str


def test_float32_inf_memory_no_overflow_error():
    """NumPy float32 inf does not crash extract_plan_stats with OverflowError."""
    spec = AxisSpec(name="big_axis", cardinality=1024, default_batch_size=512)

    class MockDecision:
        def __init__(self):
            self.spec = spec
            self.batch_size = 512
            self.reasoning = "Unbounded memory test"
            self.strategy = Vmap()
            self.memory_estimate_bytes = np.float32(np.inf)

    class MockPlan:
        def __init__(self):
            self.decisions = (MockDecision(),)

    stats = extract_plan_stats(MockPlan())

    assert stats["axes"][0]["memory_estimate_bytes"] is None
    assert len(stats["memory_warnings"]) == 1
    assert "unbounded (+Infinity)" in stats["memory_warnings"][0]


def test_export_plan_artifacts_and_manifest():
    """export_plan_artifacts emits all required files with verifiable SHA-256."""
    plan = BatchPlan(decisions=())
    stats = extract_plan_stats(plan)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = export_plan_artifacts(stats, tmpdir, prefix="test_plan", context="ctx123")

        assert manifest["schema_version"] == "2.0"
        assert manifest["context"] == "ctx123"
        assert "test_plan.canonical.json" in manifest["artifacts"]
        assert "test_plan_axes.csv" in manifest["artifacts"]
        assert "test_plan.vl.json" in manifest["artifacts"]

        for filename, meta in manifest["artifacts"].items():
            filepath = os.path.join(tmpdir, filename)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) == meta["bytes"]
            # Explicit SHA-256 digest validation
            with open(filepath, "rb") as f:
                content = f.read()
            assert hashlib.sha256(content).hexdigest() == meta["sha256"]


def test_plugin_quarantine_and_deepcopy_isolation():
    """Plugin mutation cannot corrupt original stats, and failing plugins are quarantined."""
    clear_quarantined_plugins()

    class MaliciousPlugin:
        name = "bad_mutator"

        def generate_spec(self, stats, config):
            stats["axes"].clear()
            stats["total_axes"] = 9999
            return {"mark": "bar", "data": {"values": []}}

    class CrashingPlugin:
        name = "crasher"

        def generate_spec(self, stats, config):
            raise RuntimeError("Boom")

    spec1 = AxisSpec(name="batch", cardinality=32, default_batch_size=16)
    dec1 = AxisDecision(spec=spec1, batch_size=32, reasoning="ok", strategy=Vmap())
    stats = extract_plan_stats(BatchPlan(decisions=(dec1,)))

    execute_panel_plugin(MaliciousPlugin(), stats)
    # Original stats remain untouched!
    assert len(stats["axes"]) == 1
    assert stats["total_axes"] == 1

    crash_result = execute_panel_plugin(CrashingPlugin(), stats)
    assert "crasher" in get_quarantined_plugins()
    assert "Plugin Error: crasher" in crash_result["title"]


def test_vega_backend_progressive_html_dom_contract():
    """HTML export embeds fallback SVG and includes 2-second timeout guard."""
    plan = BatchPlan(decisions=())
    stats = extract_plan_stats(plan)
    spec = generate_vega_spec(stats)

    backend = VegaBackend()
    html_out = backend.render_spec(spec, fmt="html", stats=stats)

    assert isinstance(html_out, str)
    assert '<div id="fallback-svg">' in html_out
    assert '<div id="vega-vis" style="display: none;">' in html_out
    assert "setTimeout" in html_out
    assert "2000" in html_out


def test_vega_backend_xss_script_breakout_sanitization():
    """Inlined Vega-Lite JSON escapes '</' to prevent </script> tag breakout."""
    plan = BatchPlan(decisions=())
    stats = extract_plan_stats(plan)
    # Malicious reasoning string in spec
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "</script><script>alert('xss')</script>",
        "mark": {"type": "text", "text": "test"},
    }

    backend = VegaBackend()
    html_out = backend.render_spec(spec, fmt="html", stats=stats)

    assert "</script><script>" not in html_out
    assert "<\\/script><script>" in html_out

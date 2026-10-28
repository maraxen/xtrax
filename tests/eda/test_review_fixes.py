"""Regression tests for the PR 251 review findings on xtrax.eda.

Each test pins one fixed behavior: render routing, bundle_js rejection, aggregate
overflow, manifest reproducibility, canonical JSON bools, CSV zero estimates and
the Vega fallback's error handling.
"""

import csv
import io
import json
import sys
import types

import numpy as np
import pytest

from xtrax.eda import export_canonical_json, export_plan_artifacts, render
from xtrax.eda.backends.vega import VegaBackend
from xtrax.eda.export import export_axes_csv
from xtrax.eda.spec import generate_vega_spec
from xtrax.eda.stats import extract_plan_stats
from xtrax.eda.types import PlanStatsDict
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import ChunkedMap, Vmap


@pytest.fixture
def two_axis_plan() -> BatchPlan:
    spec1 = AxisSpec(name="batch", cardinality=64, default_batch_size=128)
    spec2 = AxisSpec(name="seq", cardinality=512, default_batch_size=64)
    return BatchPlan(
        decisions=(
            AxisDecision(spec=spec1, batch_size=128, reasoning="Batch axis", strategy=Vmap()),
            AxisDecision(
                spec=spec2,
                batch_size=64,
                reasoning="Sequence axis",
                strategy=ChunkedMap(batch_size=64),
            ),
        )
    )


def _axis(name: str, cardinality: int, memory: float | None = None) -> dict:
    return {
        "name": name,
        "strategy": "Vmap",
        "cardinality": cardinality,
        "batch_size": 8,
        "reasoning": "test",
        "memory_estimate_bytes": memory,
        "varying_inputs": [],
    }


def _stats(axes: list[dict]) -> PlanStatsDict:
    return {
        "axes": axes,
        "strategy_counts": {"Vmap": len(axes)},
        "total_axes": len(axes),
        "memory_warnings": [],
        "dedup_stats": [],
        "bucket_stats": [],
    }


class TestRenderRouting:
    """Finding: render(backend="vega") bypassed viz.render's kwargs handling."""

    def test_vega_honors_stats_transform_and_metadata(self, two_axis_plan, tmp_path):
        def bump_vmap(stats: PlanStatsDict) -> PlanStatsDict:
            stats = dict(stats)
            stats["strategy_counts"] = {"Vmap": 99}
            return stats

        out = tmp_path / "plan.svg"
        result = render(
            two_axis_plan,
            fmt="svg",
            path=out,
            backend="vega",
            metadata=True,
            stats_transform=bump_vmap,
        )

        assert result is None
        assert out.exists()
        sidecar = json.loads(out.with_suffix(".json").read_text())
        assert sidecar["strategy_counts"] == {"Vmap": 99}

    def test_vega_accepts_positional_fmt(self, two_axis_plan):
        result = render(two_axis_plan, "dashboard", "svg", backend="vega")

        assert isinstance(result, bytes)
        assert b"<svg" in result[:500]

    def test_vega_rejects_missing_transform_keys(self, two_axis_plan):
        def drop_keys(stats: PlanStatsDict) -> PlanStatsDict:
            return {"axes": []}  # type: ignore[return-value]

        with pytest.raises(TypeError, match="required keys"):
            render(two_axis_plan, fmt="svg", backend="vega", stats_transform=drop_keys)


class TestBundleJsRejected:
    """Finding: bundle_js was accepted and silently ignored."""

    def test_render_rejects_bundle_js(self, two_axis_plan):
        with pytest.raises(NotImplementedError, match="bundle_js"):
            render(two_axis_plan, fmt="html", backend="vega", bundle_js=True)

    def test_matplotlib_rejects_bundle_js(self, two_axis_plan):
        with pytest.raises(NotImplementedError, match="bundle_js"):
            render(two_axis_plan, fmt="svg", backend="matplotlib", bundle_js=True)

    def test_backend_rejects_bundle_js(self):
        with pytest.raises(NotImplementedError, match="bundle_js"):
            VegaBackend().render_spec({}, fmt="html", stats=_stats([]), options={"bundle_js": True})


class TestAggregateOverflow:
    """Finding: overflow="aggregate" only retitled the chart; nothing was summarized."""

    @pytest.fixture
    def four_axes(self) -> PlanStatsDict:
        return _stats([_axis("a", 100), _axis("b", 50), _axis("c", 10), _axis("d", 1)])

    def test_aggregate_adds_summary_bar_for_dropped_axes(self, four_axes):
        spec = generate_vega_spec(four_axes, panels={"cardinality"}, max_axes=2)

        values = spec["data"]["values"]
        assert [v["name"] for v in values] == ["a", "b", "+2 more"]
        summary = values[-1]
        assert summary["cardinality"] == 11  # 10 + 1
        assert summary["strategy"] == "aggregated"
        assert "+2 remaining axes summed into one bar" in spec["title"]

    def test_truncate_drops_dropped_axes_without_summary(self, four_axes):
        spec = generate_vega_spec(
            four_axes, panels={"cardinality"}, max_axes=2, overflow="truncate"
        )

        assert [v["name"] for v in spec["data"]["values"]] == ["a", "b"]
        assert "+2 remaining truncated" in spec["title"]

    def test_no_overflow_has_no_summary(self, four_axes):
        spec = generate_vega_spec(four_axes, panels={"cardinality"}, max_axes=10)

        assert len(spec["data"]["values"]) == 4
        assert "remaining" not in spec["title"]


class TestManifest:
    """Finding: the manifest embedded a wall-clock timestamp, breaking byte-determinism."""

    def test_manifest_has_no_timestamp_and_is_reproducible(self, two_axis_plan, tmp_path):
        stats = extract_plan_stats(two_axis_plan)

        export_plan_artifacts(stats, tmp_path / "a", prefix="plan", context="ctx")
        export_plan_artifacts(stats, tmp_path / "b", prefix="plan", context="ctx")

        manifest_a = (tmp_path / "a" / "plan_manifest.json").read_bytes()
        manifest_b = (tmp_path / "b" / "plan_manifest.json").read_bytes()
        assert manifest_a == manifest_b
        assert "created_at_utc" not in json.loads(manifest_a)

    def test_skipped_parquet_is_recorded(self, two_axis_plan, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        stats = extract_plan_stats(two_axis_plan)

        manifest = export_plan_artifacts(stats, tmp_path, prefix="plan")

        assert "plan_axes.parquet" not in manifest["artifacts"]
        assert "plan_axes.parquet" in manifest["skipped"]
        assert not (tmp_path / "plan_axes.parquet").exists()


class TestCanonicalValues:
    """Finding: bools serialized as 1/0 and a 0-byte estimate exported as empty."""

    def test_bool_serializes_as_json_bool(self):
        out = export_canonical_json({"flag": True, "off": np.bool_(False), "n": 1})

        assert out == b'{"flag":true,"n":1,"off":false}'

    def test_zero_memory_estimate_exports_as_zero(self):
        rows = list(
            csv.DictReader(io.StringIO(export_axes_csv(_stats([_axis("a", 4, memory=0.0)]))))
        )

        # Written as a real value (0.0), not blank: blank is reserved for None.
        assert rows[0]["memory_estimate_bytes"] != ""
        assert float(rows[0]["memory_estimate_bytes"]) == 0.0

    def test_none_memory_estimate_exports_as_empty(self):
        rows = list(
            csv.DictReader(io.StringIO(export_axes_csv(_stats([_axis("a", 4, memory=None)]))))
        )

        assert rows[0]["memory_estimate_bytes"] == ""


class TestVegaFallback:
    """Finding: a bare except hid malformed-spec errors behind a placeholder SVG."""

    def test_conversion_error_propagates(self, monkeypatch):
        fake = types.ModuleType("vl_convert")

        def boom(spec):
            raise ValueError("malformed spec")

        fake.vegalite_to_svg = boom
        monkeypatch.setitem(sys.modules, "vl_convert", fake)

        with pytest.raises(ValueError, match="malformed spec"):
            VegaBackend()._render_fallback_svg({}, _stats([_axis("a", 4)]))

    def test_missing_vl_convert_uses_placeholder(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vl_convert", None)

        svg = VegaBackend()._render_fallback_svg({}, _stats([_axis("a", 4)]))

        assert svg.startswith("<svg")
        assert "1 active axes" in svg

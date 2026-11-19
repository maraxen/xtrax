"""Tests for xtrax.eda paths the tier-2 coverage gate (distribution/coverage_dag.toml)
was not exercising: entry-point plugin discovery, plugin quarantine edge cases, spec
panels with bucket and memory data, file-path exports, the Vega PNG/ImportError/format
branches, the Matplotlib backend, and registered backend dispatch in render().
"""

import copy
import importlib.metadata
import json
import sys

import pytest

from xtrax.eda import export_canonical_json, render
from xtrax.eda.backends.matplotlib import MatplotlibBackend
from xtrax.eda.backends.vega import VegaBackend
from xtrax.eda.export import export_axes_csv, export_axes_parquet
from xtrax.eda.plugins import (
    clear_quarantined_plugins,
    execute_panel_plugin,
    get_quarantined_plugins,
    get_registered_backends,
    get_registered_panels,
    reload_plugins,
)
from xtrax.eda.spec import _sanitize_float, generate_vega_spec
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


@pytest.fixture
def stats(two_axis_plan) -> PlanStatsDict:
    """Real stats from the plan, with dedup, bucket and memory-warning data added."""
    base = copy.deepcopy(extract_plan_stats(two_axis_plan))
    base["dedup_stats"] = [
        {
            "axis_name": "seq",
            "dedup_ratio": 0.5,
            "unique_count": 4,
            "padded_count": 8,
            "total_count": 8,
            "padding_waste": 4,
        }
    ]
    base["bucket_stats"] = [
        {"axis_name": "seq", "bucket_count": 2, "bucket_boundaries": [8, 16]},
    ]
    base["memory_warnings"] = ["batch exceeds budget", "seq exceeds budget"]
    return base


class _FakeEntryPoint:
    def __init__(self, name, target):
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


class _FakeEntryPoints:
    def __init__(self, groups):
        self._groups = groups

    def select(self, group):
        return self._groups.get(group, [])


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Install a fake entry-point table; returns a setter taking {group: [eps]}."""

    def install(groups):
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _FakeEntryPoints(groups))

    yield install


@pytest.fixture(autouse=True)
def _fresh_plugin_state():
    reload_plugins()
    clear_quarantined_plugins()
    yield
    reload_plugins()
    clear_quarantined_plugins()


class _GoodPanel:
    name = "custom"

    def generate_spec(self, stats, config):
        return {"title": "Custom", "mark": "bar"}


class _Adapter:
    name = "fake"

    def render_spec(self, spec, fmt, stats, options=None):
        return f"fake:{fmt}:{len(spec)}"


class TestPluginDiscovery:
    def test_reload_clears_caches_and_quarantine(self):
        import xtrax.eda.plugins as plugins

        plugins._PANEL_CACHE = {"stale": _GoodPanel()}
        plugins._BACKEND_CACHE = {"stale": _Adapter()}
        plugins._QUARANTINED_PLUGINS.append("old")

        reload_plugins()

        assert plugins._PANEL_CACHE is None
        assert plugins._BACKEND_CACHE is None
        assert get_quarantined_plugins() == []

    def test_panel_entry_points_are_loaded_and_instantiated(self, fake_entry_points):
        fake_entry_points({"xtrax.eda.panels": [_FakeEntryPoint("custom", _GoodPanel)]})

        panels = get_registered_panels()

        assert isinstance(panels["custom"], _GoodPanel)

    def test_broken_panel_entry_point_is_quarantined_and_skipped(self, fake_entry_points):
        fake_entry_points(
            {
                "xtrax.eda.panels": [
                    _FakeEntryPoint("broken", ImportError("no such module")),
                    _FakeEntryPoint("custom", _GoodPanel),
                ]
            }
        )

        panels = get_registered_panels()

        assert "broken" not in panels
        assert "custom" in panels
        assert "broken" in get_quarantined_plugins()

    def test_backend_entry_points_are_loaded(self, fake_entry_points):
        fake_entry_points({"xtrax.eda.backends": [_FakeEntryPoint("fake", _Adapter)]})

        backends = get_registered_backends()

        assert isinstance(backends["fake"], _Adapter)

    def test_broken_backend_entry_point_is_skipped(self, fake_entry_points):
        fake_entry_points(
            {
                "xtrax.eda.backends": [
                    _FakeEntryPoint("broken", RuntimeError("boom")),
                    _FakeEntryPoint("fake", _Adapter),
                ]
            }
        )

        backends = get_registered_backends()

        assert set(backends) == {"fake"}


class TestPluginQuarantine:
    def test_non_dict_snippet_is_quarantined(self):
        class NotADict:
            name = "scalar"

            def generate_spec(self, stats, config):
                return "not a spec"

        result = execute_panel_plugin(NotADict(), {"total_axes": 1})

        assert "scalar" in get_quarantined_plugins()
        assert "Plugin Error: scalar" in result["title"]

    def test_dict_without_vega_keys_is_quarantined(self):
        class NoMark:
            name = "unmarked"

            def generate_spec(self, stats, config):
                return {"description": "missing mark/layer/concat"}

        result = execute_panel_plugin(NoMark(), {"total_axes": 1})

        assert "unmarked" in get_quarantined_plugins()
        assert "not a valid Vega-Lite panel dict" in result["mark"]["text"]

    def test_repeated_failure_is_recorded_once(self):
        class Crasher:
            name = "crasher"

            def generate_spec(self, stats, config):
                raise ValueError("bad")

        execute_panel_plugin(Crasher(), {})
        execute_panel_plugin(Crasher(), {})

        assert get_quarantined_plugins().count("crasher") == 1

    def test_valid_plugin_snippet_passes_through(self):
        result = execute_panel_plugin(_GoodPanel(), {"total_axes": 1})

        assert result == {"title": "Custom", "mark": "bar"}
        assert get_quarantined_plugins() == []


class TestSpecPanels:
    def test_sanitize_float_maps_nan_and_non_numeric_to_none(self):
        assert _sanitize_float(float("nan")) is None
        assert _sanitize_float(float("inf")) is None
        assert _sanitize_float(2.5) == 2.5

    def test_bucket_panel_renders_bucket_data(self, stats):
        spec = generate_vega_spec(stats, panels={"bucket"})

        assert spec["title"] == "Bucket Strategy Overview"
        assert spec["data"]["values"] == [
            {"axis_name": "seq", "bucket_count": 2, "boundaries": "8, 16"}
        ]

    def test_memory_title_counts_warnings(self, stats):
        spec = generate_vega_spec(stats, panels={"memory"})

        assert spec["title"] == "Estimated Memory Consumption (2 warnings)"

    def test_truncate_overflow_states_count_in_title(self, stats):
        spec = generate_vega_spec(stats, panels={"cardinality"}, max_axes=1, overflow="truncate")

        assert "+1 remaining truncated" in spec["title"]
        assert len(spec["data"]["values"]) == 1

    def test_unknown_panel_name_is_rejected(self, stats):
        with pytest.raises(ValueError, match="Invalid panel names"):
            generate_vega_spec(stats, panels={"no_such_panel"})

    def test_empty_panel_set_yields_zero_axis_spec(self, stats):
        spec = generate_vega_spec(stats, panels=set())

        assert spec["description"] == "Empty compilation plan"

    def test_registered_plugin_panel_is_appended(self, stats, monkeypatch):
        monkeypatch.setattr(
            "xtrax.eda.spec.get_registered_panels", lambda: {"custom": _GoodPanel()}
        )

        spec = generate_vega_spec(stats, panels={"strategy", "custom"})

        assert {"title": "Custom", "mark": "bar"} in spec["vconcat"]


class TestFileExports:
    def test_canonical_json_writes_file_matching_return_value(self, stats, tmp_path):
        out = tmp_path / "stats.json"

        data = export_canonical_json(stats, path=out)

        assert out.read_bytes() == data
        assert json.loads(data)["total_axes"] == stats["total_axes"]

    def test_nan_memory_estimate_serializes_as_null(self, stats):
        stats = copy.deepcopy(stats)
        stats["axes"][0]["memory_estimate_bytes"] = float("nan")

        payload = json.loads(export_canonical_json(stats))

        assert payload["axes"][0]["memory_estimate_bytes"] is None

    def test_csv_writes_file_and_includes_bucket_boundaries(self, stats, tmp_path):
        out = tmp_path / "axes.csv"

        text = export_axes_csv(stats, path=out)

        assert out.read_bytes().decode("utf-8") == text
        assert "8;16" in text

    def test_parquet_writes_file_matching_return_value(self, stats, tmp_path):
        out = tmp_path / "axes.parquet"

        data = export_axes_parquet(stats, path=out)

        assert out.read_bytes() == data
        assert data[:4] == b"PAR1"


class TestVegaBackendBranches:
    def test_png_rasterizes_via_vl_convert(self, stats):
        spec = generate_vega_spec(stats, panels={"strategy"})

        data = VegaBackend().render_spec(spec, fmt="png", stats=stats)

        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_svg_without_vl_convert_raises_import_error(self, stats, monkeypatch):
        monkeypatch.setitem(sys.modules, "vl_convert", None)
        spec = generate_vega_spec(stats, panels={"strategy"})

        with pytest.raises(ImportError, match="requires vl-convert-python"):
            VegaBackend().render_spec(spec, fmt="svg", stats=stats)


class TestMatplotlibBackend:
    def test_svg_with_no_panels_skips_panel_injection(self, stats):
        out = MatplotlibBackend().render_spec({}, fmt="svg", stats=stats, options={"panels": []})

        assert b"<svg" in out

    def test_html_wraps_svg_in_document(self, stats):
        out = MatplotlibBackend().render_spec({}, fmt="html", stats=stats)

        assert out.startswith("<!DOCTYPE html>")
        assert "<svg" in out


class TestRegisteredBackendDispatch:
    def test_render_routes_to_registered_backend(self, two_axis_plan, monkeypatch):
        monkeypatch.setattr(
            "xtrax.eda.render_core.get_registered_backends", lambda: {"fake": _Adapter()}
        )

        result = render(two_axis_plan, fmt="svg", backend="fake")

        assert result.startswith("fake:svg:")

    def test_render_rejects_unknown_backend(self, two_axis_plan):
        with pytest.raises(ValueError, match="Invalid backend"):
            render(two_axis_plan, fmt="svg", backend="nope")

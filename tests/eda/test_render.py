"""Tests for xtrax.eda.viz render() function — criterion-based test suite.

This test module verifies the render() API against acceptance criteria from the
EDA visualization API design spec. Tests are organized by criterion number and
focus on the core behavior expected from the public rendering interface.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Skip all tests in this module if seaborn is not installed
seaborn = pytest.importorskip("seaborn")

from xtrax.eda.types import PlanStatsDict
from xtrax.eda.viz import render
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import SafeMap, Vmap


@pytest.fixture
def simple_plan() -> BatchPlan:
    """Simple 2-axis plan for basic tests.

    Criterion 1-3: Test basic rendering formats.
    """
    spec1 = AxisSpec(name="batch", cardinality=64, default_batch_size=128)
    spec2 = AxisSpec(name="seq", cardinality=512, default_batch_size=64)

    decision1 = AxisDecision(
        spec=spec1,
        batch_size=128,
        reasoning="Batch axis",
        strategy=Vmap(),
    )
    decision2 = AxisDecision(
        spec=spec2,
        batch_size=64,
        reasoning="Sequence axis",
        strategy=SafeMap(batch_size=64),
    )

    return BatchPlan(decisions=(decision1, decision2))


class TestRenderPNG:
    """Criterion 1: PNG output returns bytes with PNG magic header."""

    def test_render_png_returns_bytes(self, simple_plan):
        """Criterion 1: render(fmt="png") returns bytes with PNG signature."""
        result = render(simple_plan, fmt="png")

        # Verify return type is bytes
        assert isinstance(result, bytes)

        # Verify PNG magic bytes: \x89PNG\r\n\x1a\n
        assert result[:8] == b"\x89PNG\r\n\x1a\n"


class TestRenderSVG:
    """Criterion 2: SVG output returns bytes starting with SVG header."""

    def test_render_svg_returns_bytes(self, simple_plan):
        """Criterion 2: render(fmt="svg") returns bytes with SVG/XML header."""
        result = render(simple_plan, fmt="svg")

        # Verify return type is bytes
        assert isinstance(result, bytes)

        # SVG can start with XML declaration or <svg directly
        svg_str = result.decode("utf-8")
        assert result[:4] == b"<svg" or result[:5] == b"<?xml"


class TestRenderHTML:
    """Criterion 3: HTML output returns str with embedded SVG."""

    def test_render_html_returns_str(self, simple_plan):
        """Criterion 3: render(fmt="html") returns str containing HTML+SVG."""
        result = render(simple_plan, fmt="html")

        # Verify return type is str, not bytes
        assert isinstance(result, str)

        # HTML should contain SVG
        assert "<svg" in result


class TestStatsTransformApplied:
    """Criterion 5: stats_transform modifies stats before rendering."""

    def test_stats_transform_applied(self, simple_plan):
        """Criterion 5: stats_transform hook modifies stats before render.

        The transform receives stats dict, modifies strategy_counts,
        and the modified values appear in the output (verified by looking
        for the mutated value in the SVG).
        """

        def bump_vmap(stats: PlanStatsDict) -> PlanStatsDict:
            # Mutate strategy_counts to have a test value
            stats = dict(stats)
            stats["strategy_counts"] = {"Vmap": 99}
            return stats

        result = render(simple_plan, fmt="svg", stats_transform=bump_vmap)

        # Convert bytes to string and check for mutated value
        svg = result.decode() if isinstance(result, bytes) else result
        assert "99" in svg


class TestMetadataSidecar:
    """Criterion 6: metadata=True writes JSON sidecar with PlanStatsDict."""

    def test_metadata_sidecar(self, simple_plan):
        """Criterion 6: metadata=True writes {stem}.json sidecar.

        The sidecar must contain axes and strategy_counts keys (PlanStatsDict).
        """
        with tempfile.TemporaryDirectory() as tmp_path:
            out = Path(tmp_path) / "plan.png"

            # Render with metadata
            render(simple_plan, path=str(out), metadata=True)

            # Check output file exists
            assert out.exists()

            # Check sidecar exists
            sidecar = Path(tmp_path) / "plan.json"
            assert sidecar.exists()

            # Parse and verify structure
            data = json.loads(sidecar.read_text())
            assert "axes" in data
            assert "strategy_counts" in data


class TestLoggerCalled:
    """Criterion 7: logger.log_figure() is called with figure, fmt, step."""

    def test_logger_called(self, simple_plan):
        """Criterion 7: render() invokes logger.log_figure(figure, fmt, step).

        For PNG format, figure should be bytes. Logger receives fmt and step
        parameters.
        """
        calls = []

        class MockLogger:
            def log_figure(self, figure: bytes | str, fmt: str, step: int | None = None):
                calls.append((type(figure).__name__, fmt, step))

        render(simple_plan, fmt="png", logger=MockLogger(), step=0)

        # Logger should be called exactly once
        assert len(calls) == 1

        # Should receive bytes for PNG, format string, and step
        assert calls[0] == ("bytes", "png", 0)


class TestPanelsFilterSVG:
    """Criterion 8 + AMD-5: panels parameter filters output, marks with data-panel."""

    def test_panels_filter_svg(self, simple_plan):
        """Criterion 8 + AMD-5: panels= filters which panels render.

        SVG output should contain data-panel attributes. When panels={"strategy"}
        is set, the strategy panel attribute should be present and cardinality
        panel should be absent.
        """
        result = render(simple_plan, fmt="svg", panels={"strategy"})

        # Convert to string for inspection
        svg = result.decode() if isinstance(result, bytes) else result

        # Strategy panel should be marked in the output
        assert 'data-panel="strategy"' in svg

        # Other panels should not appear (since only strategy is requested)
        assert 'data-panel="cardinality"' not in svg

"""Tests for xtrax.eda.viz rendering API."""

import json
import tempfile
from pathlib import Path

import pytest

from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import (
    Bucket,
    DedupGather,
    SafeMap,
    Vmap,
)
from xtrax.eda.stats import extract_plan_stats
from xtrax.eda.types import PlanLogger, PlanStatsDict
from xtrax.eda.viz import render


class MockLogger:
    """Simple in-memory PlanLogger for testing."""

    def __init__(self):
        self.figures = []

    def log_figure(self, figure: bytes | str, fmt: str, step: int | None = None) -> None:
        self.figures.append({"figure": figure, "fmt": fmt, "step": step})


@pytest.fixture
def simple_plan() -> BatchPlan:
    """Create a simple 2-axis plan for testing."""
    spec1 = AxisSpec(name="batch", cardinality=32, default_batch_size=16)
    spec2 = AxisSpec(name="sequence", cardinality=128, default_batch_size=64)

    decision1 = AxisDecision(
        spec=spec1, batch_size=16, reasoning="Fits in memory", strategy=Vmap()
    )
    decision2 = AxisDecision(
        spec=spec2,
        batch_size=64,
        reasoning="SafeMap for larger axis",
        strategy=SafeMap(batch_size=64),
    )

    return BatchPlan(decisions=(decision1, decision2))


@pytest.fixture
def empty_plan() -> BatchPlan:
    """Create an empty plan with no axes."""
    return BatchPlan(decisions=())


@pytest.fixture
def dedup_plan() -> BatchPlan:
    """Create a plan with DedupGather strategy."""
    import numpy as np

    spec = AxisSpec(name="vocab", cardinality=10000, default_batch_size=64)
    unique_indices = np.array([0, 1, 2, 5, 7], dtype=np.int32)
    index_map = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int32)

    strategy = DedupGather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=5,
        k_bucket=8,
        dedup_fn=lambda xs, ui: xs[ui],
        gather_fn=lambda ys, im: ys[im],
    )

    decision = AxisDecision(
        spec=spec,
        batch_size=64,
        reasoning="Dedup for vocabulary axis",
        strategy=strategy,
    )

    return BatchPlan(decisions=(decision,))


@pytest.fixture
def bucket_plan() -> BatchPlan:
    """Create a plan with Bucket strategy."""
    spec = AxisSpec(
        name="length",
        cardinality=512,
        default_batch_size=128,
        bucket_boundaries=(64, 128, 256, 512),
    )

    strategy = Bucket(boundaries=(64, 128, 256, 512))

    decision = AxisDecision(
        spec=spec,
        batch_size=128,
        reasoning="Bucket for variable-length axis",
        strategy=strategy,
    )

    return BatchPlan(decisions=(decision,))


class TestRenderBasicPNG:
    """Test basic PNG rendering."""

    def test_render_simple_plan_png_in_memory(self, simple_plan):
        """Render simple plan to PNG bytes in memory."""
        result = render(simple_plan, fmt="png")

        assert isinstance(result, bytes)
        assert len(result) > 0
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        assert result[:4] == b"\x89PNG"

    def test_render_simple_plan_png_to_file(self, simple_plan):
        """Render simple plan to PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            result = render(simple_plan, fmt="png", path=path)

            assert result is None
            assert path.exists()
            assert path.stat().st_size > 0
            with open(path, "rb") as f:
                data = f.read()
            assert data[:4] == b"\x89PNG"

    def test_render_empty_plan_png(self, empty_plan):
        """Render empty plan without error (placeholder figure)."""
        result = render(empty_plan, fmt="png")

        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"


class TestRenderSVG:
    """Test SVG rendering."""

    def test_render_simple_plan_svg_in_memory(self, simple_plan):
        """Render simple plan to SVG bytes in memory."""
        result = render(simple_plan, fmt="svg")

        assert isinstance(result, bytes)
        assert b"<svg" in result

    def test_render_simple_plan_svg_to_file(self, simple_plan):
        """Render simple plan to SVG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.svg"
            result = render(simple_plan, fmt="svg", path=path)

            assert result is None
            assert path.exists()
            content = path.read_text()
            assert "<svg" in content

    def test_svg_contains_panel_attributes(self, simple_plan):
        """SVG output should contain data-panel attributes."""
        result = render(simple_plan, fmt="svg")

        svg_str = result.decode("utf-8")
        # Should have data-panel attributes injected
        assert 'data-panel="' in svg_str


class TestRenderHTML:
    """Test HTML rendering."""

    def test_render_simple_plan_html_in_memory(self, simple_plan):
        """Render simple plan to HTML string in memory."""
        result = render(simple_plan, fmt="html")

        assert isinstance(result, str)
        assert "<html>" in result
        assert "<svg" in result
        assert "</html>" in result

    def test_render_simple_plan_html_to_file(self, simple_plan):
        """Render simple plan to HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.html"
            result = render(simple_plan, fmt="html", path=path)

            assert result is None
            assert path.exists()
            content = path.read_text()
            assert "<html>" in content
            assert "<svg" in content

    def test_html_contains_svg(self, simple_plan):
        """HTML output should embed SVG content."""
        result = render(simple_plan, fmt="html")

        assert "<svg" in result
        assert 'data-panel="' in result


class TestMetadata:
    """Test metadata sidecar generation."""

    def test_metadata_sidecar_with_path(self, simple_plan):
        """Metadata=True writes JSON sidecar alongside output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            result = render(simple_plan, fmt="png", path=path, metadata=True)

            assert result is None
            assert path.exists()
            metadata_path = path.with_suffix(".json")
            assert metadata_path.exists()

            metadata = json.loads(metadata_path.read_text())
            assert "axes" in metadata
            assert "strategy_counts" in metadata
            assert metadata["total_axes"] == 2

    def test_metadata_without_path_raises(self, simple_plan):
        """metadata=True without path should raise ValueError."""
        with pytest.raises(ValueError, match="metadata=True requires path"):
            render(simple_plan, fmt="png", metadata=True)

    def test_metadata_dict_structure(self, simple_plan):
        """Metadata JSON should contain all expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            render(simple_plan, fmt="png", path=path, metadata=True)

            metadata = json.loads(path.with_suffix(".json").read_text())
            assert "axes" in metadata
            assert "strategy_counts" in metadata
            assert "total_axes" in metadata
            assert "memory_warnings" in metadata
            assert "dedup_stats" in metadata
            assert "bucket_stats" in metadata


class TestPanelFiltering:
    """Test panel filtering and validation."""

    def test_invalid_panel_name_raises(self, simple_plan):
        """Invalid panel name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown panel"):
            render(simple_plan, fmt="png", panels={"invalid_panel"})

    def test_valid_panel_names(self, simple_plan):
        """Valid panel names should not raise."""
        valid_panels = {"strategy", "cardinality", "dedup", "bucket", "memory", "reasoning"}
        result = render(simple_plan, fmt="png", panels=valid_panels)
        assert result is not None

    def test_subset_of_panels(self, simple_plan):
        """Rendering subset of panels should work."""
        result = render(simple_plan, fmt="png", panels={"strategy"})
        assert result is not None

    def test_empty_panels_set(self, simple_plan):
        """Empty panels set should still render."""
        result = render(simple_plan, fmt="png", panels=set())
        assert result is not None


class TestStatsTransform:
    """Test stats_transform hook."""

    def test_stats_transform_applied(self, simple_plan):
        """stats_transform should be applied to extracted stats."""
        def add_annotation(stats: PlanStatsDict) -> PlanStatsDict:
            stats["memory_warnings"] = ["Test warning added by transform"]
            return stats

        result = render(simple_plan, fmt="png", stats_transform=add_annotation)
        assert result is not None

    def test_stats_transform_missing_keys_raises(self, simple_plan):
        """stats_transform returning incomplete dict should raise TypeError."""
        def incomplete_transform(stats: PlanStatsDict) -> PlanStatsDict:
            # Return dict missing required keys
            return {}

        with pytest.raises(TypeError, match="missing"):
            render(simple_plan, fmt="png", stats_transform=incomplete_transform)

    def test_stats_transform_can_filter_axes(self, simple_plan):
        """stats_transform can filter axes from stats."""
        def keep_first_axis(stats: PlanStatsDict) -> PlanStatsDict:
            stats["axes"] = stats["axes"][:1]
            return stats

        result = render(simple_plan, fmt="png", stats_transform=keep_first_axis)
        assert result is not None


class TestLogger:
    """Test PlanLogger integration."""

    def test_logger_called_with_bytes(self, simple_plan):
        """Logger should be called for PNG/SVG with bytes figure."""
        logger = MockLogger()
        render(simple_plan, fmt="png", logger=logger)

        assert len(logger.figures) == 1
        assert isinstance(logger.figures[0]["figure"], bytes)
        assert logger.figures[0]["fmt"] == "png"
        assert logger.figures[0]["step"] is None

    def test_logger_called_with_str_for_html(self, simple_plan):
        """Logger should be called for HTML with str figure."""
        logger = MockLogger()
        render(simple_plan, fmt="html", logger=logger)

        assert len(logger.figures) == 1
        assert isinstance(logger.figures[0]["figure"], str)
        assert logger.figures[0]["fmt"] == "html"

    def test_logger_receives_step(self, simple_plan):
        """Logger should receive step parameter."""
        logger = MockLogger()
        render(simple_plan, fmt="png", logger=logger, step=42)

        assert logger.figures[0]["step"] == 42

    def test_logger_called_even_with_path(self, simple_plan):
        """Logger should be called even when writing to path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MockLogger()
            path = Path(tmpdir) / "test.png"
            render(simple_plan, fmt="png", path=path, logger=logger)

            assert len(logger.figures) == 1



class TestImportErrors:
    """Test ImportError handling for missing dependencies."""

    def test_render_requires_eda_extras(self):
        """render() raises ImportError with pip install message when seaborn is absent."""
        import sys
        import subprocess

        # Run a fresh Python process where seaborn is unavailable
        # This isolates the test from matplotlib's state
        code = """
import sys
sys.modules['seaborn'] = None
try:
    from xtrax.eda.viz import render
    sys.exit(1)  # Should not reach here
except ImportError as e:
    if 'pip install xtrax[eda]' in str(e):
        sys.exit(0)  # Expected error message
    else:
        print(f"Wrong error message: {e}", file=sys.stderr)
        sys.exit(2)
"""
        # Use project root (parent of tests/) as cwd for stable path across worktrees
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        assert result.returncode == 0, f"Test failed. stderr: {result.stderr}"


class TestSpecialCases:
    """Test special cases and edge cases."""

    def test_render_dedup_plan(self, dedup_plan):
        """Render plan with DedupGather strategy."""
        result = render(dedup_plan, fmt="png")
        assert result is not None

    def test_render_bucket_plan(self, bucket_plan):
        """Render plan with Bucket strategy."""
        result = render(bucket_plan, fmt="png")
        assert result is not None

    def test_render_empty_plan_has_placeholder(self, empty_plan):
        """Empty plan should render placeholder text."""
        result = render(empty_plan, fmt="png")
        # Just verify it renders without error
        assert result is not None

    def test_all_formats_supported(self, simple_plan):
        """All three formats should be supported."""
        png_result = render(simple_plan, fmt="png")
        svg_result = render(simple_plan, fmt="svg")
        html_result = render(simple_plan, fmt="html")

        assert isinstance(png_result, bytes)
        assert isinstance(svg_result, bytes)
        assert isinstance(html_result, str)

    def test_render_with_all_options(self, simple_plan):
        """Render with all parameters specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MockLogger()

            def dummy_transform(stats):
                return stats

            path = Path(tmpdir) / "full.png"
            result = render(
                simple_plan,
                view="dashboard",
                fmt="png",
                path=path,
                stats_transform=dummy_transform,
                metadata=True,
                logger=logger,
                step=10,
                panels={"strategy", "cardinality"},
            )

            assert result is None
            assert path.exists()
            assert path.with_suffix(".json").exists()
            assert len(logger.figures) == 1

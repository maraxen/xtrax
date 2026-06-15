"""Tests for xtrax.eda.viz render() error handling — error criteria test suite.

This test module verifies the render() API against error handling acceptance criteria
from the EDA visualization API design spec. Tests are organized by criterion number
and focus on edge cases and unhappy paths that should raise or validate gracefully.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests in this module if seaborn is not installed
seaborn = pytest.importorskip("seaborn")

from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Vmap
from xtrax.eda.viz import render
from xtrax.eda.stats import analyze_dedup, analyze_bucket
from xtrax.eda.types import PlanStatsDict


class TestRenderEmptyPlan:
    """Criterion 9: render(empty plan) returns non-empty bytes without raising."""

    def test_render_empty_plan(self):
        """Criterion 9: render(BatchPlan(decisions=()), fmt="png") returns bytes.

        An empty plan produces a placeholder panel. The output should be valid
        PNG bytes (non-empty, with PNG magic header).
        """
        result = render(BatchPlan(decisions=()), fmt="png")

        # Verify return type is bytes
        assert isinstance(result, bytes) and len(result) > 0

        # Verify PNG magic bytes: \x89PNG\r\n\x1a\n
        assert result[:8] == b"\x89PNG\r\n\x1a\n"


class TestMetadataWithoutPath:
    """Criterion 10: metadata=True with path=None raises ValueError."""

    def test_metadata_without_path_raises(self):
        """Criterion 10: render(plan, metadata=True, path=None) raises ValueError.

        The sidecar contract requires path to be set. Calling with metadata=True
        and path=None is an invalid combination and should raise ValueError with
        a clear message.
        """
        spec = AxisSpec("batch", 64, 128)
        decision = AxisDecision(
            spec=spec,
            batch_size=128,
            reasoning="test",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        with pytest.raises(ValueError, match="metadata=True requires path"):
            render(plan, metadata=True, path=None)


class TestInvalidPanelRaises:
    """Criterion 11: panels= containing unknown names raises ValueError."""

    def test_invalid_panel_raises(self):
        """Criterion 11: render(plan, panels={"invalid_panel"}) raises ValueError.

        The panels parameter accepts only valid panel names. Any unknown name
        should raise ValueError with a message listing valid options.
        """
        spec = AxisSpec("batch", 64, 128)
        decision = AxisDecision(
            spec=spec,
            batch_size=128,
            reasoning="test",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        with pytest.raises(ValueError) as exc_info:
            render(plan, panels={"invalid_panel"})

        msg = str(exc_info.value)
        assert "invalid_panel" in msg
        # Message should list valid panels
        assert any(p in msg for p in ["strategy", "cardinality", "dedup"])


class TestImportWithoutSeabornRaises:
    """Criterion 12: viz module import without seaborn raises ImportError."""

    def test_import_without_seaborn_raises(self):
        """Criterion 12: Importing xtrax.eda.viz without seaborn raises ImportError.

        This test simulates the missing seaborn scenario by temporarily removing
        it from sys.modules and reimporting the viz module. The ImportError message
        should guide users to install extras.
        """
        # Save the original state
        saved_viz = sys.modules.pop("xtrax.eda.viz", None)
        saved_sns = sys.modules.pop("seaborn", None)

        try:
            # Block seaborn import
            sys.modules["seaborn"] = None  # type: ignore

            with pytest.raises(ImportError, match="pip install xtrax\\[eda\\]"):
                # This will fail during the module import
                import importlib
                import xtrax.eda.viz as viz_module
                importlib.reload(viz_module)
        finally:
            # Restore original state
            if saved_sns is not None:
                sys.modules["seaborn"] = saved_sns
            else:
                sys.modules.pop("seaborn", None)

            if saved_viz is not None:
                sys.modules["xtrax.eda.viz"] = saved_viz


class TestStatsTransformMissingKeysRaises:
    """AMD-6: stats_transform returning incomplete dict raises TypeError."""

    def test_stats_transform_missing_key_raises(self):
        """AMD-6: render() with stats_transform returning incomplete dict.

        If stats_transform returns a dict missing required keys (axes, strategy_counts,
        etc.), render() should raise TypeError with a message about missing keys.
        """
        spec = AxisSpec("batch", 64, 128)
        decision = AxisDecision(
            spec=spec,
            batch_size=128,
            reasoning="test",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        def bad_transform(stats: PlanStatsDict) -> PlanStatsDict:
            # Return an incomplete dict missing required keys
            return {"axes": []}  # type: ignore

        with pytest.raises(TypeError, match="missing"):
            render(plan, stats_transform=bad_transform)


class TestAnalyzeDedupWrongStrategy:
    """AMD-4: analyze_dedup() with non-DedupGather strategy raises TypeError."""

    def test_analyze_dedup_wrong_strategy(self):
        """AMD-4: analyze_dedup(decision with non-DedupGather) raises TypeError.

        analyze_dedup() requires a DedupGather strategy. Passing any other
        strategy type should raise TypeError naming the received type.
        """
        spec = AxisSpec("x", 64, 128)
        decision = AxisDecision(
            spec=spec,
            batch_size=128,
            reasoning="test",
            strategy=Vmap(),  # Not DedupGather
        )

        with pytest.raises(TypeError, match="DedupGather"):
            analyze_dedup(decision)


class TestAnalyzeBucketWrongStrategy:
    """AMD-4: analyze_bucket() with non-Bucket strategy raises TypeError."""

    def test_analyze_bucket_wrong_strategy(self):
        """AMD-4: analyze_bucket(decision with non-Bucket) raises TypeError.

        analyze_bucket() requires a Bucket strategy. Passing any other strategy
        type should raise TypeError naming the received type.
        """
        spec = AxisSpec("x", 64, 128)
        decision = AxisDecision(
            spec=spec,
            batch_size=128,
            reasoning="test",
            strategy=Vmap(),  # Not Bucket
        )

        with pytest.raises(TypeError, match="Bucket"):
            analyze_bucket(decision)


__all__ = [
    "TestRenderEmptyPlan",
    "TestMetadataWithoutPath",
    "TestInvalidPanelRaises",
    "TestImportWithoutSeabornRaises",
    "TestStatsTransformMissingKeysRaises",
    "TestAnalyzeDedupWrongStrategy",
    "TestAnalyzeBucketWrongStrategy",
]

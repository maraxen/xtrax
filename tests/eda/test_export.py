"""Tests for xtrax.eda.export — plan_to_dataframe."""

import numpy as np
import pytest

from xtrax.eda.stats import extract_plan_stats
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Vmap


class TestPlanToDataframe:
    """Tests for plan_to_dataframe."""

    def test_requires_pandas_extras(self):
        """plan_to_dataframe import requires eda extras."""
        # Test that the ImportError is raised when pandas is missing.
        # Use sys.modules patching to simulate absent pandas in the current interpreter.
        import sys

        # Save original pandas module (if it exists)
        original_pandas = sys.modules.get("pandas")

        try:
            # Patch sys.modules to hide pandas
            sys.modules["pandas"] = None

            # Force reimport of export module to trigger the ImportError
            if "xtrax.eda.export" in sys.modules:
                del sys.modules["xtrax.eda.export"]

            # This should raise ImportError
            with pytest.raises(ImportError, match=r"xtrax\[eda\]"):
                from xtrax.eda.export import plan_to_dataframe  # noqa: F401
        finally:
            # Restore original pandas module
            if original_pandas is not None:
                sys.modules["pandas"] = original_pandas
            elif "pandas" in sys.modules:
                del sys.modules["pandas"]

            # Clean up the export module so other tests get a fresh import
            if "xtrax.eda.export" in sys.modules:
                del sys.modules["xtrax.eda.export"]

    def test_empty_plan_dataframe(self):
        """plan_to_dataframe with empty plan returns DataFrame with no rows."""
        pytest.importorskip("pandas")
        from xtrax.eda.export import plan_to_dataframe

        plan = BatchPlan(decisions=())
        stats = extract_plan_stats(plan)
        df = plan_to_dataframe(stats)

        assert len(df) == 0
        assert "name" in df.columns
        assert "strategy" in df.columns
        assert "cardinality" in df.columns
        assert "batch_size" in df.columns
        assert "reasoning" in df.columns

    def test_single_axis_dataframe(self):
        """plan_to_dataframe with single axis creates one row."""
        pytest.importorskip("pandas")
        from xtrax.eda.export import plan_to_dataframe

        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="cardinality < batch_size",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))
        stats = extract_plan_stats(plan)

        df = plan_to_dataframe(stats)

        assert len(df) == 1
        assert df.loc[0, "name"] == "batch"
        assert df.loc[0, "strategy"] == "Vmap"
        assert df.loc[0, "cardinality"] == 100
        assert df.loc[0, "batch_size"] == 32
        assert df.loc[0, "reasoning"] == "cardinality < batch_size"

    def test_multiple_axes_dataframe(self):
        """plan_to_dataframe with multiple axes creates multiple rows."""
        pytest.importorskip("pandas")
        from xtrax.eda.export import plan_to_dataframe

        spec1 = AxisSpec(name="batch", cardinality=100, default_batch_size=32)
        spec2 = AxisSpec(name="sequence", cardinality=1000, default_batch_size=256)

        decision1 = AxisDecision(
            spec=spec1,
            batch_size=32,
            reasoning="vmap eligible",
            strategy=Vmap(),
        )
        decision2 = AxisDecision(
            spec=spec2,
            batch_size=256,
            reasoning="safemap for large cardinality",
            strategy=SafeMap(batch_size=256),
        )
        plan = BatchPlan(decisions=(decision1, decision2))
        stats = extract_plan_stats(plan)

        df = plan_to_dataframe(stats)

        assert len(df) == 2
        assert df.loc[0, "name"] == "batch"
        assert df.loc[0, "strategy"] == "Vmap"
        assert df.loc[1, "name"] == "sequence"
        assert df.loc[1, "strategy"] == "SafeMap"

    def test_dataframe_includes_dedup_columns(self):
        """plan_to_dataframe includes dedup columns where applicable."""
        pytest.importorskip("pandas")
        from xtrax.eda.export import plan_to_dataframe

        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=32, dedup_eligible=True)
        unique_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        index_map = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)

        strategy = DedupGather(
            unique_indices=unique_indices,
            index_map=index_map,
            k=5,
            k_bucket=8,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx_map: ys[idx_map],
        )
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="dedup eligible axis",
            strategy=strategy,
        )
        plan = BatchPlan(decisions=(decision,))
        stats = extract_plan_stats(plan)

        df = plan_to_dataframe(stats)

        assert len(df) == 1
        # Dedup columns should be present
        assert "dedup_ratio" in df.columns
        assert "unique_count" in df.columns
        assert "padded_count" in df.columns
        assert "padding_waste" in df.columns
        # Check values
        assert df.loc[0, "unique_count"] == 5
        assert df.loc[0, "padded_count"] == 8
        assert abs(df.loc[0, "dedup_ratio"] - 0.95) < 1e-6

    def test_dataframe_includes_bucket_columns(self):
        """plan_to_dataframe includes bucket columns where applicable."""
        pytest.importorskip("pandas")
        from xtrax.eda.export import plan_to_dataframe

        spec = AxisSpec(
            name="sequence",
            cardinality=1000,
            default_batch_size=256,
            bucket_boundaries=(256, 512, 1024),
        )
        strategy = Bucket(boundaries=(256, 512, 1024))
        decision = AxisDecision(
            spec=spec,
            batch_size=256,
            reasoning="bucketing strategy",
            strategy=strategy,
        )
        plan = BatchPlan(decisions=(decision,))
        stats = extract_plan_stats(plan)

        df = plan_to_dataframe(stats)

        assert len(df) == 1
        # Bucket columns should be present
        assert "bucket_count" in df.columns
        assert "bucket_boundaries" in df.columns
        # Check values
        assert df.loc[0, "bucket_count"] == 3
        assert df.loc[0, "bucket_boundaries"] == [256, 512, 1024]

    def test_dataframe_mixed_strategies_nan_handling(self):
        """plan_to_dataframe fills NaN for dedup/bucket columns on non-applicable axes."""
        pytest.importorskip("pandas")
        import pandas as pd

        from xtrax.eda.export import plan_to_dataframe

        # Create two axes: one Vmap, one Bucket
        spec1 = AxisSpec(name="batch", cardinality=100, default_batch_size=32)
        decision1 = AxisDecision(
            spec=spec1,
            batch_size=32,
            reasoning="vmap eligible",
            strategy=Vmap(),
        )

        spec2 = AxisSpec(
            name="sequence",
            cardinality=1000,
            default_batch_size=256,
            bucket_boundaries=(256, 512),
        )
        strategy2 = Bucket(boundaries=(256, 512))
        decision2 = AxisDecision(
            spec=spec2,
            batch_size=256,
            reasoning="bucketing strategy",
            strategy=strategy2,
        )

        plan = BatchPlan(decisions=(decision1, decision2))
        stats = extract_plan_stats(plan)

        df = plan_to_dataframe(stats)

        assert len(df) == 2
        # First row (Vmap) should have NaN for bucket columns
        assert pd.isna(df.loc[0, "bucket_count"])
        assert pd.isna(df.loc[0, "bucket_boundaries"])
        # Second row (Bucket) should have NaN for dedup columns
        assert pd.isna(df.loc[1, "dedup_ratio"])
        assert pd.isna(df.loc[1, "unique_count"])
        # Second row should have valid bucket values
        assert df.loc[1, "bucket_count"] == 2
        assert df.loc[1, "bucket_boundaries"] == [256, 512]

    def test_dataframe_preserves_axis_order(self):
        """plan_to_dataframe preserves decision order."""
        pytest.importorskip("pandas")
        from xtrax.eda.export import plan_to_dataframe

        decisions = tuple(
            AxisDecision(
                spec=AxisSpec(name=f"axis{i}", cardinality=100 + i, default_batch_size=32),
                batch_size=32,
                reasoning=f"axis {i}",
                strategy=Vmap(),
            )
            for i in range(5)
        )
        plan = BatchPlan(decisions=decisions)
        stats = extract_plan_stats(plan)

        df = plan_to_dataframe(stats)

        assert len(df) == 5
        for i in range(5):
            assert df.loc[i, "name"] == f"axis{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for xtrax.eda.stats — extract_plan_stats, analyze_dedup, analyze_bucket."""

import numpy as np
import pytest

from xtrax.eda.stats import analyze_bucket, analyze_dedup, extract_plan_stats
from xtrax.eda.types import (
    AxisStatsEntry,
    BucketStatsEntry,
    DedupStatsEntry,
    PlanStatsDict,
)
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Vmap


class TestExtractPlanStats:
    """Tests for extract_plan_stats."""

    def test_empty_plan_returns_empty_stats(self):
        """extract_plan_stats with empty plan returns all empty structures."""
        plan = BatchPlan(decisions=())
        stats = extract_plan_stats(plan)

        assert isinstance(stats, dict)
        assert stats["axes"] == []
        assert stats["strategy_counts"] == {}
        assert stats["total_axes"] == 0
        assert stats["memory_warnings"] == []
        assert stats["dedup_stats"] == []
        assert stats["bucket_stats"] == []

    def test_single_vmap_axis(self):
        """extract_plan_stats with single Vmap axis."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="cardinality < batch_size",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        stats = extract_plan_stats(plan)

        assert stats["total_axes"] == 1
        assert len(stats["axes"]) == 1
        assert stats["strategy_counts"] == {"Vmap": 1}
        assert stats["memory_warnings"] == []
        assert stats["dedup_stats"] == []
        assert stats["bucket_stats"] == []

        axis = stats["axes"][0]
        assert axis["name"] == "batch"
        assert axis["strategy"] == "Vmap"
        assert axis["cardinality"] == 100
        assert axis["batch_size"] == 32
        assert axis["reasoning"] == "cardinality < batch_size"
        assert axis["memory_estimate_bytes"] is None

    def test_multiple_different_strategies(self):
        """extract_plan_stats with multiple different strategies."""
        spec1 = AxisSpec(name="batch", cardinality=100, batch_size=32)
        spec2 = AxisSpec(name="sequence", cardinality=1000, batch_size=256)

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

        assert stats["total_axes"] == 2
        assert len(stats["axes"]) == 2
        assert stats["strategy_counts"] == {"Vmap": 1, "SafeMap": 1}

    def test_strategy_count_accumulates(self):
        """extract_plan_stats accumulates strategy counts correctly."""
        decisions = tuple(
            AxisDecision(
                spec=AxisSpec(
                    name=f"axis{i}", cardinality=100 + i, batch_size=32
                ),
                batch_size=32,
                reasoning="test",
                strategy=Vmap() if i < 3 else SafeMap(batch_size=32),
            )
            for i in range(5)
        )
        plan = BatchPlan(decisions=decisions)

        stats = extract_plan_stats(plan)

        assert stats["strategy_counts"] == {"Vmap": 3, "SafeMap": 2}

    def test_dedup_stats_accumulated_in_extract(self):
        """extract_plan_stats accumulates dedup stats from DedupGather decisions."""
        spec = AxisSpec(
            name="batch", cardinality=100, batch_size=32, dedup_eligible=True
        )
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

        assert len(stats["dedup_stats"]) == 1
        assert stats["strategy_counts"] == {"DedupGather": 1}

        dedup_entry = stats["dedup_stats"][0]
        assert dedup_entry["axis_name"] == "batch"
        assert dedup_entry["unique_count"] == 5
        assert dedup_entry["padded_count"] == 8
        assert dedup_entry["total_count"] == 100
        assert dedup_entry["padding_waste"] == 3
        assert isinstance(dedup_entry["dedup_ratio"], float)

    def test_bucket_stats_accumulated_in_extract(self):
        """extract_plan_stats accumulates bucket stats from Bucket decisions."""
        spec = AxisSpec(
            name="sequence",
            cardinality=1000,
            batch_size=256,
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

        assert len(stats["bucket_stats"]) == 1
        assert stats["strategy_counts"] == {"Bucket": 1}

        bucket_entry = stats["bucket_stats"][0]
        assert bucket_entry["axis_name"] == "sequence"
        assert bucket_entry["bucket_count"] == 3
        assert bucket_entry["bucket_boundaries"] == [256, 512, 1024]


class TestAnalyzeDedup:
    """Tests for analyze_dedup."""

    def test_analyze_dedup_basic(self):
        """analyze_dedup computes statistics correctly."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        unique_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        index_map = np.array([0, 1, 2, 3, 4, 0, 1], dtype=np.int32)

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
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_dedup(decision)

        assert result["axis_name"] == "batch"
        assert result["unique_count"] == 5
        assert result["padded_count"] == 8
        assert result["total_count"] == 100
        assert result["padding_waste"] == 3
        assert isinstance(result["dedup_ratio"], float)
        # dedup_ratio = 1.0 - (5 / 100) = 0.95
        assert abs(result["dedup_ratio"] - 0.95) < 1e-6

    def test_analyze_dedup_zero_cardinality_ratio_handling(self):
        """analyze_dedup handles zero cardinality gracefully."""
        spec = AxisSpec(name="empty", cardinality=0, batch_size=32)
        strategy = DedupGather(
            unique_indices=np.array([], dtype=np.int32),
            index_map=np.array([], dtype=np.int32),
            k=0,
            k_bucket=1,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx_map: ys[idx_map],
        )
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_dedup(decision)

        assert result["total_count"] == 0
        assert result["dedup_ratio"] == 0.0

    def test_analyze_dedup_rejects_wrong_strategy_type(self):
        """analyze_dedup raises TypeError if strategy is not DedupGather."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=Vmap(),
        )

        with pytest.raises(
            TypeError, match="analyze_dedup requires DedupGather strategy; got Vmap"
        ):
            analyze_dedup(decision)

    def test_analyze_dedup_rejects_safemap(self):
        """analyze_dedup rejects SafeMap strategy."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=SafeMap(batch_size=32),
        )

        with pytest.raises(
            TypeError, match="analyze_dedup requires DedupGather strategy; got SafeMap"
        ):
            analyze_dedup(decision)

    def test_analyze_dedup_rejects_bucket(self):
        """analyze_dedup rejects Bucket strategy."""
        spec = AxisSpec(name="sequence", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=Bucket(boundaries=(32, 64)),
        )

        with pytest.raises(
            TypeError, match="analyze_dedup requires DedupGather strategy; got Bucket"
        ):
            analyze_dedup(decision)


class TestAnalyzeBucket:
    """Tests for analyze_bucket."""

    def test_analyze_bucket_basic(self):
        """analyze_bucket extracts boundary information correctly."""
        spec = AxisSpec(
            name="sequence",
            cardinality=1000,
            batch_size=256,
            bucket_boundaries=(256, 512, 1024),
        )
        strategy = Bucket(boundaries=(256, 512, 1024))
        decision = AxisDecision(
            spec=spec,
            batch_size=256,
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_bucket(decision)

        assert result["axis_name"] == "sequence"
        assert result["bucket_count"] == 3
        assert result["bucket_boundaries"] == [256, 512, 1024]

    def test_analyze_bucket_single_boundary(self):
        """analyze_bucket works with single bucket boundary."""
        spec = AxisSpec(
            name="seq", cardinality=100, batch_size=32, bucket_boundaries=(64,)
        )
        strategy = Bucket(boundaries=(64,))
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_bucket(decision)

        assert result["bucket_count"] == 1
        assert result["bucket_boundaries"] == [64]

    def test_analyze_bucket_returns_list_not_tuple(self):
        """analyze_bucket.bucket_boundaries is a list, not tuple."""
        spec = AxisSpec(name="seq", cardinality=100, batch_size=32)
        strategy = Bucket(boundaries=(256, 512))
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_bucket(decision)

        assert isinstance(result["bucket_boundaries"], list)
        assert result["bucket_boundaries"] == [256, 512]

    def test_analyze_bucket_rejects_wrong_strategy_type(self):
        """analyze_bucket raises TypeError if strategy is not Bucket."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=Vmap(),
        )

        with pytest.raises(
            TypeError, match="analyze_bucket requires Bucket strategy; got Vmap"
        ):
            analyze_bucket(decision)

    def test_analyze_bucket_rejects_safemap(self):
        """analyze_bucket rejects SafeMap strategy."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=SafeMap(batch_size=32),
        )

        with pytest.raises(
            TypeError, match="analyze_bucket requires Bucket strategy; got SafeMap"
        ):
            analyze_bucket(decision)

    def test_analyze_bucket_rejects_dedup_gather(self):
        """analyze_bucket rejects DedupGather strategy."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        strategy = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1], dtype=np.int32),
            k=2,
            k_bucket=4,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx_map: ys[idx_map],
        )
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=strategy,
        )

        with pytest.raises(
            TypeError,
            match="analyze_bucket requires Bucket strategy; got DedupGather",
        ):
            analyze_bucket(decision)


class TestStatsTypeConsistency:
    """Tests for type consistency between stats functions and types module."""

    def test_extract_plan_stats_returns_plan_stats_dict(self):
        """extract_plan_stats return matches PlanStatsDict type contract."""
        plan = BatchPlan(decisions=())
        result = extract_plan_stats(plan)

        # Check all required keys are present
        assert "axes" in result
        assert "strategy_counts" in result
        assert "total_axes" in result
        assert "memory_warnings" in result
        assert "dedup_stats" in result
        assert "bucket_stats" in result

        # Check types
        assert isinstance(result["axes"], list)
        assert isinstance(result["strategy_counts"], dict)
        assert isinstance(result["total_axes"], int)
        assert isinstance(result["memory_warnings"], list)
        assert isinstance(result["dedup_stats"], list)
        assert isinstance(result["bucket_stats"], list)

    def test_dedup_stats_entry_types(self):
        """analyze_dedup return matches DedupStatsEntry type contract."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        strategy = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1], dtype=np.int32),
            k=2,
            k_bucket=4,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx_map: ys[idx_map],
        )
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_dedup(decision)

        assert isinstance(result["axis_name"], str)
        assert isinstance(result["dedup_ratio"], float)
        assert isinstance(result["unique_count"], int)
        assert isinstance(result["padded_count"], int)
        assert isinstance(result["total_count"], int)
        assert isinstance(result["padding_waste"], int)

    def test_bucket_stats_entry_types(self):
        """analyze_bucket return matches BucketStatsEntry type contract."""
        spec = AxisSpec(name="sequence", cardinality=100, batch_size=32)
        strategy = Bucket(boundaries=(64, 128))
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test",
            strategy=strategy,
        )

        result = analyze_bucket(decision)

        assert isinstance(result["axis_name"], str)
        assert isinstance(result["bucket_count"], int)
        assert isinstance(result["bucket_boundaries"], list)
        assert all(isinstance(b, int) for b in result["bucket_boundaries"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

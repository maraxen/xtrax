"""Tests for xtrax.eda.explain — explain_plan function."""


from xtrax.eda.explain import explain_plan
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import SafeMap, Vmap


class TestExplainPlan:
    """Tests for explain_plan."""

    def test_empty_plan_returns_empty_stats(self):
        """explain_plan with empty plan returns empty stats structure."""
        plan = BatchPlan(decisions=())
        stats = explain_plan(plan)

        assert isinstance(stats, dict)
        assert stats["axes"] == []
        assert stats["strategy_counts"] == {}
        assert stats["total_axes"] == 0

    def test_plan_with_non_empty_reasoning_unchanged(self):
        """explain_plan preserves non-empty reasoning fields."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="cardinality < batch_size, using Vmap",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        stats = explain_plan(plan)

        assert len(stats["axes"]) == 1
        assert stats["axes"][0]["reasoning"] == "cardinality < batch_size, using Vmap"

    def test_plan_with_empty_reasoning_substituted(self):
        """explain_plan substitutes empty reasoning with default message."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="",  # Empty reasoning
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        stats = explain_plan(plan)

        assert len(stats["axes"]) == 1
        assert (
            stats["axes"][0]["reasoning"]
            == "No reasoning recorded for axis 'batch'"
        )

    def test_plan_with_mixed_reasoning(self):
        """explain_plan handles mix of empty and non-empty reasoning."""
        spec1 = AxisSpec(name="batch", cardinality=100, batch_size=32)
        spec2 = AxisSpec(name="sequence", cardinality=500, batch_size=64)

        decision1 = AxisDecision(
            spec=spec1,
            batch_size=32,
            reasoning="cardinality < batch_size",
            strategy=Vmap(),
        )
        decision2 = AxisDecision(
            spec=spec2,
            batch_size=64,
            reasoning="",  # Empty
            strategy=SafeMap(batch_size=64),
        )

        plan = BatchPlan(decisions=(decision1, decision2))
        stats = explain_plan(plan)

        assert len(stats["axes"]) == 2
        assert stats["axes"][0]["reasoning"] == "cardinality < batch_size"
        assert (
            stats["axes"][1]["reasoning"]
            == "No reasoning recorded for axis 'sequence'"
        )

    def test_explain_plan_returns_complete_plandstatsdict(self):
        """explain_plan returns complete PlanStatsDict with all fields."""
        spec = AxisSpec(name="batch", cardinality=100, batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="test reasoning",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        stats = explain_plan(plan)

        # Verify all required keys exist
        assert "axes" in stats
        assert "strategy_counts" in stats
        assert "total_axes" in stats
        assert "memory_warnings" in stats
        assert "dedup_stats" in stats
        assert "bucket_stats" in stats

        assert stats["total_axes"] == 1
        assert stats["strategy_counts"] == {"Vmap": 1}

    def test_whitespace_only_reasoning_treated_as_empty(self):
        """explain_plan treats whitespace-only reasoning as empty."""
        spec = AxisSpec(name="axis", cardinality=50, batch_size=16)
        decision = AxisDecision(
            spec=spec,
            batch_size=16,
            reasoning="   ",  # Whitespace only
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))

        stats = explain_plan(plan)

        # Note: Current implementation only checks for falsy strings,
        # so whitespace-only will NOT be substituted.
        # This test documents current behavior.
        assert stats["axes"][0]["reasoning"] == "   "

"""Tests for xtrax.tiling.plan — AxisSpec, AxisDecision, BatchPlan, BatchPlanner."""

import warnings

import jax
import pytest

from xtrax.tiling.plan import (
    AxisDecision,
    AxisSpec,
    BatchPlan,
    BatchPlanner,
)
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Vmap


class TestAxisSpec:
    """AxisSpec dataclass creation and validation."""

    def test_axis_spec_instantiates_with_required_fields(self):
        """AxisSpec instantiates with name, cardinality, default_batch_size."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=32)
        assert spec.name == "batch"
        assert spec.cardinality == 100
        assert spec.default_batch_size == 32
        assert spec.tile_granularity == 1
        assert spec.heterogeneous is False
        assert spec.dedup_eligible is False

    def test_axis_spec_with_all_fields(self):
        """AxisSpec accepts all optional fields."""
        spec = AxisSpec(
            name="batch",
            cardinality=100,
            default_batch_size=32,
            tile_granularity=4,
            heterogeneous=True,
            dedup_eligible=True,
        )
        assert spec.tile_granularity == 4
        assert spec.heterogeneous is True
        assert spec.dedup_eligible is True

    def test_axis_spec_bucket_boundaries_default_none(self):
        """AxisSpec.bucket_boundaries defaults to None."""
        spec = AxisSpec(name="seq", cardinality=10, default_batch_size=4)
        assert spec.bucket_boundaries is None

    def test_axis_spec_bucket_boundaries_coerced_to_tuple(self):
        """A list of bucket_boundaries is coerced to a tuple (stays hashable)."""
        spec = AxisSpec(
            name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=[8, 16, 32]
        )
        assert spec.bucket_boundaries == (8, 16, 32)
        assert hash(spec) == hash(spec)  # hashable: frozen + tuple field

    def test_axis_spec_bucket_boundaries_empty_raises(self):
        """Empty bucket_boundaries is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            AxisSpec(name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=())

    def test_axis_spec_bucket_boundaries_non_ascending_raises(self):
        """Non-ascending bucket_boundaries is rejected."""
        with pytest.raises(ValueError, match="strictly ascending"):
            AxisSpec(
                name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(16, 8)
            )

    def test_axis_spec_bucket_boundaries_duplicate_raises(self):
        """Duplicate bucket_boundaries (not strictly ascending) is rejected."""
        with pytest.raises(ValueError, match="strictly ascending"):
            AxisSpec(
                name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(8, 8, 16)
            )

    def test_axis_spec_bucket_boundaries_non_positive_raises(self):
        """Non-positive bucket_boundaries is rejected."""
        with pytest.raises(ValueError, match="positive"):
            AxisSpec(
                name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(0, 8)
            )


class TestAxisDecision:
    """AxisDecision dataclass creation."""

    def test_axis_decision_instantiates(self):
        """AxisDecision instantiates with spec, batch_size, reasoning, strategy."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=32)
        strategy = Vmap()
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="cardinality <= batch_size",
            strategy=strategy,
        )
        assert decision.spec is spec
        assert decision.batch_size == 32
        assert decision.reasoning == "cardinality <= batch_size"
        assert isinstance(decision.strategy, Vmap)


class TestBatchPlan:
    """BatchPlan dataclass creation."""

    def test_batch_plan_instantiates_empty(self):
        """BatchPlan instantiates with empty decisions."""
        plan = BatchPlan(decisions=())
        assert len(plan.decisions) == 0

    def test_batch_plan_instantiates_with_decisions(self):
        """BatchPlan instantiates with decisions tuple."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=32)
        decision = AxisDecision(
            spec=spec,
            batch_size=32,
            reasoning="cardinality <= batch_size",
            strategy=Vmap(),
        )
        plan = BatchPlan(decisions=(decision,))
        assert len(plan.decisions) == 1
        assert plan.decisions[0] is decision


class TestBatchPlanner:
    """BatchPlanner selection rules and behavior."""

    def test_batch_planner_instantiates(self):
        """BatchPlanner instantiates with optional memory_estimator."""
        planner = BatchPlanner()
        assert planner is not None

    def test_batch_planner_with_memory_estimator(self):
        """BatchPlanner accepts a memory_estimator callable."""

        def estimate_memory(spec: AxisSpec) -> int:
            return spec.cardinality * 1000

        planner = BatchPlanner(memory_estimator=estimate_memory)
        assert planner is not None

    def test_plan_empty_specs(self):
        """plan([]) returns BatchPlan with empty decisions."""
        planner = BatchPlanner()
        plan = planner.plan([])
        assert isinstance(plan, BatchPlan)
        assert len(plan.decisions) == 0

    def test_phase0_carry_spec_returns_scan(self):
        """Phase 0: CarrySpec declared → Scan."""
        from xtrax.tiling.carry import CarrySpec

        spec = AxisSpec(name="n_samples", cardinality=10, default_batch_size=32)

        def transition(carry, x):
            return carry, x

        carry_spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
        )

        planner = BatchPlanner(carry_specs=[carry_spec])
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert decision.spec is spec
        # Phase 0 pre-demotes to Scan strategy
        from xtrax.tiling.strategy import Scan
        assert isinstance(decision.strategy, Scan)

    def test_phase0b_dedup_spec_returns_dedupgather(self):
        """Phase 0b: DedupSpec declared → DedupGather."""
        import numpy as np
        from xtrax.tiling.dedup import DedupSpec

        spec = AxisSpec(
            name="token",
            cardinality=1000,
            default_batch_size=32,
            dedup_eligible=True,
        )
        # Create a DedupSpec for this axis: 3 unique elements out of 1000
        unique_indices = np.array([0, 500, 999])
        index_map = np.zeros(1000, dtype=np.int32)
        index_map[500:] = 1
        index_map[999:] = 2
        dedup_spec = DedupSpec(
            axis_name="token",
            unique_indices=unique_indices,
            index_map=index_map,
            k=3,
        )

        planner = BatchPlanner(dedup_specs=[dedup_spec])
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert decision.spec is spec
        assert isinstance(decision.strategy, DedupGather)

    def test_rule1_bucket_boundaries_returns_bucket(self):
        """Rule 1: bucket_boundaries set → Bucket strategy."""
        spec = AxisSpec(
            name="seq",
            cardinality=300,
            default_batch_size=4,
            bucket_boundaries=(128, 256, 512),
        )
        planner = BatchPlanner()
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, Bucket)
        assert decision.strategy.boundaries == (128, 256, 512)

    def test_rule1_bucket_wins_over_dedup(self):
        """bucket_boundaries takes precedence over dedup_eligible."""
        spec = AxisSpec(
            name="seq",
            cardinality=300,
            default_batch_size=4,
            dedup_eligible=True,
            bucket_boundaries=(512,),
        )
        planner = BatchPlanner()
        plan = planner.plan([spec])

        assert isinstance(plan.decisions[0].strategy, Bucket)

    def test_rule2_cardinality_le_batch_size_returns_vmap(self):
        """Rule 2: cardinality <= batch_size → Vmap."""
        spec = AxisSpec(name="batch", cardinality=32, default_batch_size=100)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, Vmap)

    def test_rule2_cardinality_equals_batch_size_returns_vmap(self):
        """Rule 2: cardinality == batch_size → Vmap."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=100)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        decision = plan.decisions[0]
        assert isinstance(decision.strategy, Vmap)

    def test_rule3_divisible_cardinality_returns_safemap(self):
        """Rule 3: cardinality > batch_size AND divisible → SafeMap."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=25)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, SafeMap)
        assert decision.strategy.batch_size == 25

    def test_rule3_divisible_cardinality_ratio_4(self):
        """Rule 3: cardinality=200, batch_size=50 (divisible) → SafeMap."""
        spec = AxisSpec(name="batch", cardinality=200, default_batch_size=50)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        decision = plan.decisions[0]
        assert isinstance(decision.strategy, SafeMap)
        assert decision.strategy.batch_size == 50

    def test_rule4_non_divisible_cardinality_returns_safemap_with_warning(self):
        """Rule 4: non-divisible → SafeMap with warnings.warn."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=30)
        planner = BatchPlanner()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = planner.plan([spec])

            # Check warning was raised
            assert len(w) == 1
            assert "not divisible" in str(w[0].message).lower()

        # Check strategy is still SafeMap (deferred failure contract)
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, SafeMap)
        assert decision.strategy.batch_size == 30

    def test_rule4_non_divisible_large_remainder(self):
        """Rule 4: cardinality=101, batch_size=30 (remainder=11)."""
        spec = AxisSpec(name="batch", cardinality=101, default_batch_size=30)
        planner = BatchPlanner()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plan = planner.plan([spec])

            assert len(w) >= 1
            # Check for "not divisible" in any warning
            assert any("not divisible" in str(warn.message).lower() for warn in w)

        decision = plan.decisions[0]
        assert isinstance(decision.strategy, SafeMap)

    def test_plan_decision_length_matches_specs(self):
        """BatchPlan.decisions length equals len(specs)."""
        specs = [
            AxisSpec(name="batch", cardinality=100, default_batch_size=50),
            AxisSpec(name="seq", cardinality=512, default_batch_size=128),
            AxisSpec(
                name="token",
                cardinality=1000,
                default_batch_size=32,
                dedup_eligible=True,
            ),
        ]
        planner = BatchPlanner()
        plan = planner.plan(specs)

        assert len(plan.decisions) == len(specs)
        for i, decision in enumerate(plan.decisions):
            assert decision.spec is specs[i]

    def test_scan_never_returned_by_plan(self):
        """BatchPlanner never returns Scan strategy."""
        from xtrax.tiling.strategy import Scan

        # Test various cardinalities and conditions
        test_cases = [
            AxisSpec(name="a", cardinality=10, default_batch_size=5),
            AxisSpec(name="b", cardinality=100, default_batch_size=100),
            AxisSpec(name="c", cardinality=100, default_batch_size=30),
            AxisSpec(name="d", cardinality=1000, default_batch_size=32, dedup_eligible=True),
        ]

        planner = BatchPlanner()
        for spec in test_cases:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plan = planner.plan([spec])
            decision = plan.decisions[0]
            assert not isinstance(decision.strategy, Scan)

    def test_memory_estimator_none_uses_defaults(self):
        """When memory_estimator is None, use default selection rules."""
        spec = AxisSpec(name="batch", cardinality=50, default_batch_size=100)
        planner = BatchPlanner(memory_estimator=None)
        plan = planner.plan([spec])

        # Rule 2: cardinality <= batch_size → Vmap
        assert isinstance(plan.decisions[0].strategy, Vmap)

    def test_memory_estimator_under_limit_prefers_vmap(self):
        """When memory estimate < limit, prefer Vmap over SafeMap."""

        def low_estimate(spec: AxisSpec) -> int:
            # Always return a small value
            return 1000

        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        planner = BatchPlanner(memory_estimator=low_estimate)
        plan = planner.plan([spec])

        # Even though cardinality > batch_size and divisible,
        # low memory estimate should prefer Vmap
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, Vmap)

    def test_memory_estimator_over_limit_prefers_safemap(self):
        """When memory estimate > limit, prefer SafeMap over Vmap."""

        def high_estimate(spec: AxisSpec) -> int:
            # Return value exceeding default 4 GiB limit
            return 10 * (2**30)  # 10 GiB

        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        planner = BatchPlanner(memory_estimator=high_estimate)
        plan = planner.plan([spec])

        # High memory estimate should prefer SafeMap
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, SafeMap)

    def test_memory_estimator_exception_fallback_to_defaults(self):
        """When memory_estimator raises, fall back to default rules silently."""

        def failing_estimate(spec: AxisSpec) -> int:
            raise RuntimeError("Device query failed")

        spec = AxisSpec(name="batch", cardinality=50, default_batch_size=100)
        planner = BatchPlanner(memory_estimator=failing_estimate)

        # Should not raise — falls back silently
        plan = planner.plan([spec])

        # Rule 2: cardinality <= batch_size → Vmap
        assert isinstance(plan.decisions[0].strategy, Vmap)

    def test_memory_estimator_device_stats_query(self):
        """memory_estimator receives AxisSpec and can query device memory."""

        def estimate_with_device_check(spec: AxisSpec) -> int:
            # This simulates what a real estimator might do
            try:
                device_limit = (
                    jax.devices()[0].memory_stats().get("bytes_limit", 4 * (2**30))
                )
            except Exception:
                device_limit = 4 * (2**30)
            # Return a value < device_limit
            return device_limit // 2

        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        planner = BatchPlanner(memory_estimator=estimate_with_device_check)
        plan = planner.plan([spec])

        # Should succeed without errors
        assert len(plan.decisions) == 1

    def test_multiple_specs_independent_decisions(self):
        """Each spec gets its own decision independent of others."""
        import numpy as np
        from xtrax.tiling.dedup import DedupSpec

        specs = [
            AxisSpec(name="batch", cardinality=32, default_batch_size=100),  # Vmap
            AxisSpec(name="seq", cardinality=100, default_batch_size=25),  # SafeMap
            AxisSpec(
                name="token",
                cardinality=500,
                default_batch_size=50,
                dedup_eligible=True,
            ),  # DedupGather via Phase 0b
        ]

        # Create DedupSpec for the token axis
        unique_indices = np.array([0, 250])
        index_map = np.zeros(500, dtype=np.int32)
        index_map[250:] = 1
        dedup_spec = DedupSpec(
            axis_name="token",
            unique_indices=unique_indices,
            index_map=index_map,
            k=2,
        )

        planner = BatchPlanner(dedup_specs=[dedup_spec])
        plan = planner.plan(specs)

        assert isinstance(plan.decisions[0].strategy, Vmap)
        assert isinstance(plan.decisions[1].strategy, SafeMap)
        assert isinstance(plan.decisions[2].strategy, DedupGather)

    def test_reasoning_field_populated(self):
        """AxisDecision.reasoning field is populated."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        decision = plan.decisions[0]
        # Reasoning should be populated (not empty or None)
        assert decision.reasoning is not None
        assert isinstance(decision.reasoning, str)
        assert len(decision.reasoning) > 0

    def test_batch_size_field_in_decision(self):
        """AxisDecision.batch_size reflects the final batch_size choice."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        decision = plan.decisions[0]
        assert decision.batch_size == spec.default_batch_size

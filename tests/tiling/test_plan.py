"""Tests for xtrax.tiling.plan — AxisSpec, AxisDecision, BatchPlan, BatchPlanner."""

import logging
import warnings

import jax
import pytest

from xtrax.tiling.estimators import DEFAULT_DEVICE_MEMORY_BYTES
from xtrax.tiling.plan import (
    AxisDecision,
    AxisSpec,
    BatchPlan,
    BatchPlanner,
)
from xtrax.tiling.strategy import Bucket, ChunkedMap, DedupGather, Vmap


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
            AxisSpec(name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(16, 8))

    def test_axis_spec_bucket_boundaries_duplicate_raises(self):
        """Duplicate bucket_boundaries (not strictly ascending) is rejected."""
        with pytest.raises(ValueError, match="strictly ascending"):
            AxisSpec(name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(8, 8, 16))

    def test_axis_spec_bucket_boundaries_non_positive_raises(self):
        """Non-positive bucket_boundaries is rejected."""
        with pytest.raises(ValueError, match="positive"):
            AxisSpec(name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(0, 8))


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

    def test_decision_for_returns_named_decision(self):
        """decision_for returns the AxisDecision (and its strategy) for a known axis."""
        batch = AxisSpec(name="batch", cardinality=8, default_batch_size=8)
        seq = AxisSpec(name="seq", cardinality=64, default_batch_size=8)
        plan = BatchPlanner().plan([batch, seq])
        decision = plan.decision_for("seq")
        assert decision.spec.name == "seq"
        assert isinstance(decision.strategy, ChunkedMap)
        assert decision is plan.decisions[1]

    def test_decision_for_unknown_axis_raises_keyerror(self):
        """Unknown axes raise KeyError naming the missing axis and the known ones."""
        spec = AxisSpec(name="batch", cardinality=8, default_batch_size=8)
        plan = BatchPlanner().plan([spec])
        with pytest.raises(KeyError, match="no decision for axis 'missing'") as excinfo:
            plan.decision_for("missing")
        assert "batch" in str(excinfo.value)

    def test_decision_for_exported_on_public_batchplan(self):
        """decision_for is on the BatchPlan exported from xtrax.tiling and xtrax."""
        import xtrax
        from xtrax.tiling import BatchPlan as PublicBatchPlan

        assert PublicBatchPlan.decision_for is BatchPlan.decision_for
        assert xtrax.BatchPlan.decision_for is BatchPlan.decision_for


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

    def test_phase0_carry_spec_collect_outputs_false_returns_whilecarry(self):
        """Phase 0: CarrySpec(collect_outputs=False) declared → WhileCarry."""
        from xtrax.tiling.carry import CarrySpec
        from xtrax.tiling.strategy import WhileCarry

        spec = AxisSpec(name="n_infer_steps", cardinality=10, default_batch_size=32)

        def body(carry):
            return carry + 1

        def cond(carry):
            return carry < 200

        carry_spec = CarrySpec(
            axis_name="n_infer_steps",
            init=0,
            transition=body,
            collect_outputs=False,
            cond=cond,
        )

        planner = BatchPlanner(carry_specs=[carry_spec])
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert decision.spec is spec
        assert isinstance(decision.strategy, WhileCarry)
        assert decision.strategy.init == 0
        assert decision.strategy.body is body
        assert decision.strategy.cond is cond
        assert "while-loop" in decision.reasoning
        assert "collect_outputs=False" in decision.reasoning

    def test_phase0_carry_spec_heterogeneous_axis_rejects_regardless_of_collect_outputs(self):
        """Phase 0: a CarrySpec (Scan or WhileCarry) on a heterogeneous axis always raises."""
        from xtrax.tiling.carry import CarrySpec

        spec = AxisSpec(name="n_infer_steps", cardinality=10, default_batch_size=32)

        carry_spec = CarrySpec(
            axis_name="n_infer_steps",
            init=0,
            transition=lambda carry: carry,
            collect_outputs=False,
            cond=lambda carry: carry < 10,
        )

        planner = BatchPlanner(carry_specs=[carry_spec], heterogeneous_axes={"n_infer_steps"})
        with pytest.raises(ValueError, match="heterogeneous"):
            planner.plan([spec])

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
        """Rule 3: cardinality > batch_size AND divisible → ChunkedMap."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=25)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        assert len(plan.decisions) == 1
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, ChunkedMap)
        assert decision.strategy.batch_size == 25

    def test_rule3_divisible_cardinality_ratio_4(self):
        """Rule 3: cardinality=200, batch_size=50 (divisible) → ChunkedMap."""
        spec = AxisSpec(name="batch", cardinality=200, default_batch_size=50)
        planner = BatchPlanner()
        plan = planner.plan([spec])

        decision = plan.decisions[0]
        assert isinstance(decision.strategy, ChunkedMap)
        assert decision.strategy.batch_size == 50

    def test_rule4_non_divisible_cardinality_returns_safemap_without_warning(self):
        """#5565: non-divisible → ChunkedMap, no warning; the final chunk is ragged."""
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=30)
        planner = BatchPlanner()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = planner.plan([spec])

        decision = plan.decisions[0]
        assert isinstance(decision.strategy, ChunkedMap)
        assert decision.strategy.batch_size == 30
        assert "ragged final chunk of 10" in decision.reasoning

    def test_rule4_non_divisible_large_remainder(self):
        """Rule 4: cardinality=101, batch_size=30 (remainder=11)."""
        spec = AxisSpec(name="batch", cardinality=101, default_batch_size=30)
        planner = BatchPlanner()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = planner.plan([spec])

        decision = plan.decisions[0]
        assert isinstance(decision.strategy, ChunkedMap)
        assert "ragged final chunk of 11" in decision.reasoning

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
        """When memory estimate < limit, prefer Vmap over ChunkedMap."""

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

    def test_memory_estimator_under_limit_prefers_vmap_for_a_non_divisible_axis(self):
        """#5565: divisibility no longer changes the decision. The old Rule 5 forced
        ChunkedMap here (only because dispatch was going to fail); now a non-divisible
        axis the estimator says fits gets Vmap, exactly like a divisible one."""

        def low_estimate(spec: AxisSpec) -> int:
            return 1000

        spec = AxisSpec(name="batch", cardinality=101, default_batch_size=50)
        decision = BatchPlanner(memory_estimator=low_estimate).plan([spec]).decisions[0]
        assert isinstance(decision.strategy, Vmap)
        assert "ragged final chunk of 1" in decision.reasoning

    def test_memory_estimator_over_limit_prefers_safemap(self):
        """When memory estimate > limit, prefer ChunkedMap over Vmap."""

        def high_estimate(spec: AxisSpec) -> int:
            # Return value exceeding default 4 GiB limit
            return 10 * (2**30)  # 10 GiB

        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        planner = BatchPlanner(memory_estimator=high_estimate)
        plan = planner.plan([spec])

        # High memory estimate should prefer ChunkedMap
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, ChunkedMap)

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
                device_limit = jax.devices()[0].memory_stats().get("bytes_limit", 4 * (2**30))
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
            AxisSpec(name="seq", cardinality=100, default_batch_size=25),  # ChunkedMap
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
        assert isinstance(plan.decisions[1].strategy, ChunkedMap)
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

    def test_memory_estimator_uses_reported_device_limit(self, monkeypatch):
        """A reported bytes_limit replaces the 4 GiB fallback (fraction 1.0)."""
        seen: dict[str, float] = {}

        def fake_budget(fraction: float = 0.9, device=None) -> int:
            seen["fraction"] = fraction
            return 500

        monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", fake_budget)
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)
        decision = BatchPlanner(memory_estimator=lambda spec: 1000).plan([spec]).decisions[0]
        assert seen["fraction"] == 1.0
        # 1000 bytes exceeds the reported 500 and is far below 4 GiB.
        assert isinstance(decision.strategy, ChunkedMap)

    def test_missing_device_stats_logs_documented_4gib_default(self, monkeypatch, caplog):
        """No bytes_limit: log once and keep the 4 GiB comparison."""

        def no_stats(fraction: float = 0.9, device=None) -> int:
            raise RuntimeError("no stats")

        monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", no_stats)
        monkeypatch.setattr("xtrax.tiling.plan._default_device_limit_logged", False)
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)

        def high(spec: AxisSpec) -> int:
            return 10 * (2**30)

        with caplog.at_level(logging.INFO, logger="xtrax.tiling.plan"):
            first = BatchPlanner(memory_estimator=high).plan([spec]).decisions[0]
            second = BatchPlanner(memory_estimator=high).plan([spec]).decisions[0]
        assert isinstance(first.strategy, ChunkedMap)
        assert isinstance(second.strategy, ChunkedMap)
        messages = [record.message for record in caplog.records if "4 GiB" in record.message]
        assert len(messages) == 1

    def test_fallback_limit_is_4gib_and_equal_estimate_stays_vmap(self, monkeypatch):
        """Documented 4 GiB fallback. Equal-to-limit stays Vmap; over the limit chunks."""
        assert DEFAULT_DEVICE_MEMORY_BYTES == 4 * 2**30

        def no_stats(fraction: float = 0.9, device=None) -> int:
            raise RuntimeError("no stats")

        monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", no_stats)
        monkeypatch.setattr("xtrax.tiling.plan._default_device_limit_logged", False)
        spec = AxisSpec(name="batch", cardinality=100, default_batch_size=50)

        def strategy_for(estimate: int):
            planner = BatchPlanner(memory_estimator=lambda _spec, estimate=estimate: estimate)
            return planner.plan([spec]).decisions[0].strategy

        assert isinstance(strategy_for(DEFAULT_DEVICE_MEMORY_BYTES), Vmap)
        assert isinstance(strategy_for(DEFAULT_DEVICE_MEMORY_BYTES + 1), ChunkedMap)
        assert isinstance(strategy_for(5 * 2**30), ChunkedMap)


class TestVaryingInputs:
    """Which named inputs vary along a mapped axis (#2598).

    The declaration lives on AxisSpec: BatchPlan stores that spec on each
    decision, and plan/explain already surface spec fields. Inputs omitted
    from the per-axis list are invariant along that axis.
    """

    def test_declaration_round_trips_through_plan_and_explain(self, capsys):
        """A valid varying-input list survives planning and shows up in plan/explain."""
        from types import SimpleNamespace

        from xtrax.cli.emit import emit
        from xtrax.cli.plan import print_plan_summary
        from xtrax.eda.explain import explain_plan
        from xtrax.tiling.plan import declare_varying_inputs

        bare = AxisSpec(name="batch", cardinality=8, default_batch_size=4)
        specs = declare_varying_inputs(
            (bare, AxisSpec(name="sequence", cardinality=16, default_batch_size=8)),
            {"batch": ("x", "x")},
            input_names=("params", "x"),
        )
        assert specs[0].varying_inputs == ("x",)
        assert specs[1].varying_inputs == ()

        plan = BatchPlanner().plan(specs)
        assert plan.decision_for("batch").spec.varying_inputs == ("x",)
        assert plan.decision_for("sequence").spec.varying_inputs == ()

        stats = explain_plan(plan)
        by_name = {entry["name"]: entry for entry in stats["axes"]}
        assert by_name["batch"]["varying_inputs"] == ["x"]
        assert by_name["sequence"]["varying_inputs"] == []

        print_plan_summary(plan)
        plan_text = capsys.readouterr().out
        assert "Varying inputs: x" in plan_text
        assert "Varying inputs: (none)" in plan_text

        emit(stats, plan, "text")
        explain_text = capsys.readouterr().out
        assert "Varying inputs: x" in explain_text
        assert "Varying inputs: (none)" in explain_text

        # A spec that only has the AxisSpecLike fields still explains.
        foreign = SimpleNamespace(name="batch", cardinality=4)
        decision = SimpleNamespace(spec=foreign, batch_size=4, reasoning="vmap", strategy=Vmap())
        foreign_stats = explain_plan(SimpleNamespace(decisions=(decision,)))
        assert foreign_stats["axes"][0]["varying_inputs"] == []

    def test_list_of_varying_inputs_is_stored_as_a_tuple(self):
        """A list is coerced so the frozen spec stays hashable."""
        spec = AxisSpec(
            name="batch",
            cardinality=4,
            default_batch_size=2,
            varying_inputs=["x", "mask"],
        )
        assert spec.varying_inputs == ("x", "mask")
        assert hash(spec) == hash(spec)

    def test_string_varying_inputs_raises(self):
        """A bare string is not a sequence of input names."""
        with pytest.raises(ValueError, match="varying_inputs"):
            AxisSpec(
                name="batch",
                cardinality=4,
                default_batch_size=2,
                varying_inputs="x",
            )

    def test_unknown_axis_raises(self):
        """Naming an axis that is not in the specs raises."""
        from xtrax.tiling.plan import declare_varying_inputs

        specs = (AxisSpec(name="batch", cardinality=4, default_batch_size=2),)
        with pytest.raises(ValueError, match="unknown axis 'token'") as exc:
            declare_varying_inputs(specs, {"token": ("x",)}, input_names=("x",))
        assert "batch" in str(exc.value)

    def test_unknown_input_raises(self):
        """Naming an input that is not in the known inputs raises."""
        from xtrax.tiling.plan import declare_varying_inputs

        specs = (AxisSpec(name="batch", cardinality=4, default_batch_size=2),)
        with pytest.raises(ValueError, match="unknown input 'weight'") as exc:
            declare_varying_inputs(specs, {"batch": ("weight",)}, input_names=("params", "x"))
        assert "params" in str(exc.value)
        assert "x" in str(exc.value)

    def test_unknown_input_with_empty_catalog_lists_none(self):
        """An empty input catalog still names the unknown input."""
        from xtrax.tiling.plan import declare_varying_inputs

        specs = (AxisSpec(name="batch", cardinality=4, default_batch_size=2),)
        with pytest.raises(ValueError, match="unknown input 'x'") as exc:
            declare_varying_inputs(specs, {"batch": ("x",)}, input_names=())
        assert "(none)" in str(exc.value)

    def test_unknown_axis_with_no_specs_lists_none(self):
        """An empty spec list still names the unknown axis."""
        from xtrax.tiling.plan import declare_varying_inputs

        with pytest.raises(ValueError, match="unknown axis 'batch'") as exc:
            declare_varying_inputs((), {"batch": ("x",)}, input_names=())
        assert "(none)" in str(exc.value)

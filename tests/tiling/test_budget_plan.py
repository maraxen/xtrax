"""Tests for BatchPlanner joint-budget mode (MemoryBudget greedy demotion).

Spec: .praxia/docs/specs/260706_joint-budget-batch-planner.md (AC1-AC10).
"""

import numpy as np
import pytest

from xtrax.tiling import (
    AxisSpec,
    BatchPlanner,
    BudgetInfeasibleError,
    CarrySpec,
    ChunkedMap,
    MemoryBudget,
    Scan,
    Vmap,
)
from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.roles import AmbiguousAxisError, AxisRole
from xtrax.tiling.strategy import Bucket, DedupGather


def _spec(name: str, cardinality: int = 1024, batch_size: int = 256, **kwargs) -> AxisSpec:
    return AxisSpec(name=name, cardinality=cardinality, default_batch_size=batch_size, **kwargs)


def _per_strategy_estimator(vmap_cost: int = 100, other_cost: int = 10):
    """Estimator charging vmap_cost per Vmap axis and other_cost otherwise."""

    calls: list[tuple] = []

    def estimate(decisions) -> int:
        calls.append(tuple(type(d.strategy).__name__ for d in decisions))
        return sum(vmap_cost if isinstance(d.strategy, Vmap) else other_cost for d in decisions)

    estimate.calls = calls
    return estimate


class TestMemoryBudgetValidation:
    """AC1: MemoryBudget validates fields at construction."""

    def test_valid_budget_constructs(self) -> None:
        budget = MemoryBudget(bytes=1024, estimate=lambda decisions: 0)
        assert budget.bytes == 1024

    def test_zero_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MemoryBudget(bytes=0, estimate=lambda decisions: 0)

    def test_negative_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MemoryBudget(bytes=-1, estimate=lambda decisions: 0)

    def test_bool_bytes_rejected(self) -> None:
        with pytest.raises(TypeError, match="int"):
            MemoryBudget(bytes=True, estimate=lambda decisions: 0)

    def test_non_int_bytes_rejected(self) -> None:
        with pytest.raises(TypeError, match="int"):
            MemoryBudget(bytes=1.5, estimate=lambda decisions: 0)

    def test_non_callable_estimate_rejected(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            MemoryBudget(bytes=1024, estimate=42)


class TestPlannerBudgetConstruction:
    """AC2: budget and memory_estimator are mutually exclusive."""

    def test_budget_with_memory_estimator_rejected(self) -> None:
        budget = MemoryBudget(bytes=1024, estimate=lambda decisions: 0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            BatchPlanner(memory_estimator=lambda spec: 0, budget=budget)

    def test_budget_alone_accepted(self) -> None:
        budget = MemoryBudget(bytes=1024, estimate=lambda decisions: 0)
        planner = BatchPlanner(budget=budget)
        assert planner.budget is budget


class TestUnderBudget:
    """AC3: a fitting plan keeps every eligible axis at Vmap."""

    def test_all_vmap_including_large_axes(self) -> None:
        estimate = _per_strategy_estimator()
        planner = BatchPlanner(budget=MemoryBudget(bytes=10_000, estimate=estimate))
        plan = planner.plan([_spec("a"), _spec("b"), _spec("small", cardinality=8)])
        assert all(isinstance(d.strategy, Vmap) for d in plan.decisions)
        assert len(estimate.calls) >= 1

    def test_default_rules_would_have_chosen_safemap(self) -> None:
        # Same specs without budget: cardinality > batch_size → ChunkedMap.
        plan = BatchPlanner().plan([_spec("a")])
        assert isinstance(plan.decisions[0].strategy, ChunkedMap)


class TestGreedyDemotion:
    """AC4: demotion proceeds in given order, re-estimating each step."""

    def test_stops_at_first_fit(self) -> None:
        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=210, estimate=estimate))
        plan = planner.plan([_spec("a"), _spec("b"), _spec("c")])
        strategies = [type(d.strategy) for d in plan.decisions]
        assert strategies == [ChunkedMap, Vmap, Vmap]
        # initial (300) + after demoting a (210, fits)
        assert len(estimate.calls) == 2

    def test_demotes_multiple_in_given_order(self) -> None:
        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=130, estimate=estimate))
        plan = planner.plan([_spec("a"), _spec("b"), _spec("c")])
        strategies = [type(d.strategy) for d in plan.decisions]
        assert strategies == [ChunkedMap, ChunkedMap, Vmap]
        assert "step 1" in plan.decisions[0].reasoning
        assert "step 2" in plan.decisions[1].reasoning

    def test_caller_order_expresses_priority(self) -> None:
        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=210, estimate=estimate))
        plan = planner.plan([_spec("c"), _spec("a"), _spec("b")])
        demoted = [d.spec.name for d in plan.decisions if isinstance(d.strategy, ChunkedMap)]
        assert demoted == ["c"]


class TestDemotionNoOpExclusion:
    """AC5: cardinality <= batch_size axes are never demotion candidates."""

    def test_small_axes_stay_vmap_when_over_budget(self) -> None:
        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=110, estimate=estimate))
        plan = planner.plan([_spec("small", cardinality=8), _spec("big")])
        by_name = {d.spec.name: d for d in plan.decisions}
        assert isinstance(by_name["small"].strategy, Vmap)
        assert isinstance(by_name["big"].strategy, ChunkedMap)
        assert "no-op" in by_name["small"].reasoning

    def test_only_small_axes_and_over_budget_is_infeasible(self) -> None:
        planner = BatchPlanner(budget=MemoryBudget(bytes=100, estimate=lambda decisions: 1_000))
        with pytest.raises(BudgetInfeasibleError, match="0 candidate"):
            planner.plan([_spec("small", cardinality=8)])


class TestFixedAxes:
    """AC6: carry/dedup/bucket axes are fixed, estimated, never demoted."""

    def _carry(self, name: str) -> CarrySpec:
        return CarrySpec(axis_name=name, init=0.0, transition=lambda c, x: (c, x))

    def _dedup(self, name: str) -> DedupSpec:
        return DedupSpec(
            axis_name=name,
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
        )

    def test_fixed_axes_reach_estimator_and_stay_fixed(self) -> None:
        seen: list[tuple] = []

        def estimate(decisions) -> int:
            seen.append(tuple(type(d.strategy).__name__ for d in decisions))
            return 0

        planner = BatchPlanner(
            budget=MemoryBudget(bytes=100, estimate=estimate),
            carry_specs=[self._carry("time")],
            dedup_specs=[self._dedup("seq")],
        )
        plan = planner.plan(
            [
                _spec("time", cardinality=16, batch_size=16),
                _spec("seq", cardinality=4, batch_size=4),
                _spec("len", bucket_boundaries=(128, 512)),
                _spec("batch"),
            ]
        )
        strategies = [type(d.strategy) for d in plan.decisions]
        assert strategies == [Scan, DedupGather, Bucket, Vmap]
        assert seen[0] == ("Scan", "DedupGather", "Bucket", "Vmap")

    def test_heterogeneous_carry_still_rejected(self) -> None:
        planner = BatchPlanner(
            budget=MemoryBudget(bytes=100, estimate=lambda decisions: 0),
            carry_specs=[self._carry("time")],
            heterogeneous_axes={"time"},
        )
        with pytest.raises(ValueError, match="heterogeneous"):
            planner.plan([_spec("time", cardinality=16, batch_size=16)])

    def test_unknown_role_still_rejected(self) -> None:
        planner = BatchPlanner(budget=MemoryBudget(bytes=100, estimate=lambda d: 0))
        with pytest.raises(AmbiguousAxisError):
            planner.plan([_spec("mystery", role=AxisRole.UNKNOWN)])


class TestHeterogeneousBudget:
    """AxisSpec.heterogeneous is never Vmap in joint-budget mode."""

    def test_heterogeneous_stays_chunked_when_budget_fits(self) -> None:
        seen: list[dict[str, type]] = []

        def estimate(decisions) -> int:
            seen.append({d.spec.name: type(d.strategy) for d in decisions})
            return 0

        planner = BatchPlanner(budget=MemoryBudget(bytes=10_000, estimate=estimate))
        plan = planner.plan(
            [
                _spec("het", cardinality=8, batch_size=32, heterogeneous=True),
                _spec("hom", cardinality=1024, batch_size=256),
            ]
        )
        by_name = {d.spec.name: d for d in plan.decisions}
        assert isinstance(by_name["het"].strategy, ChunkedMap)
        assert by_name["het"].strategy.batch_size == 32
        assert isinstance(by_name["hom"].strategy, Vmap)
        # Pending reasoning also contains "heterogeneous" and "joint-budget".
        # The finalized branch is the one that records the measured bytes.
        assert (
            "Vmap is invalid when element shapes vary (final estimate 0 B, budget 10000 B)"
            in by_name["het"].reasoning
        )
        assert seen
        assert all(snapshot["het"] is ChunkedMap for snapshot in seen)

    def test_heterogeneous_not_demoted_from_vmap_when_over_budget(self) -> None:
        seen: list[tuple] = []

        def estimate(decisions) -> int:
            seen.append(tuple(type(d.strategy).__name__ for d in decisions))
            return 10_000

        planner = BatchPlanner(budget=MemoryBudget(bytes=100, estimate=estimate))
        with pytest.raises(BudgetInfeasibleError, match="het=ChunkedMap"):
            planner.plan([_spec("het", heterogeneous=True), _spec("hom")])
        assert seen[0] == ("ChunkedMap", "Vmap")
        assert seen[-1] == ("ChunkedMap", "ChunkedMap")
        assert all(snapshot[0] == "ChunkedMap" for snapshot in seen)

    def test_heterogeneous_only_over_budget_names_chunked_map(self) -> None:
        planner = BatchPlanner(budget=MemoryBudget(bytes=100, estimate=lambda decisions: 10_000))
        with pytest.raises(BudgetInfeasibleError, match=r"het=ChunkedMap") as excinfo:
            planner.plan([_spec("het", heterogeneous=True)])
        assert "0 candidate" in str(excinfo.value)

    def test_heterogeneous_dedup_kept_when_budget_fits(self) -> None:
        dedup = DedupSpec(
            axis_name="het",
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
        )
        planner = BatchPlanner(
            budget=MemoryBudget(bytes=100, estimate=lambda decisions: 0),
            dedup_specs=[dedup],
        )
        plan = planner.plan([_spec("het", cardinality=4, batch_size=4, heterogeneous=True)])
        assert isinstance(plan.decisions[0].strategy, DedupGather)

    def test_over_budget_heterogeneous_dedup_falls_back_to_chunked_map(self) -> None:
        """A DedupGather that exceeds the budget is dropped; the axis stays ChunkedMap.

        cardinality <= batch_size would be Vmap for a homogeneous axis. Heterogeneous
        still forbids that after the dedup fallback.
        """
        dedup = DedupSpec(
            axis_name="het",
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
        )

        def estimate(decisions) -> int:
            if any(isinstance(decision.strategy, DedupGather) for decision in decisions):
                return 10_000
            return 1

        planner = BatchPlanner(
            budget=MemoryBudget(bytes=100, estimate=estimate),
            dedup_specs=[dedup],
        )
        with pytest.warns(RuntimeWarning, match="without dedup"):
            plan = planner.plan([_spec("het", cardinality=4, batch_size=4, heterogeneous=True)])
        decision = plan.decisions[0]
        assert isinstance(decision.strategy, ChunkedMap)
        assert decision.strategy.batch_size == 4
        assert "DedupSpec dropped" in decision.reasoning
        assert "final estimate 1 B, budget 100 B" in decision.reasoning


class TestInfeasible:
    """AC7: exhausted candidates over budget raise BudgetInfeasibleError."""

    def test_error_names_budget_estimate_and_axes(self) -> None:
        planner = BatchPlanner(budget=MemoryBudget(bytes=100, estimate=lambda decisions: 10_000))
        with pytest.raises(BudgetInfeasibleError) as excinfo:
            planner.plan([_spec("a"), _spec("b")])
        message = str(excinfo.value)
        assert "estimate 10000 B" in message
        assert "budget 100 B" in message
        assert "a=ChunkedMap" in message
        assert "b=ChunkedMap" in message


class TestEstimatorErrors:
    """AC8: estimator exceptions propagate unchanged (no silent fallback)."""

    def test_estimator_exception_propagates(self) -> None:
        def broken(decisions) -> int:
            raise RuntimeError("estimator exploded")

        planner = BatchPlanner(budget=MemoryBudget(bytes=100, estimate=broken))
        with pytest.raises(RuntimeError, match="estimator exploded"):
            planner.plan([_spec("a")])


class TestDivisibilityWarning:
    """#5565: demoting a non-divisible axis no longer warns -- dispatch handles the
    ragged final chunk, so there is no deferred failure left to warn about."""

    def test_non_divisible_demotion_does_not_warn(self) -> None:
        import warnings

        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=50, estimate=estimate))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = planner.plan([_spec("ragged", cardinality=1000, batch_size=256)])
        assert type(plan.decisions[0].strategy).__name__ == "ChunkedMap"

    def test_divisible_demotion_does_not_warn(self) -> None:
        import warnings

        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=50, estimate=estimate))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            planner.plan([_spec("even", cardinality=1024, batch_size=256)])


class TestPlanShape:
    """AC10: order preserved, reasoning non-empty and numeric in budget mode."""

    def test_order_and_reasoning(self) -> None:
        estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
        planner = BatchPlanner(budget=MemoryBudget(bytes=210, estimate=estimate))
        specs = [_spec("a"), _spec("small", cardinality=8), _spec("b")]
        plan = planner.plan(specs)
        assert [d.spec.name for d in plan.decisions] == ["a", "small", "b"]
        for decision in plan.decisions:
            assert decision.reasoning
            assert "budget" in decision.reasoning

    def test_deterministic(self) -> None:
        def run() -> list[str]:
            estimate = _per_strategy_estimator(vmap_cost=100, other_cost=10)
            planner = BatchPlanner(budget=MemoryBudget(bytes=130, estimate=estimate))
            plan = planner.plan([_spec("a"), _spec("b"), _spec("c")])
            return [type(d.strategy).__name__ for d in plan.decisions]

        assert run() == run()

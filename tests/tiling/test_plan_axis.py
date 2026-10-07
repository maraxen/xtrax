"""Tests for plan_axis, the single-axis BatchPlanner wrapper."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.tiling import (
    AxisSpec,
    BatchPlanner,
    BudgetInfeasibleError,
    ChunkedMap,
    MemoryBudget,
    Vmap,
    plan_axis,
)
from xtrax.tiling.estimators import lowered_memory_estimate


def _spec(**kwargs) -> AxisSpec:
    fields = {"name": "batch", "cardinality": 8, "default_batch_size": 2}
    fields.update(kwargs)
    return AxisSpec(**fields)


class TestPlanAxisIntEstimate:
    """Int estimate is bytes per live element, scaled by cardinality or tile."""

    def test_under_budget_keeps_vmap(self) -> None:
        # Vmap live cost = 100 * 8 = 800; ChunkedMap would be 100 * 2 = 200.
        strategy = plan_axis(_spec(), estimate=100, budget=800)
        assert isinstance(strategy, Vmap)

    def test_over_full_cardinality_demotes_to_chunked_map(self) -> None:
        # 100 * 8 = 800 > 500 >= 100 * 2.
        strategy = plan_axis(_spec(), estimate=100, budget=500)
        assert isinstance(strategy, ChunkedMap)
        assert strategy.batch_size == 2

    def test_below_tile_is_infeasible(self) -> None:
        with pytest.raises(BudgetInfeasibleError):
            plan_axis(_spec(), estimate=100, budget=100)

    def test_heterogeneous_axis_is_never_vmap(self) -> None:
        strategy = plan_axis(
            _spec(cardinality=4, default_batch_size=32, heterogeneous=True),
            estimate=1,
            budget=10**9,
        )
        assert isinstance(strategy, ChunkedMap)
        assert strategy.batch_size == 32

    def test_matches_batch_planner_strategy(self) -> None:
        spec = _spec()
        strategy = plan_axis(spec, estimate=100, budget=500)

        def joint(decisions) -> int:
            return sum(
                (d.spec.cardinality if isinstance(d.strategy, Vmap) else d.batch_size) * 100
                for d in decisions
            )

        direct = BatchPlanner(budget=MemoryBudget(bytes=500, estimate=joint)).plan([spec])
        assert type(strategy) is type(direct.decisions[0].strategy)

    def test_negative_estimate_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes per element"):
            plan_axis(_spec(), estimate=-1, budget=1000)

    def test_abstract_args_with_int_estimate_rejected(self) -> None:
        with pytest.raises(TypeError, match="abstract args"):
            plan_axis(_spec(), object(), estimate=10, budget=1000)


class TestPlanAxisCallableEstimate:
    """Callable estimate is measured with lowered_memory_estimate, then scaled."""

    def test_scales_lowered_bytes_like_the_int_form(self) -> None:
        def add_one(x):
            return x + 1.0

        abstract = jax.ShapeDtypeStruct((4,), jnp.float32)
        measured = lowered_memory_estimate(add_one, abstract)
        assert measured > 0
        spec = _spec()
        kept = plan_axis(spec, abstract, estimate=add_one, budget=measured * spec.cardinality)
        assert isinstance(kept, Vmap)
        demoted = plan_axis(
            spec,
            abstract,
            estimate=add_one,
            budget=measured * spec.default_batch_size + 1,
        )
        assert isinstance(demoted, ChunkedMap)
        assert demoted.batch_size == spec.default_batch_size

    def test_memo_skips_recompile(self, monkeypatch) -> None:
        def add_one(x):
            return x + 1.0

        abstract = jax.ShapeDtypeStruct((4,), jnp.float32)
        memo: dict = {}
        plan_axis(_spec(), abstract, estimate=add_one, budget=10**18, memo=memo)
        assert len(memo) == 1

        import xtrax.tiling.plan as plan_mod

        def _boom(*args, **kwargs):
            raise AssertionError("lowered_memory_estimate should have been served from memo")

        monkeypatch.setattr(plan_mod, "lowered_memory_estimate", _boom)
        strategy = plan_axis(_spec(), abstract, estimate=add_one, budget=10**18, memo=memo)
        assert isinstance(strategy, Vmap)

    def test_callable_without_abstract_args_rejected(self) -> None:
        with pytest.raises(TypeError, match="abstract args"):
            plan_axis(_spec(), estimate=lambda x: x, budget=1000)

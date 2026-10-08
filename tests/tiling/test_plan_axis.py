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
        # 1024 * 4 = 4096 B under Vmap; ChunkedMap keeps one tile of 64 * 4 = 256 B.
        spec = _spec(cardinality=1024, default_batch_size=64)
        per_element = 4
        budget = 256
        strategy = plan_axis(spec, estimate=per_element, budget=budget)

        def joint(decisions) -> int:
            total = 1
            for decision in decisions:
                extent = (
                    decision.spec.cardinality
                    if isinstance(decision.strategy, Vmap)
                    else decision.batch_size
                )
                total *= extent
            return per_element * total

        direct = BatchPlanner(budget=MemoryBudget(bytes=budget, estimate=joint)).plan([spec])
        direct_strategy = direct.decisions[0].strategy
        assert isinstance(strategy, ChunkedMap)
        assert isinstance(direct_strategy, ChunkedMap)
        assert strategy.batch_size == 64
        assert strategy.batch_size == direct_strategy.batch_size

    def test_bool_estimate_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes per element"):
            plan_axis(_spec(), estimate=True, budget=1000)
        with pytest.raises(ValueError, match="bytes per element"):
            plan_axis(_spec(), estimate=False, budget=1000)

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

    def test_memo_distinguishes_shape_and_dtype(self, monkeypatch) -> None:
        """Shape and dtype are part of the memo key, so each signature is measured once."""

        def add_one(x):
            return x + 1.0

        measured = {
            ((4,), "float32"): 16,
            ((64,), "float32"): 256,
            ((4,), "float16"): 8,
        }

        def fake_lowered(fn, *args):
            arg = args[0]
            return measured[(tuple(arg.shape), jnp.dtype(arg.dtype).name)]

        import xtrax.tiling.plan as plan_mod

        monkeypatch.setattr(plan_mod, "lowered_memory_estimate", fake_lowered)
        memo: dict = {}
        spec = _spec()
        abstracts = (
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((64,), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float16),
        )
        for abstract in abstracts:
            plan_axis(spec, abstract, estimate=add_one, budget=10**18, memo=memo)

        assert len(memo) == 3
        assert set(memo.values()) == {16, 256, 8}

    def test_memo_does_not_collide_across_functions(self, monkeypatch) -> None:
        """Two callables with the same abstract args keep distinct memo entries."""

        def narrow(x):
            return x + 1.0

        def wide(x):
            return x + 2.0

        def fake_lowered(fn, *args):
            return 10 if fn is narrow else 400

        import xtrax.tiling.plan as plan_mod

        monkeypatch.setattr(plan_mod, "lowered_memory_estimate", fake_lowered)
        abstract = jax.ShapeDtypeStruct((4,), jnp.float32)
        memo: dict = {}
        plan_axis(_spec(), abstract, estimate=narrow, budget=10**18, memo=memo)
        plan_axis(_spec(), abstract, estimate=wide, budget=10**18, memo=memo)

        assert len(memo) == 2
        assert set(memo.values()) == {10, 400}
        assert {key[0] for key in memo} == {narrow, wide}

    def test_callable_without_abstract_args_rejected(self) -> None:
        with pytest.raises(TypeError, match="abstract args"):
            plan_axis(_spec(), estimate=lambda x: x, budget=1000)

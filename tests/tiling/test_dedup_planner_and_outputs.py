"""#5175 (plan() dedup collisions; DedupGather under MemoryBudget) and #5217
(numeric output-equivalence of the dedup path)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.dedup_synthesis import (
    DedupOutputMismatchError,
    DedupSpecCollisionError,
    DedupSpecVerificationError,
    verify_dedup_outputs,
)
from xtrax.tiling.plan import AxisSpec, BatchPlanner
from xtrax.tiling.strategy import DedupGather, SafeMap


def _ds(k: int, n: int = 100) -> DedupSpec:
    return DedupSpec(
        axis_name="b",
        unique_indices=np.arange(k, dtype=np.int32),
        index_map=(np.arange(n) % k).astype(np.int32),
        k=k,
    )


SPEC = AxisSpec(name="b", cardinality=100, default_batch_size=10, dedup_eligible=True)


# --- #5175 part 1 ------------------------------------------------------------------------


def test_duplicate_dedup_specs_raise_instead_of_last_wins():
    """Measured before the fix: k=30 then k=5 silently planned k=5."""
    with pytest.raises(DedupSpecCollisionError, match="'b'"):
        BatchPlanner(dedup_specs=[_ds(30), _ds(5)]).plan([SPEC])


def test_single_dedup_spec_still_plans_dedup_gather():
    plan = BatchPlanner(dedup_specs=[_ds(30)]).plan([SPEC])
    assert isinstance(plan.decisions[0].strategy, DedupGather)
    assert plan.decisions[0].batch_size == 30


# --- #5175 part 2: DedupGather never makes a feasible budget plan infeasible ------------


def _cost(decisions) -> int:  # noqa: ANN001
    total = 0
    for d in decisions:
        if isinstance(d.strategy, DedupGather):
            total += 1_000  # a working set the estimate says is too big
        elif isinstance(d.strategy, SafeMap):
            total += d.batch_size * 10
        else:  # Vmap
            total += d.spec.cardinality * 10
    return total


def test_without_dedup_this_budget_is_feasible():
    """The baseline the next test must not regress: SafeMap(10) costs 100 <= 150."""
    plan = BatchPlanner(budget=MemoryBudget(bytes=150, estimate=_cost)).plan([SPEC])
    assert isinstance(plan.decisions[0].strategy, SafeMap)


def test_dedup_is_dropped_as_a_last_resort_rather_than_failing_the_plan():
    planner = BatchPlanner(dedup_specs=[_ds(30)], budget=MemoryBudget(bytes=150, estimate=_cost))
    with pytest.warns(RuntimeWarning, match="without dedup"):
        plan = planner.plan([SPEC])
    decision = plan.decisions[0]
    assert isinstance(decision.strategy, SafeMap)
    assert "DedupSpec dropped" in decision.reasoning


def test_dedup_is_kept_when_it_fits():
    planner = BatchPlanner(dedup_specs=[_ds(30)], budget=MemoryBudget(bytes=5_000, estimate=_cost))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plan = planner.plan([SPEC])
    assert isinstance(plan.decisions[0].strategy, DedupGather)


def test_still_infeasible_without_dedup_raises_and_says_so():
    planner = BatchPlanner(dedup_specs=[_ds(30)], budget=MemoryBudget(bytes=1, estimate=_cost))
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(BudgetInfeasibleError, match="DedupGather already dropped"),
    ):
        planner.plan([SPEC])


def test_other_axes_are_demoted_before_dedup_is_dropped():
    """Dedup is the LAST resort: an ordinary axis that can absorb the overrun goes first."""
    other = AxisSpec(name="o", cardinality=100, default_batch_size=10)

    def cost(decisions) -> int:  # noqa: ANN001
        by = {d.spec.name: d for d in decisions}
        dedup = 50 if isinstance(by["b"].strategy, DedupGather) else 999
        o = by["o"]
        return dedup + (100 if isinstance(o.strategy, SafeMap) else 1_000)

    planner = BatchPlanner(dedup_specs=[_ds(30)], budget=MemoryBudget(bytes=200, estimate=cost))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plan = planner.plan([SPEC, other])
    by = {d.spec.name: d for d in plan.decisions}
    assert isinstance(by["b"].strategy, DedupGather)
    assert isinstance(by["o"].strategy, SafeMap)


# --- #5217 --------------------------------------------------------------------------------


def _batch(n: int, k: int, d: int, seed: int = 0):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((k, d)).astype(np.float32)
    index_map = rng.integers(0, k, size=n).astype(np.int32)
    xs = jnp.asarray(base[index_map])
    uniq = np.array([np.flatnonzero(index_map == j)[0] for j in range(k)], dtype=np.int32)
    spec = DedupSpec(axis_name="b", unique_indices=uniq, index_map=index_map, k=k)
    w = jnp.asarray(rng.standard_normal((d, d)).astype(np.float32))
    return spec, xs, w


def test_outputs_equal_within_tolerance_even_where_not_bitwise_equal():
    """The case that makes 'never bitwise' necessary (measured 260930: 3.3e-6)."""
    spec, xs, w = _batch(2_000, 7, 128)

    def fn(r):  # noqa: ANN001, ANN202
        return jnp.tanh(r @ w).sum() + jnp.sin(r).mean()

    result = verify_dedup_outputs(spec, fn, xs)
    assert 0.0 < result.max_abs_error < 1e-4  # genuinely not bitwise-equal here
    assert (result.n_rows, result.k) == (2_000, 7)


def test_a_spec_that_merges_distinct_rows_is_caught():
    spec, xs, w = _batch(64, 4, 16)
    wrong = DedupSpec(
        axis_name="b",
        unique_indices=spec.unique_indices,
        index_map=(spec.index_map + 1) % spec.k,  # every row mapped to the wrong canon
        k=spec.k,
    )
    with pytest.raises(DedupOutputMismatchError) as info:
        verify_dedup_outputs(wrong, lambda r: r @ w, xs)
    assert info.value.n_bad > 0
    assert info.value.max_abs_error > 1e-3


def test_integer_outputs_compare_exactly_and_nan_matches_nan():
    spec, xs, _ = _batch(32, 4, 8)

    def fn(r):  # noqa: ANN001, ANN202
        return {"count": jnp.sum(r > 0).astype(jnp.int32), "nan": r[0] * jnp.nan}

    assert verify_dedup_outputs(spec, fn, xs).max_abs_error == 0.0


def test_structural_mismatch_is_reported_before_evaluating_fn():
    spec, xs, _ = _batch(32, 4, 8)
    called = []
    with pytest.raises(DedupSpecVerificationError):
        verify_dedup_outputs(spec, lambda r: called.append(1) or r, xs[:16])
    assert called == []


# --- review-driven cases ---------------------------------------------------------------


def test_equal_infinities_are_equal_and_inf_vs_finite_is_not():
    """Masked logits / log(0) produce +-inf on both paths; inf-inf is NaN, not a diff."""
    spec, xs, _ = _batch(32, 4, 8)
    ok = verify_dedup_outputs(spec, lambda r: jnp.where(r > 0, r, -jnp.inf), xs)
    assert ok.max_abs_error == 0.0
    wrong = DedupSpec(
        axis_name="b",
        unique_indices=spec.unique_indices,
        index_map=(spec.index_map + 1) % spec.k,
        k=spec.k,
    )
    with pytest.raises(DedupOutputMismatchError):
        verify_dedup_outputs(wrong, lambda r: jnp.where(r > 0, r, -jnp.inf), xs)


def test_eager_and_jitted_modes_both_verify():
    spec, xs, w = _batch(64, 4, 16)
    for jit in (True, False):
        assert verify_dedup_outputs(spec, lambda r: jnp.tanh(r @ w), xs, jit=jit).n_rows == 64


def test_fn_with_no_array_outputs_is_a_clear_error():
    spec, xs, _ = _batch(16, 4, 8)
    with pytest.raises(ValueError, match="no array leaves"):
        verify_dedup_outputs(spec, lambda r: {}, xs)


def _two_dedup_axes():  # noqa: ANN202
    a = AxisSpec(name="a", cardinality=100, default_batch_size=10, dedup_eligible=True)
    b = AxisSpec(name="b", cardinality=100, default_batch_size=10, dedup_eligible=True)
    specs = [
        DedupSpec(
            axis_name=n,
            unique_indices=np.arange(30, dtype=np.int32),
            index_map=(np.arange(100) % 30).astype(np.int32),
            k=30,
        )
        for n in ("a", "b")
    ]
    return a, b, specs


def test_dedup_axes_are_released_one_at_a_time_in_spec_order():
    """Releasing the first dedup axis is enough, so the second keeps its dedup."""
    a, b, specs = _two_dedup_axes()

    def cost(decisions) -> int:  # noqa: ANN001
        return sum(600 if isinstance(d.strategy, DedupGather) else 100 for d in decisions)

    planner = BatchPlanner(dedup_specs=specs, budget=MemoryBudget(bytes=800, estimate=cost))
    with pytest.warns(RuntimeWarning, match="'a'") as record:
        plan = planner.plan([a, b])
    by = {d.spec.name: d for d in plan.decisions}
    assert not isinstance(by["a"].strategy, DedupGather)
    assert isinstance(by["b"].strategy, DedupGather)
    assert not any("'b'" in str(w.message) for w in record)


def test_bucket_bounded_dedup_axis_falls_back_to_its_bucket_strategy():
    spec = AxisSpec(
        name="b",
        cardinality=100,
        default_batch_size=10,
        dedup_eligible=True,
        bucket_boundaries=(50, 100),
    )

    def cost(decisions) -> int:  # noqa: ANN001
        return 1_000 if any(isinstance(d.strategy, DedupGather) for d in decisions) else 10

    planner = BatchPlanner(dedup_specs=[_ds(30)], budget=MemoryBudget(bytes=100, estimate=cost))
    with pytest.warns(RuntimeWarning):
        plan = planner.plan([spec])
    assert type(plan.decisions[0].strategy).__name__ == "Bucket"
    assert "DedupSpec dropped" in plan.decisions[0].reasoning


def test_dropped_unknown_role_dedup_axis_still_fails_loud():
    from xtrax.tiling.roles import AmbiguousAxisError, AxisRole

    spec = AxisSpec(
        name="b",
        cardinality=100,
        default_batch_size=10,
        dedup_eligible=True,
        role=AxisRole.UNKNOWN,
    )
    planner = BatchPlanner(dedup_specs=[_ds(30)], budget=MemoryBudget(bytes=150, estimate=_cost))
    with pytest.raises(AmbiguousAxisError):
        planner.plan([spec])

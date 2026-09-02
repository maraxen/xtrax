"""Composer routing: the four supported strategies, and the two that are not."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export.composer import (
    ComposerError,
    MultiAxisCompositionError,
    UnsupportedStrategyError,
    build_traceable_callable,
    compose_single_axis,
)
from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap, WhileCarry


def _decision(name: str, strategy: object, *, cardinality: int = 8) -> AxisDecision:
    return AxisDecision(
        spec=AxisSpec(name=name, cardinality=cardinality, default_batch_size=0),
        batch_size=0,
        reasoning="test",
        strategy=strategy,
    )


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


class TestUnsupportedStrategies:
    def test_bucket_raises_naming_the_strategy(self):
        with pytest.raises(UnsupportedStrategyError, match="'Bucket'"):
            compose_single_axis(lambda x: x, _decision("a", Bucket(boundaries=(4, 8))))

    def test_bucket_points_at_the_supported_set(self):
        with pytest.raises(UnsupportedStrategyError, match="Vmap/SafeMap/Scan/DedupGather"):
            compose_single_axis(lambda x: x, _decision("a", Bucket(boundaries=(4, 8))))

    def test_bucket_explains_the_workaround(self):
        with pytest.raises(UnsupportedStrategyError, match="bucketize"):
            compose_single_axis(lambda x: x, _decision("a", Bucket(boundaries=(4, 8))))

    def test_while_carry_raises_naming_the_strategy(self):
        with pytest.raises(UnsupportedStrategyError, match="'WhileCarry'"):
            compose_single_axis(lambda x: x, _decision("a", WhileCarry()))

    def test_decision_without_a_strategy_raises_naming_the_axis(self):
        class _NoStrategy:
            strategy = None
            spec = AxisSpec(name="orphan", cardinality=4, default_batch_size=0)

        with pytest.raises(UnsupportedStrategyError, match="'orphan'"):
            compose_single_axis(lambda x: x, _NoStrategy())

    def test_while_carry_explains_the_workaround(self):
        with pytest.raises(UnsupportedStrategyError, match="unbounded trip count"):
            compose_single_axis(lambda x: x, _decision("a", WhileCarry()))


class TestMapAxes:
    @pytest.mark.parametrize("strategy", [Vmap(), SafeMap(batch_size=2)])
    def test_round_trips_a_map_axis(self, strategy):
        xs = jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3)
        composed = compose_single_axis(lambda x: x * 2.0, _decision("a", strategy))
        np.testing.assert_allclose(np.asarray(composed(xs)), np.asarray(xs) * 2.0)

    def test_traces_under_jit(self):
        xs = jnp.ones((8, 3), dtype=jnp.float32)
        composed = compose_single_axis(lambda x: x + 1.0, _decision("a", Vmap()))
        np.testing.assert_allclose(np.asarray(jax.jit(composed)(xs)), np.asarray(xs) + 1.0)

    def test_fuse_is_applied_after_the_axis(self):
        xs = jnp.ones((8, 3), dtype=jnp.float32)
        boundary = AxisBoundary(fuse=lambda ys: jnp.sum(ys, axis=0))
        composed = compose_single_axis(lambda x: x, _decision("a", Vmap()), boundary)
        np.testing.assert_allclose(np.asarray(composed(xs)), np.full((3,), 8.0))


class TestScanAxis:
    def test_returns_the_stacked_ys_not_the_carry(self):
        xs = jnp.arange(8, dtype=jnp.float32)
        composed = compose_single_axis(
            lambda c, x: (c + x, c),
            _decision("a", Scan(init=None)),
            scan_init=jnp.float32(0.0),
        )
        out = np.asarray(composed(xs))
        assert out.shape == (8,)
        np.testing.assert_allclose(out, np.cumsum(np.arange(8)) - np.arange(8))

    def test_falls_back_to_the_strategys_own_init(self):
        xs = jnp.arange(4, dtype=jnp.float32)
        composed = compose_single_axis(
            lambda c, x: (c + x, c), _decision("a", Scan(init=jnp.float32(0.0)))
        )
        assert np.asarray(composed(xs)).shape == (4,)

    def test_missing_init_raises_a_composer_error(self):
        with pytest.raises(ComposerError, match="initial carry"):
            compose_single_axis(lambda c, x: (c, x), _decision("a", Scan(init=None)))


class TestDedupGatherAxis:
    def test_round_trips_dedup_map_gather_in_order(self):
        """The indices are host NumPy baked in as closure constants."""
        unique_indices = np.array([0, 2], dtype=np.int32)
        index_map = np.array([0, 1, 0, 1], dtype=np.int32)
        strategy = DedupGather(
            unique_indices=unique_indices,
            index_map=index_map,
            k=2,
            k_bucket=2,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx: ys[idx],
        )
        xs = jnp.array([[1.0], [2.0], [1.0], [2.0]], dtype=jnp.float32)
        composed = compose_single_axis(lambda x: x * 10.0, _decision("a", strategy, cardinality=4))
        out = np.asarray(composed(xs))
        assert out.shape == (4, 1)
        np.testing.assert_allclose(out, np.array([[10.0], [10.0], [10.0], [10.0]]))


class TestBuildTraceableCallable:
    def test_routes_the_single_axis(self):
        plan = _Plan([_decision("batch", Vmap())])
        composed = build_traceable_callable(lambda x: x * 3.0, plan)
        xs = jnp.ones((8, 2), dtype=jnp.float32)
        np.testing.assert_allclose(np.asarray(composed(xs)), np.full((8, 2), 3.0))

    def test_picks_the_boundary_by_axis_name(self):
        plan = _Plan([_decision("batch", Vmap())])
        boundaries = {"batch": AxisBoundary(fuse=lambda ys: jnp.sum(ys, axis=0))}
        composed = build_traceable_callable(lambda x: x, plan, boundaries)
        assert np.asarray(composed(jnp.ones((8, 2), jnp.float32))).shape == (2,)

    def test_ignores_a_boundary_keyed_to_another_axis(self):
        plan = _Plan([_decision("batch", Vmap())])
        boundaries = {"other": AxisBoundary(fuse=lambda ys: jnp.sum(ys, axis=0))}
        composed = build_traceable_callable(lambda x: x, plan, boundaries)
        assert np.asarray(composed(jnp.ones((8, 2), jnp.float32))).shape == (8, 2)

    def test_empty_plan_raises(self):
        with pytest.raises(ComposerError, match="no axis decisions"):
            build_traceable_callable(lambda x: x, _Plan([]))

    def test_uncertified_two_axis_shape_raises_naming_the_shape(self):
        """Vmap-over-Scan is the one certified nesting; Vmap-over-Vmap is not."""
        plan = _Plan([_decision("outer", Vmap()), _decision("inner", Vmap())])
        with pytest.raises(MultiAxisCompositionError, match="Vmap-over-Vmap"):
            build_traceable_callable(lambda x: x, plan)

    def test_three_axis_plan_raises_naming_the_axes(self):
        plan = _Plan([_decision("a", Vmap()), _decision("b", Scan()), _decision("c", Vmap())])
        with pytest.raises(MultiAxisCompositionError, match="'a', 'b', 'c'"):
            build_traceable_callable(lambda x: x, plan)

    def test_composes_unaware_of_materialize(self):
        """Stripping lives in pipeline.py, so the composer sees the real sink."""
        seen: list[int] = []

        class _Sink:
            ordered = False

            def __call__(self, x) -> None:
                seen.append(1)

        plan = _Plan([_decision("batch", Vmap())])
        boundaries = {"batch": AxisBoundary(sink=_Sink(), materialize=True)}
        composed = build_traceable_callable(lambda x: x, plan, boundaries)
        composed(jnp.ones((8, 2), jnp.float32))
        assert seen, "composer must call the sink it was handed"

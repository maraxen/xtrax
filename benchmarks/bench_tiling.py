import jax.numpy as jnp
import pytest

from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.strategy import DedupGather, SafeMap, Vmap


def _simple_fn(x):
    return x * 2.0


def _make_vmap():
    return Vmap()


def _make_safe_map():
    return SafeMap(batch_size=8)


def _make_dedup():
    dedup_fn = lambda xs: (  # noqa: E731
        jnp.unique(xs, size=8, axis=0),
        jnp.zeros(32, dtype=jnp.int32),
    )
    gather_fn = lambda ys, idx: ys[idx]  # noqa: E731
    return DedupGather(
        dedup_fn=dedup_fn,
        gather_fn=gather_fn,
        k_bucket=8,
    )


STRATEGIES = {
    "vmap": _make_vmap,
    "safe_map": _make_safe_map,
    "dedup": _make_dedup,
}


@pytest.mark.parametrize("strategy_name", ["vmap", "safe_map", "dedup"])
def test_tiling_dispatch_overhead(benchmark, strategy_name):
    strategy = STRATEGIES[strategy_name]()
    xs = jnp.ones((32, 4))
    benchmark(make_axis_dispatch, strategy, _simple_fn, xs)

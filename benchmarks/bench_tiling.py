import jax.numpy as jnp
import pytest

from xtrax.tiling.dispatch import axis_dispatch, make_axis_dispatch
from xtrax.tiling.strategy import DedupGather, SafeMap, Vmap


def _simple_fn(x):
    return x * 2.0


def _make_vmap():
    return Vmap()


def _make_safe_map():
    return SafeMap(batch_size=8)


def _make_dedup():
    # Create minimal DedupGather for benchmarking
    unique_indices = jnp.arange(8, dtype=jnp.int32)  # 8 unique elements
    index_map = jnp.arange(32, dtype=jnp.int32) % 8  # Map 32 to 8 buckets
    dedup_fn = lambda xs, indices: xs[indices]  # noqa: E731
    gather_fn = lambda ys, idx: ys[idx]  # noqa: E731
    return DedupGather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=8,
        k_bucket=8,
        dedup_fn=dedup_fn,
        gather_fn=gather_fn,
    )


STRATEGIES = {
    "vmap": _make_vmap,
    "safe_map": _make_safe_map,
    "dedup": _make_dedup,
}


@pytest.mark.parametrize("strategy_name", ["vmap", "safe_map", "dedup"])
def test_tiling_dispatch_overhead(benchmark, strategy_name):
    # Declaration protocol for XTRAX_BENCH_RECORD_DIR emission (see
    # xtrax.profiling.bench): stage 1 micro-bench; scale basis stated
    # explicitly so the record is never mistaken for molecular n_atoms.
    benchmark.extra_info.update(
        {
            "xtrax_stage": 1,
            "xtrax_n_atoms": 32,
            "xtrax_scale_basis": "batch_rows",
            # DedupGather rides the backward-compatible eager shim; Vmap/
            # SafeMap go through the factory. Recorded so cross-strategy
            # comparisons know the entry point differed.
            "xtrax_dispatch_entry": (
                "axis_dispatch" if strategy_name == "dedup" else "make_axis_dispatch"
            ),
        }
    )
    strategy = STRATEGIES[strategy_name]()
    xs = jnp.ones((32, 4))
    if strategy_name == "dedup":
        # DedupGather is handled by axis_dispatch (backward-compatible eager shim)
        benchmark(axis_dispatch, strategy, _simple_fn, xs)
    else:
        # Vmap/SafeMap use the factory-style make_axis_dispatch
        dispatch = make_axis_dispatch(strategy)
        benchmark(dispatch, _simple_fn, xs)

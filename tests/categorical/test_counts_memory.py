"""Gate S1-G4 (exact path): tiling bounds, planner, and the row-block iterator.

Memory is measured with ``compile().memory_analysis()`` and never by executing
large arrays. The memory shapes (P=80, K=4, T=10000) are compiled only. Exact
values are checked on small shapes against the numpy int64 oracle.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.categorical.test_counts_exact import _oracle, _oracle_pairs
from xtrax.categorical import (
    CategoricalCodes,
    cooccurrence_counts,
    encode_codes,
    iter_cooccurrence_row_blocks,
)
from xtrax.categorical import plan as plan_mod
from xtrax.categorical.plan import (
    CooccurrencePlan,
    estimate_cooccurrence_bytes,
    per_tile_bytes,
    plan_cooccurrence,
)
from xtrax.tiling.budget import BudgetInfeasibleError


def _manual_plan(
    *,
    t_chunk: int,
    row_block: int,
    n_frames: int,
    n_a: int,
    n_b: int,
    k: int,
    r: int = 1,
    r_tile: int = 1,
    weight_itemsize: int = 1,
    code_itemsize: int = 1,
) -> CooccurrencePlan:
    """Build a matmul plan with explicit tile sizes and its spec §5 estimate."""
    t_padded = -(-n_frames // t_chunk) * t_chunk
    est = estimate_cooccurrence_bytes(
        method="matmul",
        n_replicates=r,
        n_frames=n_frames,
        n_a=n_a,
        k_a=k,
        n_b=n_b,
        k_b=k,
        row_block=row_block,
        n_pairs_blk=row_block * n_b,
        t_chunk=t_chunk,
        r_tile=r_tile,
        pair_tile=1,
        weight_itemsize=weight_itemsize,
        code_itemsize=code_itemsize,
    )
    return CooccurrencePlan(
        method="matmul",
        r_tile=r_tile,
        t_chunk=t_chunk,
        row_block=row_block,
        pair_tile=1,
        r_padded=-(-r // r_tile) * r_tile,
        t_padded=t_padded,
        estimate_bytes=est,
        budget_bytes=None,
    )


def _temp_bytes(
    ca: CategoricalCodes, cb: CategoricalCodes, w: jax.Array, plan: CooccurrencePlan
) -> int:
    """Compiled temp size of the kernel under an explicit plan (compile only)."""

    def run(a, b, weights):
        return cooccurrence_counts(a, b, weights=weights, plan=plan, check=False)

    compiled = jax.jit(run).lower(ca, cb, w).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes)


# S1-G4 memory shapes: P=80, K=4, T=10000 (compile only, no execution).
_P, _K, _T = 80, 4, 10000


@pytest.fixture(scope="module")
def memory_inputs() -> tuple[CategoricalCodes, CategoricalCodes, jax.Array]:
    rng = np.random.default_rng(0)
    ca = encode_codes(rng.integers(0, _K, size=(_P, _T)), _K)
    cb = encode_codes(rng.integers(0, _K, size=(_P, _T)), _K)
    return ca, cb, jnp.ones((1, _T), dtype=jnp.int8)


def test_tiled_temp_at_most_quarter_of_untiled(memory_inputs) -> None:
    """S1-G4: the tiled kernel's temp is <= 0.25x the untiled one at P=80, K=4, T=10000."""
    ca, cb, w = memory_inputs
    untiled = _temp_bytes(
        ca, cb, w, _manual_plan(t_chunk=_T, row_block=_P, n_frames=_T, n_a=_P, n_b=_P, k=_K)
    )
    tiled = _temp_bytes(
        ca, cb, w, _manual_plan(t_chunk=128, row_block=_P, n_frames=_T, n_a=_P, n_b=_P, k=_K)
    )
    assert tiled <= 0.25 * untiled, f"tiled temp {tiled} vs untiled {untiled}"


def test_temp_within_per_tile_term_and_estimate_within_4x(memory_inputs) -> None:
    """S1-G4: temp <= the estimate's per_tile term, and the estimate <= 4x temp (both recorded)."""
    ca, cb, w = memory_inputs
    plan = _manual_plan(t_chunk=128, row_block=_P, n_frames=_T, n_a=_P, n_b=_P, k=_K)
    temp = _temp_bytes(ca, cb, w, plan)
    per_tile = per_tile_bytes(
        method="matmul",
        n_a=_P,
        k_a=_K,
        n_b=_P,
        k_b=_K,
        row_block=_P,
        t_chunk=128,
        r_tile=1,
        pair_tile=1,
        weight_itemsize=1,
    )
    print(f"recorded: temp={temp} per_tile={per_tile} estimate={plan.estimate_bytes}")
    assert temp <= per_tile, f"temp {temp} exceeds per_tile {per_tile}"
    assert plan.estimate_bytes <= 4 * temp, f"estimate {plan.estimate_bytes} vs temp {temp}"


def test_infeasible_int_budget_raises(memory_inputs) -> None:
    """A budget below the smallest tile plan's estimate raises BudgetInfeasibleError."""
    ca, _, _ = memory_inputs
    # The full result is ~410 KB, so it fits; the smallest tile plan needs ~1.6 MB of inputs.
    with pytest.raises(BudgetInfeasibleError, match="no tile plan fits"):
        cooccurrence_counts(ca, ca, byte_budget=500_000)


def test_output_too_large_raises_and_names_the_iterator() -> None:
    """If the full result alone exceeds the budget, the error names the row-block iterator."""
    ca = encode_codes(np.zeros((80, 4), dtype=np.int64), 4)
    with pytest.raises(BudgetInfeasibleError, match="iter_cooccurrence_row_blocks"):
        cooccurrence_counts(ca, ca, byte_budget=1000)


def test_byte_budget_none_never_queries_the_device(monkeypatch) -> None:
    """byte_budget=None uses the unbudgeted defaults and must not touch device memory."""

    def boom(*args, **kwargs):
        raise AssertionError("device queried with byte_budget=None")

    monkeypatch.setattr(plan_mod, "device_budget_bytes", boom)
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    out = cooccurrence_counts(ca, ca, method="matmul")
    assert out.joint.shape == (1, 2, 2, 3, 3)


def test_byte_budget_device_on_cpu_raises_without_fallback() -> None:
    """On a backend without memory_stats, 'device' raises and says to pass an int."""
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-only: relies on the CPU backend lacking memory_stats")
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    with pytest.raises(RuntimeError, match="integer byte_budget"):
        cooccurrence_counts(ca, ca, byte_budget="device")


def test_planner_unbudgeted_defaults_match_spec() -> None:
    p = plan_cooccurrence(
        n_replicates=3,
        n_frames=1500,
        n_a=5,
        k_a=4,
        n_b=6,
        k_b=4,
        n_pairs=30,
        weight_itemsize=1,
        code_itemsize=1,
        method="matmul",
        layout="dense",
        platform="cpu",
    )
    assert (p.r_tile, p.t_chunk, p.row_block, p.budget_bytes) == (1, 1024, 5, None)
    assert p.t_padded == 2048 and p.r_padded == 3


def test_planner_budget_search_respects_budget_and_is_exact() -> None:
    """A budget that forces row blocks is honored, and the executed counts stay exact."""
    rng = np.random.default_rng(5)
    pa, pb, k, t, r = 6, 4, 3, 300, 2
    xa = rng.integers(0, k, size=(pa, t))
    xb = rng.integers(0, k, size=(pb, t))
    w = rng.integers(0, 100, size=(r, t)).astype(np.int32)

    def est(row_block: int) -> int:
        return estimate_cooccurrence_bytes(
            method="matmul",
            n_replicates=r,
            n_frames=t,
            n_a=pa,
            k_a=k,
            n_b=pb,
            k_b=k,
            row_block=row_block,
            n_pairs_blk=row_block * pb,
            t_chunk=128,
            r_tile=1,
            pair_tile=1,
            weight_itemsize=4,
            code_itemsize=1,
        )

    # A budget between the one-row and the full-block estimates forces row blocks.
    budget = (est(1) + est(pa)) // 2
    assert est(1) <= budget < est(pa)
    plan = plan_cooccurrence(
        n_replicates=r,
        n_frames=t,
        n_a=pa,
        k_a=k,
        n_b=pb,
        k_b=k,
        n_pairs=pa * pb,
        weight_itemsize=4,
        code_itemsize=1,
        method="matmul",
        layout="dense",
        byte_budget=budget,
        platform="cpu",
    )
    assert plan.estimate_bytes <= budget
    assert plan.row_block < pa, "the budget should force row blocks at this size"
    out = cooccurrence_counts(
        encode_codes(xa, k), encode_codes(xb, k), weights=w, byte_budget=budget, method="matmul"
    )
    ref = _oracle(xa, k, xb, k, w)
    for field in ("joint", "marginal_a", "marginal_b", "total"):
        assert np.array_equal(np.asarray(getattr(out, field)), ref[field])


def test_executed_tiles_match_plan_in_jaxpr() -> None:
    """The plan's tile sizes equal the scan lengths and map batch sizes in the jaxpr."""
    pa, pb, t, r = 5, 3, 1000, 4
    ca = encode_codes(np.zeros((pa, t), dtype=np.int64), 3)
    cb = encode_codes(np.zeros((pb, t), dtype=np.int64), 3)
    plan = CooccurrencePlan(
        method="matmul",
        r_tile=2,
        t_chunk=128,
        row_block=2,
        pair_tile=1,
        r_padded=4,
        t_padded=1024,
        estimate_bytes=0,
        budget_bytes=None,
    )
    w = jnp.ones((r, t), dtype=jnp.int8)

    def run(a, b, weights):
        return cooccurrence_counts(a, b, weights=weights, plan=plan, check=False)

    jaxpr = jax.make_jaxpr(run)(ca, cb, w).jaxpr
    lengths = set()
    stack = [jaxpr]
    while stack:
        j = stack.pop()
        for eqn in j.eqns:
            if eqn.primitive.name == "scan":
                lengths.add(int(eqn.params["length"]))
            for v in eqn.params.values():
                for sub in v if isinstance(v, (list, tuple)) else [v]:
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None:
                        stack.append(inner)
                    elif hasattr(sub, "eqns"):
                        stack.append(sub)
    n_frame_chunks = plan.t_padded // plan.t_chunk
    n_replicate_steps = plan.r_padded // plan.r_tile
    n_row_blocks = -(-pa // plan.row_block)
    expected = {n_frame_chunks, n_replicate_steps, n_row_blocks}
    assert expected <= lengths, f"expected scan lengths {expected} in jaxpr, found {lengths}"
    assert n_frame_chunks == 8 and n_replicate_steps == 2 and n_row_blocks == 3


def test_plan_argument_mismatch_is_rejected() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    bad_t = CooccurrencePlan(
        method="matmul",
        r_tile=1,
        t_chunk=4,
        row_block=2,
        pair_tile=1,
        r_padded=1,
        t_padded=8,
        estimate_bytes=0,
        budget_bytes=None,
    )
    with pytest.raises(ValueError, match="disagrees|different frame count"):
        cooccurrence_counts(ca, ca, plan=bad_t, method="scatter")


@pytest.mark.parametrize("r_tile", [1, 2, 3])
def test_r_tile_does_not_change_exact_counts(r_tile: int) -> None:
    rng = np.random.default_rng(9)
    pa, pb, t, r, k = 3, 2, 200, 3, 4
    xa = rng.integers(0, k, size=(pa, t))
    xb = rng.integers(0, k, size=(pb, t))
    w = rng.integers(0, 50, size=(r, t)).astype(np.int32)
    ca, cb = encode_codes(xa, k), encode_codes(xb, k)
    plan = CooccurrencePlan(
        method="matmul",
        r_tile=r_tile,
        t_chunk=64,
        row_block=1,
        pair_tile=1,
        r_padded=-(-r // r_tile) * r_tile,
        t_padded=256,
        estimate_bytes=0,
        budget_bytes=None,
    )
    out = cooccurrence_counts(ca, cb, weights=w, plan=plan, method="matmul")
    ref = _oracle(xa, k, xb, k, w)
    assert np.array_equal(np.asarray(out.joint), ref["joint"])


def test_row_block_iterator_matches_full_counts_dense() -> None:
    rng = np.random.default_rng(12)
    pa, pb, t, r, k = 5, 3, 120, 2, 3
    xa = rng.integers(0, k, size=(pa, t))
    xb = rng.integers(0, k, size=(pb, t))
    w = rng.integers(0, 60, size=(r, t)).astype(np.int32)
    ca, cb = encode_codes(xa, k), encode_codes(xb, k)
    ref = _oracle(xa, k, xb, k, w)
    joint = np.zeros_like(ref["joint"])
    seen = []
    for rows, blk in iter_cooccurrence_row_blocks(ca, cb, row_block=2, weights=w):
        joint[:, rows] = np.asarray(blk.joint)
        seen.append((rows.start, rows.stop))
        assert np.array_equal(np.asarray(blk.marginal_b), ref["marginal_b"])
        assert np.array_equal(np.asarray(blk.total), ref["total"])
        assert isinstance(blk.joint, np.ndarray)
    assert seen == [(0, 2), (2, 4), (4, 5)]
    assert np.array_equal(joint, ref["joint"])


def test_row_block_iterator_self_layout_with_shifts() -> None:
    rng = np.random.default_rng(13)
    p, t, r, k = 4, 90, 2, 3
    xa = rng.integers(0, k, size=(p, t))
    w = rng.integers(0, 40, size=(r, t)).astype(np.int32)
    sa = rng.integers(-150, 150, size=(r, p)).astype(np.int32)
    ca = encode_codes(xa, k)
    ref = _oracle(xa, k, xa, k, w, shifts_a=sa, shifts_b=sa)
    joint = np.zeros_like(ref["joint"])
    for rows, blk in iter_cooccurrence_row_blocks(ca, None, row_block=3, weights=w, shifts_a=sa):
        joint[:, rows] = np.asarray(blk.joint)
    assert np.array_equal(joint, ref["joint"])


def test_row_block_iterator_with_shifts_b_dense() -> None:
    rng = np.random.default_rng(14)
    pa, pb, t, r, k = 3, 2, 60, 2, 4
    xa = rng.integers(0, k, size=(pa, t))
    xb = rng.integers(0, k, size=(pb, t))
    w = rng.integers(0, 30, size=(r, t)).astype(np.int32)
    sa = rng.integers(-80, 80, size=(r, pa)).astype(np.int32)
    sb = rng.integers(-80, 80, size=(r, pb)).astype(np.int32)
    ca, cb = encode_codes(xa, k), encode_codes(xb, k)
    ref = _oracle(xa, k, xb, k, w, shifts_a=sa, shifts_b=sb)
    joint = np.zeros_like(ref["joint"])
    for rows, blk in iter_cooccurrence_row_blocks(
        ca, cb, row_block=2, weights=w, shifts_a=sa, shifts_b=sb
    ):
        joint[:, rows] = np.asarray(blk.joint)
    assert np.array_equal(joint, ref["joint"])


def test_row_block_iterator_rejects_pairs_and_bad_block() -> None:
    ca = encode_codes(np.zeros((2, 4), dtype=np.int64), 3)
    with pytest.raises(NotImplementedError, match="pair blocks"):
        next(iter_cooccurrence_row_blocks(ca, ca, row_block=1, pairs=np.zeros((1, 2), np.int32)))
    with pytest.raises(ValueError, match="row_block"):
        next(iter_cooccurrence_row_blocks(ca, ca, row_block=0))


def test_pairs_oracle_helper_agrees_with_dense_gather() -> None:
    """Sanity of the shared oracle helper used by the pairs tests."""
    rng = np.random.default_rng(15)
    xa = rng.integers(0, 3, size=(2, 20))
    w = rng.integers(0, 9, size=(1, 20))
    pairs = np.array([[1, 0], [0, 1]], dtype=np.int32)
    dense = _oracle(xa, 3, xa, 3, w)
    part = _oracle_pairs(xa, 3, xa, 3, w, pairs)
    assert np.array_equal(part["joint"][:, 0], dense["joint"][:, 1, 0])
    assert np.array_equal(part["joint"][:, 1], dense["joint"][:, 0, 1])


# --- Review #277 (major): scatter feasibility is judged at pair_tile=1 ---------------------

_SCATTER_KW = dict(
    n_replicates=1,
    n_frames=128,
    n_a=4,
    k_a=8,
    n_b=400,
    k_b=8,
    weight_itemsize=4,
    code_itemsize=4,
    method="scatter",
    platform="cpu",
)


def _scatter_min_estimate(layout: str, n_pairs: int) -> int:
    rb = 1 if layout == "dense" else _SCATTER_KW["n_a"]
    blk = rb * _SCATTER_KW["n_b"] if layout == "dense" else n_pairs
    return estimate_cooccurrence_bytes(
        method="scatter",
        n_replicates=1,
        n_frames=128,
        n_a=_SCATTER_KW["n_a"],
        k_a=8,
        n_b=_SCATTER_KW["n_b"],
        k_b=8,
        row_block=rb,
        n_pairs_blk=blk,
        t_chunk=128,
        r_tile=1,
        pair_tile=1,
        weight_itemsize=4,
        code_itemsize=4,
        differentiable=False,
        layout=layout,
    )


@pytest.mark.parametrize(("layout", "n_pairs"), [("dense", 1600), ("pairs", 1200)])
def test_scatter_budget_above_minimum_plans_instead_of_raising(layout: str, n_pairs: int) -> None:
    """A budget that holds the pair_tile=1 minimum must yield a plan that fits it."""
    minimum = _scatter_min_estimate(layout, n_pairs)
    budget = minimum + 1024
    p = plan_cooccurrence(n_pairs=n_pairs, layout=layout, byte_budget=budget, **_SCATTER_KW)
    assert p.estimate_bytes <= budget
    assert p.pair_tile >= 1


@pytest.mark.parametrize(("layout", "n_pairs"), [("dense", 1600), ("pairs", 1200)])
def test_scatter_budget_below_minimum_still_raises(layout: str, n_pairs: int) -> None:
    minimum = _scatter_min_estimate(layout, n_pairs)
    with pytest.raises(BudgetInfeasibleError):
        plan_cooccurrence(n_pairs=n_pairs, layout=layout, byte_budget=minimum - 1, **_SCATTER_KW)


# --- Review #277 (nit): the GPU branch of select_method is pinned ----------------------------


@pytest.mark.parametrize("platform", ["gpu", "cuda", "rocm"])
def test_select_method_gpu_branch(platform: str) -> None:
    from xtrax.categorical.plan import KK_CROSSOVER_GPU, select_method

    k = int(np.floor(np.sqrt(KK_CROSSOVER_GPU)))
    assert select_method("auto", "dense", 4, 4, k, k, platform) == ("matmul", "gpu_provisional")
    big = KK_CROSSOVER_GPU + 1
    assert select_method("auto", "dense", 4, 4, big, 1, platform) == ("scatter", "gpu_provisional")
    # A pairs list shorter than the dense grid prefers scatter even for small K.
    assert select_method("auto", "pairs", 2, 4, 2, 2, platform) == ("scatter", "gpu_provisional")
    # A full-grid pairs list uses the dense rule's K test, but only dense gets matmul.
    assert select_method("auto", "pairs", 4, 4, 2, 2, platform) == ("scatter", "gpu_provisional")
    assert select_method("matmul", "dense", 4, 4, big, 1, platform) == ("matmul", "explicit")

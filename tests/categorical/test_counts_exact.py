"""Gate S1-G2 (matmul subset, dense and self layouts): exact int32 counts.

Reference: numpy int64 via ``np.roll`` and ``np.add.at``. Comparisons use
``np.array_equal``; there are no tolerances anywhere in this file.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jcore

from xtrax.categorical import CooccurrenceCounts, cooccurrence_counts, encode_codes
from xtrax.categorical.counts import KK_CROSSOVER_CPU

_WEIGHT_CAP = {"int8": 127, "int16": 1000, "int32": 100_000}


def _oracle(
    xa: np.ndarray,
    ka: int,
    xb: np.ndarray,
    kb: int,
    w: np.ndarray,
    shifts_a: np.ndarray | None = None,
    shifts_b: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Numpy int64 reference for dense counts, shifts following ``np.roll``."""
    xa = np.asarray(xa, dtype=np.int64)
    xb = np.asarray(xb, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    n_rep = w.shape[0]
    pa, pb = xa.shape[0], xb.shape[0]
    joint = np.zeros((n_rep, pa, pb, ka, kb), dtype=np.int64)
    ma = np.zeros((n_rep, pa, ka), dtype=np.int64)
    mb = np.zeros((n_rep, pb, kb), dtype=np.int64)
    for r in range(n_rep):
        sa = np.stack(
            [np.roll(xa[i], 0 if shifts_a is None else int(shifts_a[r, i])) for i in range(pa)]
        )
        sb = np.stack(
            [np.roll(xb[j], 0 if shifts_b is None else int(shifts_b[r, j])) for j in range(pb)]
        )
        for i in range(pa):
            np.add.at(ma[r, i], sa[i], w[r])
            for j in range(pb):
                np.add.at(joint[r, i, j], (sa[i], sb[j]), w[r])
        for j in range(pb):
            np.add.at(mb[r, j], sb[j], w[r])
    return {"joint": joint, "marginal_a": ma, "marginal_b": mb, "total": w.sum(axis=1)}


def _case(seed: int, pa: int, pb: int, ka: int, kb: int, t: int, r: int, wname: str):
    rng = np.random.default_rng(seed)
    xa = rng.integers(0, ka, size=(pa, t))
    xb = rng.integers(0, kb, size=(pb, t))
    w = rng.integers(0, _WEIGHT_CAP[wname] + 1, size=(r, t))
    return xa, xb, w


def _assert_counts_equal(out: CooccurrenceCounts, ref: dict[str, np.ndarray]) -> None:
    for field in ("joint", "marginal_a", "marginal_b", "total"):
        got = np.asarray(getattr(out, field))
        assert got.dtype == np.int32, f"{field} dtype {got.dtype}"
        assert np.array_equal(got, ref[field]), f"{field} differs from int64 oracle"


@pytest.mark.parametrize(
    ("ka", "kb"),
    [(3, 3), (3, 200), (300, 4)],
)
@pytest.mark.parametrize("wname", ["int8", "int16", "int32"])
@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("t", [50, 1500])
def test_matmul_dense_exact_vs_int64_oracle(ka: int, kb: int, wname: str, r: int, t: int) -> None:
    """Dense layout, explicit matmul. T=1500 exercises frame padding (t_chunk=1024)."""
    xa, xb, w = _case(seed=ka * 7 + kb + r + t, pa=2, pb=3, ka=ka, kb=kb, t=t, r=r, wname=wname)
    out = cooccurrence_counts(
        encode_codes(xa, ka),
        encode_codes(xb, kb),
        weights=w.astype(wname),
        method="matmul",
    )
    assert out.layout == "dense"
    assert out.method == "matmul"
    assert out.method_rule == "explicit"
    assert out.joint.shape == (r, 2, 3, ka, kb)
    assert out.marginal_a.shape == (r, 2, ka)
    assert out.marginal_b.shape == (r, 3, kb)
    assert out.total.shape == (r,)
    _assert_counts_equal(out, _oracle(xa, ka, xb, kb, w))


def test_default_weights_are_ones_and_total_is_frame_count() -> None:
    xa, xb, _ = _case(seed=1, pa=2, pb=2, ka=4, kb=5, t=37, r=1, wname="int8")
    out = cooccurrence_counts(encode_codes(xa, 4), encode_codes(xb, 5), method="matmul")
    ones = np.ones((1, 37), dtype=np.int64)
    _assert_counts_equal(out, _oracle(xa, 4, xb, 5, ones))
    assert int(out.total[0]) == 37


def test_1d_weights_broadcast_to_one_replicate() -> None:
    xa, xb, w = _case(seed=2, pa=1, pb=2, ka=3, kb=3, t=20, r=1, wname="int16")
    out = cooccurrence_counts(
        encode_codes(xa, 3), encode_codes(xb, 3), weights=w[0].astype("int16"), method="matmul"
    )
    assert out.joint.shape == (1, 1, 2, 3, 3)
    _assert_counts_equal(out, _oracle(xa, 3, xb, 3, w))


def test_self_layout_diagonal_equals_marginal_and_matches_oracle() -> None:
    """Self layout: codes_b None; diagonal block (a, a) equals diag(marginal_a[a])."""
    xa, _, w = _case(seed=3, pa=3, pb=3, ka=4, kb=4, t=64, r=2, wname="int32")
    out = cooccurrence_counts(encode_codes(xa, 4), weights=w.astype("int32"), method="matmul")
    assert out.joint.shape == (2, 3, 3, 4, 4)
    _assert_counts_equal(out, _oracle(xa, 4, xa, 4, w))
    joint = np.asarray(out.joint)
    marg = np.asarray(out.marginal_a)
    for a in range(3):
        block = joint[:, a, a]
        assert np.array_equal(block, np.stack([np.diag(marg[r, a]) for r in range(2)]))


def test_totals_above_2_to_the_24_are_exact() -> None:
    """Large totals (spec §8 S1-G2): one cell is exactly 2**24 + 1, total near 2**30."""
    t = 64
    xa = np.zeros((1, t), dtype=np.int64)
    xb = np.ones((1, t), dtype=np.int64)
    xb[0, :2] = 0  # frames 0 and 1 share code 0 for b
    w = np.full((1, t), 2**24, dtype=np.int64)
    w[0, 1] = 1
    out = cooccurrence_counts(
        encode_codes(xa, 2),
        encode_codes(xb, 2),
        weights=w.astype(np.int32),
        method="matmul",
    )
    joint = np.asarray(out.joint)
    assert joint[0, 0, 0, 0, 0] == 2**24 + 1
    _assert_counts_equal(out, _oracle(xa, 2, xb, 2, w))
    assert int(out.total[0]) == t * 2**24 - (2**24 - 1)


def test_auto_selects_matmul_on_cpu_below_crossover() -> None:
    xa, xb, w = _case(seed=4, pa=2, pb=2, ka=3, kb=3, t=30, r=1, wname="int8")
    assert 3 * 3 <= KK_CROSSOVER_CPU
    out = cooccurrence_counts(encode_codes(xa, 3), encode_codes(xb, 3), weights=w.astype("int8"))
    assert out.method == "matmul"
    assert out.method_rule == "cpu_provisional"
    _assert_counts_equal(out, _oracle(xa, 3, xb, 3, w))


def test_auto_above_crossover_needs_scatter_not_yet_landed() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU crossover rule")
    xa = np.zeros((1, 4), dtype=np.int64)
    with pytest.raises(NotImplementedError, match="task 1.4"):
        cooccurrence_counts(encode_codes(xa, 16), encode_codes(xa, 16))


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"strategy": "float"}, NotImplementedError, "task 1.7"),
        ({"differentiable": True}, NotImplementedError, "task 1.7"),
        ({"pairs": np.zeros((1, 2), dtype=np.int32)}, NotImplementedError, "task 1.4"),
        ({"byte_budget": 1 << 20}, NotImplementedError, "task 1.5"),
        ({"method": "bogus"}, ValueError, "method"),
    ],
)
def test_out_of_scope_options_raise(kwargs: dict, exc: type, match: str) -> None:
    ca = encode_codes(np.zeros((2, 4), dtype=np.int64), 3)
    with pytest.raises(exc, match=match):
        cooccurrence_counts(ca, ca, **kwargs)


def test_float_weights_rejected_on_exact_path() -> None:
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 3)
    with pytest.raises(TypeError, match="differentiable=True"):
        cooccurrence_counts(ca, ca, weights=np.ones(4, dtype=np.float32))


def test_soft_or_raw_codes_rejected_on_exact_path() -> None:
    with pytest.raises(TypeError, match="CategoricalCodes"):
        cooccurrence_counts(np.zeros((1, 4), dtype=np.int32))


def test_weight_and_code_shape_mismatches_raise() -> None:
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 3)
    cb = encode_codes(np.zeros((1, 5), dtype=np.int64), 3)
    with pytest.raises(ValueError, match="share T"):
        cooccurrence_counts(ca, cb)
    with pytest.raises(ValueError, match="weights"):
        cooccurrence_counts(ca, ca, weights=np.ones(5, dtype=np.int32))


def test_static_index_size_guards_raise_before_allocation() -> None:
    big = encode_codes(np.zeros((1, 2), dtype=np.int64), 65536)
    with pytest.raises(ValueError, match=r"Ka\*Kb"):
        cooccurrence_counts(big, big, method="matmul")
    wide = encode_codes(np.zeros((2, 2), dtype=np.int64), 2**31 - 1)
    with pytest.raises(ValueError, match=r"Pa\*Ka"):
        cooccurrence_counts(wide, wide, method="matmul")


def _walk_eqns(jaxpr):
    for eqn in jaxpr.eqns:
        yield eqn
        for value in eqn.params.values():
            subs = value if isinstance(value, (list, tuple)) else [value]
            for sub in subs:
                if isinstance(sub, jcore.ClosedJaxpr):
                    yield from _walk_eqns(sub.jaxpr)
                elif isinstance(sub, jcore.Jaxpr):
                    yield from _walk_eqns(sub)


def test_every_dot_general_accumulates_in_int32() -> None:
    """Jaxpr check (spec §8): every dot_general on the exact path is int32 out."""
    ca = encode_codes(np.zeros((2, 1500), dtype=np.int64), 5)
    cb = encode_codes(np.zeros((3, 1500), dtype=np.int64), 4)
    w = jnp.ones((2, 1500), dtype=jnp.int8)

    def run(a_codes, b_codes, weights):
        return cooccurrence_counts(a_codes, b_codes, weights=weights, method="matmul")

    jaxpr = jax.make_jaxpr(run)(ca, cb, w)
    dots = [e for e in _walk_eqns(jaxpr.jaxpr) if e.primitive.name == "dot_general"]
    assert dots, "expected at least one dot_general in the matmul path"
    for eqn in dots:
        assert eqn.params["preferred_element_type"] == jnp.int32
        assert eqn.outvars[0].aval.dtype == jnp.int32


def _shift_regime(rng: np.random.Generator, r: int, p: int, t: int, regime: str) -> np.ndarray:
    """Shifts for one regime: random in [-3T, 3T], zero, or wrap-around extremes."""
    if regime == "zero":
        return np.zeros((r, p), dtype=np.int64)
    if regime == "wrap":
        base = np.array([t + 3, -t - 1, 0, 2 * t, -2 * t - 5, 1])
        return np.resize(base, (r, p)).astype(np.int64)
    return rng.integers(-3 * t, 3 * t + 1, size=(r, p))


@pytest.mark.parametrize("regime", ["random", "zero", "wrap"])
@pytest.mark.parametrize("wname", ["int8", "int16", "int32"])
@pytest.mark.parametrize("t", [50, 1500])
def test_dense_shifts_exact_vs_roll_oracle(regime: str, wname: str, t: int) -> None:
    """Shifts follow np.roll for any integer shift, both sides shifted, R=3."""
    rng = np.random.default_rng(11 + t)
    ka, kb, pa, pb, r = 4, 5, 2, 3, 3
    xa = rng.integers(0, ka, size=(pa, t))
    xb = rng.integers(0, kb, size=(pb, t))
    w = rng.integers(0, _WEIGHT_CAP[wname] + 1, size=(r, t))
    sa = _shift_regime(rng, r, pa, t, regime)
    sb = _shift_regime(rng, r, pb, t, regime)
    out = cooccurrence_counts(
        encode_codes(xa, ka),
        encode_codes(xb, kb),
        weights=w.astype(wname),
        shifts_a=sa.astype(np.int32),
        shifts_b=sb.astype(np.int32),
        method="matmul",
    )
    _assert_counts_equal(out, _oracle(xa, ka, xb, kb, w, shifts_a=sa, shifts_b=sb))


def test_only_shifts_a_leaves_b_unshifted() -> None:
    rng = np.random.default_rng(21)
    t, ka, kb = 40, 3, 3
    xa = rng.integers(0, ka, size=(2, t))
    xb = rng.integers(0, kb, size=(2, t))
    w = rng.integers(0, 50, size=(2, t))
    sa = np.array([[5, -7], [0, 41]], dtype=np.int64)
    out = cooccurrence_counts(
        encode_codes(xa, ka),
        encode_codes(xb, kb),
        weights=w.astype(np.int16),
        shifts_a=sa.astype(np.int32),
        method="matmul",
    )
    _assert_counts_equal(out, _oracle(xa, ka, xb, kb, w, shifts_a=sa))


def test_self_layout_shifts_apply_to_both_sides() -> None:
    """Self layout: shifts_a applies to both sides; the diagonal equals diag(marginal_a)."""
    rng = np.random.default_rng(31)
    t, k, p, r = 64, 4, 3, 2
    xa = rng.integers(0, k, size=(p, t))
    w = rng.integers(0, 100_000, size=(r, t))
    sa = rng.integers(-200, 200, size=(r, p))
    out = cooccurrence_counts(
        encode_codes(xa, k),
        weights=w.astype(np.int32),
        shifts_a=sa.astype(np.int32),
        method="matmul",
    )
    _assert_counts_equal(out, _oracle(xa, k, xa, k, w, shifts_a=sa, shifts_b=sa))
    joint = np.asarray(out.joint)
    marg = np.asarray(out.marginal_a)
    for a in range(p):
        for rep in range(r):
            assert np.array_equal(joint[rep, a, a], np.diag(marg[rep, a]))


def test_single_weight_row_broadcasts_over_shift_replicates() -> None:
    """Weights (1, T) broadcast across R=3 shift replicates."""
    rng = np.random.default_rng(41)
    t, k = 30, 3
    xa = rng.integers(0, k, size=(2, t))
    xb = rng.integers(0, k, size=(2, t))
    w1 = rng.integers(0, 100, size=(1, t))
    sa = rng.integers(0, t, size=(3, 2))
    out = cooccurrence_counts(
        encode_codes(xa, k),
        encode_codes(xb, k),
        weights=w1.astype(np.int8),
        shifts_a=sa.astype(np.int32),
        method="matmul",
    )
    assert out.joint.shape[0] == 3
    _assert_counts_equal(out, _oracle(xa, k, xb, k, np.repeat(w1, 3, axis=0), shifts_a=sa))


def test_shifts_b_rejected_in_self_layout() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    with pytest.raises(ValueError, match="self layout"):
        cooccurrence_counts(ca, shifts_b=np.zeros((1, 2), dtype=np.int32))


@pytest.mark.parametrize(
    ("shifts", "match", "exc"),
    [
        (np.zeros((1, 3), dtype=np.int32), r"\(R, 2\)", ValueError),
        (np.zeros((1, 2), dtype=np.float32), "integer-valued", TypeError),
    ],
)
def test_shift_shape_and_dtype_errors(shifts: np.ndarray, match: str, exc: type) -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    with pytest.raises(exc, match=match):
        cooccurrence_counts(ca, ca, shifts_a=shifts)


def test_mismatched_replicates_between_weights_and_shifts_raise() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    with pytest.raises(ValueError, match="replicate counts"):
        cooccurrence_counts(
            ca,
            ca,
            weights=np.ones((3, 8), dtype=np.int8),
            shifts_a=np.zeros((2, 2), dtype=np.int32),
        )

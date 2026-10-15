"""Gate S1-G5 (differentiable path, spec §8) and the S1-G3 ``jit(grad)`` variant.

Items 1-6 of S1-G5 cover the ``custom_vjp`` (``differentiable=True``) and the
``strategy="float"`` bilinear form. Per-field Jacobians are compared with the
closed-form numpy float64 Jacobian. Outputs are linear in the weights, so the
Jacobian entries are exact 0/1 integers. No float32 ``check_grads`` (spec A14).

The ``jit(grad)`` cases are the S1-G3 variant that task 1.5 deferred to this
task. The total-``2**31`` case is an int-path check with no gradient, so it stays
in ``test_counts_checks.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.categorical import CategoricalCodes, cooccurrence_counts, encode_codes
from xtrax.categorical.plan import CooccurrencePlan, plan_cooccurrence

R = 3
T = 8
K_A = 3
K_B = 4
SHIFTS_CROSS_A = np.array([[0, 3], [-2, 9], [1, 1]])
SHIFTS_CROSS_B = np.array([[2, 0, -1], [5, -8, 3], [0, 0, 0]])
SHIFTS_SELF = np.array([[0, 3, 1], [-2, 9, 4], [1, 1, -5]])
CROSS_PAIRS = np.array([[0, 1], [1, 0], [1, 2], [0, 0], [1, 2]])
SELF_PAIRS = np.array([[0, 1], [2, 0], [1, 1], [2, 2], [0, 1]])
LAYOUTS = ("dense", "pairs", "self", "self_pairs")


def _case(layout: str) -> dict:
    """Build one fixture: hard codes, soft assignments, shifts and pairs for a layout."""
    rng = np.random.default_rng(261015)
    self_layout = layout.startswith("self")
    use_pairs = layout.endswith("pairs")
    if self_layout:
        xa = rng.integers(0, K_A, size=(3, T))
        soft_a = rng.dirichlet(np.ones(K_A), size=(3, T))
        return {
            "xa": xa,
            "xb": None,
            "ka": K_A,
            "kb": K_A,
            "sa": SHIFTS_SELF,
            "sb": None,
            "pairs": SELF_PAIRS if use_pairs else None,
            "self": True,
            "soft_a": soft_a,
            "soft_b": None,
        }
    xa = rng.integers(0, K_A, size=(2, T))
    xb = rng.integers(0, K_B, size=(3, T))
    return {
        "xa": xa,
        "xb": xb,
        "ka": K_A,
        "kb": K_B,
        "sa": SHIFTS_CROSS_A,
        "sb": SHIFTS_CROSS_B,
        "pairs": CROSS_PAIRS if use_pairs else None,
        "self": False,
        "soft_a": rng.dirichlet(np.ones(K_A), size=(2, T)),
        "soft_b": rng.dirichlet(np.ones(K_B), size=(3, T)),
    }


def _weights() -> np.ndarray:
    """Integer-valued float32 weights ``(R, T)`` with one zero frame and no zero replicate."""
    rng = np.random.default_rng(7)
    w = rng.integers(0, 4, size=(R, T)).astype(np.float32)
    w[:, 0] = np.maximum(w[:, 0], 1.0)
    w[1, 3] = 0.0
    return w


def _run(case: dict, weights, *, soft: bool = False, **kwargs):
    """Call ``cooccurrence_counts`` on a fixture with hard codes or soft assignments."""
    if soft:
        ca = jnp.asarray(case["soft_a"], dtype=jnp.float32)
        cb = None if case["self"] else jnp.asarray(case["soft_b"], dtype=jnp.float32)
    else:
        ca = encode_codes(case["xa"], case["ka"])
        cb = None if case["self"] else encode_codes(case["xb"], case["kb"])
    return cooccurrence_counts(
        ca,
        cb,
        weights=weights,
        shifts_a=case["sa"],
        shifts_b=None if case["self"] else case["sb"],
        pairs=case["pairs"],
        **kwargs,
    )


def _shifted_soft(x: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Return ``out[r, t, p, :] = x[p, (t - s[r,p]) mod T, :]`` in float64, shape ``(R,T,P,K)``."""
    r_n, p_n = shifts.shape
    t_n = x.shape[1]
    t = np.arange(t_n)
    out = np.empty((r_n, t_n, p_n, x.shape[2]))
    for r in range(r_n):
        for p in range(p_n):
            out[r, :, p, :] = x[p][(t - shifts[r, p]) % t_n]
    return out


def _closed_form_jacobian(field: str, case: dict) -> np.ndarray:
    """Closed-form float64 Jacobian of one output field with respect to ``w`` (shape ``+(R, T)``).

    ``d joint[r,...] / d w[r',t] = delta(r,r') * o_a[r,t,...] o_b[r,t,...]``, and the
    marginals and total follow the same pattern with one factor, or with ones.
    """
    xb = case["xa"] if case["self"] else case["xb"]
    sb = case["sa"] if case["self"] else case["sb"]
    oa = _shifted_soft(np.eye(case["ka"])[case["xa"]], case["sa"])
    ob = _shifted_soft(np.eye(case["kb"])[xb], sb)
    pa, ka = oa.shape[2], oa.shape[3]
    pb, kb = ob.shape[2], ob.shape[3]
    if field == "total":
        jac = np.zeros((R, R, T))
        for r in range(R):
            jac[r, r, :] = 1.0
        return jac
    if field == "marginal_a":
        jac = np.zeros((R, pa, ka, R, T))
        for r in range(R):
            jac[r, :, :, r, :] = oa[r].transpose(1, 2, 0)
        return jac
    if field == "marginal_b":
        jac = np.zeros((R, pb, kb, R, T))
        for r in range(R):
            jac[r, :, :, r, :] = ob[r].transpose(1, 2, 0)
        return jac
    if case["pairs"] is not None:
        ia, ib = case["pairs"][:, 0], case["pairs"][:, 1]
        jac = np.zeros((R, len(ia), ka, kb, R, T))
        for r in range(R):
            jac[r, :, :, :, r, :] = np.einsum("tpi,tpj->pijt", oa[r][:, ia, :], ob[r][:, ib, :])
        return jac
    jac = np.zeros((R, pa, pb, ka, kb, R, T))
    for r in range(R):
        jac[r, :, :, :, :, r, :] = np.einsum("tai,tbj->abijt", oa[r], ob[r])
    return jac


def _jax_jacobian(field: str, case: dict, weights, **kwargs) -> np.ndarray:
    """Reverse-mode Jacobian of one field with respect to the weights, as float64."""

    def f(w):
        return getattr(_run(case, w, differentiable=True, **kwargs), field)

    return np.asarray(jax.jacrev(f)(jnp.asarray(weights)), dtype=np.float64)


def _entropy_sum(counts: jax.Array, total: jax.Array) -> jax.Array:
    """Sum of ``-p log p`` over all cells, with ``p = counts / total`` per replicate.

    The double ``where`` keeps the gradient finite at ``p = 0``.
    """
    n = total.reshape((-1,) + (1,) * (counts.ndim - 1))
    p = counts / n
    pos = p > 0
    safe = jnp.where(pos, p, 1.0)
    return -jnp.sum(jnp.where(pos, p * jnp.log(safe), 0.0))


def _mi_loss(out) -> jax.Array:
    """``L = sum(Ha) + sum(Hb) - sum(Hab)`` from the four fields, in the active dtype."""
    return (
        _entropy_sum(out.marginal_a, out.total)
        + _entropy_sum(out.marginal_b, out.total)
        - _entropy_sum(out.joint, out.total)
    )


def _marginal_loss(out) -> jax.Array:
    """Marginals-only loss: a gradient here must not depend on the joint path."""
    return _entropy_sum(out.marginal_a, out.total) + _entropy_sum(out.marginal_b, out.total)


def _raw_codes(values: np.ndarray, n_categories: int) -> CategoricalCodes:
    """Codes without ``encode_codes``'s host range check, to exercise the traced C1."""
    return CategoricalCodes(codes=jnp.asarray(values.astype(np.uint8)), n_categories=n_categories)


# --- S1-G5 item 1: grad w.r.t. codes and integer weights raises TypeError ----------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grad_wrt_integer_weights_raises_type_error(layout: str) -> None:
    case = _case(layout)
    w_int = jnp.asarray(_weights().astype(np.int32))
    with pytest.raises(TypeError):
        jax.grad(lambda w: _run(case, w, differentiable=True).joint.sum())(w_int)


def test_grad_wrt_codes_raises_type_error() -> None:
    case = _case("dense")
    cb = encode_codes(case["xb"], case["kb"])

    def loss(raw_codes: jax.Array) -> jax.Array:
        ca = CategoricalCodes(codes=raw_codes, n_categories=case["ka"])
        return cooccurrence_counts(
            ca, cb, weights=_weights(), differentiable=True, shifts_a=case["sa"]
        ).joint.sum()

    with pytest.raises(TypeError):
        jax.grad(loss)(jnp.asarray(case["xa"], dtype=jnp.uint8))


# --- S1-G5 item 2: float32 outputs equal the int32 results ---------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_differentiable_outputs_are_float32_and_equal_int_results(layout: str) -> None:
    case = _case(layout)
    w = _weights()
    diff = _run(case, w, differentiable=True)
    exact = _run(case, w.astype(np.int32))
    for field in ("joint", "marginal_a", "marginal_b", "total"):
        out = np.asarray(getattr(diff, field))
        assert out.dtype == np.float32, field
        assert np.array_equal(out, np.asarray(getattr(exact, field))), field
    assert exact.joint.dtype == jnp.int32


def test_float_weight_300_is_exact_on_differentiable_path() -> None:
    """int8 storage would wrap 300, so the float weights must be cast to int32 (spec S1-G3)."""
    case = _case("dense")
    w = np.full((1, T), 300.0, dtype=np.float32)
    diff = _run(case, w, differentiable=True)
    exact = _run(case, w.astype(np.int32))
    assert np.array_equal(np.asarray(diff.joint), np.asarray(exact.joint))
    assert np.asarray(diff.total)[0] == 300.0 * T


# --- S1-G5 item 3: per-field Jacobians equal the closed-form float64 Jacobian --------


@pytest.mark.parametrize("field", ("joint", "marginal_a", "marginal_b", "total"))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_jacobian_matches_closed_form_per_field(layout: str, field: str) -> None:
    case = _case(layout)
    expected = _closed_form_jacobian(field, case)
    got = _jax_jacobian(field, case, _weights())
    assert np.any(expected != 0.0), "closed-form Jacobian is all zero (A11 trap)"
    assert np.any(got != 0.0), "custom_vjp Jacobian is all zero: a silent zero gradient"
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


# --- Tiling of the backward pass (spec §4.6, §5): the plan's sizes must not change the answer


@pytest.mark.parametrize("method", ("matmul", "scatter"))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_backward_with_row_blocks_frame_padding_and_replicate_padding(
    layout: str, method: str
) -> None:
    """Non-dividing sizes: row_block=1 (dense), t_chunk=3 of T=8, r_tile=2 of R=3, pair_tile=2."""
    case = _case(layout)
    n_a = case["xa"].shape[0]
    # Dense splits the a rows into blocks of one; pairs are one block (row_block == Pa).
    row_block = n_a if layout in ("pairs", "self_pairs") else 1
    plan = CooccurrencePlan(
        method=method,
        r_tile=2,
        t_chunk=3,
        row_block=row_block,
        pair_tile=2,
        r_padded=4,
        t_padded=9,
        estimate_bytes=0,
        budget_bytes=None,
    )
    w = _weights()
    for field in ("joint", "marginal_a"):
        got = np.asarray(
            jax.jacrev(
                lambda w_: getattr(
                    _run(case, w_, differentiable=True, method=method, plan=plan), field
                )
            )(jnp.asarray(w)),
            dtype=np.float64,
        )
        assert np.array_equal(got, _closed_form_jacobian(field, case)), field


def test_plan_estimate_includes_backward_term() -> None:
    kwargs = {
        "n_replicates": R,
        "n_frames": T,
        "n_a": 2,
        "k_a": K_A,
        "n_b": 3,
        "k_b": K_B,
        "n_pairs": 6,
        "weight_itemsize": 4,
        "code_itemsize": 1,
        "method": "matmul",
        "layout": "dense",
        "platform": "cpu",
    }
    fwd = plan_cooccurrence(**kwargs)
    bwd = plan_cooccurrence(**kwargs, differentiable=True)
    # r_tile * row_block * Pb * t_chunk * 4, with r_tile=1, row_block=Pa=2, Pb=3, t_chunk=T=8.
    assert bwd.estimate_bytes - fwd.estimate_bytes == 1 * 2 * 3 * 8 * 4


# --- S1-G5 item 4: MI-form loss, custom_vjp against strategy="float" autodiff ---------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_mi_form_gradient_matches_float_strategy(layout: str) -> None:
    case = _case(layout)
    w = jnp.asarray(_weights())
    g_diff = np.asarray(
        jax.grad(lambda w_: _mi_loss(_run(case, w_, differentiable=True, method="matmul")))(w)
    )
    g_float = np.asarray(jax.grad(lambda w_: _mi_loss(_run(case, w_, strategy="float")))(w))
    assert np.any(g_diff != 0.0)
    np.testing.assert_allclose(g_diff, g_float, rtol=1e-5, atol=1e-6 * np.max(np.abs(g_float)))


@pytest.mark.parametrize("layout", ("dense", "self"))
def test_marginals_only_loss_has_nonzero_gradient_matching_float(layout: str) -> None:
    case = _case(layout)
    w = jnp.asarray(_weights())
    g_diff = np.asarray(
        jax.grad(lambda w_: _marginal_loss(_run(case, w_, differentiable=True, method="matmul")))(w)
    )
    g_float = np.asarray(jax.grad(lambda w_: _marginal_loss(_run(case, w_, strategy="float")))(w))
    assert np.any(g_diff != 0.0)
    np.testing.assert_allclose(g_diff, g_float, rtol=1e-5, atol=1e-6 * np.max(np.abs(g_float)))


# --- S1-G5 item 5: forward mode raises TypeError on the exact differentiable path ---


def test_jvp_on_differentiable_path_raises_type_error() -> None:
    case = _case("dense")
    w = jnp.asarray(_weights())
    with pytest.raises(TypeError):
        jax.jvp(lambda w_: _run(case, w_, differentiable=True).joint, (w,), (w,))


def test_jacfwd_on_differentiable_path_raises_type_error() -> None:
    case = _case("dense")
    w = jnp.asarray(_weights())
    with pytest.raises(TypeError):
        jax.jacfwd(lambda w_: _run(case, w_, differentiable=True).total)(w)


# --- S1-G5 item 6: strategy="float" on soft assignments, against the f64 analytic gradient --


def _soft_analytic_gradient(case: dict, weights: np.ndarray, cot: dict) -> np.ndarray:
    """``dL/dw`` in float64 for ``L = sum(Gj*joint) + sum(Ga*ma) + sum(Gb*mb) + sum(Gt*total)``."""
    oa = _shifted_soft(case["soft_a"], case["sa"])
    if case["self"]:
        ob = oa
    else:
        ob = _shifted_soft(case["soft_b"], case["sb"])
    if case["pairs"] is None:
        dw = np.einsum("rabij,rtai,rtbj->rt", cot["joint"], oa, ob)
    else:
        ia, ib = case["pairs"][:, 0], case["pairs"][:, 1]
        dw = np.einsum("rpij,rtpi,rtpj->rt", cot["joint"], oa[:, :, ia, :], ob[:, :, ib, :])
    dw = dw + np.einsum("rai,rtai->rt", cot["marginal_a"], oa)
    dw = dw + np.einsum("rbj,rtbj->rt", cot["marginal_b"], ob)
    return dw + cot["total"][:, None]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_float_strategy_soft_gradient_matches_f64_analytic(layout: str) -> None:
    case = _case(layout)
    w = _weights()
    base = _run(case, w, soft=True, strategy="float")
    rng = np.random.default_rng(99)
    cot = {
        "joint": rng.normal(size=base.joint.shape),
        "marginal_a": rng.normal(size=base.marginal_a.shape),
        "marginal_b": rng.normal(size=base.marginal_b.shape),
        "total": rng.normal(size=base.total.shape),
    }

    def loss(w_: jax.Array) -> jax.Array:
        out = _run(case, w_, soft=True, strategy="float")
        return (
            jnp.sum(jnp.asarray(cot["joint"], jnp.float32) * out.joint)
            + jnp.sum(jnp.asarray(cot["marginal_a"], jnp.float32) * out.marginal_a)
            + jnp.sum(jnp.asarray(cot["marginal_b"], jnp.float32) * out.marginal_b)
            + jnp.sum(jnp.asarray(cot["total"], jnp.float32) * out.total)
        )

    got = np.asarray(jax.grad(loss)(jnp.asarray(w)), dtype=np.float64)
    expected = _soft_analytic_gradient(case, w, cot)
    assert np.any(expected != 0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5 * np.max(np.abs(expected)))


# --- Float strategy on hard codes: exact for integer weights below 2**24 -----------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_float_strategy_hard_codes_equal_exact_counts(layout: str) -> None:
    case = _case(layout)
    w = _weights()
    flt = _run(case, w, strategy="float")
    exact = _run(case, w.astype(np.int32))
    assert flt.joint.dtype == jnp.float32
    for field in ("joint", "marginal_a", "marginal_b", "total"):
        assert np.array_equal(np.asarray(getattr(flt, field)), np.asarray(getattr(exact, field))), (
            field
        )


# --- Routing and option errors on the new paths -------------------------------------


def test_soft_codes_with_differentiable_exact_raise_type_error() -> None:
    case = _case("dense")
    with pytest.raises(TypeError, match="CategoricalCodes"):
        _run(case, _weights(), soft=True, differentiable=True)


def test_float_strategy_rejects_scatter_and_budget() -> None:
    case = _case("dense")
    with pytest.raises(ValueError, match="method"):
        _run(case, _weights(), strategy="float", method="scatter")
    with pytest.raises(ValueError, match="untiled"):
        _run(case, _weights(), strategy="float", byte_budget=10**9)


# --- S1-G3 variant under jit(grad): check=True raises under trace ------------------------


def _grad_under_jit(ca, cb, w, *, check: bool = True, **kwargs):
    def loss(w_: jax.Array) -> jax.Array:
        out = cooccurrence_counts(ca, cb, weights=w_, check=check, method="matmul", **kwargs)
        return jnp.sum(out.joint) + jnp.sum(out.total)

    return jax.block_until_ready(jax.jit(jax.grad(loss))(jnp.asarray(w)))


def test_jit_grad_clean_inputs_match_eager() -> None:
    case = _case("dense")
    ca = encode_codes(case["xa"], case["ka"])
    cb = encode_codes(case["xb"], case["kb"])
    w = _weights()

    def loss(w_: jax.Array) -> jax.Array:
        out = cooccurrence_counts(
            ca, cb, weights=w_, differentiable=True, method="matmul", shifts_a=case["sa"]
        )
        return jnp.sum(out.joint) + jnp.sum(out.total)

    eager = np.asarray(jax.grad(loss)(jnp.asarray(w)))
    jitted = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(w)))
    assert np.array_equal(eager, jitted)


def test_jit_grad_out_of_range_code_raises_c1() -> None:
    xa = np.zeros((2, 4), dtype=np.int64)
    xa[1, 2] = 3  # K=3, so code 3 is out of range
    ca = _raw_codes(xa, 3)
    cb = encode_codes(np.zeros((1, 4), dtype=np.int64), 3)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C1 failed"):
        _grad_under_jit(ca, cb, np.ones((1, 4), dtype=np.float32), differentiable=True)


def test_jit_grad_negative_weight_raises_c2() -> None:
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 2)
    w = np.array([[1.0, -1.0, 1.0, 1.0]], dtype=np.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C2 failed: weights must be >= 0"):
        _grad_under_jit(ca, ca, w, differentiable=True)


def test_jit_grad_non_integer_float_weight_raises_c2() -> None:
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 2)
    w = np.array([[1.0, 1.5, 1.0, 1.0]], dtype=np.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C2 failed"):
        _grad_under_jit(ca, ca, w, differentiable=True)


def test_jit_grad_total_two_to_the_24_plus_one_raises_c3() -> None:
    """Two frames summing to 2**24 + 1; each is below 2**24, so only the running total trips."""
    ca = encode_codes(np.zeros((1, 2), dtype=np.int64), 2)
    w = np.array([[2.0**23, 2.0**23 + 1.0]], dtype=np.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C3 failed"):
        _grad_under_jit(ca, ca, w, differentiable=True)


def test_jit_grad_float_strategy_total_two_to_the_24_plus_one_raises_c3() -> None:
    ca = encode_codes(np.zeros((1, 2), dtype=np.int64), 2)
    w = np.array([[2.0**23, 2.0**23 + 1.0]], dtype=np.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C3 failed"):
        _grad_under_jit(ca, ca, w, strategy="float")


def test_jit_grad_non_integer_float_strategy_has_no_total_bound() -> None:
    """Non-integer weights make the float strategy approximate, so no C3 bound applies."""
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 2)
    # 2**22 + 0.5 is exact in float32, and four of them total 2**24 + 2 (above the bound).
    w = np.full((1, 4), 2.0**22 + 0.5, dtype=np.float32)
    grad = _grad_under_jit(ca, ca, w, strategy="float")
    assert np.all(np.isfinite(grad))


def test_jit_grad_negative_weight_is_silent_when_check_is_off() -> None:
    """C4: check=False skips the traced checks, so no error is raised (results undefined)."""
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 2)
    w = np.array([[1.0, -1.0, 1.0, 1.0]], dtype=np.float32)
    _grad_under_jit(ca, ca, w, check=False, differentiable=True)


def test_jit_forward_float_weight_300_is_exact() -> None:
    case = _case("dense")
    ca = encode_codes(case["xa"], case["ka"])
    cb = encode_codes(case["xb"], case["kb"])
    w = np.full((1, T), 300.0, dtype=np.float32)
    traced = jax.jit(lambda w_: cooccurrence_counts(ca, cb, weights=w_, differentiable=True).joint)(
        jnp.asarray(w)
    )
    eager = cooccurrence_counts(ca, cb, weights=w.astype(np.int32)).joint
    assert np.array_equal(np.asarray(traced), np.asarray(eager))

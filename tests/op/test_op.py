"""Tests for xtrax.op, the standalone kernel seam (backlog #6098).

Kernels are Pallas in interpret mode (CPU). A Context with ``kernels=True`` and
``backend="cpu"`` selects them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xtrax
from tests.op._kernels import (
    make_binned_sum_kernel,
    make_binned_sum_reference,
    make_hist_kernel,
    make_hist_reference,
    matmul_kernel,
    matmul_reference,
    scale_kernel,
    scale_reference,
)
from xtrax.intent import Context, IntentError
from xtrax.kernel_seam import VjpKernel, op
from xtrax.testing.kernel_parity import assert_kernel_ran, scale_error_control

CPU_KERNELS = Context(kernels=True, backend="cpu")


def kop(*args, **kwargs):
    """``op`` bound to a Context that selects the CPU kernels."""
    return op(*args, context=CPU_KERNELS, **kwargs)


NBINS = 5


def _problem():
    kx, kw = jax.random.split(jax.random.PRNGKey(0))
    return jax.random.normal(kx, (16, 24)), jax.random.normal(kw, (24, 16))


def _close(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return all(
        np.allclose(u, v, atol=1e-5 * max(1.0, float(np.max(np.abs(v)))), rtol=1e-5)
        for u, v in zip(la, lb, strict=True)
    )


def _mm_vjp(bwd_scale=None):
    def fwd(x, w):
        return matmul_kernel(x, w), (x, w)

    def bwd(res, ct):
        x, w = res
        return matmul_kernel(ct, w.T), matmul_kernel(x.T, ct)

    vk = VjpKernel(fwd, bwd)
    return vk if bwd_scale is None else scale_error_control(vk, bwd_scale)


# --- public surface ------------------------------------------------------------------------


def test_exported_from_xtrax():
    assert xtrax.op is op
    assert xtrax.VjpKernel is VjpKernel
    assert not hasattr(xtrax, "op_context")  # no ambient context: bind with Op.with_context


def test_op_keeps_reference_metadata():
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel})
    assert mm.name == "matmul_reference"
    assert op(matmul_reference, name="mm").name == "mm"


# --- explicit context binding --------------------------------------------------------------


def test_with_context_returns_a_new_op_and_leaves_the_original_unbound():
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel})
    bound = mm.with_context(CPU_KERNELS)
    assert bound is not mm and bound != mm and hash(bound) != hash(mm)
    assert mm.context == Context() and bound.context == CPU_KERNELS
    assert (bound.reference, bound.kernels, bound.name) == (mm.reference, mm.kernels, mm.name)
    assert mm.select().form == "reference" and bound.select().form == "kernel"


def test_op_wrapping_an_op_keeps_its_own_fields():
    inner = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()}, name="inner")
    outer = op(inner, kernels={"cpu": matmul_kernel}, name="outer")
    assert outer.name == "outer" and outer.context == Context()
    assert set(outer.kernels) == {"cpu"} and outer.vjp_kernels == {}
    assert outer.__wrapped__ is inner


def test_op_is_immutable():
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel})
    with pytest.raises(AttributeError, match="immutable"):
        mm.context = CPU_KERNELS


def test_context_keyword_equals_with_context():
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel}, context=CPU_KERNELS)
    assert mm.context == CPU_KERNELS and mm.select().form == "kernel"


def test_jit_caches_cannot_cross_contexts():
    """Differently bound ops are different callables, so each jit traces its own selection.

    The motivating failure: with a shared Op whose Context came from ambient state, one
    ``jax.jit(my_op)`` first traced under ``kernels=True`` was reused unchanged under
    ``kernels=False``, silently running a path the caller did not ask for.
    """
    x, w = _problem()
    unbound = op(matmul_reference, kernels={"cpu": matmul_kernel})
    bound = unbound.with_context(CPU_KERNELS)
    jit_bound, jit_unbound = jax.jit(bound), jax.jit(unbound)
    assert "pallas_call" in str(jax.make_jaxpr(jit_bound)(x, w))
    assert "pallas_call" not in str(jax.make_jaxpr(jit_unbound)(x, w))
    # the other order, same input shapes: still no cross-talk
    assert "pallas_call" not in str(jax.make_jaxpr(jit_unbound)(x, w))
    assert "pallas_call" in str(jax.make_jaxpr(jit_bound)(x, w))
    assert _close(jit_bound(x, w), jit_unbound(x, w))


# --- selection rules -----------------------------------------------------------------------


def test_default_context_runs_reference():
    x, w = _problem()
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel})
    sel = mm.select()
    assert (sel.form, sel.primal, sel.reason) == ("reference", "reference", "kernels=False")
    assert "pallas_call" not in str(jax.make_jaxpr(mm)(x, w))


def test_kernel_selected_for_resolved_backend_and_actually_runs():
    x, w = _problem()
    mm = kop(matmul_reference, kernels={"cpu": matmul_kernel})
    sel = mm.select()
    assert (sel.form, sel.primal) == ("kernel", "kernel:cpu")
    assert_kernel_ran(mm, x, w)
    assert _close(mm(x, w), matmul_reference(x, w))


def test_resolved_backend_defaults_to_jax_default_backend():
    mm = op(matmul_reference, kernels={jax.default_backend(): matmul_kernel})
    assert mm.select(Context(kernels=True)).primal == f"kernel:{jax.default_backend()}"


def test_export_target_forces_reference_even_when_kernel_exists():
    x, w = _problem()
    export_ctx = Context(target="export", export_format="stablehlo", kernels=True, backend="cpu")
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel}, context=export_ctx)
    sel = mm.select()
    assert (sel.form, sel.reason) == ("reference", "export target")
    assert "pallas_call" not in str(jax.make_jaxpr(mm)(x, w))


def test_kernels_false_forces_reference():
    mm = op(matmul_reference, kernels={"cpu": matmul_kernel})
    sel = mm.select(Context(kernels=False, backend="cpu"))
    assert (sel.form, sel.reason) == ("reference", "kernels=False")


def test_no_kernel_for_backend_runs_reference():
    x, w = _problem()
    mm = kop(matmul_reference, kernels={"gpu": matmul_kernel})
    sel = mm.select()
    assert (sel.form, sel.reason) == ("reference", "no kernel for backend 'cpu'")
    assert "pallas_call" not in str(jax.make_jaxpr(mm)(x, w))


def test_vjp_kernel_takes_precedence_for_same_backend():
    mm = op(
        matmul_reference,
        kernels={"cpu": matmul_kernel},
        vjp_kernels={"cpu": _mm_vjp()},
    )
    assert mm.select(CPU_KERNELS).form == "vjp_kernel"
    assert mm.select(Context(kernels=True, backend="gpu")).form == "reference"


def test_kernel_signature_mismatch_is_an_intent_error():
    x, w = _problem()
    bad = kop(matmul_reference, kernels={"cpu": lambda a, b: matmul_kernel(a, b)[:, :8]}, name="mm")
    with pytest.raises(IntentError, match="mm") as err:
        bad(x, w)
    assert err.value.site == "mm" and err.value.path == "primal"
    wrong_dtype = kop(
        matmul_reference, kernels={"cpu": lambda a, b: matmul_kernel(a, b).astype(jnp.float16)}
    )
    with pytest.raises(IntentError, match="float16"):
        wrong_dtype(x, w)


# --- custom_jvp form: ported from probe q6 -------------------------------------------------


@pytest.fixture
def mm_pair():
    return kop(matmul_reference, kernels={"cpu": matmul_kernel}), matmul_reference


def _loss(fn):
    def loss(x, w):
        return jnp.sum(jnp.tanh(fn(x, w)))

    return loss


def test_jvp_form_forward_and_grad(mm_pair):
    mm, ref = mm_pair
    x, w = _problem()
    assert _close(mm(x, w), ref(x, w))
    assert _close(jax.grad(_loss(mm), (0, 1))(x, w), jax.grad(_loss(ref), (0, 1))(x, w))


def test_jvp_form_jvp(mm_pair):
    mm, ref = mm_pair
    x, w = _problem()
    tx = jnp.ones_like(x)
    got = jax.jvp(lambda a: mm(a, w), (x,), (tx,))[1]
    assert _close(got, jax.jvp(lambda a: ref(a, w), (x,), (tx,))[1])


def test_jvp_form_hessian_jacfwd_of_jacrev(mm_pair):
    mm, ref = mm_pair
    x, w = _problem()
    xs, ws = x[:8, :8], w[:8, :8]
    got = jax.jacfwd(jax.jacrev(lambda a: _loss(mm)(a, ws)))(xs)
    assert _close(got, jax.hessian(lambda a: _loss(ref)(a, ws))(xs))


def test_jvp_form_vmap_grad_and_jit_grad(mm_pair):
    mm, ref = mm_pair
    x, w = _problem()
    xb = jnp.stack([x, 2 * x])
    vg = jax.vmap(jax.grad(lambda a: _loss(mm)(a, w)))(xb)
    jg = jax.jit(jax.grad(lambda a: _loss(mm)(a, w)))(x)
    assert _close(vg, jax.vmap(jax.grad(lambda a: _loss(ref)(a, w)))(xb))
    assert _close(jg, jax.grad(lambda a: _loss(ref)(a, w))(x))


def test_jvp_form_forward_jaxpr_keeps_pallas_call_under_grad(mm_pair):
    mm, _ = mm_pair
    x, w = _problem()
    assert_kernel_ran(mm, x, w)
    assert_kernel_ran(jax.grad(_loss(mm)), x, w)


def test_control_derivative_comes_from_the_reference_not_the_kernel():
    """A reference that differs from the kernel changes the gradient: the checks can go red."""
    x, w = _problem()
    doubled = kop(lambda a, b: 2.0 * (a @ b), kernels={"cpu": matmul_kernel})
    got = jax.grad(_loss(doubled))(x, w)
    assert not _close(got, jax.grad(_loss(matmul_reference))(x, w))


def test_elementwise_kernel_scalar_derivative():
    sc = kop(scale_reference, kernels={"cpu": scale_kernel})
    x = jnp.arange(8.0)
    assert_kernel_ran(sc, x)
    assert _close(jax.grad(lambda a: jnp.sum(sc(a)))(x), 3.0 * jnp.ones(8))


# --- integer outputs: float0 (probe result in test_op_probes.py) --------------------------


def test_integer_only_output_inside_grad():
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    hist = kop(make_hist_reference(NBINS), kernels={"cpu": make_hist_kernel(NBINS)})

    def loss(x):
        counts = hist(idx)
        return jnp.sum(x * counts / jnp.sum(counts))

    x = jnp.arange(1.0, NBINS + 1.0)
    assert_kernel_ran(hist, idx)
    counts = hist(idx)
    got = jax.grad(loss)(x)
    jit_got = jax.jit(jax.grad(loss))(x)
    _, tangent = jax.jvp(hist, (idx,), (np.zeros(idx.shape, jax.dtypes.float0),))
    assert counts.dtype == jnp.int32
    np.testing.assert_array_equal(counts, [2, 2, 1, 0, 3])
    np.testing.assert_allclose(got, np.array([2, 2, 1, 0, 3]) / 8.0, rtol=1e-6)
    np.testing.assert_allclose(jit_got, got, rtol=1e-6)
    assert tangent.dtype == jax.dtypes.float0


def test_mixed_float_and_int_outputs_with_int_input():
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    ref = make_binned_sum_reference(NBINS)
    bs = kop(ref, kernels={"cpu": make_binned_sum_kernel(NBINS)})
    x = jnp.arange(1.0, 9.0)

    def loss(fn):
        def inner(a):
            sums, counts = fn(a, idx)
            return jnp.sum(sums**2 / (counts + 1))

        return inner

    assert_kernel_ran(bs, x, idx)
    got = jax.grad(loss(bs))(x)
    assert _close(got, jax.grad(loss(ref))(x))


# --- custom_vjp form -----------------------------------------------------------------------


def test_vjp_form_forward_grad_and_kernel_in_both_passes():
    x, w = _problem()
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()})
    assert _close(mm(x, w), matmul_reference(x, w))
    got = jax.grad(_loss(mm), (0, 1))(x, w)
    jit_got = jax.jit(jax.grad(_loss(mm), (0, 1)))(x, w)
    n_pallas = str(jax.make_jaxpr(jax.grad(_loss(mm), (0, 1)))(x, w)).count("pallas_call")
    want = jax.grad(_loss(matmul_reference), (0, 1))(x, w)
    assert _close(got, want) and _close(jit_got, want)
    assert n_pallas >= 3  # forward, dx, dw


def test_vjp_form_vmap_grad():
    x, w = _problem()
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()})
    xb = jnp.stack([x, 2 * x])
    got = jax.vmap(jax.grad(lambda a: _loss(mm)(a, w)))(xb)
    assert _close(got, jax.vmap(jax.grad(lambda a: _loss(matmul_reference)(a, w)))(xb))
    assert_kernel_ran(jax.vmap(jax.grad(lambda a: _loss(mm)(a, w))), xb)


def test_control_vjp_backward_kernel_is_what_runs_in_reverse_mode():
    """Corrupt the backward kernel by 0.1%: grad goes red, the forward value stays green."""
    x, w = _problem()
    bad = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp(bwd_scale=1.001)})
    assert _close(bad(x, w), matmul_reference(x, w))
    got = jax.grad(_loss(bad), (0, 1))(x, w)
    assert not _close(got, jax.grad(_loss(matmul_reference), (0, 1))(x, w))


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(
            lambda f, x, w: jax.jvp(lambda a: f(a, w), (x,), (jnp.ones_like(x),)), id="jvp"
        ),
        pytest.param(lambda f, x, w: jax.jacfwd(f)(x, w), id="jacfwd"),
        pytest.param(
            lambda f, x, w: jax.jit(
                lambda a: jax.jvp(lambda b: f(b, w), (a,), (jnp.ones_like(a),))
            )(x),
            id="jit_jvp",
        ),
        pytest.param(
            lambda f, x, w: jax.vmap(
                lambda a: jax.jvp(lambda b: f(b, w), (a,), (jnp.ones_like(a),))
            )(jnp.stack([x, x])),
            id="vmap_jvp",
        ),
    ],
)
def test_vjp_form_forward_mode_raises_intent_error(transform):
    x, w = _problem()
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()}, name="mm")
    with pytest.raises(IntentError) as err:
        transform(mm, x, w)
    message = str(err.value)
    assert "mm" in message and "cpu" in message
    assert "vjp kernel selected" in message and "forward mode needs the JVP form" in message
    assert (err.value.site, err.value.path) == ("mm", "diff")


def test_vjp_form_second_order_reverse_over_reverse_uses_the_reference():
    x, w = _problem()
    xs, ws = x[:8, :8], w[:8, :8]
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()})
    got = jax.jacrev(jax.jacrev(lambda a: _loss(mm)(a, ws)))(xs)
    cross = jax.jacrev(jax.grad(_loss(mm), 0), 1)(xs, ws)
    ref_loss = _loss(matmul_reference)
    assert _close(got, jax.hessian(lambda a: ref_loss(a, ws))(xs))
    want_cross = jax.jacrev(jax.grad(ref_loss, 0), 1)(xs, ws)
    assert _close(cross, want_cross)
    assert np.max(np.abs(want_cross)) > 1e-3


def test_vjp_form_second_order_forward_over_reverse():
    x, w = _problem()
    xs, ws = x[:8, :8], w[:8, :8]
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()})
    got = jax.jacfwd(jax.jacrev(lambda a: _loss(mm)(a, ws)))(xs)
    assert _close(got, jax.hessian(lambda a: _loss(matmul_reference)(a, ws))(xs))
    assert_kernel_ran(jax.jacfwd(jax.jacrev(lambda a: _loss(mm)(a, ws))), xs)


def test_non_kernel_backend_keeps_forward_mode():
    """Same model: reverse-only on the vjp-kernel backend, fully differentiable elsewhere."""
    x, w = _problem()
    mm = kop(matmul_reference, vjp_kernels={"gpu": _mm_vjp()})
    tx = jnp.ones_like(x)
    assert mm.select().form == "reference"
    got = jax.jvp(lambda a: mm(a, w), (x,), (tx,))[1]
    assert _close(got, jax.jvp(lambda a: matmul_reference(a, w), (x,), (tx,))[1])


def test_jvp_kernel_backend_keeps_forward_mode_when_vjp_kernel_is_on_another_backend():
    x, w = _problem()
    mm = kop(matmul_reference, kernels={"cpu": matmul_kernel}, vjp_kernels={"gpu": _mm_vjp()})
    tx = jnp.ones_like(x)
    assert mm.select().form == "kernel"
    got = jax.jvp(lambda a: mm(a, w), (x,), (tx,))[1]
    assert _close(got, jax.jvp(lambda a: matmul_reference(a, w), (x,), (tx,))[1])


def test_vjp_form_with_integer_input_and_integer_output():
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    ref = make_binned_sum_reference(NBINS)
    kern = make_binned_sum_kernel(NBINS)

    def fwd(x, i):
        return kern(x, i), (i,)

    def bwd(res, ct):
        (i,) = res
        ct_sums, _ = ct
        return ct_sums[i], None  # no cotangent for the integer input

    bs = kop(ref, vjp_kernels={"cpu": VjpKernel(fwd, bwd)})
    x = jnp.arange(1.0, 9.0)

    def loss(fn):
        def inner(a):
            sums, counts = fn(a, idx)
            return jnp.sum(sums**2 / (counts + 1))

        return inner

    got = jax.grad(loss(bs))(x)
    assert _close(got, jax.grad(loss(ref))(x))
    assert_kernel_ran(bs, x, idx)


# --- nested ops ----------------------------------------------------------------------------


def _tanh_mm_reference(inner):
    def reference(x, w):
        return jnp.tanh(inner(x, w))

    return reference


def _tanh_mm_kernel(x, w):
    return jnp.tanh(matmul_kernel(x, w))


def _tanh_mm_plain(x, w):
    return jnp.tanh(matmul_reference(x, w))


def test_nested_vjp_op_inside_a_jvp_form_reference_differentiates_under_grad():
    """The outer JVP rule differentiates its reference; the inner op runs its reference there."""
    x, w = _problem()
    inner = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()}, name="inner")
    outer = kop(_tanh_mm_reference(inner), kernels={"cpu": _tanh_mm_kernel}, name="outer")
    got = jax.grad(_loss(outer), (0, 1))(x, w)
    assert _close(got, jax.grad(_loss(_tanh_mm_plain), (0, 1))(x, w))
    assert _close(jax.jit(jax.grad(_loss(outer), (0, 1)))(x, w), got)
    assert_kernel_ran(outer, x, w)


def test_nested_vjp_op_inside_a_vjp_form_reference_differentiates_second_order():
    x, w = _problem()
    xs, ws = x[:8, :8], w[:8, :8]
    inner = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()}, name="inner")

    def fwd(a, b):
        out = _tanh_mm_kernel(a, b)
        return out, (a, b, out)

    def bwd(res, ct):
        a, b, out = res
        g = ct * (1 - out**2)
        return matmul_kernel(g, b.T), matmul_kernel(a.T, g)

    outer = kop(_tanh_mm_reference(inner), vjp_kernels={"cpu": VjpKernel(fwd, bwd)})
    got = jax.jacfwd(jax.jacrev(lambda a: _loss(outer)(a, ws)))(xs)
    assert _close(got, jax.hessian(lambda a: _loss(_tanh_mm_plain)(a, ws))(xs))


def test_user_forward_mode_through_a_nested_vjp_op_still_raises():
    """Only the seam's own reference derivatives switch inner ops to their reference."""
    x, w = _problem()
    inner = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()}, name="inner")
    with pytest.raises(IntentError, match="forward mode needs the JVP form"):
        jax.jvp(lambda a: jnp.tanh(inner(a, w)), (x,), (jnp.ones_like(x),))


@pytest.fixture
def no_direct_linearize():
    old = jax.config.jax_use_direct_linearize
    jax.config.update("jax_use_direct_linearize", False)
    yield
    jax.config.update("jax_use_direct_linearize", old)


def test_vjp_form_without_direct_linearize_reverse_ok_and_eager_forward_still_intent_error(
    no_direct_linearize,
):
    """JVPTracer is also the reverse-mode tracer here, so the pre-check must not fire on grad."""
    x, w = _problem()
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()}, name="mm")
    got = jax.grad(_loss(mm), (0, 1))(x, w)
    with pytest.raises(IntentError, match="forward mode needs the JVP form"):
        jax.jvp(lambda a: mm(a, w), (x,), (jnp.ones_like(x),))
    assert _close(got, jax.grad(_loss(matmul_reference), (0, 1))(x, w))


def test_vjp_form_without_direct_linearize_jit_jvp_gives_the_generic_jax_error(
    no_direct_linearize,
):
    """Documented limitation: under jit the error comes at lowering, outside the op."""
    x, w = _problem()
    mm = kop(matmul_reference, vjp_kernels={"cpu": _mm_vjp()})
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jit(lambda a: jax.jvp(lambda b: mm(b, w), (a,), (jnp.ones_like(a),)))(x)

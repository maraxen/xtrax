"""Probes of the two assumptions the xtrax.op spec marked unverified (261023, section 4).

1. An integer-output ``custom_jvp`` with ``float0`` tangents works inside ``grad`` of a program
   that reads the counts.
2. ``custom_jvp`` whose tangent map is a ``custom_vjp`` (``bwd`` = the backward kernel) keeps
   forward mode and gives kernel reverse mode, with the second-order cotangent taken from the
   reference's autodiff.

Both run on CPU in Pallas interpret mode. The probes build their own minimal ``custom_jvp`` so the
finding does not depend on the shipped op. Each ends with a corruption control.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.op._kernels import (
    make_binned_sum_kernel,
    make_binned_sum_reference,
    make_hist_kernel,
    make_hist_reference,
    matmul_kernel,
    matmul_reference,
)

NBINS = 5


def _float0_zeros(x):
    return np.zeros(np.shape(x), jax.dtypes.float0)


def _tangent_or_float0(x, t):
    return t if jnp.issubdtype(jnp.result_type(x), jnp.inexact) else _float0_zeros(x)


def _proto_jvp_op(reference, kernel, *, corrupt=1.0):
    """Minimal ``custom_jvp``: primal = kernel, JVP = ``jax.jvp(reference)``."""

    @jax.custom_jvp
    def f(*args):
        return kernel(*args)

    @f.defjvp
    def f_jvp(primals, tangents):
        tangents = tuple(_tangent_or_float0(p, t) for p, t in zip(primals, tangents, strict=True))
        out = f(*primals)
        _, tout = jax.jvp(reference, primals, tangents)
        return out, jax.tree.map(lambda t: t if t.dtype == jax.dtypes.float0 else t * corrupt, tout)

    return f


# --- Probe 1: float0 -----------------------------------------------------------------------


def test_probe_float0_int_only_output_inside_grad():
    """An op returning only int32 counts, read inside a differentiated program."""
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    op = _proto_jvp_op(make_hist_reference(NBINS), make_hist_kernel(NBINS))

    def loss(x, count_fn):
        counts = count_fn(idx)
        return jnp.sum(x * counts / jnp.sum(counts))

    x = jnp.arange(1.0, NBINS + 1.0)
    got = jax.grad(loss)(x, op)
    want = jax.grad(loss)(x, make_hist_reference(NBINS))
    np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_allclose(got, np.array([2, 2, 1, 0, 3]) / 8.0, rtol=1e-6)
    # the counts are int32 and the kernel (not the reference) produced them
    assert op(idx).dtype == jnp.int32
    assert "pallas_call" in str(jax.make_jaxpr(op)(idx))
    # forward mode: the output tangent is float0
    _, tout = jax.jvp(op, (idx,), (_float0_zeros(idx),))
    assert tout.dtype == jax.dtypes.float0 and tout.shape == (NBINS,)
    # jit(grad) and vmap(grad) as well
    np.testing.assert_allclose(jax.jit(jax.grad(loss), static_argnums=1)(x, op), want, rtol=1e-6)
    np.testing.assert_allclose(
        jax.vmap(jax.grad(loss), in_axes=(0, None))(jnp.stack([x, 2 * x]), op),
        jnp.stack([want, want]),  # the loss is linear in x, so the gradient is x-independent
        rtol=1e-6,
    )


def test_probe_float0_mixed_outputs_and_int_input():
    """Float and int outputs plus an int input: both float0 directions at once."""
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    ref = make_binned_sum_reference(NBINS)
    op = _proto_jvp_op(ref, make_binned_sum_kernel(NBINS))
    x = jnp.arange(1.0, 9.0)

    def loss(x_, fn):
        sums, counts = fn(x_, idx)
        return jnp.sum(sums**2 / (counts + 1))

    got = jax.grad(loss)(x, op)
    want = jax.grad(loss)(x, ref)
    np.testing.assert_allclose(got, want, rtol=1e-5)
    # a jvp whose int input carries a float0 tangent
    (_, _), (ts, tc) = jax.jvp(op, (x, idx), (jnp.ones_like(x), _float0_zeros(idx)))
    np.testing.assert_allclose(
        ts, jax.jvp(ref, (x, idx), (jnp.ones_like(x), _float0_zeros(idx)))[1][0]
    )
    assert tc.dtype == jax.dtypes.float0


def test_probe_float0_control_corrupted_jvp_turns_red():
    """Negative control: scaling the JVP by 2 must break the grad comparison."""
    idx = jnp.array([0, 1, 1, 4, 4, 4, 2, 0], jnp.int32)
    ref = make_binned_sum_reference(NBINS)
    bad = _proto_jvp_op(ref, make_binned_sum_kernel(NBINS), corrupt=2.0)
    x = jnp.arange(1.0, 9.0)

    def loss(x_, fn):
        sums, counts = fn(x_, idx)
        return jnp.sum(sums**2 / (counts + 1))

    assert not np.allclose(jax.grad(loss)(x, bad), jax.grad(loss)(x, ref), rtol=1e-3)


# --- Probe 2: custom_jvp whose tangent map is a custom_vjp ---------------------------------


def _alt_op(reference, kernel, bwd_kernel, *, bwd_corrupt=None):
    """``custom_jvp`` with a ``custom_vjp`` tangent map; ``bwd`` is the backward kernel.

    ``tangent_map(primals, tangents)`` is linear in ``tangents`` and equals the reference's JVP.
    Its backward pass returns the kernel cotangent for ``tangents`` and the reference's
    autodiff cotangent for ``primals`` (second order).
    """

    @jax.custom_vjp
    def tangent_map(primals, tangents):
        return jax.jvp(reference, primals, tangents)[1]

    def tm_fwd(primals, tangents):
        return tangent_map(primals, tangents), (primals, tangents)

    def tm_bwd(res, ct):
        primals, tangents = res
        d_tangents = bwd_kernel(primals, ct)
        if bwd_corrupt is not None:
            d_tangents = bwd_corrupt(d_tangents)
        _, pull = jax.vjp(lambda p: jax.jvp(reference, p, tangents)[1], primals)
        (d_primals,) = pull(ct)
        return d_primals, d_tangents

    tangent_map.defvjp(tm_fwd, tm_bwd)

    @jax.custom_jvp
    def f(*args):
        return kernel(*args)

    @f.defjvp
    def f_jvp(primals, tangents):
        return f(*primals), tangent_map(primals, tangents)

    return f


def _mm_bwd(primals, ct):
    x, w = primals
    return matmul_kernel(ct, w.T), matmul_kernel(x.T, ct)


def _problem():
    kx, kw = jax.random.split(jax.random.PRNGKey(0))
    return jax.random.normal(kx, (16, 24)), jax.random.normal(kw, (24, 16))


def _loss(fn):
    def loss(x, w):
        return jnp.sum(jnp.tanh(fn(x, w)))

    return loss


def _close(a, b):
    a, b = jax.tree.leaves(a), jax.tree.leaves(b)
    return all(
        np.allclose(u, v, atol=1e-5 * max(1.0, float(np.max(np.abs(v)))), rtol=1e-5)
        for u, v in zip(a, b, strict=True)
    )


def test_probe_alt_forward_mode_works():
    """Forward mode evaluates the tangent map directly; the custom_vjp is never differentiated."""
    x, w = _problem()
    op = _alt_op(matmul_reference, matmul_kernel, _mm_bwd)
    tx = jnp.ones_like(x)
    assert _close(
        jax.jvp(lambda a: op(a, w), (x,), (tx,))[1],
        jax.jvp(lambda a: matmul_reference(a, w), (x,), (tx,))[1],
    )


@pytest.mark.parametrize("transform", ["grad", "jit_grad", "vmap_grad", "jacrev"])
def test_probe_alt_reverse_mode_fails_to_transpose(transform):
    """FAIL: reverse mode transposes the JVP rule's tangent output; custom_vjp_call cannot.

    jax 0.11.1. This is the probe result that rules the alternative out: the spec's first-cut
    forward-mode IntentError stays.
    """
    x, w = _problem()
    op = _alt_op(matmul_reference, matmul_kernel, _mm_bwd)
    loss = _loss(op)
    grad_fn = {
        "grad": jax.grad(loss),
        "jit_grad": jax.jit(jax.grad(loss)),
        "vmap_grad": lambda a, b: jax.vmap(lambda a_: jax.grad(loss)(a_, b))(jnp.stack([a, a])),
        "jacrev": jax.jacrev(lambda a, b: jnp.tanh(op(a, b))),
    }[transform]
    with pytest.raises(NotImplementedError, match="custom_vjp_call"):
        grad_fn(x, w)


def test_probe_alt_control_plain_custom_jvp_form_transposes():
    """Control: with the same inputs, the plain custom_jvp form differentiates in reverse."""
    x, w = _problem()
    plain = _proto_jvp_op(matmul_reference, matmul_kernel)
    assert _close(
        jax.grad(_loss(plain), (0, 1))(x, w), jax.grad(_loss(matmul_reference), (0, 1))(x, w)
    )

"""Throwaway spike: an xtrax.loop intent primitive that lowers to while_loop unless
differentiated, in which case its JVP rule emits a scan.

Checks:
  s1 jit forward jaxpr/lowering uses while (no scan) and matches a reference scan
  s2 jax.grad works, matches grad of the scan reference, and the grad jaxpr uses scan
  s3 vmap works
  s4 jax.export of the forward (cpu) works and the module contains no custom_call
"""

import functools

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

loop_p = Primitive("xtrax_loop")
loop_p.multiple_results = True


def _while_impl(*carry, body, n_steps):
    def cond(state):
        return state[0] < n_steps

    def step(state):
        i, c = state
        return i + 1, body(c)

    _, out = jax.lax.while_loop(cond, step, (jnp.int32(0), tuple(carry)))
    return list(out)


def _scan_impl(*carry, body, n_steps):
    out, _ = jax.lax.scan(lambda c, _: (body(c), None), tuple(carry), None, length=n_steps)
    return list(out)


loop_p.def_impl(_while_impl)
loop_p.def_abstract_eval(lambda *avals, body, n_steps: list(avals))
mlir.register_lowering(loop_p, mlir.lower_fun(_while_impl, multiple_results=True))


def _jvp(primals, tangents, *, body, n_steps):
    tangents = [ad.instantiate_zeros(t) for t in tangents]
    f = functools.partial(_scan_impl, body=body, n_steps=n_steps)
    out, t_out = jax.jvp(lambda *c: tuple(f(*c)), tuple(primals), tuple(tangents))
    return list(out), list(t_out)


ad.primitive_jvps[loop_p] = _jvp


def _batch(args, dims, *, body, n_steps):
    size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
    args = [
        jnp.moveaxis(a, d, 0) if d is not None else jnp.broadcast_to(a, (size, *a.shape))
        for a, d in zip(args, dims)
    ]
    vbody = jax.vmap(body)
    out = loop_p.bind(*args, body=lambda c: tuple(vbody(c)), n_steps=n_steps)
    return out, [0] * len(out)


batching.primitive_batchers[loop_p] = _batch


def xloop(body, init, n_steps):
    """Intent: iterate body n_steps times over a tuple carry."""
    return tuple(loop_p.bind(*init, body=body, n_steps=n_steps))


def body(c):
    (x,) = c
    return (jnp.sin(x) * 0.9 + 0.1 * x,)


def f_loop(x):
    return xloop(body, (x,), 64)[0].sum()


def f_ref(x):
    return _scan_impl(x, body=body, n_steps=64)[0].sum()


x = jnp.linspace(-1.0, 1.0, 8)

fwd = jax.make_jaxpr(jax.jit(f_loop))(x)
low = jax.jit(f_loop).lower(x).as_text()
s1 = bool(jnp.allclose(jax.jit(f_loop)(x), f_ref(x))) and "while" in low
print(
    "s1 forward matches ref:",
    bool(jnp.allclose(jax.jit(f_loop)(x), f_ref(x))),
    "| lowered has while:",
    "while" in low,
)

g = jax.jit(jax.grad(f_loop))(x)
g_ref = jax.grad(f_ref)(x)
gj = str(jax.make_jaxpr(jax.grad(f_loop))(x))
s2 = bool(jnp.allclose(g, g_ref)) and "scan" in gj
print(
    "s2 grad matches ref:",
    bool(jnp.allclose(g, g_ref)),
    "| grad jaxpr has scan:",
    "scan" in gj,
    "| has while:",
    "while[" in gj,
)

xb = jnp.stack([x, x * 0.5])
vb = jax.vmap(f_loop)(xb)
s3 = bool(jnp.allclose(vb, jax.vmap(f_ref)(xb)))
print("s3 vmap matches ref:", s3)

try:
    exp = jax.export.export(jax.jit(f_loop), platforms=["cpu"])(x)
    txt = exp.mlir_module()
    s4 = "custom_call" not in txt and bool(jnp.allclose(exp.call(x), f_ref(x)))
    print("s4 export ok, no custom_call:", "custom_call" not in txt, "| roundtrip matches:", s4)
except Exception as e:  # noqa: BLE001
    s4 = False
    print("s4 export FAILED:", type(e).__name__, str(e)[:200])

print("RESULT", "PASS" if all([s1, s2, s3, s4]) else "FAIL", dict(s1=s1, s2=s2, s3=s3, s4=s4))

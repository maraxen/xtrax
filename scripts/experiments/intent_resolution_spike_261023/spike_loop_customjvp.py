"""Throwaway spike: intent loop via PUBLIC jax.custom_jvp.

primal = while_loop; jvp rule = jvp of the scan form. Reverse mode should come
from transposing the (scan) JVP. Checks forward jaxpr, grad, jvp, hessian
(reverse-over-reverse and forward-over-reverse), vmap of grad, remat, export.
"""

import functools

import jax
import jax.numpy as jnp


def body(x):
    return jnp.sin(x) * 0.9 + 0.1 * x


def _while(x, n_steps):
    _, out = jax.lax.while_loop(
        lambda s: s[0] < n_steps, lambda s: (s[0] + 1, body(s[1])), (jnp.int32(0), x)
    )
    return out


def _scan(x, n_steps):
    out, _ = jax.lax.scan(lambda c, _: (body(c), None), x, None, length=n_steps)
    return out


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def xloop(x, n_steps):
    return _while(x, n_steps)


@xloop.defjvp
def _jvp(n_steps, primals, tangents):
    return jax.jvp(lambda c: _scan(c, n_steps), primals, tangents)


def f_loop(x):
    return xloop(x, 16).sum()


def f_ref(x):
    return _scan(x, 16).sum()


def ok(a, b):
    return bool(jnp.allclose(a, b, atol=1e-5))


x = jnp.linspace(-1.0, 1.0, 6)
fwd = str(jax.make_jaxpr(f_loop)(x))
print(
    "j1 forward matches",
    ok(jax.jit(f_loop)(x), f_ref(x)),
    "| while",
    "while[" in fwd,
    "| scan",
    "scan[" in fwd,
)
gj = str(jax.make_jaxpr(jax.grad(f_loop))(x))
print(
    "j2 grad matches",
    ok(jax.jit(jax.grad(f_loop))(x), jax.grad(f_ref)(x)),
    "| grad jaxpr scan",
    "scan[" in gj,
    "| while",
    "while[" in gj,
)
t = jnp.ones_like(x)
print("j3 jvp matches", ok(jax.jvp(f_loop, (x,), (t,))[1], jax.jvp(f_ref, (x,), (t,))[1]))
for name, h in [
    ("jacrev(jacrev)", jax.jacrev(jax.jacrev(f_loop))),
    ("jacfwd(jacrev)", jax.hessian(f_loop)),
]:
    try:
        ref = jax.hessian(f_ref)(x)
        print(f"j4 {name} matches", ok(h(x), ref))
    except Exception as e:  # noqa: BLE001
        print(f"j4 {name} FAILS:", type(e).__name__, str(e)[:120])
xb = jnp.stack([x, 0.5 * x])
print("j5 vmap(grad) matches", ok(jax.vmap(jax.grad(f_loop))(xb), jax.vmap(jax.grad(f_ref))(xb)))
print("j6 grad(remat) matches", ok(jax.grad(jax.checkpoint(f_loop))(x), jax.grad(f_ref)(x)))
exp = jax.export.export(jax.jit(f_loop), platforms=["cpu"])(x)
print(
    "j7 export roundtrip",
    ok(exp.call(x), f_ref(x)),
    "| custom_call",
    "custom_call" in exp.mlir_module(),
)

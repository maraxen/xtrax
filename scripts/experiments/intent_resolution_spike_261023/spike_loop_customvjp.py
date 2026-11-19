"""Throwaway spike: the same intent loop on PUBLIC API only (jax.custom_vjp).

Primal (not differentiated) path runs a while_loop; under reverse-mode the fwd
rule runs a scan and bwd uses the scan's VJP. Checks forward jaxpr has a while
and no scan, grad matches the scan reference, vmap works, export works, and
what forward-mode (jvp) does.
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


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def xloop(x, n_steps):
    return _while(x, n_steps)


def _fwd(x, n_steps):
    out, vjp = jax.vjp(lambda c: _scan(c, n_steps), x)
    return out, vjp


def _bwd(n_steps, vjp, g):
    return vjp(g)


xloop.defvjp(_fwd, _bwd)


def f_loop(x):
    return xloop(x, 64).sum()


def f_ref(x):
    return _scan(x, 64).sum()


x = jnp.linspace(-1.0, 1.0, 8)
fwd = str(jax.make_jaxpr(f_loop)(x))
print(
    "c1 forward: matches",
    bool(jnp.allclose(jax.jit(f_loop)(x), f_ref(x))),
    "| while",
    "while[" in fwd,
    "| scan",
    "scan[" in fwd,
)
gj = str(jax.make_jaxpr(jax.grad(f_loop))(x))
print(
    "c2 grad matches",
    bool(jnp.allclose(jax.jit(jax.grad(f_loop))(x), jax.grad(f_ref)(x))),
    "| grad jaxpr scan",
    "scan[" in gj,
    "| while",
    "while[" in gj,
)
xb = jnp.stack([x, x * 0.5])
print("c3 vmap matches", bool(jnp.allclose(jax.vmap(f_loop)(xb), jax.vmap(f_ref)(xb))))
exp = jax.export.export(jax.jit(f_loop), platforms=["cpu"])(x)
print(
    "c4 export roundtrip",
    bool(jnp.allclose(exp.call(x), f_ref(x))),
    "| custom_call",
    "custom_call" in exp.mlir_module(),
)
try:
    jax.jvp(f_loop, (x,), (jnp.ones_like(x),))
    print("c5 jvp: works")
except Exception as e:  # noqa: BLE001
    print("c5 jvp FAILS:", type(e).__name__, str(e)[:120])

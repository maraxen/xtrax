"""Throwaway: custom_jvp loop whose body closes over differentiated params.

c1 naive closure (body captures w): does grad wrt w work, error, or silently drop?
c2 explicit threading: xloop(body, (w, x)) with w passed as a carried/const arg.
c3 jax.closure_convert: hoist closed-over values into explicit args automatically.
"""

import functools

import jax
import jax.numpy as jnp


def make_xloop():
    @functools.partial(jax.custom_jvp, nondiff_argnums=(0, 3))
    def xloop(body, consts, x, n_steps):
        _, out = jax.lax.while_loop(
            lambda s: s[0] < n_steps, lambda s: (s[0] + 1, body(consts, s[1])), (jnp.int32(0), x)
        )
        return out

    @xloop.defjvp
    def _jvp(body, n_steps, primals, tangents):
        def scan_form(consts, x):
            out, _ = jax.lax.scan(lambda c, _: (body(consts, c), None), x, None, length=n_steps)
            return out

        return jax.jvp(scan_form, primals, tangents)

    return xloop


xloop = make_xloop()
x0 = jnp.linspace(-1.0, 1.0, 6)


def ref(w):
    out, _ = jax.lax.scan(lambda c, _: (jnp.sin(c * w) * 0.9, None), x0, None, length=16)
    return out.sum()


g_ref = jax.grad(ref)(jnp.float32(1.3))


def naive(w):
    @functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
    def loop(x, n):
        _, out = jax.lax.while_loop(
            lambda s: s[0] < n, lambda s: (s[0] + 1, jnp.sin(s[1] * w) * 0.9), (0, x)
        )
        return out

    @loop.defjvp
    def _j(n, p, t):
        return jax.jvp(
            lambda c: jax.lax.scan(lambda a, _: (jnp.sin(a * w) * 0.9, None), c, None, length=n)[0],
            p,
            t,
        )

    return loop(x0, 16).sum()


try:
    g = jax.grad(naive)(jnp.float32(1.3))
    print(
        "c1 naive closure: grad",
        float(g),
        "ref",
        float(g_ref),
        "| correct:",
        bool(jnp.allclose(g, g_ref)),
    )
except Exception as e:  # noqa: BLE001
    print("c1 naive closure: RAISES", type(e).__name__, str(e)[:140])


def threaded(w):
    return xloop(lambda consts, c: jnp.sin(c * consts) * 0.9, w, x0, 16).sum()


g2 = jax.grad(threaded)(jnp.float32(1.3))
print("c2 explicit consts: correct:", bool(jnp.allclose(g2, g_ref)))


def converted(w):
    def body_closed(c):
        return jnp.sin(c * w) * 0.9

    fn, consts = jax.closure_convert(body_closed, x0)
    return xloop(lambda cs, c: fn(c, *cs), tuple(consts), x0, 16).sum()


g3 = jax.grad(converted)(jnp.float32(1.3))
print("c3 closure_convert: correct:", bool(jnp.allclose(g3, g_ref)))

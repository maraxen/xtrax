"""Throwaway spike: intent loop that COLLECTS per-step outputs (scan with ys).

Primal = while_loop writing ys into a preallocated buffer (dynamic_update_index);
jvp rule = jvp of the lax.scan form. Body closes over params (closure_convert).
Checks forward jaxpr (while, no scan), outputs equal lax.scan, grad / jvp /
vmap(grad) equal the scan reference, export round-trips.
"""

import functools

import jax
import jax.numpy as jnp


def make_loop(length):
    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def run(step, consts, init):
        y0 = jax.eval_shape(lambda c: step(consts, c)[1], init)
        ys0 = jax.tree_util.tree_map(lambda s: jnp.zeros((length, *s.shape), s.dtype), y0)

        def body(state):
            i, c, ys = state
            c2, y = step(consts, c)
            ys = jax.tree_util.tree_map(
                lambda b, v: jax.lax.dynamic_update_index_in_dim(b, v, i, 0), ys, y
            )
            return i + 1, c2, ys

        _, c, ys = jax.lax.while_loop(lambda s: s[0] < length, body, (jnp.int32(0), init, ys0))
        return c, ys

    @run.defjvp
    def _jvp(step, primals, tangents):
        def scan_form(consts, init):
            return jax.lax.scan(lambda c, _: step(consts, c), init, None, length=length)

        return jax.jvp(scan_form, primals, tangents)

    def loop(body_closed, init):
        fn, consts = jax.closure_convert(body_closed, init)
        return run(lambda cs, c: fn(c, *cs), tuple(consts), init)

    return loop


LEN = 32
loop = make_loop(LEN)
x0 = jnp.linspace(-1.0, 1.0, 5)


def decode(w):
    def step(c):
        c2 = jnp.tanh(c * w + 0.1)
        return c2, c2.sum()  # carry, per-step output ("token score")

    c, ys = loop(step, x0)
    return c.sum() + ys.sum()


def decode_ref(w):
    def step(c, _):
        c2 = jnp.tanh(c * w + 0.1)
        return c2, c2.sum()

    c, ys = jax.lax.scan(step, x0, None, length=LEN)
    return c.sum() + ys.sum()


def ok(a, b):
    return bool(jnp.allclose(a, b, atol=1e-5))


w = jnp.float32(0.7)
fj = str(jax.make_jaxpr(decode)(w))
print(
    "y1 forward matches",
    ok(jax.jit(decode)(w), decode_ref(w)),
    "| while",
    "while[" in fj,
    "| scan",
    "scan[" in fj,
)
gj = str(jax.make_jaxpr(jax.grad(decode))(w))
print(
    "y2 grad matches",
    ok(jax.grad(decode)(w), jax.grad(decode_ref)(w)),
    "| scan",
    "scan[" in gj,
    "| while",
    "while[" in gj,
)
print("y3 jvp matches", ok(jax.jvp(decode, (w,), (1.0,))[1], jax.jvp(decode_ref, (w,), (1.0,))[1]))
ws = jnp.array([0.3, 0.7, 1.1])
print(
    "y4 vmap(grad) matches", ok(jax.vmap(jax.grad(decode))(ws), jax.vmap(jax.grad(decode_ref))(ws))
)
exp = jax.export.export(jax.jit(decode), platforms=["cpu"])(w)
print(
    "y5 export roundtrip",
    ok(exp.call(w), decode_ref(w)),
    "| custom_call",
    "custom_call" in exp.mlir_module(),
)

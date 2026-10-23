"""q6 (throwaway): kernel primal + reference JVP via jax.custom_jvp.

Design question for the xtrax.op seam: can an op take its derivative from the
reference's JVP (custom_jvp) instead of a hand-written custom_vjp? custom_vjp
breaks forward mode (jvp/jacfwd/hessian), which aminx uses. The JVP of the
reference is linear in the tangents, so JAX should transpose it for reverse
mode. Checks grad, jvp, jacfwd(jacrev), vmap(grad), jit against the dense
reference, and that the forward jaxpr still contains the kernel.
"""

import logging
import os

import jax
import jax.numpy as jnp
from common import blockify, bsmm_raw, expand_mask, make_problem, unblockify

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("q6")


def reference(x, blocks, mask):
    return x @ (unblockify(blocks) * expand_mask(mask))


@jax.custom_jvp
def op(x, blocks, mask):
    return bsmm_raw(x, blocks, mask, interpret=True)


@op.defjvp
def _op_jvp(primals, tangents):
    x, blocks, mask = primals
    tx, tb, _ = tangents
    tmask = jnp.zeros(mask.shape, jax.dtypes.float0)
    y = op(x, blocks, mask)
    _, ty = jax.jvp(reference, (x, blocks, mask), (tx, tb, tmask))
    return y, ty * (2.0 if os.environ.get("Q6_CORRUPT") else 1.0)


def main():
    x, w, mask = make_problem()
    blocks = blockify(w)
    xs = x[:32, :64]
    ws = blockify(w[:64, :64])
    ms = mask[:2, :2]
    def close(a, b):
        atol = 1e-5 * max(1.0, float(jnp.max(jnp.abs(b))))
        return bool(jnp.allclose(a, b, atol=atol, rtol=1e-5))

    def lk(x_, b_):
        return jnp.sum(jnp.tanh(op(x_, b_, mask)))

    def lr(x_, b_):
        return jnp.sum(jnp.tanh(reference(x_, b_, mask)))

    results = {}
    results["forward"] = close(op(x, blocks, mask), reference(x, blocks, mask))
    gk = jax.grad(lk, argnums=(0, 1))(x, blocks)
    gr = jax.grad(lr, argnums=(0, 1))(x, blocks)
    results["grad"] = all(close(a, b) for a, b in zip(gk, gr, strict=True))
    tx = jnp.ones_like(x)
    results["jvp"] = close(
        jax.jvp(lambda a: op(a, blocks, mask), (x,), (tx,))[1],
        jax.jvp(lambda a: reference(a, blocks, mask), (x,), (tx,))[1],
    )

    def lks(x_):
        return jnp.sum(jnp.tanh(op(x_, ws, ms)))

    def lrs(x_):
        return jnp.sum(jnp.tanh(reference(x_, ws, ms)))

    results["hessian"] = close(jax.jacfwd(jax.jacrev(lks))(xs[:, :]), jax.hessian(lrs)(xs))
    xb = jnp.stack([x, 2 * x])
    results["vmap_grad"] = close(
        jax.vmap(jax.grad(lambda a: lk(a, blocks)))(xb),
        jax.vmap(jax.grad(lambda a: lr(a, blocks)))(xb),
    )
    results["jit_grad"] = close(jax.jit(jax.grad(lambda a: lk(a, blocks)))(x), gr[0])
    fwd_jaxpr = str(jax.make_jaxpr(lambda a: op(a, blocks, mask))(x))
    results["forward_uses_kernel"] = "pallas_call" in fwd_jaxpr
    for k, v in results.items():
        log.info("%-20s %s", k, "PASS" if v else "FAIL")
    log.info("ALL %s", "PASS" if all(results.values()) else "FAIL")


if __name__ == "__main__":
    main()

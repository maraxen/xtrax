import logging

import jax
import jax.numpy as jnp
from common import blockify, bsmm_raw, dense_ref, expand_mask, make_problem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q2")


@jax.custom_vjp
def bsmm(x, w, mask):
    """Dense-W interface: blockify W, run the Pallas kernel."""
    return bsmm_raw(x, blockify(w), mask)


def _fwd(x, w, mask):
    return bsmm(x, w, mask), (x, w, mask)


def _bwd(res, g):
    x, w, mask = res
    em = expand_mask(mask)
    dx = g @ (w * em).T
    dw = (x.T @ g) * em
    return dx, dw, None  # mask is integer-valued data: no cotangent


bsmm.defvjp(_fwd, _bwd)

x, w, mask = make_problem()


def loss_kernel(x, w):
    return jnp.sum(bsmm(x, w, mask))


def loss_dense(x, w):
    return jnp.sum(dense_ref(x, w, mask))


gx_k, gw_k = jax.grad(loss_kernel, argnums=(0, 1))(x, w)
gx_d, gw_d = jax.grad(loss_dense, argnums=(0, 1))(x, w)
ok_x = bool(jnp.allclose(gx_k, gx_d, atol=1e-5, rtol=1e-5))
ok_w = bool(jnp.allclose(gw_k, gw_d, atol=1e-5, rtol=1e-5))
log.info(
    "max|dx err|=%.3e max|dW err|=%.3e",
    float(jnp.max(jnp.abs(gx_k - gx_d))),
    float(jnp.max(jnp.abs(gw_k - gw_d))),
)

# Information only: plain autodiff through the RAW pallas_call (no custom_vjp).
try:
    from common import blockify
    from common import bsmm_raw as _raw

    jax.grad(lambda xx: jnp.sum(_raw(xx, blockify(w), mask)))(x)
    log.info("INFO raw pallas_call autodiff (no custom_vjp): succeeded")
except Exception as e:  # noqa: BLE001
    log.info(
        "INFO raw pallas_call autodiff (no custom_vjp): FAILS with %s: %s",
        type(e).__name__,
        str(e)[:200],
    )

passed = ok_x and ok_w
print(f"RESULT q2: {'PASS' if passed else 'FAIL'} dx_ok={ok_x} dW_ok={ok_w}")

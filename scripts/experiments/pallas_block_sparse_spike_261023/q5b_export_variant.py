"""Q5 follow-up: mask as an ordinary blocked input (no scalar prefetch), to see
whether the cuda (Mosaic GPU) non-interpret export can lower it."""

import logging

import jax
import jax.numpy as jnp
from common import BK, BM, BN, blockify, dense_ref, make_problem
from jax import export
from jax.experimental import pallas as pl

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q5b")


def _kernel(x_ref, w_ref, m_ref, o_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    @pl.when(m_ref[0, 0] != 0)
    def _acc():
        o_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)


def bsmm_blockmask(x, blocks, mask, *, interpret):
    m, k = x.shape
    kb, nb, _, _ = blocks.shape
    n = nb * BN
    return pl.pallas_call(
        _kernel,
        grid=(m // BM, n // BN, k // BK),
        in_specs=[
            pl.BlockSpec((BM, BK), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((None, None, BK, BN), lambda i, j, kk: (kk, j, 0, 0)),
            pl.BlockSpec((1, 1), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        interpret=interpret,
    )(x, blocks, mask.astype(jnp.int32))


x, w, mask = make_problem()
blocks = blockify(w)
x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
b_spec = jax.ShapeDtypeStruct(blocks.shape, blocks.dtype)
m_spec = jax.ShapeDtypeStruct(mask.shape, jnp.int32)

# Interpret parity for the variant first (must be correct before export matters).
y_i = bsmm_blockmask(x, blocks, mask, interpret=True)
var_parity = bool(jnp.allclose(y_i, dense_ref(x, w, mask), atol=1e-5, rtol=1e-5))
log.info("blockmask variant interpret parity=%s", var_parity)

try:
    f = jax.jit(lambda a, b, c: bsmm_blockmask(a, b, c, interpret=False))
    export.export(f, platforms=["cuda"])(x_spec, b_spec, m_spec)
    cuda_ok = True
    log.info("cuda non-interpret export of blockmask variant OK")
except Exception as e:  # noqa: BLE001
    cuda_ok = False
    log.info(
        "cuda non-interpret export of blockmask variant FAILED: %s: %s",
        type(e).__name__,
        str(e)[:800],
    )

print(
    f"RESULT q5b: {'PASS' if cuda_ok and var_parity else 'FAIL'} "
    f"blockmask_interpret_parity={var_parity} cuda_noninterp_export={'OK' if cuda_ok else 'FAILED'}"
)

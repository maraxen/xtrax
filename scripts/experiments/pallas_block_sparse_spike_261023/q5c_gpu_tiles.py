"""Q5 follow-up: is the Mosaic GPU dot_general failure about tile size or dtype?
Probe the blockmask kernel at 64-wide tiles, f32 and bf16 operands, cuda export only."""

import logging

import jax
import jax.numpy as jnp
from jax import export
from jax.experimental import pallas as pl

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q5c")


def make_kernel(bm, bk, bn):
    def _kernel(x_ref, w_ref, m_ref, o_ref):
        k = pl.program_id(2)

        @pl.when(k == 0)
        def _init():
            o_ref[...] = jnp.zeros_like(o_ref)

        @pl.when(m_ref[0, 0] != 0)
        def _acc():
            o_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    return _kernel


def call(x, blocks, mask, bm, bk, bn):
    m, k = x.shape
    n = blocks.shape[1] * bn
    return pl.pallas_call(
        make_kernel(bm, bk, bn),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((None, None, bk, bn), lambda i, j, kk: (kk, j, 0, 0)),
            pl.BlockSpec((1, 1), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        interpret=False,
    )(x, blocks, mask)


results = {}
for tile in (32, 64, 128):
    for dt in (jnp.float32, jnp.bfloat16):
        M, K, N = 256, 256, 256
        xs = jax.ShapeDtypeStruct((M, K), dt)
        bs = jax.ShapeDtypeStruct((K // tile, N // tile, tile, tile), dt)
        ms = jax.ShapeDtypeStruct((K // tile, N // tile), jnp.int32)
        f = jax.jit(lambda a, b, c, t=tile: call(a, b, c, t, t, t))
        key = f"tile{tile}_{jnp.dtype(dt).name}"
        try:
            export.export(f, platforms=["cuda"])(xs, bs, ms)
            results[key] = "OK"
        except Exception as e:  # noqa: BLE001
            results[key] = f"{type(e).__name__}: {str(e)[:160]}"
        log.info("%s -> %s", key, results[key])

n_ok = sum(v == "OK" for v in results.values())
print(
    f"RESULT q5c: {'PASS' if n_ok else 'FAIL'} cuda_export_ok_configs={n_ok}/{len(results)} "
    + "; ".join(f"{k}={'OK' if v == 'OK' else 'ERR'}" for k, v in results.items())
)

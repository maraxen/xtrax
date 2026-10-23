"""Shared helpers for the 261023 Pallas block-sparse feasibility spike (throwaway)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M, K, N = 128, 256, 256
BM = BK = BN = 32
SEED = 0


def make_problem(seed=SEED, density=0.5, m=M, k=K, n=N, bm=BM, bk=BK, bn=BN):
    kx, kw, km = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = jax.random.normal(kx, (m, k), dtype=jnp.float32)
    w = jax.random.normal(kw, (k, n), dtype=jnp.float32)
    kb, nb = k // bk, n // bn
    mask = jax.random.bernoulli(km, density, (kb, nb))
    return x, w, mask


def expand_mask(mask, bk=BK, bn=BN):
    return jnp.kron(mask.astype(jnp.float32), jnp.ones((bk, bn), jnp.float32))


def dense_ref(x, w, mask, bk=BK, bn=BN):
    return x @ (w * expand_mask(mask, bk, bn))


def blockify(w, bk=BK, bn=BN):
    """(K, N) -> (Kb, Nb, bk, bn) dense block storage."""
    k, n = w.shape
    return w.reshape(k // bk, bk, n // bn, bn).transpose(0, 2, 1, 3)


def unblockify(blocks):
    kb, nb, bk, bn = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(kb * bk, nb * bn)


def _kernel(mask_ref, x_ref, w_ref, o_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    # Skip the whole block matmul when the block is masked out.
    @pl.when(mask_ref[k, pl.program_id(1)] != 0)
    def _acc():
        o_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)


def bsmm_raw(x, blocks, mask, *, bm=BM, bk=BK, bn=BN, interpret=True):
    """y = x @ W where W is given as (Kb, Nb, bk, bn) blocks gated by (Kb, Nb) mask.

    Mask is passed via scalar prefetch (PrefetchScalarGridSpec).
    """
    m, k = x.shape
    kb, nb, _, _ = blocks.shape
    n = nb * bn
    grid = (m // bm, n // bn, k // bk)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk, mask_ref: (i, kk)),
            pl.BlockSpec((None, None, bk, bn), lambda i, j, kk, mask_ref: (kk, j, 0, 0)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk, mask_ref: (i, j)),
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        interpret=interpret,
    )(mask.astype(jnp.int32), x, blocks)

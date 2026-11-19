"""Small Pallas kernels (interpret mode, CPU) shared by the xtrax.op tests."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BM = BN = BK = 8


def _mm_body(x_ref, w_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)


def matmul_kernel(x, w):
    """Tiled ``x @ w`` Pallas kernel. Shapes must be multiples of 8."""
    m, k = x.shape
    n = w.shape[1]
    return pl.pallas_call(
        _mm_body,
        grid=(m // BM, n // BN, k // BK),
        in_specs=[
            pl.BlockSpec((BM, BK), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((BK, BN), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        interpret=True,
    )(x, w)


def matmul_reference(x, w):
    return x @ w


def _scale_body(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 3.0


def scale_kernel(x):
    """Elementwise ``3 * x`` (single block)."""
    return pl.pallas_call(
        _scale_body,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x)


def scale_reference(x):
    return 3.0 * x


def make_hist_kernel(nbins):
    """Integer-count histogram kernel: ``counts[b] = #(idx == b)``, int32."""

    def body(idx_ref, o_ref):
        hit = idx_ref[...][:, None] == jnp.arange(nbins, dtype=jnp.int32)[None, :]
        o_ref[...] = jnp.sum(hit.astype(jnp.int32), axis=0)

    def kernel(idx):
        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((nbins,), jnp.int32),
            interpret=True,
        )(idx)

    return kernel


def make_hist_reference(nbins):
    def reference(idx):
        return jnp.sum(
            (idx[:, None] == jnp.arange(nbins, dtype=jnp.int32)[None, :]), axis=0
        ).astype(jnp.int32)

    return reference


def make_binned_sum_kernel(nbins):
    """Mixed outputs: per-bin float sum of ``x`` and int32 per-bin counts, int ``idx`` input."""

    def body(x_ref, idx_ref, s_ref, c_ref):
        hit = idx_ref[...][:, None] == jnp.arange(nbins, dtype=jnp.int32)[None, :]
        s_ref[...] = jnp.sum(jnp.where(hit, x_ref[...][:, None], 0.0), axis=0)
        c_ref[...] = jnp.sum(hit.astype(jnp.int32), axis=0)

    def kernel(x, idx):
        return pl.pallas_call(
            body,
            out_shape=(
                jax.ShapeDtypeStruct((nbins,), x.dtype),
                jax.ShapeDtypeStruct((nbins,), jnp.int32),
            ),
            interpret=True,
        )(x, idx)

    return kernel


def make_binned_sum_reference(nbins):
    def reference(x, idx):
        hit = idx[:, None] == jnp.arange(nbins, dtype=jnp.int32)[None, :]
        return jnp.sum(jnp.where(hit, x[:, None], 0.0), axis=0), jnp.sum(
            hit.astype(jnp.int32), axis=0
        )

    return reference

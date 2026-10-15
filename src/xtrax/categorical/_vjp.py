"""Differentiable exact counts: the ``custom_vjp`` of spec §4.6.

The forward runs the int32 kernel (``counts._execute``) on ``w.astype(int32)``
and casts all four outputs to float32. Every output is linear in the weights,
so the backward is the closed-form cotangent: each cotangent is gathered at the
realized (shifted) codes, frame-tiled with the plan's own tile sizes so that its
temporaries match the planner's ``differentiable`` term (spec §5).

Reverse mode only. ``jax.jvp`` and ``jacfwd`` raise ``TypeError`` on this path
(spec A33); callers who need forward mode use ``strategy="float"``.
Integer codes and shifts receive no cotangent (``None``); ``jax.grad`` with
respect to an integer input raises ``TypeError`` before this module runs.

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§4.6, §5, S1-G5)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int, Shaped

from xtrax.categorical.counts import _execute, _shifted_codes
from xtrax.categorical.plan import CooccurrencePlan


class DiffConfig(NamedTuple):
    """Static configuration of one differentiable call.

    It is a ``custom_vjp`` nondiff argument, so every field must be hashable.

    Attributes:
        layout: ``"dense"`` or ``"pairs"``.
        method: ``"matmul"`` or ``"scatter"``, as chosen by the caller.
        plan: The tile plan the kernel and the backward both execute.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        n_frames: Unpadded frame count ``T``.
    """

    layout: str
    method: str
    plan: CooccurrencePlan
    ka: int
    kb: int
    n_frames: int


def _forward(
    cfg: DiffConfig,
    w: jax.Array,
    xa: jax.Array,
    xb: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    pair_a: jax.Array | None,
    pair_b: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the int32 kernel and cast all four outputs to float32 (spec §4.6).

    The cast ``w.astype(int32)`` is exact because C2 (integer-valued, non-negative)
    is enforced before this call. The float32 outputs are exact while C3 holds.
    """
    joint, ma, mb, tot = _execute(
        xa,
        xb,
        w.astype(jnp.int32),
        sa,
        sb,
        pair_a,
        pair_b,
        layout=cfg.layout,
        method=cfg.method,
        plan=cfg.plan,
        ka=cfg.ka,
        kb=cfg.kb,
        n_frames=cfg.n_frames,
    )
    return (
        joint.astype(jnp.float32),
        ma.astype(jnp.float32),
        mb.astype(jnp.float32),
        tot.astype(jnp.float32),
    )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _exact_counts(
    cfg: DiffConfig,
    w: jax.Array,
    xa: jax.Array,
    xb: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    pair_a: jax.Array | None,
    pair_b: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Float32 exact counts with a reverse-mode gradient with respect to ``w``."""
    return _forward(cfg, w, xa, xb, sa, sb, pair_a, pair_b)


def _exact_fwd(
    cfg: DiffConfig,
    w: jax.Array,
    xa: jax.Array,
    xb: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    pair_a: jax.Array | None,
    pair_b: jax.Array | None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple]:
    """Forward rule: the outputs plus the residuals the backward needs."""
    out = _forward(cfg, w, xa, xb, sa, sb, pair_a, pair_b)
    return out, (w, xa, xb, sa, sb, pair_a, pair_b)


def _exact_bwd(cfg: DiffConfig, res: tuple, g: tuple) -> tuple:
    """Backward rule: the weight cotangent. Integer inputs get no cotangent."""
    w, xa, xb, sa, sb, pair_a, pair_b = res
    dw = _weight_cotangent(cfg, xa, xb, sa, sb, pair_a, pair_b, g)
    return (dw.astype(w.dtype), None, None, None, None, None, None)


_exact_counts.defvjp(_exact_fwd, _exact_bwd)


def exact_counts(
    cfg: DiffConfig,
    w: Float[Array, "R T"],  # noqa: F722 (jaxtyping shape)
    xa: Shaped[Array, "Pa T"],  # noqa: F722
    xb: Shaped[Array, "Pb T"],  # noqa: F722
    sa: Int[Array, "R Pa"],  # noqa: F722
    sb: Int[Array, "R Pb"],  # noqa: F722
    pair_a: Int[Array, "n_pairs"] | None,  # noqa: F821 (jaxtyping shape name)
    pair_b: Int[Array, "n_pairs"] | None,  # noqa: F821
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return float32 ``(joint, marginal_a, marginal_b, total)`` with a custom VJP.

    Args:
        cfg: Static configuration (layout, method, plan, sizes).
        w: float32 weights ``(R, T)``, already broadcast to ``R`` replicates and
            already checked (C2, C3).
        xa: Codes ``(Pa, T)``, unpadded, any unsigned or int dtype.
        xb: Codes ``(Pb, T)``.
        sa: int32 shifts ``(R, Pa)``.
        sb: int32 shifts ``(R, Pb)``.
        pair_a: int32 ``a`` index per pair, or None for the dense layout.
        pair_b: int32 ``b`` index per pair, or None for the dense layout.

    Returns:
        The four float32 fields, shaped as ``CooccurrenceCounts`` documents them.
    """
    return _exact_counts(cfg, w, xa, xb, sa, sb, pair_a, pair_b)


def _dense_joint_cotangent(gj_r: jax.Array, xa_k: jax.Array, xb_k: jax.Array) -> jax.Array:
    """Sum over the block's ``(a, b)`` entries of ``G[a, b, x_a, x_b]`` for one frame chunk.

    Args:
        gj_r: Joint cotangent of one replicate and one row block, ``(rb, Pb, Ka, Kb)``.
        xa_k: Shifted codes of the block's ``a`` rows, ``(rb, tc)``.
        xb_k: Shifted codes of all ``b`` variables, ``(Pb, tc)``.

    Returns:
        ``(tc,)`` contribution to ``dL/dw`` for these frames.
    """
    rb, pb = gj_r.shape[0], gj_r.shape[1]
    ia = jnp.arange(rb)[:, None, None]
    ib = jnp.arange(pb)[None, :, None]
    vals = gj_r[ia, ib, xa_k[:, None, :], xb_k[None, :, :]]
    return vals.sum(axis=(0, 1))


def _pairs_joint_cotangent(
    gj_r: jax.Array,
    xa_k: jax.Array,
    xb_k: jax.Array,
    *,
    pa_idx: jax.Array,
    pb_idx: jax.Array,
    pair_tile: int,
) -> jax.Array:
    """Pairs analogue of ``_dense_joint_cotangent``, gathered ``pair_tile`` pairs at a time.

    Args:
        gj_r: Joint cotangent of one replicate, ``(n_pt, Ka, Kb)``. The pair count is
            padded to a multiple of ``pair_tile`` with zeros.
        xa_k: Shifted codes of all ``a`` variables, ``(Pa, tc)``.
        xb_k: Shifted codes of all ``b`` variables, ``(Pb, tc)``.
        pa_idx: ``a`` index per padded pair, ``(n_pt,)``.
        pb_idx: ``b`` index per padded pair, ``(n_pt,)``.
        pair_tile: Pairs per ``lax.map`` step.

    Returns:
        ``(tc,)`` contribution to ``dL/dw`` for these frames.
    """
    n_tiles = gj_r.shape[0] // pair_tile
    ka, kb = gj_r.shape[1], gj_r.shape[2]
    tiles = (
        gj_r.reshape(n_tiles, pair_tile, ka, kb),
        pa_idx.reshape(n_tiles, pair_tile),
        pb_idx.reshape(n_tiles, pair_tile),
    )
    lane = jnp.arange(pair_tile)[:, None]

    def tile(args: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        g_t, ia_t, ib_t = args
        return g_t[lane, xa_k[ia_t], xb_k[ib_t]].sum(axis=0)

    return lax.map(tile, tiles).sum(axis=0)


def _weight_cotangent(
    cfg: DiffConfig,
    xa: jax.Array,
    xb: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    pair_a: jax.Array | None,
    pair_b: jax.Array | None,
    g: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    """Return ``dL/dw`` ``(R, T)`` from the four output cotangents (spec §4.6).

    ``dL/dw[r,t] = sum_ab G[r,a,b,x_a(t-s_a), x_b(t-s_b)] + sum_a gm_a[r,a,x_a(...)]
    + sum_b gm_b[r,b,x_b(...)] + gn[r]``, with the joint term over all entries
    (dense) or the selected pairs. The sum over row blocks is a scan with a carry,
    and replicates run through ``lax.map`` with the plan's ``r_tile``.
    """
    g_joint, g_ma, g_mb, g_tot = g
    plan = cfg.plan
    n_frames, ka = cfg.n_frames, cfg.ka
    r = g_tot.shape[0]
    pa, pb = xa.shape[0], xb.shape[0]
    r_pad, t_pad, t_chunk = plan.r_padded, plan.t_padded, plan.t_chunk
    rb = plan.row_block
    n_rb = -(-pa // rb)
    pa_pad = n_rb * rb
    t_all = jnp.arange(t_pad, dtype=jnp.int32).reshape(t_pad // t_chunk, t_chunk)

    def pad_reps(x: jax.Array) -> jax.Array:
        return jnp.pad(x, [(0, r_pad - r)] + [(0, 0)] * (x.ndim - 1))

    sb_p = pad_reps(sb)
    gmb_p = pad_reps(g_mb)
    gn_p = pad_reps(g_tot)
    sa_p = jnp.pad(pad_reps(sa), ((0, 0), (0, pa_pad - pa)))
    gma_p = jnp.pad(pad_reps(g_ma), ((0, 0), (0, pa_pad - pa), (0, 0)))
    xa_blocks = jnp.pad(xa, ((0, pa_pad - pa), (0, 0))).reshape(n_rb, rb, n_frames)
    sa_blocks = sa_p.reshape(r_pad, n_rb, rb).transpose(1, 0, 2)
    gma_blocks = gma_p.reshape(r_pad, n_rb, rb, ka).transpose(1, 0, 2, 3)

    if cfg.layout == "dense":
        gj = jnp.pad(g_joint, ((0, r_pad - r), (0, pa_pad - pa), (0, 0), (0, 0), (0, 0)))
        gj_blocks = gj.reshape(r_pad, n_rb, rb, pb, ka, cfg.kb).transpose(1, 0, 2, 3, 4, 5)
        joint_fn = _dense_joint_cotangent
    else:
        # Pairs are one block (row_block == Pa, validated in counts._validate_plan).
        assert pair_a is not None and pair_b is not None
        n_pairs = g_joint.shape[1]
        n_pt = -(-n_pairs // plan.pair_tile) * plan.pair_tile
        gj = jnp.pad(g_joint, ((0, r_pad - r), (0, n_pt - n_pairs), (0, 0), (0, 0)))
        gj_blocks = gj[None]
        pa_idx = jnp.pad(pair_a, (0, n_pt - n_pairs))
        pb_idx = jnp.pad(pair_b, (0, n_pt - n_pairs))
        joint_fn = partial(
            _pairs_joint_cotangent, pa_idx=pa_idx, pb_idx=pb_idx, pair_tile=plan.pair_tile
        )

    def per_block(
        xa_blk: jax.Array, sa_blk: jax.Array, gj_blk: jax.Array, gma_blk: jax.Array
    ) -> jax.Array:
        def per_rep(args: tuple[jax.Array, ...]) -> jax.Array:
            sa_r, sb_r, gj_r, gma_r = args

            def chunk(_: None, t_k: jax.Array) -> tuple[None, jax.Array]:
                xa_k = _shifted_codes(xa_blk, sa_r, t_k, n_frames)
                xb_k = _shifted_codes(xb, sb_r, t_k, n_frames)
                joint_part = joint_fn(gj_r, xa_k, xb_k)
                marg_a = gma_r[jnp.arange(rb)[:, None], xa_k].sum(axis=0)
                return None, joint_part + marg_a

            _, out = lax.scan(chunk, None, t_all)
            return out.reshape(t_pad)

        return lax.map(per_rep, (sa_blk, sb_p, gj_blk, gma_blk), batch_size=plan.r_tile)

    def block_step(acc: jax.Array, blk: tuple[jax.Array, ...]) -> tuple[jax.Array, None]:
        return acc + per_block(*blk), None

    acc, _ = lax.scan(
        block_step,
        jnp.zeros((r_pad, t_pad), dtype=jnp.float32),
        (xa_blocks, sa_blocks, gj_blocks, gma_blocks),
    )

    def per_rep_b(args: tuple[jax.Array, ...]) -> jax.Array:
        sb_r, gmb_r, gn_r = args

        def chunk(_: None, t_k: jax.Array) -> tuple[None, jax.Array]:
            xb_k = _shifted_codes(xb, sb_r, t_k, n_frames)
            marg_b = gmb_r[jnp.arange(pb)[:, None], xb_k].sum(axis=0)
            return None, marg_b + gn_r

        _, out = lax.scan(chunk, None, t_all)
        return out.reshape(t_pad)

    dw_b = lax.map(per_rep_b, (sb_p, gmb_p, gn_p), batch_size=plan.r_tile)
    return (acc + dw_b)[:r, :n_frames]

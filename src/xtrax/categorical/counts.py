"""Exact integer weighted co-occurrence counts over categorical codes.

Stage 1 covers the ``exact_int`` path with the ``matmul`` and ``scatter``
methods, the dense, pairs and self layouts, circular shifts, traced
preconditions, the row-block iterator and the byte-budget planner. Counts are
int32 throughout: one-hots and weights are integer, every dot product sets
``preferred_element_type=int32``, scatter-adds accumulate in int32, and no float
touches the counts.

Frames are gathered per chunk inside the scan (spec §4.2 step 1), so no
``(P, T)`` shifted copy is ever materialized.

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§3, §4.2-4.5, §5, §6)
"""

from collections.abc import Iterator
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import ArrayLike

from xtrax.categorical.codes import CategoricalCodes
from xtrax.categorical.plan import (
    KK_CROSSOVER_CPU,
    KK_CROSSOVER_GPU,
    CooccurrencePlan,
    plan_cooccurrence,
    select_method,
)

_INT31 = 2**31


class CooccurrenceCounts(eqx.Module):
    """Exact co-occurrence counts for one call of ``cooccurrence_counts``.

    Attributes:
        joint: Dense layout ``(R, Pa, Pb, Ka, Kb)``; pairs layout
            ``(R, n_pairs, Ka, Kb)``. int32.
        marginal_a: Shape ``(R, Pa, Ka)``, int32.
        marginal_b: Shape ``(R, Pb, Kb)``, int32.
        total: Shape ``(R,)``, int32; the summed weight per replicate.
        layout: ``"dense"`` (all ``Pa*Pb`` variable pairs) or ``"pairs"``
            (only the requested rows).
        method: Method actually run, ``"matmul"`` or ``"scatter"``.
        method_rule: Why that method ran: ``explicit``, ``cpu_provisional``,
            ``cpu_measured``, ``gpu_provisional`` or ``gpu_measured``.
    """

    joint: jax.Array
    marginal_a: jax.Array
    marginal_b: jax.Array
    total: jax.Array
    layout: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    method_rule: str = eqx.field(static=True)


def _prepare_weights(weights: ArrayLike | None, n_frames: int) -> jax.Array:
    """Return integer weights of shape ``(R, T)``; ones when ``weights`` is None."""
    if weights is None:
        return jnp.ones((1, n_frames), dtype=jnp.int8)
    w = jnp.asarray(weights)
    if not jnp.issubdtype(w.dtype, jnp.integer):
        raise TypeError(
            f"exact_int requires integer weights, got dtype {w.dtype}; "
            "pass integer weights, or differentiable=True"
        )
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2 or w.shape[1] != n_frames:
        raise ValueError(f"weights must have shape (T,) or (R, T) with T={n_frames}, got {w.shape}")
    if not (w.dtype == jnp.int8 or w.dtype == jnp.int16 or w.dtype == jnp.int32):
        w = w.astype(jnp.int32)
    return w


def _onehot(codes_chunk: jax.Array, n_categories: int) -> jax.Array:
    """Dense int8 one-hot of an int32 ``(P, tc)`` chunk, shape ``(tc, P*K)``.

    Column ``a*K + k`` is 1 where variable ``a`` has code ``k`` in that frame.
    """
    tc = codes_chunk.shape[1]
    eq = codes_chunk.T[:, :, None] == jnp.arange(n_categories, dtype=jnp.int32)
    return eq.reshape(tc, -1).astype(jnp.int8)


def _shifted_codes(x: jax.Array, shifts: jax.Array, t_idx: jax.Array, n_frames: int) -> jax.Array:
    """Gather ``x[p, (t - s[p]) mod T]`` for the frame indices ``t`` in ``t_idx``.

    Matches ``np.roll(x[p], s[p])``. Shifts are reduced with ``jnp.mod`` first
    so ``t - s`` stays in range for any int32 shift. Padded frames (``t >= T``)
    read index 0; their weight is zero.

    Args:
        x: int32 codes ``(P, T)``, unpadded.
        shifts: int32 shifts ``(P,)`` for one replicate.
        t_idx: int32 absolute frame indices, typically one chunk of
            ``arange(T_pad)``.
        n_frames: Unpadded frame count ``T``.

    Returns:
        int32 codes ``(P, len(t_idx))``.
    """
    s_mod = jnp.mod(shifts, n_frames)[:, None]
    idx = jnp.mod(t_idx[None, :] - s_mod, n_frames)
    idx = jnp.where((t_idx < n_frames)[None, :], idx, 0)
    # Cast after the gather: the storage dtype stays in HBM, and only one chunk
    # is widened to int32 for arithmetic (spec §4.1).
    return jnp.take_along_axis(x, idx, axis=1).astype(jnp.int32)


@partial(jax.jit, static_argnames=("ka", "kb", "t_chunk", "n_frames", "r_tile"))
def _matmul_block(
    xa: jax.Array,
    xb: jax.Array,
    w: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    *,
    ka: int,
    kb: int,
    t_chunk: int,
    n_frames: int,
    r_tile: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Matmul counts for one block of ``a`` rows, with circular shifts.

    Args:
        xa: int32 codes ``(rb, T)`` for this block, unpadded.
        xb: int32 codes ``(Pb, T)``, unpadded.
        w: Integer weights ``(R_pad, T_pad)``, zero-padded.
        sa: int32 shifts ``(R_pad, rb)`` for this block.
        sb: int32 shifts ``(R_pad, Pb)``.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        t_chunk: Frames per scan step; divides ``T_pad``.
        n_frames: Unpadded frame count ``T``.
        r_tile: Replicates per ``lax.map`` step; divides ``R_pad``.

    Returns:
        ``(joint, marginal_a, marginal_b, total)`` in int32: ``joint``
        ``(R_pad, rb, Pb, Ka, Kb)``, ``marginal_a`` ``(R_pad, rb, Ka)``,
        ``marginal_b`` ``(R_pad, Pb, Kb)``, ``total`` ``(R_pad,)``.
    """
    rb, pb = xa.shape[0], xb.shape[0]
    r, t_pad = w.shape
    n_chunks = t_pad // t_chunk
    wdtype = w.dtype
    fa, fb = rb * ka, pb * kb
    t_all = jnp.arange(t_pad, dtype=jnp.int32).reshape(n_chunks, t_chunk)
    w_c = w.reshape(r, n_chunks, t_chunk)

    def one_replicate(args: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        w_r, sa_r, sb_r = args

        def step(carry: tuple[jax.Array, ...], chunk: tuple[jax.Array, ...]) -> tuple:
            joint, ma, mb, tot = carry
            t_k, w_k = chunk
            xa_k = _shifted_codes(xa, sa_r, t_k, n_frames)
            xb_k = _shifted_codes(xb, sb_r, t_k, n_frames)
            oa = _onehot(xa_k, ka)
            ob = _onehot(xb_k, kb).astype(wdtype)
            wc = w_k[:, None]
            ow_a = oa.astype(wdtype) * wc
            joint = joint + lax.dot_general(
                ow_a,
                ob,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.int32,
            )
            ma = ma + jnp.sum(ow_a, axis=0, dtype=jnp.int32)
            mb = mb + jnp.sum(ob * wc, axis=0, dtype=jnp.int32)
            tot = tot + jnp.sum(w_k, dtype=jnp.int32)
            return (joint, ma, mb, tot), None

        init = (
            jnp.zeros((fa, fb), dtype=jnp.int32),
            jnp.zeros((fa,), dtype=jnp.int32),
            jnp.zeros((fb,), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        carry, _ = lax.scan(step, init, (t_all, w_r))
        return carry

    joint, ma, mb, tot = lax.map(one_replicate, (w_c, sa, sb), batch_size=r_tile)
    joint = joint.reshape(r, rb, ka, pb, kb).transpose(0, 1, 3, 2, 4)
    return joint, ma.reshape(r, rb, ka), mb.reshape(r, pb, kb), tot


@partial(jax.jit, static_argnames=("ka", "kb", "t_chunk", "n_frames", "pair_tile", "r_tile"))
def _scatter_block(
    xa: jax.Array,
    xb: jax.Array,
    w: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    pair_a: jax.Array,
    pair_b: jax.Array,
    *,
    ka: int,
    kb: int,
    t_chunk: int,
    n_frames: int,
    pair_tile: int,
    r_tile: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-pair composite bincount for one block of ``a`` rows, with shifts.

    Each pair ``p = (a, b)`` accumulates ``hist[p, x_a*Kb + x_b] += w`` over
    frames, in int32. Weights are cast to int32 before the add: ``bincount``
    would accumulate in the weight dtype and wrap (spec A26). Pairs run in
    ``pair_tile`` chunks through ``lax.map`` inside the frame scan.

    Args:
        xa: int32 codes ``(rb, T)`` for this block, unpadded.
        xb: int32 codes ``(Pb, T)``, unpadded.
        w: Integer weights ``(R_pad, T_pad)``, zero-padded.
        sa: int32 shifts ``(R_pad, rb)`` for this block.
        sb: int32 shifts ``(R_pad, Pb)``.
        pair_a: int32 block-local ``a`` row per pair, length a multiple of
            ``pair_tile``. Padding entries are 0 and the caller slices them off.
        pair_b: int32 ``b`` index per pair, same length as ``pair_a``.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        t_chunk: Frames per scan step; divides ``T_pad``.
        n_frames: Unpadded frame count ``T``.
        pair_tile: Pairs per ``lax.map`` step.
        r_tile: Replicates per ``lax.map`` step; divides ``R_pad``.

    Returns:
        ``(hist, marginal_a, marginal_b, total)`` in int32: ``hist``
        ``(R_pad, n_pairs_padded, Ka, Kb)``, the marginals as in ``_matmul_block``.
    """
    rb, pb = xa.shape[0], xb.shape[0]
    r, t_pad = w.shape
    n_chunks = t_pad // t_chunk
    n_tiles = pair_a.shape[0] // pair_tile
    kk = ka * kb
    t_all = jnp.arange(t_pad, dtype=jnp.int32).reshape(n_chunks, t_chunk)
    w_c = w.reshape(r, n_chunks, t_chunk)
    rows = jnp.arange(pair_tile, dtype=jnp.int32)[:, None]
    a_rows = jnp.arange(rb, dtype=jnp.int32)[:, None]
    b_rows = jnp.arange(pb, dtype=jnp.int32)[:, None]

    def one_replicate(args: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        w_r, sa_r, sb_r = args

        def step(carry: tuple[jax.Array, ...], chunk: tuple[jax.Array, ...]) -> tuple:
            hist, ma, mb, tot = carry
            t_k, w_k = chunk
            xa_k = _shifted_codes(xa, sa_r, t_k, n_frames)
            xb_k = _shifted_codes(xb, sb_r, t_k, n_frames)
            w32 = w_k.astype(jnp.int32)
            comp = (xa_k[pair_a] * kb + xb_k[pair_b]).reshape(n_tiles, pair_tile, t_chunk)

            def add_tile(args_t: tuple[jax.Array, jax.Array]) -> jax.Array:
                h, c = args_t
                return h.at[rows, c].add(w32[None, :])

            hist = lax.map(add_tile, (hist, comp))
            ma = ma.at[a_rows, xa_k].add(w32[None, :])
            mb = mb.at[b_rows, xb_k].add(w32[None, :])
            tot = tot + jnp.sum(w32, dtype=jnp.int32)
            return (hist, ma, mb, tot), None

        init = (
            jnp.zeros((n_tiles, pair_tile, kk), dtype=jnp.int32),
            jnp.zeros((rb, ka), dtype=jnp.int32),
            jnp.zeros((pb, kb), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        (hist, ma, mb, tot), _ = lax.scan(step, init, (t_all, w_r))
        return hist.reshape(-1, ka, kb), ma, mb, tot

    return lax.map(one_replicate, (w_c, sa, sb), batch_size=r_tile)


def _prepare_shifts(shifts: ArrayLike | None, n_vars: int, name: str) -> jax.Array | None:
    """Return int32 shifts ``(R, P)``, or None when no shifts were given."""
    if shifts is None:
        return None
    s = jnp.asarray(shifts)
    if not jnp.issubdtype(s.dtype, jnp.integer):
        raise TypeError(f"{name} must be integer-valued, got dtype {s.dtype}")
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2 or s.shape[1] != n_vars:
        raise ValueError(f"{name} must have shape (R, {n_vars}), got {s.shape}")
    return s.astype(jnp.int32)


def _common_replicates(sizes: list[tuple[str, int]]) -> int:
    """Return the shared replicate count; a size of 1 broadcasts to the others."""
    reps = max(n for _, n in sizes)
    if any(n not in (1, reps) for _, n in sizes):
        detail = ", ".join(f"{k}={v}" for k, v in sizes)
        raise ValueError(f"replicate counts must agree or be 1; got {detail}")
    return reps


def _broadcast_shifts(s: jax.Array | None, n_rep: int, n_vars: int) -> jax.Array:
    """Return shifts ``(n_rep, n_vars)``; zeros (no shift) when ``s`` is None."""
    if s is None:
        return jnp.zeros((n_rep, n_vars), dtype=jnp.int32)
    return jnp.broadcast_to(s, (n_rep, n_vars))


def _dense_pair_index(pa: int, pb: int) -> tuple[jax.Array, jax.Array]:
    """Return ``(a, b)`` index columns of the dense grid, row-major ``(Pa, Pb)``."""
    ia = jnp.repeat(jnp.arange(pa, dtype=jnp.int32), pb)
    ib = jnp.tile(jnp.arange(pb, dtype=jnp.int32), pa)
    return ia, ib


def _prepare_pairs(pairs: ArrayLike, pa: int, pb: int) -> tuple[jax.Array, jax.Array]:
    """Validate ``(n_pairs, 2)`` pair rows and return int32 ``(a, b)`` columns.

    Concrete pairs are range-checked here. Traced pairs get shape and dtype
    checks only; their range is checked under trace by ``_check_pairs`` when
    ``check=True``.

    Args:
        pairs: Integer rows ``(a, b)`` with ``0 <= a < Pa`` and ``0 <= b < Pb``.
        pa: Number of ``a`` variables.
        pb: Number of ``b`` variables.

    Returns:
        ``(pair_a, pair_b)``, int32 arrays of length ``n_pairs``.

    Raises:
        TypeError: If ``pairs`` is not integer-valued.
        ValueError: If the shape is not ``(n, 2)`` with ``n >= 1``, or a concrete
            index is out of range.
    """
    traced = isinstance(pairs, jax.core.Tracer)
    arr = pairs if traced else np.asarray(pairs)
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise TypeError(f"pairs must be integer-valued, got dtype {arr.dtype}")
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"pairs must have shape (n_pairs, 2) with n_pairs >= 1, got {arr.shape}")
    a, b = arr[:, 0], arr[:, 1]
    if not traced and (a.min() < 0 or a.max() >= pa or b.min() < 0 or b.max() >= pb):
        raise ValueError(
            f"pairs index out of range: a must lie in [0, {pa}), b in [0, {pb}); "
            f"found a in [{a.min()}, {a.max()}], b in [{b.min()}, {b.max()}]"
        )
    return jnp.asarray(a, dtype=jnp.int32), jnp.asarray(b, dtype=jnp.int32)


def _pad_to_tile(pair_a: jax.Array, pair_b: jax.Array, tile: int) -> tuple[jax.Array, jax.Array]:
    """Pad pair columns with ``(0, 0)`` rows up to a multiple of ``tile``."""
    extra = -(-pair_a.shape[0] // tile) * tile - pair_a.shape[0]
    return jnp.pad(pair_a, (0, extra)), jnp.pad(pair_b, (0, extra))


def _check_static_sizes(pa: int, ka: int, pb: int, kb: int) -> None:
    """Reject index spaces that would overflow int32 (spec §3.1, §4.1)."""
    for name, size in (("Pa*Ka", pa * ka), ("Pb*Kb", pb * kb), ("Ka*Kb", ka * kb)):
        if size >= _INT31:
            raise ValueError(f"{name}={size} must be < 2**31 for int32 index arithmetic")


def _check_preconditions(
    xa: jax.Array, ka: int, xb: jax.Array, kb: int, w: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Enforce C1-C3 under trace with ``eqx.error_if`` (spec §3.2).

    The returned arrays must be the ones the kernel consumes, so the check is
    not dead-code eliminated.

    C1 codes in ``[0, K)``. C2 weights ``>= 0``. C3 total ``< 2**31``. The total
    check uses a running int32 sum. Each weight is in ``[0, 2**31)``, so the
    first partial sum that reaches ``2**31`` wraps negative, and any total of
    ``2**31`` or more is detected exactly.

    Args:
        xa: int32 codes ``(Pa, T)``.
        ka: Categories of ``a``.
        xb: int32 codes ``(Pb, T)``.
        kb: Categories of ``b``.
        w: Integer weights ``(R, T)``.

    Returns:
        ``(xa, xb, w)`` with the checks attached.
    """
    xa = eqx.error_if(xa, jnp.any((xa < 0) | (xa >= ka)), "C1 failed: codes_a outside [0, Ka)")
    xb = eqx.error_if(xb, jnp.any((xb < 0) | (xb >= kb)), "C1 failed: codes_b outside [0, Kb)")
    w = eqx.error_if(w, jnp.any(w < 0), "C2 failed: weights must be >= 0")
    running = jnp.cumsum(w, axis=1, dtype=jnp.int32)
    w = eqx.error_if(w, jnp.any(running < 0), "C3 failed: total >= 2**31 overflows int32")
    return xa, xb, w


def _check_pairs(
    pair_a: jax.Array, pair_b: jax.Array, pa: int, pb: int
) -> tuple[jax.Array, jax.Array]:
    """Enforce the pair index range under trace with ``eqx.error_if``."""
    bad = (pair_a < 0) | (pair_a >= pa) | (pair_b < 0) | (pair_b >= pb)
    pair_a = eqx.error_if(pair_a, jnp.any(bad), "pairs index out of range")
    return pair_a, pair_b


def _validate_plan(
    plan: CooccurrencePlan, chosen: str, n_rep: int, n_frames: int, pa: int, layout: str
) -> None:
    """Reject a caller-supplied plan that does not describe this call."""
    if plan.method != chosen:
        raise ValueError(
            f"plan.method={plan.method!r} disagrees with the method this call resolves to "
            f"({chosen!r})"
        )
    if plan.t_chunk < 1 or plan.t_padded != -(-n_frames // plan.t_chunk) * plan.t_chunk:
        raise ValueError(f"plan was built for a different frame count (T={n_frames})")
    if plan.r_tile < 1 or plan.r_padded != -(-n_rep // plan.r_tile) * plan.r_tile:
        raise ValueError(f"plan was built for a different replicate count (R={n_rep})")
    if not 1 <= plan.row_block <= pa:
        raise ValueError(f"plan.row_block={plan.row_block} must lie in [1, Pa={pa}]")
    if layout == "pairs" and plan.row_block != pa:
        raise ValueError("the pairs layout is not row-blocked; plan.row_block must equal Pa")
    if plan.pair_tile < 1:
        raise ValueError(f"plan.pair_tile must be >= 1, got {plan.pair_tile}")


def _execute(
    xa: jax.Array,
    xb: jax.Array,
    w: jax.Array,
    sa: jax.Array,
    sb: jax.Array,
    pair_a: jax.Array | None,
    pair_b: jax.Array | None,
    *,
    layout: str,
    method: str,
    plan: CooccurrencePlan,
    ka: int,
    kb: int,
    n_frames: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the planned kernel and slice the padding off.

    The ``a`` variables are split into ``n_rb`` row blocks of ``plan.row_block``
    through ``lax.map``. The pairs layout is one block, and its pairs are
    selected from the dense result (matmul) or computed directly (scatter).

    Returns:
        ``(joint, marginal_a, marginal_b, total)``, sliced to the true ``R``,
        ``Pa`` and ``Pb``.
    """
    pa, pb = xa.shape[0], xb.shape[0]
    r = w.shape[0]
    rb = plan.row_block
    n_rb = -(-pa // rb)
    pa_pad = n_rb * rb
    r_pad = plan.r_padded

    w = jnp.pad(w, ((0, r_pad - r), (0, plan.t_padded - n_frames)))
    sa = jnp.pad(sa, ((0, r_pad - r), (0, pa_pad - pa)))
    sb = jnp.pad(sb, ((0, r_pad - r), (0, 0)))
    xa_blocks = jnp.pad(xa, ((0, pa_pad - pa), (0, 0))).reshape(n_rb, rb, n_frames)
    sa_blocks = sa.reshape(r_pad, n_rb, rb).transpose(1, 0, 2)

    if method == "matmul":

        def block(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, ...]:
            xa_blk, sa_blk = args
            return _matmul_block(
                xa_blk,
                xb,
                w,
                sa_blk,
                sb,
                ka=ka,
                kb=kb,
                t_chunk=plan.t_chunk,
                n_frames=n_frames,
                r_tile=plan.r_tile,
            )

        joint_b, ma_b, mb_b, tot_b = lax.map(block, (xa_blocks, sa_blocks))
        joint = joint_b.transpose(1, 0, 2, 3, 4, 5).reshape(r_pad, pa_pad, pb, ka, kb)
        joint = joint[:r, :pa]
        if layout == "pairs":
            joint = joint[:, pair_a, pair_b]
    else:
        if layout == "dense":
            blk_a, blk_b = _dense_pair_index(rb, pb)
        else:
            blk_a, blk_b = pair_a, pair_b
        n_blk = blk_a.shape[0]
        blk_a, blk_b = _pad_to_tile(blk_a, blk_b, plan.pair_tile)

        def block(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, ...]:
            xa_blk, sa_blk = args
            return _scatter_block(
                xa_blk,
                xb,
                w,
                sa_blk,
                sb,
                blk_a,
                blk_b,
                ka=ka,
                kb=kb,
                t_chunk=plan.t_chunk,
                n_frames=n_frames,
                pair_tile=plan.pair_tile,
                r_tile=plan.r_tile,
            )

        hist_b, ma_b, mb_b, tot_b = lax.map(block, (xa_blocks, sa_blocks))
        if layout == "dense":
            hist = hist_b[:, :, :n_blk].reshape(n_rb, r_pad, rb, pb, ka, kb)
            joint = hist.transpose(1, 0, 2, 3, 4, 5).reshape(r_pad, pa_pad, pb, ka, kb)
            joint = joint[:r, :pa]
        else:
            joint = hist_b[0, :r, :n_blk]

    ma = ma_b.transpose(1, 0, 2, 3).reshape(r_pad, pa_pad, ka)[:r, :pa]
    return joint, ma, mb_b[0, :r], tot_b[0, :r]


def cooccurrence_counts(
    codes_a: CategoricalCodes | jax.Array,
    codes_b: CategoricalCodes | jax.Array | None = None,
    *,
    weights: ArrayLike | None = None,
    shifts_a: ArrayLike | None = None,
    shifts_b: ArrayLike | None = None,
    pairs: ArrayLike | None = None,
    strategy: Literal["exact_int", "float"] = "exact_int",
    differentiable: bool = False,
    method: Literal["auto", "matmul", "scatter"] = "auto",
    byte_budget: int | Literal["device"] | None = None,
    plan: CooccurrencePlan | None = None,
    check: bool = True,
    mesh: object | None = None,
) -> CooccurrenceCounts:
    """Exact weighted co-occurrence counts between two sets of categorical codes.

    Stage 1 implements ``strategy="exact_int"``, ``differentiable=False``, both
    methods, circular shifts, the dense, pairs and self layouts, the traced
    preconditions C1-C3, and the planner with optional byte budget. Options
    outside Stage 1 raise ``NotImplementedError`` until their task lands.

    Args:
        codes_a: Hard codes for the ``a`` variables, ``(Pa, T)``, as
            ``CategoricalCodes``.
        codes_b: Hard codes for the ``b`` variables, ``(Pb, T)``. ``None`` selects
            the self layout, in which ``a`` and ``b`` are the same variables.
        weights: Non-negative integer weights, ``(R, T)`` or ``(T,)``. ``None``
            means ones with ``R = 1``.
        shifts_a: Per-replicate circular shifts ``(R, Pa)``. Variable ``a`` in
            replicate ``r`` reads ``codes_a[a, (t - s) mod T]``, as ``np.roll``
            does, for any integer shift. ``None`` means no shift.
        shifts_b: Per-replicate circular shifts ``(R, Pb)``, same semantics.
            Must be None in the self layout, where ``shifts_a`` applies to both.
        pairs: ``(n_pairs, 2)`` integer rows ``(a, b)`` selecting the pairs
            layout: ``joint`` is then ``(R, n_pairs, Ka, Kb)``. Rows may repeat.
            In the self layout the pair indices refer to ``codes_a``. May be
            traced; then ``check=True`` range-checks it under trace.
        strategy: ``"exact_int"`` only in Stage 1.
        differentiable: Must be False in Stage 1 (custom_vjp lands in task 1.7).
        method: ``"auto"``, ``"matmul"`` or ``"scatter"``. ``auto`` follows spec
            §4.4 (the pairs layout with fewer pairs than ``Pa*Pb`` prefers
            scatter). The rule is provisional until the probe in task 1.6.
        byte_budget: None for the unbudgeted defaults (the device is not
            queried), an int of bytes, or ``"device"`` for half of the device
            allocator limit. The budget is soft; see ``plan_cooccurrence``.
        plan: A ``CooccurrencePlan`` from ``plan_cooccurrence`` for the same
            shapes. Mutually exclusive with ``byte_budget``.
        check: Enforce C1-C3 (and traced pair range) under trace with
            ``eqx.error_if``. ``False`` skips them, for benchmarks only: results
            are then undefined if a precondition fails.
        mesh: Must be None in Stage 1 (sharding is Stage 1b).

    Returns:
        ``CooccurrenceCounts`` with int32 fields.

    Raises:
        TypeError: If ``codes_a``/``codes_b`` are not ``CategoricalCodes``, or
            weights or pairs are not integer-valued.
        ValueError: If shapes disagree, the self layout gets ``shifts_b``, a
            static index-size guard fails, pairs are out of range or a plan does
            not match this call.
        BudgetInfeasibleError: If ``byte_budget`` cannot hold the result or any tile plan.
        NotImplementedError: For any option outside the Stage 1 scope above.
    """
    if strategy != "exact_int":
        raise NotImplementedError("strategy='float' lands in task 1.7")
    if differentiable:
        raise NotImplementedError("differentiable=True lands in task 1.7")
    if mesh is not None:
        raise NotImplementedError("mesh sharding is Stage 1b")
    if byte_budget is not None and plan is not None:
        raise ValueError("pass byte_budget or plan, not both; a plan already fixes the tiles")

    if not isinstance(codes_a, CategoricalCodes):
        raise TypeError(
            "codes_a must be CategoricalCodes (build with encode_codes); "
            "soft or raw arrays are not accepted on the exact_int path"
        )
    self_layout = codes_b is None
    if self_layout:
        codes_b = codes_a
    elif not isinstance(codes_b, CategoricalCodes):
        raise TypeError("codes_b must be CategoricalCodes or None (self layout)")

    ka = codes_a.n_categories
    kb = codes_b.n_categories
    xa = jnp.asarray(codes_a.codes)
    xb = jnp.asarray(codes_b.codes)
    pa, n_frames = xa.shape
    pb = xb.shape[0]
    if xb.shape[1] != n_frames:
        raise ValueError(f"codes_a and codes_b must share T; got {n_frames} and {xb.shape[1]}")
    if n_frames < 1:
        raise ValueError("at least one frame is required")
    _check_static_sizes(pa, ka, pb, kb)

    sa = _prepare_shifts(shifts_a, pa, "shifts_a")
    if self_layout and shifts_b is not None:
        raise ValueError("shifts_b must be None in the self layout; shifts_a applies to both sides")
    sb = sa if self_layout else _prepare_shifts(shifts_b, pb, "shifts_b")

    w = _prepare_weights(weights, n_frames)
    if check:
        xa, xb, w = _check_preconditions(xa, ka, xb, kb, w)
    n_rep = _common_replicates(
        [("weights", w.shape[0])]
        + [(name, s.shape[0]) for name, s in (("shifts_a", sa), ("shifts_b", sb)) if s is not None]
    )
    w = jnp.broadcast_to(w, (n_rep, n_frames))
    sa = _broadcast_shifts(sa, n_rep, pa)
    sb = _broadcast_shifts(sb, n_rep, pb)

    layout: Literal["dense", "pairs"] = "dense" if pairs is None else "pairs"
    pair_a: jax.Array | None = None
    pair_b: jax.Array | None = None
    if pairs is not None:
        pair_a, pair_b = _prepare_pairs(pairs, pa, pb)
        if check:
            pair_a, pair_b = _check_pairs(pair_a, pair_b, pa, pb)
    n_pairs = pa * pb if pairs is None else int(pair_a.shape[0])

    platform = jax.default_backend()
    chosen, rule = select_method(method, layout, n_pairs, pa * pb, ka, kb, platform)
    if plan is None:
        plan = plan_cooccurrence(
            n_replicates=n_rep,
            n_frames=n_frames,
            n_a=pa,
            k_a=ka,
            n_b=pb,
            k_b=kb,
            n_pairs=n_pairs,
            weight_itemsize=w.dtype.itemsize,
            code_itemsize=codes_a.codes.dtype.itemsize,
            method=method,
            layout=layout,
            byte_budget=byte_budget,
            platform=platform,
        )
    else:
        _validate_plan(plan, chosen, n_rep, n_frames, pa, layout)

    joint, ma, mb, tot = _execute(
        xa,
        xb,
        w,
        sa,
        sb,
        pair_a,
        pair_b,
        layout=layout,
        method=chosen,
        plan=plan,
        ka=ka,
        kb=kb,
        n_frames=n_frames,
    )
    return CooccurrenceCounts(
        joint=joint,
        marginal_a=ma,
        marginal_b=mb,
        total=tot,
        layout=layout,
        method=chosen,
        method_rule=rule,
    )


def iter_cooccurrence_row_blocks(
    codes_a: CategoricalCodes,
    codes_b: CategoricalCodes | None = None,
    *,
    row_block: int,
    **same_kwargs: object,
) -> Iterator[tuple[slice, CooccurrenceCounts]]:
    """Yield dense counts one block of ``codes_a`` rows at a time (host generator).

    Each block ``[start, stop)`` of ``codes_a`` rows is counted against all of
    ``codes_b``, and the result is moved to host memory. ``marginal_b`` and
    ``total`` repeat in every block. In the self layout the block is counted
    against all of ``codes_a``, with ``shifts_a`` applied to both sides as in
    ``cooccurrence_counts``.

    Args:
        codes_a: Hard codes for the ``a`` variables.
        codes_b: Hard codes for the ``b`` variables, or None for the self layout.
        row_block: Variable rows of ``codes_a`` per block; at least 1.
        **same_kwargs: The remaining ``cooccurrence_counts`` arguments
            (``weights``, ``shifts_a``, ``shifts_b``, ``method``, ``byte_budget``,
            ``check``). Each block is planned separately.

    Yields:
        ``(rows, counts)``: ``rows`` is the ``slice`` of ``codes_a`` rows and
        ``counts`` is a ``CooccurrenceCounts`` with numpy arrays.

    Raises:
        TypeError: If ``codes_a`` is not ``CategoricalCodes``.
        ValueError: If ``row_block < 1``, ``plan`` is passed, or ``shifts_b`` is
            passed in the self layout.
        NotImplementedError: If ``pairs`` is passed. Pair blocks are a Stage 2
            concern (spec §6.5).
    """
    if not isinstance(codes_a, CategoricalCodes):
        raise TypeError("codes_a must be CategoricalCodes (build with encode_codes)")
    if same_kwargs.get("pairs") is not None:
        raise NotImplementedError(
            "row blocks split codes_a rows; pair blocks are not implemented in Stage 1"
        )
    if same_kwargs.get("plan") is not None:
        raise ValueError("each block is planned separately; do not pass plan")
    if row_block < 1:
        raise ValueError(f"row_block must be >= 1, got {row_block}")
    self_layout = codes_b is None
    if self_layout and same_kwargs.get("shifts_b") is not None:
        raise ValueError("shifts_b must be None in the self layout; shifts_a applies to both sides")

    n_a = codes_a.codes.shape[0]
    shifts_a = same_kwargs.pop("shifts_a", None)
    shifts_b = same_kwargs.pop("shifts_b", None)
    full_shifts = None if shifts_a is None else jnp.asarray(shifts_a)
    for start in range(0, n_a, row_block):
        stop = min(start + row_block, n_a)
        block = CategoricalCodes(codes=codes_a.codes[start:stop], n_categories=codes_a.n_categories)
        sa = None if full_shifts is None else full_shifts[..., start:stop]
        if self_layout:
            out = cooccurrence_counts(
                block, codes_a, shifts_a=sa, shifts_b=full_shifts, **same_kwargs
            )
        else:
            out = cooccurrence_counts(block, codes_b, shifts_a=sa, shifts_b=shifts_b, **same_kwargs)
        yield slice(start, stop), jax.device_get(out)


__all__ = [
    "KK_CROSSOVER_CPU",
    "KK_CROSSOVER_GPU",
    "CooccurrenceCounts",
    "cooccurrence_counts",
    "iter_cooccurrence_row_blocks",
]

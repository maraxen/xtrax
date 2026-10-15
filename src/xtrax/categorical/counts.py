"""Exact integer weighted co-occurrence counts over categorical codes.

Stage 1 covers the ``exact_int`` path with the ``matmul`` and ``scatter``
methods, the dense, pairs and self layouts, and circular shifts. Counts are
int32 throughout: one-hots and weights are integer, every dot product sets
``preferred_element_type=int32``, scatter-adds accumulate in int32, and no float
touches the counts.

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§3, §4.2-4.4, §6)
"""

from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import ArrayLike

from xtrax.categorical.codes import CategoricalCodes
from xtrax.categorical.plan import CooccurrencePlan

# Method-selection crossovers (spec §4.4). CPU value is provisional until the
# bathos-registered probe (task 1.6) re-derives it.
KK_CROSSOVER_CPU = 64
KK_CROSSOVER_GPU = 16

# Unbudgeted frame-chunk cap (spec §5, `byte_budget=None` defaults).
_FRAME_CHUNK = 1024
# Unbudgeted pair-tile cap for scatter (spec §5 `pair_tile=256`).
_PAIR_TILE = 256
_INT31 = 2**31


class CooccurrenceCounts(eqx.Module):
    """Exact co-occurrence counts for one call of ``cooccurrence_counts``.

    Attributes:
        joint: Dense layout, shape ``(R, Pa, Pb, Ka, Kb)``, int32.
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


def _select_method(
    method: Literal["auto", "matmul", "scatter"],
    layout: Literal["dense", "pairs"],
    n_pairs: int,
    n_dense_pairs: int,
    ka: int,
    kb: int,
    platform: str,
) -> tuple[str, str]:
    """Resolve ``method="auto"`` to a concrete method and its provenance rule.

    Spec §4.4. A pairs layout with fewer pairs than the dense ``Pa*Pb`` grid
    prefers ``scatter``, because ``matmul`` would compute the whole dense grid
    and gather rows. The spec's "<<" is unquantified, so ``n_pairs < Pa*Pb`` is
    the literal reading used here. Otherwise the platform crossover applies.

    Args:
        method: Caller's choice.
        layout: ``"dense"`` or ``"pairs"``.
        n_pairs: Number of pairs computed.
        n_dense_pairs: ``Pa*Pb``, the size of the dense grid.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        platform: ``jax.default_backend()``.

    Returns:
        ``(method, method_rule)``.

    Raises:
        ValueError: If ``method`` is not one of the three allowed values.
    """
    if method not in ("auto", "matmul", "scatter"):
        raise ValueError(f"method must be 'auto', 'matmul' or 'scatter', got {method!r}")
    if method != "auto":
        return method, "explicit"
    if layout == "pairs" and n_pairs < n_dense_pairs:
        rule = "cpu_provisional" if platform == "cpu" else "gpu_provisional"
        return "scatter", rule
    if platform == "cpu":
        chosen = "matmul" if ka * kb <= KK_CROSSOVER_CPU else "scatter"
        return chosen, "cpu_provisional"
    dense_small = layout == "dense" and ka * kb <= KK_CROSSOVER_GPU
    return ("matmul" if dense_small else "scatter"), "gpu_provisional"


def _onehot(codes_chunk: jax.Array, n_categories: int) -> jax.Array:
    """Dense int8 one-hot of an int32 ``(P, tc)`` chunk, shape ``(tc, P*K)``.

    Column ``a*K + k`` is 1 where variable ``a`` has code ``k`` in that frame.
    """
    tc = codes_chunk.shape[1]
    eq = codes_chunk.T[:, :, None] == jnp.arange(n_categories, dtype=jnp.int32)
    return eq.reshape(tc, -1).astype(jnp.int8)


def _shifted_codes(x: jax.Array, shifts: jax.Array, t_idx: jax.Array, n_frames: int) -> jax.Array:
    """Gather ``x[p, (t - s[p]) mod T]`` for every padded frame index ``t``.

    Matches ``np.roll(x[p], s[p])``. Shifts are reduced with ``jnp.mod`` first
    so ``t - s`` stays in range for any int32 shift. Padded frames (``t >= T``)
    read index 0; their weight is zero.

    Args:
        x: int32 codes ``(P, T)``.
        shifts: int32 shifts ``(P,)`` for one replicate.
        t_idx: ``arange(T_pad)`` in int32.
        n_frames: Unpadded frame count ``T``.

    Returns:
        int32 codes ``(P, T_pad)``.
    """
    s_mod = jnp.mod(shifts, n_frames)[:, None]
    idx = jnp.mod(t_idx[None, :] - s_mod, n_frames)
    idx = jnp.where((t_idx < n_frames)[None, :], idx, 0)
    return jnp.take_along_axis(x, idx, axis=1)


@partial(jax.jit, static_argnames=("ka", "kb", "t_chunk", "n_frames"))
def _matmul_dense(
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Matmul counts with per-replicate circular shifts.

    Args:
        xa: int32 codes ``(Pa, T)``, unpadded.
        xb: int32 codes ``(Pb, T)``, unpadded.
        w: Integer weights ``(R, T_pad)``, zero-padded.
        sa: int32 shifts ``(R, Pa)``; zeros for no shift.
        sb: int32 shifts ``(R, Pb)``; zeros for no shift.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        t_chunk: Frames per scan step; divides ``T_pad``.
        n_frames: Unpadded frame count ``T``.

    Returns:
        ``(joint, marginal_a, marginal_b, total)`` in int32, with ``joint``
        shaped ``(R, Pa, Pb, Ka, Kb)``. Frames are scanned in chunks of
        ``t_chunk`` with an int32 carry; replicates run sequentially through
        ``lax.map`` (``r_tile=1``).
    """
    pa = xa.shape[0]
    pb = xb.shape[0]
    r, t_pad = w.shape
    n_chunks = t_pad // t_chunk
    wdtype = w.dtype
    fa, fb = pa * ka, pb * kb
    t_idx = jnp.arange(t_pad, dtype=jnp.int32)

    def one_replicate(args: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        w_r, sa_r, sb_r = args
        xa_r = _shifted_codes(xa, sa_r, t_idx, n_frames)
        xb_r = _shifted_codes(xb, sb_r, t_idx, n_frames)
        xa_c = xa_r.reshape(pa, n_chunks, t_chunk).transpose(1, 0, 2)
        xb_c = xb_r.reshape(pb, n_chunks, t_chunk).transpose(1, 0, 2)
        w_c = w_r.reshape(n_chunks, t_chunk)

        def step(carry: tuple[jax.Array, ...], chunk: tuple[jax.Array, ...]) -> tuple:
            joint, ma, mb, tot = carry
            xa_chunk, xb_chunk, w_chunk = chunk
            oa = _onehot(xa_chunk, ka)
            ob = _onehot(xb_chunk, kb).astype(wdtype)
            wc = w_chunk[:, None]
            ow_a = oa.astype(wdtype) * wc
            joint = joint + lax.dot_general(
                ow_a,
                ob,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.int32,
            )
            ma = ma + jnp.sum(ow_a, axis=0, dtype=jnp.int32)
            mb = mb + jnp.sum(ob * wc, axis=0, dtype=jnp.int32)
            tot = tot + jnp.sum(w_chunk, dtype=jnp.int32)
            return (joint, ma, mb, tot), None

        init = (
            jnp.zeros((fa, fb), dtype=jnp.int32),
            jnp.zeros((fa,), dtype=jnp.int32),
            jnp.zeros((fb,), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        carry, _ = lax.scan(step, init, (xa_c, xb_c, w_c))
        return carry

    joint, ma, mb, tot = lax.map(one_replicate, (w, sa, sb))
    joint = joint.reshape(r, pa, ka, pb, kb).transpose(0, 1, 3, 2, 4)
    return joint, ma.reshape(r, pa, ka), mb.reshape(r, pb, kb), tot


@partial(jax.jit, static_argnames=("ka", "kb", "t_chunk", "n_frames", "pair_tile"))
def _scatter_counts(
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-pair composite bincount with per-replicate circular shifts.

    Each pair ``p = (a, b)`` accumulates ``hist[p, x_a*Kb + x_b] += w`` over
    frames, in int32. Weights are cast to int32 before the add: ``bincount``
    would accumulate in the weight dtype and wrap (spec A26). Pairs run in
    ``pair_tile`` chunks through ``lax.map`` inside the frame scan.

    Args:
        xa: int32 codes ``(Pa, T)``, unpadded.
        xb: int32 codes ``(Pb, T)``, unpadded.
        w: Integer weights ``(R, T_pad)``, zero-padded.
        sa: int32 shifts ``(R, Pa)``; zeros for no shift.
        sb: int32 shifts ``(R, Pb)``; zeros for no shift.
        pair_a: int32 ``a`` index per pair, length a multiple of ``pair_tile``.
            Padding entries are 0 and their rows are sliced off by the caller.
        pair_b: int32 ``b`` index per pair, same length as ``pair_a``.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        t_chunk: Frames per scan step; divides ``T_pad``.
        n_frames: Unpadded frame count ``T``.
        pair_tile: Pairs per ``lax.map`` step.

    Returns:
        ``(joint, marginal_a, marginal_b, total)`` in int32, with ``joint``
        shaped ``(R, n_pairs_padded, Ka, Kb)`` and the marginals as in
        ``_matmul_dense``.
    """
    pa, pb = xa.shape[0], xb.shape[0]
    r, t_pad = w.shape
    n_chunks = t_pad // t_chunk
    n_tiles = pair_a.shape[0] // pair_tile
    kk = ka * kb
    t_idx = jnp.arange(t_pad, dtype=jnp.int32)
    rows = jnp.arange(pair_tile, dtype=jnp.int32)[:, None]
    a_rows = jnp.arange(pa, dtype=jnp.int32)[:, None]
    b_rows = jnp.arange(pb, dtype=jnp.int32)[:, None]

    def one_replicate(args: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        w_r, sa_r, sb_r = args
        xa_r = _shifted_codes(xa, sa_r, t_idx, n_frames)
        xb_r = _shifted_codes(xb, sb_r, t_idx, n_frames)
        xa_c = xa_r.reshape(pa, n_chunks, t_chunk).transpose(1, 0, 2)
        xb_c = xb_r.reshape(pb, n_chunks, t_chunk).transpose(1, 0, 2)
        w_c = w_r.reshape(n_chunks, t_chunk).astype(jnp.int32)

        def step(carry: tuple[jax.Array, ...], chunk: tuple[jax.Array, ...]) -> tuple:
            hist, ma, mb, tot = carry
            xa_k, xb_k, w_k = chunk
            comp = xa_k[pair_a] * kb + xb_k[pair_b]
            comp = comp.reshape(n_tiles, pair_tile, t_chunk)

            def add_tile(args_t: tuple[jax.Array, jax.Array]) -> jax.Array:
                h, c = args_t
                return h.at[rows, c].add(w_k[None, :])

            hist = lax.map(add_tile, (hist, comp))
            ma = ma.at[a_rows, xa_k].add(w_k[None, :])
            mb = mb.at[b_rows, xb_k].add(w_k[None, :])
            tot = tot + jnp.sum(w_k, dtype=jnp.int32)
            return (hist, ma, mb, tot), None

        init = (
            jnp.zeros((n_tiles, pair_tile, kk), dtype=jnp.int32),
            jnp.zeros((pa, ka), dtype=jnp.int32),
            jnp.zeros((pb, kb), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        (hist, ma, mb, tot), _ = lax.scan(step, init, (xa_c, xb_c, w_c))
        return hist.reshape(-1, ka, kb), ma, mb, tot

    return lax.map(one_replicate, (w, sa, sb))


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

    Pairs must be a concrete host array in Stage 1. Traced pairs need the
    traced range check, which lands with task 1.5.

    Args:
        pairs: Integer rows ``(a, b)`` with ``0 <= a < Pa`` and ``0 <= b < Pb``.
        pa: Number of ``a`` variables.
        pb: Number of ``b`` variables.

    Returns:
        ``(pair_a, pair_b)``, int32 arrays of length ``n_pairs``.

    Raises:
        NotImplementedError: If ``pairs`` is traced.
        TypeError: If ``pairs`` is not integer-valued.
        ValueError: If the shape is not ``(n, 2)`` with ``n >= 1``, or an index
            is out of range.
    """
    if isinstance(pairs, jax.core.Tracer):
        raise NotImplementedError(
            "traced pairs need the traced range check, which lands in task 1.5; "
            "pass a concrete array"
        )
    host = np.asarray(pairs)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError(f"pairs must be integer-valued, got dtype {host.dtype}")
    if host.ndim != 2 or host.shape[1] != 2 or host.shape[0] < 1:
        raise ValueError(f"pairs must have shape (n_pairs, 2) with n_pairs >= 1, got {host.shape}")
    a, b = host[:, 0], host[:, 1]
    if a.min() < 0 or a.max() >= pa or b.min() < 0 or b.max() >= pb:
        raise ValueError(
            f"pairs index out of range: a must lie in [0, {pa}), b in [0, {pb}); "
            f"found a in [{a.min()}, {a.max()}], b in [{b.min()}, {b.max()}]"
        )
    return jnp.asarray(a, dtype=jnp.int32), jnp.asarray(b, dtype=jnp.int32)


def _pad_pairs(pair_a: jax.Array, pair_b: jax.Array) -> tuple[jax.Array, jax.Array, int]:
    """Pad pair columns to a multiple of the pair tile; return the tile size.

    Padding rows point at pair ``(0, 0)``; the caller slices them off.
    """
    n = pair_a.shape[0]
    tile = min(_PAIR_TILE, n)
    n_pad = -(-n // tile) * tile
    extra = n_pad - n
    return jnp.pad(pair_a, (0, extra)), jnp.pad(pair_b, (0, extra)), tile


def _check_static_sizes(pa: int, ka: int, pb: int, kb: int) -> None:
    """Reject index spaces that would overflow int32 (spec §3.1, §4.1)."""
    for name, size in (("Pa*Ka", pa * ka), ("Pb*Kb", pb * kb), ("Ka*Kb", ka * kb)):
        if size >= _INT31:
            raise ValueError(f"{name}={size} must be < 2**31 for int32 index arithmetic")


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

    Stage 1 implements ``strategy="exact_int"``, ``differentiable=False``,
    the ``matmul`` and ``scatter`` methods, circular shifts, and the dense,
    pairs and self layouts with unbudgeted defaults (``byte_budget=None``).
    Every other option named in the signature raises ``NotImplementedError``
    until its task lands (spec §10).

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
            In the self layout the pair indices refer to ``codes_a``. Must be a
            concrete array in Stage 1.
        strategy: ``"exact_int"`` only in Stage 1.
        differentiable: Must be False in Stage 1 (custom_vjp lands in task 1.7).
        method: ``"auto"``, ``"matmul"`` or ``"scatter"``. ``auto`` follows
            spec §4.4 (the pairs layout with fewer pairs than ``Pa*Pb`` prefers
            scatter).
        byte_budget: Must be None in Stage 1 (planner lands in task 1.5).
        plan: Must be None in Stage 1 (planner lands in task 1.5).
        check: Traced preconditions C1-C3 (spec §3.2) are enforced in task 1.5.
            Until then this argument is accepted and not yet enforced.
        mesh: Must be None in Stage 1 (sharding is Stage 1b).

    Returns:
        ``CooccurrenceCounts`` with int32 fields.

    Raises:
        TypeError: If ``codes_a``/``codes_b`` are not ``CategoricalCodes``, or
            weights are not integer-valued.
        ValueError: If shapes disagree, the self layout gets ``shifts_b``, or a
            static index-size guard fails.
        NotImplementedError: For any option outside the Stage 1 scope above.
    """
    if strategy != "exact_int":
        raise NotImplementedError("strategy='float' lands in task 1.7")
    if differentiable:
        raise NotImplementedError("differentiable=True lands in task 1.7")
    if byte_budget is not None:
        raise NotImplementedError("byte_budget lands in task 1.5")
    if plan is not None:
        raise NotImplementedError("plan lands in task 1.5")
    if mesh is not None:
        raise NotImplementedError("mesh sharding is Stage 1b")

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
    xa = jnp.asarray(codes_a.codes).astype(jnp.int32)
    xb = jnp.asarray(codes_b.codes).astype(jnp.int32)
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
    n_rep = _common_replicates(
        [("weights", w.shape[0])]
        + [(name, s.shape[0]) for name, s in (("shifts_a", sa), ("shifts_b", sb)) if s is not None]
    )
    w = jnp.broadcast_to(w, (n_rep, n_frames))
    sa = _broadcast_shifts(sa, n_rep, pa)
    sb = _broadcast_shifts(sb, n_rep, pb)

    layout: Literal["dense", "pairs"] = "dense" if pairs is None else "pairs"
    if pairs is None:
        pair_a, pair_b = _dense_pair_index(pa, pb)
    else:
        pair_a, pair_b = _prepare_pairs(pairs, pa, pb)
    n_pairs = int(pair_a.shape[0])
    chosen, rule = _select_method(method, layout, n_pairs, pa * pb, ka, kb, jax.default_backend())

    t_chunk = min(_FRAME_CHUNK, n_frames)
    t_pad = -(-n_frames // t_chunk) * t_chunk
    w = jnp.pad(w, ((0, 0), (0, t_pad - n_frames)))

    if chosen == "matmul":
        joint, ma, mb, tot = _matmul_dense(
            xa, xb, w, sa, sb, ka=ka, kb=kb, t_chunk=t_chunk, n_frames=n_frames
        )
        if layout == "pairs":
            joint = joint[:, pair_a, pair_b]
    else:
        pair_a_p, pair_b_p, pair_tile = _pad_pairs(pair_a, pair_b)
        hist, ma, mb, tot = _scatter_counts(
            xa,
            xb,
            w,
            sa,
            sb,
            pair_a_p,
            pair_b_p,
            ka=ka,
            kb=kb,
            t_chunk=t_chunk,
            n_frames=n_frames,
            pair_tile=pair_tile,
        )
        joint = hist[:, :n_pairs]
        if layout == "dense":
            joint = joint.reshape(joint.shape[0], pa, pb, ka, kb)
    return CooccurrenceCounts(
        joint=joint,
        marginal_a=ma,
        marginal_b=mb,
        total=tot,
        layout=layout,
        method=chosen,
        method_rule=rule,
    )


__all__ = [
    "KK_CROSSOVER_CPU",
    "KK_CROSSOVER_GPU",
    "CooccurrenceCounts",
    "cooccurrence_counts",
]

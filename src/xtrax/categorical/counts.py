"""Exact integer weighted co-occurrence counts over categorical codes.

Stage 1 covers the ``exact_int`` path with the ``matmul`` method and the dense
and self layouts. Counts are int32 throughout: one-hots and weights are
integer, every dot product sets ``preferred_element_type=int32``, and no float
touches the counts.

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§3, §4.2, §6)
"""

from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
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
_INT31 = 2**31


class CooccurrenceCounts(eqx.Module):
    """Exact co-occurrence counts for one call of ``cooccurrence_counts``.

    Attributes:
        joint: Dense layout, shape ``(R, Pa, Pb, Ka, Kb)``, int32.
        marginal_a: Shape ``(R, Pa, Ka)``, int32.
        marginal_b: Shape ``(R, Pb, Kb)``, int32.
        total: Shape ``(R,)``, int32; the summed weight per replicate.
        layout: ``"dense"`` (pairs layout lands in task 1.4).
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
    ka: int,
    kb: int,
    platform: str,
) -> tuple[str, str]:
    """Resolve ``method="auto"`` to a concrete method and its provenance rule."""
    if method not in ("auto", "matmul", "scatter"):
        raise ValueError(f"method must be 'auto', 'matmul' or 'scatter', got {method!r}")
    if method != "auto":
        return method, "explicit"
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


@partial(jax.jit, static_argnames=("ka", "kb", "t_chunk"))
def _matmul_dense(
    xa: jax.Array, xb: jax.Array, w: jax.Array, *, ka: int, kb: int, t_chunk: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Matmul counts for zero-padded int32 codes ``(P, T_pad)`` and weights ``(R, T_pad)``.

    Frames are scanned in chunks of ``t_chunk`` with an int32 carry; replicates
    run sequentially through ``lax.map`` (``r_tile=1``).
    """
    pa, t_pad = xa.shape
    pb = xb.shape[0]
    r = w.shape[0]
    n_chunks = t_pad // t_chunk
    wdtype = w.dtype
    fa, fb = pa * ka, pb * kb

    xa_c = xa.reshape(pa, n_chunks, t_chunk).transpose(1, 0, 2)
    xb_c = xb.reshape(pb, n_chunks, t_chunk).transpose(1, 0, 2)
    w_c = w.reshape(r, n_chunks, t_chunk)

    def one_replicate(w_r: jax.Array) -> tuple[jax.Array, ...]:
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
        carry, _ = lax.scan(step, init, (xa_c, xb_c, w_r))
        return carry

    joint, ma, mb, tot = lax.map(one_replicate, w_c)
    joint = joint.reshape(r, pa, ka, pb, kb).transpose(0, 1, 3, 2, 4)
    return joint, ma.reshape(r, pa, ka), mb.reshape(r, pb, kb), tot


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
    the ``matmul`` method, and the dense and self layouts with unbudgeted
    defaults (``byte_budget=None``). Every other option named in the signature
    raises ``NotImplementedError`` until its task lands (see the spec, §10).

    Args:
        codes_a: Hard codes for the ``a`` variables, ``(Pa, T)``, as
            ``CategoricalCodes``.
        codes_b: Hard codes for the ``b`` variables, ``(Pb, T)``. ``None`` selects
            the self layout, in which ``a`` and ``b`` are the same variables.
        weights: Non-negative integer weights, ``(R, T)`` or ``(T,)``. ``None``
            means ones with ``R = 1``.
        shifts_a: Per-replicate circular shifts ``(R, Pa)``; task 1.3.
        shifts_b: Per-replicate circular shifts ``(R, Pb)``; task 1.3. Must be
            None in the self layout.
        pairs: ``(n_pairs, 2)`` pair rows; pairs layout lands in task 1.4.
        strategy: ``"exact_int"`` only in Stage 1.
        differentiable: Must be False in Stage 1 (custom_vjp lands in task 1.7).
        method: ``"auto"``, ``"matmul"`` or ``"scatter"``. ``scatter`` lands in
            task 1.4, so ``auto`` resolving to it raises for now.
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
    if pairs is not None:
        raise NotImplementedError("pairs layout lands in task 1.4")
    if shifts_a is not None or shifts_b is not None:
        raise NotImplementedError("circular shifts land in task 1.3")
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

    w = _prepare_weights(weights, n_frames)
    layout: Literal["dense", "pairs"] = "dense"
    chosen, rule = _select_method(method, layout, ka, kb, jax.default_backend())
    if chosen != "matmul":
        raise NotImplementedError(f"method {chosen!r} lands in task 1.4")

    t_chunk = min(_FRAME_CHUNK, n_frames)
    t_pad = -(-n_frames // t_chunk) * t_chunk
    pad = ((0, 0), (0, t_pad - n_frames))
    xa = jnp.pad(xa, pad)
    xb = jnp.pad(xb, pad)
    w = jnp.pad(w, pad)

    joint, ma, mb, tot = _matmul_dense(xa, xb, w, ka=ka, kb=kb, t_chunk=t_chunk)
    return CooccurrenceCounts(
        joint=joint,
        marginal_a=ma,
        marginal_b=mb,
        total=tot,
        layout=layout,
        method="matmul",
        method_rule=rule,
    )


__all__ = [
    "KK_CROSSOVER_CPU",
    "KK_CROSSOVER_GPU",
    "CooccurrenceCounts",
    "cooccurrence_counts",
]

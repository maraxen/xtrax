#!/usr/bin/env python3
"""Benchmark five ways to compute pairwise joint histograms + mutual information.

Spec ``.praxia/docs/specs/261023_kernel-seam-xtrax-op.md`` section 3 (backlog #6099),
task 261023_equinox-decouple-pallas. One invocation measures ONE cell
(form, entropy dtype, shape, K, tiles) in its own process and writes a stamped
artifact; ``--plan`` prints every cell as a shell command so the job script can wrap
each in its own ``timeout`` (preemption-safe, per the global long-computation rule).

Forms (all take int32 states ``(n_res, n_frames)`` and ``pairs (n_pairs, 2)`` and
return per-pair MI ``(n_pairs,)`` in nats, plus a debug path returning exact int32
joint counts ``(S, K, K)`` for a few pairs):

- ``A``  port of denxity's ``fused_mi_matrix`` path: per-pair composite-index
  histogram + entropy, chunked over pairs with ``chunked_map(batch_size=...)`` and
  denxity's pad-and-strip (``chunked_e_step``). Deviation from the source: counts come
  from ``zeros(K*K, count_dtype).at[composite].add(1)`` (explicit dtype) instead of
  ``jnp.bincount``, which returns int64 when x64 is on; in-range inputs make the two
  identical. Entropy is fused per pair, so counts never leave the map body.
- ``B``  plain XLA: ``lax.map`` over pair chunks, ``fori_loop`` over frame tiles,
  ``einsum("pfk,pfl->pkl", onehot(s_i), onehot(s_j))`` accumulated in float32 (or
  int32 for ``--onehot-dtype int8``). Float32 counts are exact only below 2**24; the
  script asserts ``n_frames < 2**24`` (55,738 and 28,007 frames are fine).
- ``C_triton``  Pallas, Triton backend: grid over pair tiles, an in-kernel
  ``fori_loop`` over frame tiles, one-hot matmul per tile accumulated in float32. The
  Triton backend maps every grid axis to a parallel CUDA axis, so a frame-tile grid axis
  cannot accumulate; hence the frame loop lives inside the kernel body.
- ``C_mosaic``  Pallas, Mosaic GPU backend: grid ``(pair tile, frame tile)`` with the
  frame axis declared ``"sequential"`` and the ``(K, K)`` output block accumulated
  across it (the block-matmul pattern of the 261023 spike q1-q4). Runs on CPU only in
  interpret mode here; real Mosaic GPU lowering of this body is UNVERIFIED.
- ``D``  Pallas, Triton backend: grid ``(pair, frame tile)``; each program histograms its
  frame tile with ``jax.experimental.pallas.triton.atomic_add`` into the pair's int32
  ``K*K`` row in global memory (the output is aliased to a zeros input). These are
  global-memory atomics, not a privatised per-block histogram.

B, C and D gather ``states[i]``, ``states[j]`` for a pair chunk in XLA (as A does) and
compute MI from the ``(chunk, K, K)`` counts in XLA, so the entropy is outside the Pallas
kernels. Fusing the entropy into the kernel is a follow-up, not measured here.

Entropy dtype is a cell parameter (``float64`` or ``float32``); x64 is enabled in every
cell so the two are compared under one environment. Marginal entropies use the same dtype.

Facts measured on jax 0.11.1 (2026-10-26) that the run environment relies on:

- Pallas on GPU defaults to the Mosaic GPU backend
  (``jax/_src/pallas/pallas_call.py:920-971``, ``config.jax_pallas_use_mosaic_gpu``).
- ``compiler_params=jax.experimental.pallas.triton.CompilerParams()`` selects the Triton
  backend, which is DEPRECATED in 0.11.1 (``deprecations.warn("jax-pallas-triton")`` at
  ``pallas_call.py:~962``) and accepts only GPUs in
  ``jax/_src/pallas/triton/gpu_info.py``'s registry. H100 and H200 are in it, Turing is
  not (why this runs on Engaging H200, not titanix).
- ``pl.pallas_call`` for Mosaic GPU is itself deprecated in favour of ``plgpu.kernel``
  (``pallas_call.py:~943``); ``C_mosaic`` uses ``pallas_call`` because it is the only
  form that keeps the BlockSpec/grid structure of the spike.
- ``jax.experimental.pallas.triton.atomic_add`` and
  ``jax.experimental.pallas.mosaic_gpu.atomic_add`` both exist
  (``jax/experimental/pallas/{triton,mosaic_gpu}.py``).

Provenance of form A: denxity ``src/denxity/graph/mi_kernel.py`` (``_pair_mi``,
``_fused_mi_values``, ``fused_mi_matrix``), ``src/denxity/graph/_graph_common.py``
(``_entropy``, ``compute_marginal_entropies``) and ``src/denxity/_tiling.py``
(``chunked_e_step``), all read at denxity origin/main commit
832ad2ad1cfa46c68453c4f6f0835ba0aaa03429 (read 2026-10-26).

Synthetic data: ``n_res`` residues x ``n_frames`` frames of int32 states, a fixed seed,
``--n-used`` states per residue (``<= K``, skewed Dirichlet marginals) and a shared latent
class that couples residues so most MI values are nonzero (an all-zero MI is the silent
failure of the retired flat ``vmap(bincount)``, ``mist.py:1598-1636``).

Per cell: compile (first call) wall seconds and backend-compile seconds from
``jax.monitoring``; median of ``--reps`` timed calls after ``--warmup`` calls, each with
``block_until_ready``; recompiles after the first call (a ``jax.monitoring`` compile-event
count, expected 0); peak device bytes from ``device.memory_stats()``. Correctness on a
sampled pair subset (first, last, random): exact counts vs host NumPy, row sums ==
``n_frames``, max |MI - float64 NumPy MI|. The script reports these and decides nothing.

Usage (one cell; ``BTH_RESULTS_PATH`` receives the flat result when set):
    uv run --no-sync python scripts/benchmarks/pairwise_mi_forms.py \\
        --form B --entropy-dtype float64 --shape full --K 64 --out-dir OUT

All cells as commands (each should be wrapped in its own ``timeout`` by the job script):
    uv run --no-sync python scripts/benchmarks/pairwise_mi_forms.py --plan \\
        --out-dir OUT --Ks 32,64,128 --timeout-s 3600

CPU smoke test (Pallas forms in interpret mode, tiny shape):
    uv run --no-sync python scripts/benchmarks/pairwise_mi_forms.py \\
        --form C_triton --shape custom --n-res 12 --n-frames 200 --K 8 --interpret \\
        --reps 2 --warmup 1 --out-dir OUT
"""

import argparse
import dataclasses
import datetime
import hashlib
import itertools
import json
import logging
import os
import shlex
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

from xtrax import chunked_map

log = logging.getLogger("pairwise_mi_forms")

SCRIPT_PATH = Path(__file__).resolve()
REPO_RELPATH = "scripts/benchmarks/pairwise_mi_forms.py"
DENXITY_COMMIT = "832ad2ad1cfa46c68453c4f6f0835ba0aaa03429"

FORMS = ("A", "B", "C_triton", "C_mosaic", "D")
BACKEND = {
    "A": "xla",
    "B": "xla",
    "C_triton": "pallas_triton",
    "C_mosaic": "pallas_mosaic_gpu",
    "D": "pallas_triton",
}
ENTROPY_DTYPES = ("float64", "float32")
# (n_res, n_frames, n_pairs); n_pairs 0 means all n_res*(n_res-1)/2 pairs.
SHAPES = {
    "full": (423, 28007, 0),  # 89,253 pairs, mist.py:730,1601
    "tree": (423, 28007, 422),  # 422 random pairs (MIST edge subset)
    "boundary": (300, 55738, 0),  # 44,850 pairs x 55,738 frames = 2.50e9 > 2.45e9 (mist.py:1604)
}
ONEHOT_DTYPES = {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "int8": jnp.int8}
COUNT_DTYPES = {"int32": jnp.int32, "int64": jnp.int64}
CORRUPTIONS = ("", "shift_state", "extra_count", "mi_scale")
# Per-form config keys that influence the measurement; only these enter the config hash.
FORM_KEYS = {
    "A": ("a_batch_size", "count_dtype"),
    "B": ("pair_chunk", "frame_tile", "onehot_dtype"),
    "C_triton": ("pair_chunk", "pair_tile", "frame_tile", "onehot_dtype", "num_warps"),
    "C_mosaic": ("pair_chunk", "pair_tile", "frame_tile", "onehot_dtype"),
    "D": ("pair_chunk", "frame_tile", "num_warps"),
}
F32_EXACT_LIMIT = 2**24
MIN_KP = {"C_triton": 16, "C_mosaic": 64}  # padded K; Triton dot needs >=16, wgmma M>=64


# --------------------------------------------------------------------------- config


@dataclasses.dataclass(frozen=True)
class Cell:
    """Everything that defines one cell; its hash stamps the artifact."""

    form: str
    entropy_dtype: str
    shape: str
    n_res: int
    n_frames: int
    n_pairs: int
    K: int
    n_used: int
    seed: int
    n_sample: int
    reps: int
    warmup: int
    interpret: bool
    corrupt: str
    a_batch_size: int
    count_dtype: str
    pair_chunk: int
    pair_tile: int
    frame_tile: int
    onehot_dtype: str
    num_warps: int

    def hash_dict(self) -> dict[str, Any]:
        """Config entries that affect the result, per form (irrelevant tile knobs excluded)."""
        base = {
            "form": self.form,
            "entropy_dtype": self.entropy_dtype,
            "shape": self.shape,
            "n_res": self.n_res,
            "n_frames": self.n_frames,
            "n_pairs": self.n_pairs,
            "K": self.K,
            "n_used": self.n_used,
            "seed": self.seed,
            "n_sample": self.n_sample,
            "reps": self.reps,
            "warmup": self.warmup,
            "interpret": self.interpret,
            "corrupt": self.corrupt,
            "denxity_commit": DENXITY_COMMIT,
        }
        for key in FORM_KEYS[self.form]:
            base[key] = getattr(self, key)
        return base

    def config_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.hash_dict(), sort_keys=True).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def script_sha256() -> str:
    return sha256_file(SCRIPT_PATH)


def cell_id(cell: Cell) -> str:
    return (
        f"{cell.form}_{cell.entropy_dtype}_{cell.shape}_K{cell.K}"
        f"_{cell.config_sha256()[:8]}_{script_sha256()[:8]}"
    )


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def validate(cell: Cell) -> None:
    """Fail fast on configurations that cannot be measured faithfully."""
    if cell.form not in FORMS:
        raise ValueError(f"unknown form {cell.form!r}")
    if cell.entropy_dtype not in ENTROPY_DTYPES:
        raise ValueError(f"entropy_dtype must be one of {ENTROPY_DTYPES}")
    if not 1 <= cell.n_used <= cell.K:
        raise ValueError(f"n_used={cell.n_used} must be in [1, K={cell.K}]")
    if cell.corrupt not in CORRUPTIONS:
        raise ValueError(f"corrupt must be one of {CORRUPTIONS}")
    if cell.onehot_dtype not in ONEHOT_DTYPES:
        raise ValueError(f"onehot_dtype must be one of {tuple(ONEHOT_DTYPES)}")
    if cell.form in ("C_triton", "C_mosaic") and cell.onehot_dtype == "int8":
        raise ValueError("int8 one-hot is implemented for form B only")
    if cell.n_frames >= F32_EXACT_LIMIT:
        raise ValueError(f"n_frames={cell.n_frames} >= 2**24: float32 counts are not exact")
    if cell.pair_chunk % cell.pair_tile:
        raise ValueError("pair_chunk must be a multiple of pair_tile")
    if cell.form in ("C_triton", "D") and (cell.frame_tile & (cell.frame_tile - 1)):
        raise ValueError("Triton frame_tile must be a power of two")
    if cell.form == "C_triton" and (cell.pair_tile & (cell.pair_tile - 1)):
        raise ValueError("Triton pair_tile must be a power of two")


# --------------------------------------------------------------------------- data


def make_states(n_res: int, n_frames: int, K: int, n_used: int, seed: int) -> np.ndarray:
    """Synthetic int32 states ``(n_res, n_frames)`` in ``[0, n_used)``, fixed seed.

    Each residue draws from its own Dirichlet(0.5) marginal; with probability 0.3 a
    frame instead takes a state determined by a shared latent class, which couples the
    residues. ``K`` only enters through ``n_used <= K``.
    """
    assert 1 <= n_used <= K
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(n_used, 0.5), size=n_res)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((n_res, n_frames), dtype=np.float32)
    own = np.empty((n_res, n_frames), dtype=np.int32)
    for r in range(n_res):  # host-side data generation, not a measured loop
        own[r] = np.minimum(np.searchsorted(cdf[r], u[r]), n_used - 1)
    latent = rng.integers(0, 6, n_frames, dtype=np.int64)
    mult = rng.integers(1, 5, n_res, dtype=np.int64)
    coupled = ((latent[None, :] * mult[:, None]) % n_used).astype(np.int32)
    use_latent = rng.random((n_res, n_frames), dtype=np.float32) < 0.3
    return np.where(use_latent, coupled, own).astype(np.int32)


def make_pairs(n_res: int, n_pairs: int, seed: int) -> np.ndarray:
    """All ``i < j`` pairs (row-major, as denxity), or ``n_pairs`` random distinct ones."""
    pairs = np.stack(np.triu_indices(n_res, k=1), axis=-1).astype(np.int32)
    if n_pairs and n_pairs < pairs.shape[0]:
        rng = np.random.default_rng(seed + 1)
        keep = np.sort(rng.choice(pairs.shape[0], size=n_pairs, replace=False))
        pairs = pairs[keep]
    return pairs


def sample_indices(n_pairs: int, n_sample: int, seed: int) -> np.ndarray:
    """Sorted pair indices including the first and last pair."""
    if n_pairs <= n_sample:
        return np.arange(n_pairs)
    rng = np.random.default_rng(seed + 2)
    mid = rng.choice(np.arange(1, n_pairs - 1), size=n_sample - 2, replace=False)
    return np.sort(np.concatenate([[0, n_pairs - 1], mid]))


# --------------------------------------------------------------------------- math


def _entropy(p: jax.Array, dtype: Any) -> jax.Array:
    """-sum(p log p) over the last axis in ``dtype`` with 0 log 0 = 0 (denxity ``_entropy``)."""
    p = p.astype(dtype)
    safe_p = jnp.where(p > 0, p, jnp.asarray(1.0, dtype))
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(safe_p), jnp.asarray(0.0, dtype)), axis=-1)


def entropy_from_counts(counts: jax.Array, dtype: Any) -> jax.Array:
    """Entropy of each row of ``counts (..., M)`` using denxity's 1e-9 total floor."""
    total = counts.sum(axis=-1).astype(dtype)
    p = counts.astype(dtype) / jnp.maximum(total, jnp.asarray(1e-9, dtype))[..., None]
    return _entropy(p, dtype)


def marginal_entropies(states: jax.Array, K: int, dtype: Any) -> jax.Array:
    """Per-residue marginal entropy; ``states`` may hold only in-range values."""

    def one(row: jax.Array) -> jax.Array:
        counts = jnp.zeros(K, jnp.int32).at[row].add(1)
        return entropy_from_counts(counts, dtype)

    return jax.vmap(one)(states)


def mi_from_chunk_counts(
    counts: jax.Array, marg_i: jax.Array, marg_j: jax.Array, dtype: Any
) -> jax.Array:
    """``max(H_i + H_j - H_ij, 0)`` for counts ``(P, K, K)``."""
    h_ij = entropy_from_counts(counts.reshape(counts.shape[0], -1), dtype)
    return jnp.maximum(marg_i + marg_j - h_ij, jnp.asarray(0.0, dtype))


# --------------------------------------------------------------------------- forms


@dataclasses.dataclass(frozen=True)
class Form:
    """A built form: prepared device inputs plus a jitted MI path and a counts path."""

    states_dev: jax.Array
    mi_fn: Callable[[jax.Array, jax.Array], jax.Array]
    counts_fn: Callable[[jax.Array, jax.Array], jax.Array]


def _corrupt_states(states: jax.Array, cell: Cell) -> jax.Array:
    """Negative-control hook: relabel states by +1 (mod K). Preserves MI, breaks counts."""
    if cell.corrupt == "shift_state":
        # Relabelling by an affine map keeps the entropy but moves every count; padding
        # sentinels (-1) must stay put so frame padding still contributes nothing.
        return jnp.where(states >= 0, (states + 1) % cell.K, states)
    return states


def _chunk_size(n: int, cell: Cell) -> int:
    """Pair-chunk length for ``n`` pairs: ``pair_chunk`` capped to ``n`` rounded to a tile."""
    tile = cell.pair_tile if cell.form in ("C_triton", "C_mosaic") else 1
    return min(cell.pair_chunk, _round_up(max(n, 1), tile))


def _map_pair_chunks(
    chunk_fn: Callable[[jax.Array], jax.Array], pairs: jax.Array, chunk: int
) -> jax.Array:
    """``lax.map`` ``chunk_fn`` over pair chunks; pad rows are pair (0, 0) and stripped."""
    n = pairs.shape[0]
    padded = _round_up(n, chunk)
    if padded != n:
        pairs = jnp.pad(pairs, ((0, padded - n), (0, 0)))
    out = lax.map(chunk_fn, pairs.reshape(padded // chunk, chunk, 2))
    return out.reshape((padded,) + out.shape[2:])[:n]


def _chunked_e_step(fn: Callable[[jax.Array], Any], x: jax.Array, batch_size: int) -> Any:
    """denxity ``_tiling.chunked_e_step``: pad to a multiple of batch_size, map, strip."""
    n = x.shape[0]
    remainder = n % batch_size
    pad_amount = 0 if remainder == 0 else batch_size - remainder
    if pad_amount > 0:
        x = jnp.pad(x, [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1), mode="constant")
    return jax.tree.map(lambda leaf: leaf[:n], chunked_map(fn, x, batch_size=batch_size))


def _pad_frames(states: np.ndarray, frame_tile: int) -> np.ndarray:
    """Pad frames with the sentinel -1 (one-hot of -1 is all zeros) to a tile multiple."""
    n_frames = states.shape[1]
    pad = _round_up(n_frames, frame_tile) - n_frames
    if pad == 0:
        return states
    return np.pad(states, ((0, 0), (0, pad)), constant_values=-1)


def _onehot_matmul_counts(
    si: jax.Array, sj: jax.Array, K: int, frame_tile: int, ohdt: Any
) -> jax.Array:
    """Counts ``(P, K, K)`` int32 via ``einsum('pfk,pfl->pkl')`` over frame tiles (form B)."""
    pc, f_pad = si.shape
    int_path = ohdt == jnp.int8
    acc_dt = jnp.int32 if int_path else jnp.float32

    def body(t: jax.Array, acc: jax.Array) -> jax.Array:
        a = lax.dynamic_slice_in_dim(si, t * frame_tile, frame_tile, axis=1)
        b = lax.dynamic_slice_in_dim(sj, t * frame_tile, frame_tile, axis=1)
        oh_i = jax.nn.one_hot(a, K, dtype=ohdt)
        oh_j = jax.nn.one_hot(b, K, dtype=ohdt)
        return acc + jnp.einsum("pfk,pfl->pkl", oh_i, oh_j, preferred_element_type=acc_dt)

    acc = lax.fori_loop(0, f_pad // frame_tile, body, jnp.zeros((pc, K, K), acc_dt))
    return acc.astype(jnp.int32)


def _c_triton_counts(si: jax.Array, sj: jax.Array, cell: Cell) -> jax.Array:
    """Counts ``(P, K, K)`` int32 with a Pallas Triton kernel (form C_triton)."""
    pc, f_pad = si.shape
    pt, ft, K = cell.pair_tile, cell.frame_tile, cell.K
    kp = max(MIN_KP["C_triton"], _next_pow2(K))
    n_t = f_pad // ft
    ohdt = ONEHOT_DTYPES[cell.onehot_dtype]

    def kernel(si_ref: Any, sj_ref: Any, o_ref: Any) -> None:
        rows = lax.broadcasted_iota(jnp.int32, (kp, ft), 0)
        cols = lax.broadcasted_iota(jnp.int32, (ft, kp), 1)
        for q in range(pt):

            def body(t: jax.Array, acc: jax.Array, q: int = q) -> jax.Array:
                sl = pl.ds(t * ft, ft)
                a = si_ref[q, sl]
                b = sj_ref[q, sl]
                oh_i = (rows == a[None, :]).astype(ohdt)
                oh_j = (b[:, None] == cols).astype(ohdt)
                return acc + jnp.dot(oh_i, oh_j, preferred_element_type=jnp.float32)

            acc = lax.fori_loop(0, n_t, body, jnp.zeros((kp, kp), jnp.float32))
            o_ref[q, :, :] = acc.astype(jnp.int32)

    out = pl.pallas_call(
        kernel,
        grid=(pc // pt,),
        in_specs=[pl.BlockSpec((pt, f_pad), lambda p: (p, 0))] * 2,
        out_specs=pl.BlockSpec((pt, kp, kp), lambda p: (p, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((pc, kp, kp), jnp.int32),
        compiler_params=pltriton.CompilerParams(num_warps=cell.num_warps),
        interpret=cell.interpret,
    )(si, sj)
    return out[:, :K, :K]


def _c_mosaic_counts(si: jax.Array, sj: jax.Array, cell: Cell) -> jax.Array:
    """Counts ``(P, K, K)`` int32 with a Pallas Mosaic GPU kernel (form C_mosaic)."""
    from jax.experimental.pallas import mosaic_gpu as plgpu

    pc, f_pad = si.shape
    pt, ft, K = cell.pair_tile, cell.frame_tile, cell.K
    kp = max(MIN_KP["C_mosaic"], _next_pow2(K))
    ohdt = ONEHOT_DTYPES[cell.onehot_dtype]

    def kernel(si_ref: Any, sj_ref: Any, o_ref: Any) -> None:
        @pl.when(pl.program_id(1) == 0)
        def _init() -> None:
            o_ref[...] = jnp.zeros_like(o_ref)

        rows = lax.broadcasted_iota(jnp.int32, (kp, ft), 0)
        cols = lax.broadcasted_iota(jnp.int32, (ft, kp), 1)
        for q in range(pt):
            a = si_ref[q, :]
            b = sj_ref[q, :]
            oh_i = (rows == a[None, :]).astype(ohdt)
            oh_j = (b[:, None] == cols).astype(ohdt)
            o_ref[q, :, :] += jnp.dot(oh_i, oh_j, preferred_element_type=jnp.float32)

    out = pl.pallas_call(
        kernel,
        grid=(pc // pt, f_pad // ft),
        in_specs=[pl.BlockSpec((pt, ft), lambda p, f: (p, f))] * 2,
        out_specs=pl.BlockSpec((pt, kp, kp), lambda p, f: (p, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((pc, kp, kp), jnp.float32),
        compiler_params=plgpu.CompilerParams(dimension_semantics=("parallel", "sequential")),
        interpret=cell.interpret,
    )(si, sj)
    return out[:, :K, :K].astype(jnp.int32)


def _d_counts(si: jax.Array, sj: jax.Array, cell: Cell) -> jax.Array:
    """Counts ``(P, K, K)`` int32 via Triton ``atomic_add`` histograms (form D)."""
    pc, f_pad = si.shape
    ft, K = cell.frame_tile, cell.K

    def kernel(si_ref: Any, sj_ref: Any, _zeros_ref: Any, o_ref: Any) -> None:
        a = si_ref[0, :]
        b = sj_ref[0, :]
        valid = (a >= 0) & (b >= 0)
        idx = jnp.where(valid, a * K + b, 0)
        if cell.interpret:
            # The interpreter's discharge rule for atomic_add (jax/_src/pallas/triton/
            # primitives.py:312-350) rejects ``mask`` and, for an array index, lowers to
            # ``ref.at[idx].set(ref[idx] + val)``, which drops colliding updates. A
            # scalar-index loop is exact there; it tests indexing, aliasing, masking and
            # grid logic but NOT the vector atomic lowering used on the GPU.
            def body(t: jax.Array, carry: None) -> None:
                pltriton.atomic_add(o_ref, (idx[t],), valid[t].astype(jnp.int32))
                return carry

            lax.fori_loop(0, ft, body, None)
        else:
            pltriton.atomic_add(o_ref, (idx,), jnp.ones((ft,), jnp.int32), mask=valid)

    out = pl.pallas_call(
        kernel,
        grid=(pc, f_pad // ft),
        in_specs=[
            pl.BlockSpec((1, ft), lambda p, f: (p, f)),
            pl.BlockSpec((1, ft), lambda p, f: (p, f)),
            pl.BlockSpec((None, K * K), lambda p, f: (p, 0)),
        ],
        out_specs=pl.BlockSpec((None, K * K), lambda p, f: (p, 0)),
        out_shape=jax.ShapeDtypeStruct((pc, K * K), jnp.int32),
        input_output_aliases={2: 0},
        compiler_params=pltriton.CompilerParams(num_warps=cell.num_warps),
        interpret=cell.interpret,
    )(si, sj, jnp.zeros((pc, K * K), jnp.int32))
    return out.reshape(pc, K, K)


def _chunk_counts(cell: Cell) -> Callable[[jax.Array, jax.Array], jax.Array]:
    ohdt = ONEHOT_DTYPES[cell.onehot_dtype]
    if cell.form == "B":
        return lambda si, sj: _onehot_matmul_counts(si, sj, cell.K, cell.frame_tile, ohdt)
    if cell.form == "C_triton":
        return lambda si, sj: _c_triton_counts(si, sj, cell)
    if cell.form == "C_mosaic":
        return lambda si, sj: _c_mosaic_counts(si, sj, cell)
    if cell.form == "D":
        return lambda si, sj: _d_counts(si, sj, cell)
    raise ValueError(cell.form)


def build_form(cell: Cell, states_np: np.ndarray) -> Form:
    """Prepare device inputs and the two jitted paths for ``cell.form``."""
    edt = jnp.dtype(cell.entropy_dtype)
    K = cell.K
    n_frames = states_np.shape[1]

    if cell.form == "A":
        cdt = COUNT_DTYPES[cell.count_dtype]
        states_dev = jnp.asarray(states_np, jnp.int32)

        def pair_counts(row: jax.Array, states: jax.Array) -> jax.Array:
            composite = states[row[0]] * K + states[row[1]]
            return jnp.zeros(K * K, cdt).at[composite].add(1)

        def mi_fn(states: jax.Array, pairs: jax.Array) -> jax.Array:
            states = _corrupt_states(states, cell)
            marg = marginal_entropies(states, K, edt)

            def pair_mi(row: jax.Array) -> jax.Array:
                h_ij = entropy_from_counts(pair_counts(row, states), edt)
                return jnp.maximum(marg[row[0]] + marg[row[1]] - h_ij, jnp.asarray(0.0, edt))

            mi = _chunked_e_step(pair_mi, pairs, cell.a_batch_size)
            return mi * jnp.asarray(1.001, edt) if cell.corrupt == "mi_scale" else mi

        def counts_fn(states: jax.Array, pairs: jax.Array) -> jax.Array:
            states = _corrupt_states(states, cell)
            fn = partial(pair_counts, states=states)
            counts = _chunked_e_step(fn, pairs, cell.a_batch_size).astype(jnp.int32)
            counts = counts.reshape(-1, K, K)
            if cell.corrupt == "extra_count":
                counts = counts.at[:, 0, 0].add(1)
            return counts

        return Form(states_dev, jax.jit(mi_fn), jax.jit(counts_fn))

    chunk_counts = _chunk_counts(cell)
    states_dev = jnp.asarray(_pad_frames(states_np, cell.frame_tile), jnp.int32)

    def gather_chunk_counts(states: jax.Array, chunk_pairs: jax.Array) -> jax.Array:
        return chunk_counts(states[chunk_pairs[:, 0]], states[chunk_pairs[:, 1]])

    def mi_fn(states: jax.Array, pairs: jax.Array) -> jax.Array:
        states = _corrupt_states(states, cell)
        marg = marginal_entropies(states[:, :n_frames], K, edt)

        def chunk_mi(chunk_pairs: jax.Array) -> jax.Array:
            counts = gather_chunk_counts(states, chunk_pairs)
            return mi_from_chunk_counts(
                counts, marg[chunk_pairs[:, 0]], marg[chunk_pairs[:, 1]], edt
            )

        mi = _map_pair_chunks(chunk_mi, pairs, _chunk_size(pairs.shape[0], cell))
        return mi * jnp.asarray(1.001, edt) if cell.corrupt == "mi_scale" else mi

    def counts_fn(states: jax.Array, pairs: jax.Array) -> jax.Array:
        states = _corrupt_states(states, cell)
        counts = _map_pair_chunks(
            lambda cp: gather_chunk_counts(states, cp), pairs, _chunk_size(pairs.shape[0], cell)
        )
        if cell.corrupt == "extra_count":
            counts = counts.at[:, 0, 0].add(1)
        return counts

    return Form(states_dev, jax.jit(mi_fn), jax.jit(counts_fn))


# --------------------------------------------------------------------------- reference


def numpy_reference(states: np.ndarray, pairs: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact int64 counts ``(S, K, K)`` and float64 MI ``(S,)`` for ``pairs`` on the host."""
    n_frames = states.shape[1]
    cache: dict[int, float] = {}

    def marg(r: int) -> float:
        if r not in cache:
            p = np.bincount(states[r], minlength=K) / max(n_frames, 1e-9)
            nz = p[p > 0]
            cache[r] = float(-(nz * np.log(nz)).sum())
        return cache[r]

    counts = np.empty((len(pairs), K, K), dtype=np.int64)
    mi = np.empty(len(pairs), dtype=np.float64)
    for n, (i, j) in enumerate(pairs):
        c = np.bincount(states[i].astype(np.int64) * K + states[j], minlength=K * K).reshape(K, K)
        counts[n] = c
        p = c.ravel() / max(c.sum(), 1e-9)
        nz = p[p > 0]
        mi[n] = max(marg(int(i)) + marg(int(j)) + float((nz * np.log(nz)).sum()), 0.0)
    return counts, mi


# --------------------------------------------------------------------------- measuring


class CompileCounter:
    """Counts backend compiles via ``jax.monitoring`` duration events."""

    EVENT = "/jax/core/compile/backend_compile_duration"
    _installed = False
    n = 0
    seconds = 0.0

    @classmethod
    def install(cls) -> None:
        if cls._installed:
            return

        def listener(event: str, duration: float, **_: Any) -> None:
            if event == cls.EVENT:
                cls.n += 1
                cls.seconds += float(duration)

        jax.monitoring.register_event_duration_secs_listener(listener)
        cls._installed = True

    @classmethod
    def snapshot(cls) -> tuple[int, float]:
        return cls.n, cls.seconds


class Heartbeat:
    """Fsynced phase log written between (never inside) timed regions."""

    def __init__(self, path: Path | None) -> None:
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def beat(self, phase: str, **extra: Any) -> None:
        log.info("phase %s %s", phase, extra or "")
        if self._fh is None:
            return
        rec = {"t": time.time(), "phase": phase, **extra}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def _peak_bytes() -> int:
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return -1
    return int(stats.get("peak_bytes_in_use", -1))


def blank_result(cell: Cell) -> dict[str, Any]:
    dev = jax.devices()[0]
    n_res, n_frames, _ = cell.n_res, cell.n_frames, cell.n_pairs
    return {
        "form": cell.form,
        "backend": BACKEND[cell.form],
        "entropy_dtype": cell.entropy_dtype,
        "shape": cell.shape,
        "n_res": n_res,
        "n_pairs": -1,
        "n_frames": n_frames,
        "K": cell.K,
        "n_used": cell.n_used,
        "compile_s": -1.0,
        "compile_backend_s": -1.0,
        "compile_events_first": -1,
        "step_median_s": -1.0,
        "step_cv": -1.0,
        "step_times_s": [],
        "recompiles": -1,
        "counts_exact": False,
        "rowsums_ok": False,
        "mi_max_abs_diff": -1.0,
        "n_sampled": 0,
        "mi_nonfinite": -1,
        "mi_all_zero": False,
        "mi_min": -1.0,
        "mi_mean": -1.0,
        "peak_bytes": -1,
        "error": "",
        "interpret": cell.interpret,
        "corrupt": cell.corrupt,
        "pair_chunk": cell.pair_chunk,
        "pair_tile": cell.pair_tile,
        "frame_tile": cell.frame_tile,
        "onehot_dtype": cell.onehot_dtype,
        "count_dtype": cell.count_dtype,
        "a_batch_size": cell.a_batch_size,
        "num_warps": cell.num_warps,
        "seed": cell.seed,
        "jax_version": jax.__version__,
        "platform": dev.platform,
        "device_kind": dev.device_kind,
        "config_sha256": cell.config_sha256(),
        "script_sha256": script_sha256(),
        "denxity_commit": DENXITY_COMMIT,
        "mi_npy": "",
    }


def measure_cell(cell: Cell, hb: Heartbeat, mi_out: Path | None) -> dict[str, Any]:
    """Run one cell with x64 enabled (scoped). Errors are recorded in ``error``."""
    with jax.enable_x64(True):
        return _measure_cell(cell, hb, mi_out)


def _measure_cell(cell: Cell, hb: Heartbeat, mi_out: Path | None) -> dict[str, Any]:
    """Run one cell. Errors are recorded in ``error``; earlier fields are kept."""
    result = blank_result(cell)
    stage = "setup"
    try:
        validate(cell)
        CompileCounter.install()
        hb.beat("data")
        states_np = make_states(cell.n_res, cell.n_frames, cell.K, cell.n_used, cell.seed)
        pairs_np = make_pairs(cell.n_res, cell.n_pairs, cell.seed)
        n_pairs = int(pairs_np.shape[0])
        result["n_pairs"] = n_pairs
        sample = sample_indices(n_pairs, cell.n_sample, cell.seed)
        ref_counts, ref_mi = numpy_reference(states_np, pairs_np[sample], cell.K)
        result["n_sampled"] = int(len(sample))

        stage = "build"
        hb.beat("build")
        form = build_form(cell, states_np)
        pairs_dev = jnp.asarray(pairs_np, jnp.int32)
        jax.block_until_ready((form.states_dev, pairs_dev))

        stage = "compile"
        hb.beat("compile_start")
        n0, s0 = CompileCounter.snapshot()
        t0 = time.perf_counter()
        out = jax.block_until_ready(form.mi_fn(form.states_dev, pairs_dev))
        result["compile_s"] = time.perf_counter() - t0
        n1, s1 = CompileCounter.snapshot()
        result["compile_events_first"] = n1 - n0
        result["compile_backend_s"] = s1 - s0
        hb.beat("compile_done", compile_s=result["compile_s"])

        stage = "time"
        for w in range(cell.warmup):
            jax.block_until_ready(form.mi_fn(form.states_dev, pairs_dev))
            hb.beat("warmup", i=w)
        times = []
        for r in range(cell.reps):
            t0 = time.perf_counter()
            out = jax.block_until_ready(form.mi_fn(form.states_dev, pairs_dev))
            times.append(time.perf_counter() - t0)
            hb.beat("rep", i=r, seconds=times[-1])
        n2, _ = CompileCounter.snapshot()
        result["recompiles"] = n2 - n1
        result["step_times_s"] = [float(t) for t in times]
        result["step_median_s"] = float(statistics.median(times))
        mean = statistics.fmean(times)
        result["step_cv"] = float(statistics.pstdev(times) / mean) if mean > 0 else -1.0
        result["peak_bytes"] = _peak_bytes()

        stage = "mi_check"
        mi = np.asarray(out, dtype=np.float64)
        if mi.shape != (n_pairs,):
            raise ValueError(f"MI shape {mi.shape} != ({n_pairs},)")
        result["mi_nonfinite"] = int((~np.isfinite(mi)).sum())
        result["mi_all_zero"] = bool(np.all(mi == 0.0))
        result["mi_min"] = float(np.nanmin(mi))
        result["mi_mean"] = float(np.nanmean(mi))
        result["mi_max_abs_diff"] = float(np.max(np.abs(mi[sample] - ref_mi)))
        if mi_out is not None:
            np.save(mi_out, mi)
            result["mi_npy"] = mi_out.name

        stage = "counts_check"
        hb.beat("counts_check")
        counts = np.asarray(
            jax.block_until_ready(form.counts_fn(form.states_dev, jnp.asarray(pairs_np[sample])))
        )
        result["counts_exact"] = bool(
            counts.shape == ref_counts.shape and np.array_equal(counts, ref_counts)
        )
        result["rowsums_ok"] = bool(
            counts.shape[0] == len(sample)
            and np.all(counts.reshape(counts.shape[0], -1).sum(axis=1) == cell.n_frames)
        )
        hb.beat("done")
    except Exception as exc:  # recorded, not raised: the cell's artifact is the failure record
        log.error("cell failed in stage %s:\n%s", stage, traceback.format_exc())
        result["error"] = f"{stage}: {type(exc).__name__}: {str(exc)[:600]}"
        hb.beat("error", stage=stage)
    return result


# --------------------------------------------------------------------------- stamping


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def find_reusable(
    out_dir: Path, cid: str, cfg_sha: str, scr_sha: str, retry_errors: bool
) -> dict[str, Any] | None:
    """Return the stored result if the stamp matches config, script and artifact hashes."""
    stamp_path = out_dir / f"{cid}.stamp.json"
    art_path = out_dir / f"{cid}.json"
    if not stamp_path.is_file() or not art_path.is_file():
        return None
    try:
        stamp = json.loads(stamp_path.read_text())
        if stamp["config_sha256"] != cfg_sha or stamp["script_sha256"] != scr_sha:
            return None
        if stamp["artifact_sha256"]["json"] != sha256_file(art_path):
            return None
        npy = stamp["artifact_sha256"].get("mi_npy")
        if npy and sha256_file(out_dir / f"{cid}.mi.npy") != npy:
            return None
        result = json.loads(art_path.read_text())
    except (KeyError, ValueError, OSError):
        return None
    if retry_errors and result.get("error"):
        return None
    return result


def record_reuse(out_dir: Path, cid: str, stamp_path: Path) -> None:
    rec = {"t": _utc_now(), "cell_id": cid, "reused_from": str(stamp_path)}
    with (out_dir / "reuse_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_artifacts(out_dir: Path, cid: str, cell: Cell, result: dict[str, Any]) -> Path:
    """Write the result JSON, then the completion stamp (the stamp is the commit point)."""
    art_path = out_dir / f"{cid}.json"
    _atomic_write_text(art_path, json.dumps(result, indent=2, allow_nan=False))
    shas = {"json": sha256_file(art_path)}
    npy_path = out_dir / f"{cid}.mi.npy"
    if result["mi_npy"] and npy_path.is_file():
        shas["mi_npy"] = sha256_file(npy_path)
    stamp = {
        "cell_id": cid,
        "config_sha256": cell.config_sha256(),
        "script_sha256": script_sha256(),
        "artifact_sha256": shas,
        "completed_utc": _utc_now(),
    }
    stamp_path = out_dir / f"{cid}.stamp.json"
    _atomic_write_text(stamp_path, json.dumps(stamp, indent=2))
    return stamp_path


# --------------------------------------------------------------------------- CLI


def _csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def _csv(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def add_cell_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--form", choices=FORMS)
    p.add_argument("--entropy-dtype", choices=ENTROPY_DTYPES, default="float64")
    p.add_argument("--shape", choices=(*SHAPES, "custom"), default="full")
    p.add_argument("--n-res", type=int, help="override (shape custom)")
    p.add_argument("--n-frames", type=int, help="override (shape custom)")
    p.add_argument("--n-pairs", type=int, help="override; 0 = all pairs")
    p.add_argument("--K", type=int, default=32, help="states per residue slot (padded K)")
    p.add_argument("--n-used", type=int, help="states actually used per residue; default K")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sample", type=int, default=64, help="pairs checked against NumPy")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--interpret", action="store_true", help="Pallas interpret mode (CPU)")
    p.add_argument("--corrupt", choices=CORRUPTIONS, default="", help="negative-control hook")
    p.add_argument("--a-batch-size", type=int, default=4096, help="form A lax.map batch size")
    p.add_argument("--count-dtype", choices=tuple(COUNT_DTYPES), default="int32", help="form A")
    p.add_argument("--pair-chunk", type=int, default=2048, help="B/C/D pairs per outer chunk")
    p.add_argument("--pair-tile", type=int, default=4, help="C: pairs per Pallas program")
    p.add_argument("--frame-tile", type=int, default=1024, help="B/C/D frames per tile")
    p.add_argument("--onehot-dtype", choices=tuple(ONEHOT_DTYPES), default="bfloat16")
    p.add_argument("--num-warps", type=int, default=4, help="Triton forms")


def cell_from_args(args: argparse.Namespace) -> Cell:
    n_res, n_frames, n_pairs = SHAPES[args.shape] if args.shape in SHAPES else (None, None, 0)
    n_res = args.n_res if args.n_res is not None else n_res
    n_frames = args.n_frames if args.n_frames is not None else n_frames
    n_pairs = args.n_pairs if args.n_pairs is not None else n_pairs
    if n_res is None or n_frames is None:
        raise SystemExit("--shape custom needs --n-res and --n-frames")
    return Cell(
        form=args.form,
        entropy_dtype=args.entropy_dtype,
        shape=args.shape,
        n_res=int(n_res),
        n_frames=int(n_frames),
        n_pairs=int(n_pairs),
        K=args.K,
        n_used=args.n_used if args.n_used is not None else args.K,
        seed=args.seed,
        n_sample=args.n_sample,
        reps=args.reps,
        warmup=args.warmup,
        interpret=args.interpret,
        corrupt=args.corrupt,
        a_batch_size=args.a_batch_size,
        count_dtype=args.count_dtype,
        pair_chunk=args.pair_chunk,
        pair_tile=args.pair_tile,
        frame_tile=args.frame_tile,
        onehot_dtype=args.onehot_dtype,
        num_warps=args.num_warps,
    )


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    """Run (or reuse) the single cell described by ``args`` and return its result."""
    cell = cell_from_args(args)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = cell_id(cell)
    if not args.force:
        reused = find_reusable(
            out_dir, cid, cell.config_sha256(), script_sha256(), args.retry_errors
        )
        if reused is not None:
            stamp_path = out_dir / f"{cid}.stamp.json"
            record_reuse(out_dir, cid, stamp_path)
            log.info("reused %s from %s", cid, stamp_path)
            return reused
    hb = Heartbeat(out_dir / f"{cid}.heartbeat.jsonl")
    try:
        result = measure_cell(cell, hb, out_dir / f"{cid}.mi.npy")
    finally:
        hb.close()
    write_artifacts(out_dir, cid, cell, result)
    return result


def plan_cells(args: argparse.Namespace) -> list[list[str]]:
    """Every cell as an argv (without interpreter), deduplicated by config hash."""
    forms = _csv(args.forms) or [f for f in FORMS if f != "C_mosaic"]
    shapes = _csv(args.shapes) or list(SHAPES)
    seen: set[str] = set()
    cells: list[list[str]] = []
    grid = itertools.product(
        forms,
        _csv(args.entropy_dtypes),
        shapes,
        _csv_ints(args.Ks),
        _csv_ints(args.pair_chunks),
        _csv_ints(args.pair_tiles),
        _csv_ints(args.frame_tiles),
        _csv(args.onehot_dtypes),
        _csv_ints(args.a_batch_sizes),
    )
    for form, edt, shape, K, pchunk, ptile, ftile, ohdt, abatch in grid:
        ns = argparse.Namespace(
            **{
                **vars(args),
                "form": form,
                "entropy_dtype": edt,
                "shape": shape,
                "K": K,
                "pair_chunk": pchunk,
                "pair_tile": ptile,
                "frame_tile": ftile,
                "onehot_dtype": ohdt,
                "a_batch_size": abatch,
                "n_res": None,
                "n_frames": None,
                "n_pairs": None,
                "n_used": args.n_used,
            }
        )
        cell = cell_from_args(ns)
        key = cell.config_sha256()
        if key in seen:
            continue
        seen.add(key)
        argv = [
            "--form", form, "--entropy-dtype", edt, "--shape", shape, "--K", str(K),
            "--seed", str(args.seed), "--reps", str(args.reps), "--warmup", str(args.warmup),
            "--out-dir", str(args.out_dir),
        ]  # fmt: skip
        if args.n_used is not None:
            argv += ["--n-used", str(args.n_used)]
        if args.interpret:
            argv.append("--interpret")
        if form == "A":
            argv += ["--a-batch-size", str(abatch)]
        else:
            argv += ["--pair-chunk", str(pchunk), "--frame-tile", str(ftile)]
            if form in ("B", "C_triton", "C_mosaic"):
                argv += ["--onehot-dtype", ohdt]
            if form in ("C_triton", "C_mosaic"):
                argv += ["--pair-tile", str(ptile)]
            if form in ("C_triton", "D"):
                argv += ["--num-warps", str(args.num_warps)]
        cells.append(argv)
    return cells


def print_plan(args: argparse.Namespace) -> None:
    py = shlex.split(args.python_cmd)
    for argv in plan_cells(args):
        cmd = [
            "env", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "timeout", str(args.timeout_s), *py, REPO_RELPATH, *argv,
        ]  # fmt: skip
        print(shlex.join(cmd))


def lower_check(args: argparse.Namespace, *, check_cell: bool = True) -> dict[str, Any]:
    """Lower one form's chunk-counts function for CUDA on a CPU-only host (no GPU run).

    Uses an abstract GPU device (``jax.sharding.AbstractDevice``) so the Triton backend can
    pick its compute capability. This exercises the Pallas-to-Triton-IR lowering, which
    catches shape, dtype and unsupported-op errors; it does not run PTX compilation or the
    kernel, so a pass here is necessary and not sufficient for a GPU run. For
    ``C_mosaic`` it reproduces the ``dot_general`` lowering failure of jax 0.11.1.
    """
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    cell = cell_from_args(args)
    out: dict[str, Any] = {
        "form": cell.form,
        "K": cell.K,
        "abstract_device_kind": args.abstract_device_kind,
        "lowers": False,
        "triton_custom_call": False,
        "error": "",
    }
    try:
        if check_cell:
            validate(cell)
        if cell.form == "A" or cell.interpret:
            raise ValueError("lower-check needs a B/C/D form without --interpret")
        chunk = _chunk_size(cell.pair_chunk, cell)
        f_pad = _round_up(cell.n_frames, cell.frame_tile)
        spec = jax.ShapeDtypeStruct((chunk, f_pad), jnp.int32)
        mesh = AbstractMesh(
            (), (), abstract_device=AbstractDevice(args.abstract_device_kind, None, "gpu")
        )
        with use_abstract_mesh(mesh), jax.enable_x64(True):
            text = (
                jax.jit(_chunk_counts(cell))
                .trace(spec, spec)
                .lower(lowering_platforms=("cuda",))
                .as_text()
            )
        out["lowers"] = True
        out["triton_custom_call"] = "gpu.triton" in text
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cell_args(p)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="re-run even if a stamp matches")
    p.add_argument("--retry-errors", action="store_true", help="do not reuse errored cells")
    p.add_argument("--plan", action="store_true", help="print all cells as commands and exit")
    p.add_argument(
        "--lower-check",
        action="store_true",
        help="lower --form for CUDA on this host with an abstract GPU (no GPU needed)",
    )
    p.add_argument("--abstract-device-kind", default="NVIDIA H200", help="lower-check")
    p.add_argument("--forms", default="", help="plan: comma list (default all but C_mosaic)")
    p.add_argument("--shapes", default="", help="plan: comma list (default all)")
    p.add_argument("--entropy-dtypes", default="float64,float32", help="plan")
    p.add_argument("--Ks", default="32", help="plan: comma list of K")
    p.add_argument("--pair-chunks", default="2048", help="plan: comma list")
    p.add_argument("--pair-tiles", default="4", help="plan: comma list")
    p.add_argument("--frame-tiles", default="1024", help="plan: comma list")
    p.add_argument("--onehot-dtypes", default="bfloat16", help="plan: comma list")
    p.add_argument("--a-batch-sizes", default="4096", help="plan: comma list")
    p.add_argument("--timeout-s", type=int, default=3600, help="plan: per-cell timeout")
    p.add_argument("--python-cmd", default="uv run --no-sync python", help="plan")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.plan:
        print_plan(args)
        return 0
    if args.form is None:
        raise SystemExit("--form is required (or use --plan)")
    if args.lower_check:
        report = lower_check(args)
        print(json.dumps(report, indent=2))
        return 0 if report["lowers"] else 1
    result = run_one(args)
    text = json.dumps(result, allow_nan=False)
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(text)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

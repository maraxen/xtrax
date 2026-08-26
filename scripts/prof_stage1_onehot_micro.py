#!/usr/bin/env python3
"""Stage 1 one-hot probe: CPU micro-execution + named-scope attribution.

P1 of .praxia/docs/specs/260825_jax-optimizing-skill-scope.md (Tier-3 exemplar).
One jitted program hosts BOTH categorical-encoding variants: "materialized"
consumes a prebuilt dense f32[N,K] one-hot INPUT (exactly what a host-side
materialization pipeline would feed device_put), while "onthefly" receives
compact int32[N] class indices and encodes via jax.nn.one_hot in-graph. Each
variant sits under its own jax.named_scope -- the single-executable pattern
from prof_stage1_tiling_micro.py (D8/D9): the compiled HLO text carries the
scope paths, the executed trace carries per-thunk durations, and
xtrax.profiling.trace's two-input attribution joins them. One executable also
means one thunk-name namespace, so per-variant labels cannot collide.
Caveat stated plainly: this isolates GRAPH compute cost; the H2D transfer
difference (8*K bytes/row extra for the dense feed) is NOT inside the traced
region -- compare the two executables' end-to-end walls for that.

Beyond the traced combined program, each variant is ALSO timed separately in
untraced steady-state loops; those wall times land as advisory metrics
(materialized_seconds / onthefly_seconds) -- the attribution-backed numbers
live in `scopes`. A numerics-parity metric (max abs output diff between the
variants on identical indices) is captured in-record: any Tier-3 change that
could alter results carries its own parity evidence (scope doc Q1 resolution).

Stage-1 records support STRUCTURAL and DISPATCH_COUNT claims only; a
TERM_RANKING over them fails closed by design (CPU ranking is structurally
invalid on GPU). Re-run this driver at stage>=2 on a GPU for rankable pairs.

Usage:
    XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage1_onehot_micro.py \
        [--out-dir outputs/profiling/stage1] [--trials 20]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from xtrax.profiling.claims import permitted_claims
from xtrax.profiling.emitters import (
    ATTRIBUTION_NAMED_SCOPE,
    attribution_from_scopes,
    emit_probe_record,
)
from xtrax.profiling.trace import (
    parse_dispatch_counts,
    parse_scopes,
    scope_map_from_hlo_text,
)

ROOT = Path(__file__).resolve().parents[1]

# Scope-label registry lives in drivers, never in the library package (D8).
LABEL_MATERIALIZED = "onehot_materialized"
LABEL_ONTHEFLY = "onehot_onthefly"
KNOWN_LABELS = frozenset({LABEL_MATERIALIZED, LABEL_ONTHEFLY})

# Parity tolerance: the variants may fuse differently, so bitwise equality is
# not expected; divergence beyond this absolute tolerance fails the driver
# loudly instead of emitting a record whose parity metric launders a bug.
PARITY_TOLERANCE = 1e-5


def _kernel(one_hot_f32: jax.Array, w: jax.Array) -> jax.Array:
    """Representative gather-classify kernel: one-hot @ weights + row reduce."""
    return ((one_hot_f32 @ w) ** 2).sum(axis=-1)


def _build_program(rows: int, classes: int, cols: int):
    """One executable hosting both encoding variants under their own scopes.

    The materialized scope consumes a DENSE f32[N,K] input (no one_hot call);
    the onthefly scope encodes int32[N] indices in-graph. Same weights both
    sides so graph-compute cost is directly attributable per scope.
    """
    rng = jax.random.key(0)
    w = jax.random.normal(rng, (classes, cols))
    oh = jnp.ones((rows, classes))
    idx = jax.random.randint(rng, (rows,), 0, classes, dtype=jnp.int32)

    def program(oh_arg, idx_arg):
        with jax.named_scope(LABEL_MATERIALIZED):
            a = _kernel(oh_arg, w)
        with jax.named_scope(LABEL_ONTHEFLY):
            b = _kernel(jax.nn.one_hot(idx_arg, classes), w)
        return a + b

    return program, oh, idx


def _variant_fns(rows: int, classes: int, cols: int):
    """Standalone per-variant functions for the untraced paired wall timing."""
    rng = jax.random.key(0)
    w = jax.random.normal(rng, (classes, cols))

    def materialized(oh_arg):
        return _kernel(oh_arg, w)

    def onthefly(idx_arg):
        return _kernel(jax.nn.one_hot(idx_arg, classes), w)

    return materialized, onthefly


def _load_trace_events(trace_dir: Path) -> list[dict]:
    trace_files = sorted(trace_dir.rglob("*.trace.json.gz"))
    if not trace_files:
        raise SystemExit(f"no *.trace.json.gz written under {trace_dir}")
    with gzip.open(trace_files[-1], "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise SystemExit(f"{trace_files[-1]} has no traceEvents list")
    print(f"loaded {len(events)} events from {trace_files[-1].name}")
    return events


def _timed_steady_state(fn, x, warmup: int, trials: int) -> float:
    """Untraced steady-state wall seconds over `trials` calls (compile excluded)."""
    compiled = jax.jit(fn).lower(x).compile()
    for _ in range(max(warmup, 1)):
        compiled(x).block_until_ready()
    start = time.perf_counter()
    for _ in range(trials):
        compiled(x).block_until_ready()
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "stage1",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args(argv)

    # --- Parity gate (outside any timed/trace region) ----------------------
    # Compare the two REAL computations over IDENTICAL class assignments:
    # feed the dense one_hot(idx_zero) matrix to the materialized kernel and
    # the same indices to the onthefly kernel; outputs must agree.
    program, oh, idx = _build_program(args.rows, args.classes, args.cols)
    materialized_fn, onthefly_fn = _variant_fns(args.rows, args.classes, args.cols)
    idx_zero = jnp.zeros_like(idx)
    out_mat = jax.jit(materialized_fn)(jax.nn.one_hot(idx_zero, args.classes))
    out_mat = out_mat.block_until_ready()
    out_fly = jax.jit(onthefly_fn)(idx_zero).block_until_ready()
    parity_max_abs_diff = float(jnp.max(jnp.abs(out_mat - out_fly)))
    if not (math.isfinite(parity_max_abs_diff) and parity_max_abs_diff <= PARITY_TOLERANCE):
        raise SystemExit(
            f"parity gate FAILED: max|materialized - onthefly| = "
            f"{parity_max_abs_diff!r} exceeds tolerance {PARITY_TOLERANCE} "
            "-- the variants disagree numerically; fix before measuring."
        )

    # --- Combined-program trace (single executable, disjoint labels) -------
    compiled = jax.jit(program).lower(oh, idx).compile()
    for _ in range(max(args.warmup, 1)):
        compiled(oh, idx).block_until_ready()

    # Under the repo-wide ignore glob outputs/profiling/**/_traces/ (traces
    # stay off-repo); the onehot/ subdir avoids clobbering tiling-probe traces.
    trace_dir = args.out_dir / "_traces" / "onehot"
    trace_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for _ in range(args.trials):
            compiled(oh, idx).block_until_ready()
    total_step_seconds = time.perf_counter() - start

    hlo_text = compiled.as_text()
    hlo_path = args.out_dir / "hlo_as_text_stage1_onehot_micro.txt"
    hlo_path.write_text(hlo_text)

    events = _load_trace_events(trace_dir)
    scope_map = scope_map_from_hlo_text(hlo_text, KNOWN_LABELS)
    measured = parse_scopes(events, scope_map)
    # Fill ALL known labels: absent from trace -> None, never 0.0.
    scopes = {label: measured.get(label) for label in sorted(KNOWN_LABELS)}
    counts = parse_dispatch_counts(events)

    # --- Untraced per-variant steady-state walls (paired, advisory) --------
    # NOTE (stated limitation): the materialized wall here feeds the DENSE
    # matrix device-side; it excludes host build + H2D transfer of that
    # matrix, which is precisely the cost a materialization pipeline would
    # add. Treat this ratio as graph-compute-only evidence.
    warm = max(args.warmup, 1)
    materialized_seconds = _timed_steady_state(materialized_fn, oh, warm, args.trials)
    onthefly_seconds = _timed_steady_state(onthefly_fn, idx, warm, args.trials)

    metrics: dict[str, float | int | str] = {
        "total_step_seconds": total_step_seconds,
        "materialized_seconds": materialized_seconds,
        "onthefly_seconds": onthefly_seconds,
        "parity_max_abs_diff": parity_max_abs_diff,
        **counts,
    }
    path = args.out_dir / "stage1_onehot_micro.json"
    record = emit_probe_record(
        path=path,
        probe_id="stage1_onehot_micro",
        stage=1,
        # Scale axis is the padded leading batch rows, per D8 convention.
        n_atoms=args.rows,
        platform="cpu",
        metrics=metrics,
        scopes=scopes,
        attribution_method={
            **attribution_from_scopes(scopes, method=ATTRIBUTION_NAMED_SCOPE),
        },
        config={
            "kernel": "onehot_matmul_sq_rowsum_x2",
            "variants": "materialized+onthefly",
            "n_rows": str(args.rows),
            "n_classes": str(args.classes),
            "n_cols": str(args.cols),
            "n_trials": str(args.trials),
            "axis_note": "n_atoms == padded leading batch rows",
            "parity_note": f"max abs diff vs tolerance {PARITY_TOLERANCE}, gated pre-trace",
        },
    )

    print(f"wrote {path}")
    print(f"HLO text -> {hlo_path.name} ({len(hlo_text)} chars)")
    for label, value in scopes.items():
        if value is None:
            print(f"  {label}: ABSENT from trace")
        else:
            seconds, n_occ = value
            pct = 100.0 * seconds / total_step_seconds
            print(f"  {label}: {seconds:.6f}s over {n_occ} occ (~{pct:.2f}% of wall)")
    print(f"dispatch: {counts}")
    print(
        f"untraced walls: materialized={materialized_seconds:.6f}s "
        f"onthefly={onthefly_seconds:.6f}s "
        f"(ratio={materialized_seconds / onthefly_seconds:.3f})"
    )
    permitted = sorted(permitted_claims(record), key=lambda c: c.name)
    print(f"permitted claims for this record: {[c.name for c in permitted]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 1 tiling probe: CPU micro under jax.profiler.trace + HLO attribution.

Phase B of .praxia/docs/specs/
260824_upstream-profiling-probe-tooling-from-prolix.md. One jitted program
applies all three tiling strategies (Vmap / SafeMap / DedupGather) to the
representative kernel, each under its own jax.named_scope. The compiled HLO
text carries the scope paths (op_name metadata); the executed trace carries
per-thunk durations under post-fusion hlo_op names -- xtrax.profiling.trace's
two-input attribution joins them. Emits ONE Stage-1 ProbeRecord whose scopes
dict fills ALL known labels (None = label expected but absent from the trace,
never 0.0), plus dispatch-count metrics.

Stage-1 records support STRUCTURAL and DISPATCH_COUNT claims only; a
TERM_RANKING over them fails closed by design (CPU ranking is structurally
invalid on GPU). D9 spike (scope doc): event shapes verified on jax 0.10.2.

Usage:
    XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage1_tiling_micro.py \
        [--out-dir outputs/profiling/stage1] [--trials 20]
"""

from __future__ import annotations

import argparse
import gzip
import json
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
from xtrax.tiling.dispatch import axis_dispatch, make_axis_dispatch
from xtrax.tiling.strategy import DedupGather, SafeMap, Vmap

ROOT = Path(__file__).resolve().parents[1]

# xtrax scope-label registry (D8: vocab lives in drivers, never in the
# library package). One label per tiling strategy.
LABEL_VMAP = "tiling_vmap"
LABEL_SAFEMAP = "tiling_safemap"
LABEL_DEDUP_GATHER = "tiling_dedup_gather"
KNOWN_LABELS = frozenset({LABEL_VMAP, LABEL_SAFEMAP, LABEL_DEDUP_GATHER})


def _core(xs):
    return (jnp.sin(xs) * 2.0).sum(axis=-1)


def _build_program(rows: int, cols: int, safemap_batch: int, dedup_k_bucket: int):
    xs = jnp.ones((rows, cols))
    unique_indices = jnp.arange(dedup_k_bucket, dtype=jnp.int32)
    index_map = jnp.arange(rows, dtype=jnp.int32) % dedup_k_bucket
    dedup = DedupGather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=dedup_k_bucket,
        k_bucket=dedup_k_bucket,
        dedup_fn=lambda x, i: x[i],
        gather_fn=lambda y, i: y[i],
    )
    vmap_iter = make_axis_dispatch(Vmap())
    safemap_iter = make_axis_dispatch(SafeMap(batch_size=safemap_batch))

    def program(xv, xsd):
        with jax.named_scope(LABEL_VMAP):
            a = vmap_iter(_core, xv).sum()
        with jax.named_scope(LABEL_SAFEMAP):
            b = safemap_iter(_core, xsd).sum()
        with jax.named_scope(LABEL_DEDUP_GATHER):
            c = axis_dispatch(dedup, _core, xv).sum()
        return a + b + c

    return program, xs


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "stage1",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--safemap-batch", type=int, default=8)
    parser.add_argument("--dedup-k-bucket", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args(argv)

    program, xs = _build_program(args.rows, args.cols, args.safemap_batch, args.dedup_k_bucket)
    # Compile + warm up OUTSIDE the timed/trace region (prolix stage-1 rule:
    # n_compilations inside the window must reflect steady-state dispatch).
    compiled = jax.jit(program).lower(xs, xs).compile()
    for _ in range(max(args.warmup, 1)):
        compiled(xs, xs).block_until_ready()

    trace_dir = args.out_dir / "_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for _ in range(args.trials):
            compiled(xs, xs).block_until_ready()
    total_step_seconds = time.perf_counter() - start

    hlo_text = compiled.as_text()
    hlo_path = args.out_dir / "hlo_as_text_stage1_tiling_micro.txt"
    hlo_path.write_text(hlo_text)

    events = _load_trace_events(trace_dir)
    scope_map = scope_map_from_hlo_text(hlo_text, KNOWN_LABELS)
    measured = parse_scopes(events, scope_map)
    # Fill ALL known labels: absent from trace -> None, never 0.0.
    scopes = {label: measured.get(label) for label in sorted(KNOWN_LABELS)}
    counts = parse_dispatch_counts(events)

    metrics: dict[str, float | int | str] = {
        "total_step_seconds": total_step_seconds,
        **counts,
    }
    path = args.out_dir / "stage1_tiling_micro.json"
    record = emit_probe_record(
        path=path,
        probe_id="stage1_tiling_micro",
        stage=1,
        n_atoms=args.rows,
        platform="cpu",
        metrics=metrics,
        scopes=scopes,
        attribution_method={
            **attribution_from_scopes(scopes, method=ATTRIBUTION_NAMED_SCOPE),
            # Labels recovered only via op_name metadata would carry the
            # weaker guarantee; this driver attributes everything through
            # named_scope paths in the HLO text or leaves them None.
        },
        config={
            "kernel": "sin_mul_rowsum_x3",
            "strategies": "vmap+safemap+dedup_gather",
            "n_rows": str(args.rows),
            "n_cols": str(args.cols),
            "safemap_batch": str(args.safemap_batch),
            "dedup_k_bucket": str(args.dedup_k_bucket),
            "n_trials": str(args.trials),
            "axis_note": "n_atoms == padded leading batch rows",
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
    permitted = sorted(permitted_claims(record), key=lambda c: c.name)
    print(f"permitted claims for this record: {[c.name for c in permitted]}")

    # Fail-closed demo: discovery + TERM_RANKING over Stage<=1 records MUST
    # raise (CPU ranking is structurally invalid on GPU). This proves the
    # claim gate is alive end-to-end on xtrax-native records.
    from xtrax.profiling.claims import (
        ClaimClass,
        ClaimValidityError,
        assert_claim_supported,
    )
    from xtrax.profiling.report import discover_records

    discovered = discover_records(root=ROOT)
    print(f"discovered {len(discovered)} record(s) under outputs/profiling/")
    try:
        assert_claim_supported(discovered, ClaimClass.TERM_RANKING)
    except ClaimValidityError as exc:
        print("claim gate alive (expected raise):")
        print(f"  {exc}")
    else:
        raise SystemExit(
            "TERM_RANKING over stage0/1 records did NOT raise -- claim gate regression"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

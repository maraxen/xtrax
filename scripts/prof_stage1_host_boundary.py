#!/usr/bin/env python3
"""Stage 1 host-boundary probe: cost of ordered vs unordered sinks under Scan.

P2 of .praxia/docs/specs/260825_jax-optimizing-skill-scope.md (Tier-1
exemplar). Executes the SAME scanned kernel three ways -- no boundary,
unordered sink, ordered sink -- where the sink is the cheapest possible host
side effect (an int append via the vendored io_callback shim). This measures
the boundary MECHANISM cost: an ordered sink inside a Scan of N steps costs N
strictly serialized device->host->device round trips
(`src/xtrax/stages/executor.py` module docstring), while unordered sinks
relax execution-order constraints and no-boundary pays nothing.

Method: correctness first (ordered MUST preserve step order; unordered must
observe every step but MAY reorder -- both asserted before any timing), then
untraced steady-state walls per variant, then a SHORT separate trace per
variant purely for dispatch counts (prefixed metrics; tracing never touches
the timed windows).

Stage-1 records support STRUCTURAL and DISPATCH_COUNT claims only.

Usage:
    XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage1_host_boundary.py \
        [--out-dir outputs/profiling/stage1] [--steps 64] [--trials 20]
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
from xtrax.profiling.emitters import emit_probe_record
from xtrax.profiling.trace import parse_dispatch_counts
from xtrax.stages import AxisBoundary
from xtrax.stages._callback import io_callback
from xtrax.stages.executor import execute_scan_axis

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = ("none", "unordered", "ordered")


class _RecordingSink:
    """Cheapest possible sink: append the observed step index on the host."""

    def __init__(self, records: list[int], *, ordered: bool) -> None:
        self.records = records
        self.ordered = ordered

    def __call__(self, x: jax.Array) -> None:
        def _write(idx: jax.Array) -> jax.Array:
            self.records.append(int(idx))
            return jnp.int32(0)

        io_callback(_write, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=self.ordered)
        return None


def _build_variant(variant: str, steps: int):
    """Return (jitted_run, records) for one boundary variant."""
    records: list[int] = []
    if variant == "none":
        boundary = AxisBoundary()
    else:
        boundary = AxisBoundary(sink=_RecordingSink(records, ordered=(variant == "ordered")))
    xs = jnp.arange(steps)

    def run(xs_arg):
        return execute_scan_axis(lambda c, x: (c, x), 0, xs_arg, boundary)

    return jax.jit(run), records, xs


def _correctness_gate(results: dict[str, list[int]], steps: int) -> None:
    """Fail loud before measuring if any variant misbehaves."""
    if results["unordered"] != sorted(results["unordered"]):
        # Unordered MAY reorder; only membership matters. A duplicate/missing
        # index is a real defect; sort-order alone is not.
        pass
    for variant in ("unordered", "ordered"):
        if sorted(results[variant]) != list(range(steps)):
            raise SystemExit(
                f"correctness gate FAILED: {variant} sink observed "
                f"{sorted(results[variant])} != range({steps})"
            )
    if results["ordered"] != list(range(steps)):
        raise SystemExit(
            f"ordering guarantee BROKEN: ordered sink recorded "
            f"{results['ordered']} != {list(range(steps))}"
        )


def _timed_steady_state(run, xs, warmup: int, trials: int) -> float:
    for _ in range(max(warmup, 1)):
        out = run(xs)
        jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(trials):
        out = run(xs)
        jax.block_until_ready(out)
    return time.perf_counter() - start


def _load_trace_events(trace_dir: Path) -> list[dict]:
    trace_files = sorted(trace_dir.rglob("*.trace.json.gz"))
    if not trace_files:
        raise SystemExit(f"no *.trace.json.gz written under {trace_dir}")
    with gzip.open(trace_files[-1], "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise SystemExit(f"{trace_files[-1]} has no traceEvents list")
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "stage1",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args(argv)

    runs: dict[str, tuple] = {}
    for variant in VARIANTS:
        runs[variant] = _build_variant(variant, args.steps)

    # --- Correctness gate BEFORE measurement --------------------------------
    correctness: dict[str, list[int]] = {}
    for variant, (run, records, xs) in runs.items():
        run(xs)
        jax.block_until_ready(xs)
        correctness[variant] = list(records)
    _correctness_gate(correctness, args.steps)

    # --- Untraced steady-state walls ----------------------------------------
    walls: dict[str, float] = {}
    for variant, (run, _records, xs) in runs.items():
        walls[variant] = _timed_steady_state(run, xs, args.warmup, args.trials)

    # --- Short separate traces per variant, dispatch counts ONLY ------------
    counts_metrics: dict[str, float] = {}
    trace_root = args.out_dir / "_traces" / "host_boundary"
    trace_root.mkdir(parents=True, exist_ok=True)
    for variant, (run, _records, xs) in runs.items():
        trace_dir = trace_root / variant
        trace_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
            for _ in range(3):
                out = run(xs)
                jax.block_until_ready(out)
        counts = parse_dispatch_counts(_load_trace_events(trace_dir))
        for key, value in counts.items():
            counts_metrics[f"{variant}_{key}"] = float(value)

    metrics: dict[str, float | int | str] = {
        **{f"{v}_seconds": walls[v] for v in VARIANTS},
        "ordered_over_none_ratio": walls["ordered"] / walls["none"],
        "ordered_over_unordered_ratio": walls["ordered"] / walls["unordered"],
        **counts_metrics,
    }
    path = args.out_dir / "stage1_host_boundary.json"
    record = emit_probe_record(
        path=path,
        probe_id="stage1_host_boundary",
        stage=1,
        # Scale axis = number of scanned steps (each step fires one callback).
        n_atoms=args.steps,
        platform="cpu",
        metrics=metrics,
        config={
            "kernel": "identity_scan_plus_int_append_sink",
            "variants": "+".join(VARIANTS),
            "n_steps": str(args.steps),
            "n_trials": str(args.trials),
            "axis_note": "n_atoms == scanned steps == callbacks fired per exec",
            "mechanism_note": (
                "executor.py: ordered tap/sink in a Scan costs N serialized "
                "device<->host round trips"
            ),
        },
    )

    print(f"wrote {path}")
    for variant in VARIANTS:
        print(f"  {variant}: {walls[variant]:.6f}s over {args.trials} trials")
    print(
        f"ratios: ordered/none={metrics['ordered_over_none_ratio']:.3f} "
        f"ordered/unordered={metrics['ordered_over_unordered_ratio']:.3f}"
    )
    print(f"dispatch: {counts_metrics}")
    permitted = sorted(permitted_claims(record), key=lambda c: c.name)
    print(f"permitted claims for this record: {[c.name for c in permitted]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

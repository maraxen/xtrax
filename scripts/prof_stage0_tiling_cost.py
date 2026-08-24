#!/usr/bin/env python3
"""Stage 0 tiling probe: XLA cost analysis over tiling strategies. Never executes.

Phase B of .praxia/docs/specs/
260824_upstream-profiling-probe-tooling-from-prolix.md. Emits one Stage-0
ProbeRecord per tiling strategy (Vmap / SafeMap / DedupGather) applied to a
representative batched kernel, via jax.jit(fn).lower().compile().cost_analysis()
-- the same never-execute pattern as prolix's prof_stage0_cost_analysis.py.

Stage 0 records support STRUCTURAL claims only (see xtrax.profiling.claims);
they deliberately cannot back a term ranking -- that needs an executed,
GPU-measured Stage-2 trace.

Usage:
    XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage0_tiling_cost.py \
        [--out-dir outputs/profiling/stage0] [--rows 256] [--cols 32]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from xtrax.profiling.emitters import emit_probe_record
from xtrax.tiling.dispatch import axis_dispatch, make_axis_dispatch
from xtrax.tiling.strategy import DedupGather, SafeMap, Vmap

ROOT = Path(__file__).resolve().parents[1]


def _core(xs):
    """Representative batched kernel: elementwise nonlinearity + row reduce."""
    return (jnp.sin(xs) * 2.0).sum(axis=-1)


def _build_programs(rows: int, cols: int, safemap_batch: int, dedup_k_bucket: int):
    """One zero-arg-closure jittable program per strategy over the kernel."""
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

    def via_iterator(dispatcher):
        return lambda x: dispatcher(_core, x)

    return {
        "vmap": via_iterator(make_axis_dispatch(Vmap())),
        "safemap": via_iterator(
            make_axis_dispatch(SafeMap(batch_size=safemap_batch))
        ),
        "dedup_gather": lambda x: axis_dispatch(dedup, _core, x),
    }, xs


def _numeric_cost_metrics(analysis: dict) -> dict[str, float]:
    """Normalize cost_analysis entries to float-valued metrics; drop the rest."""
    metrics: dict[str, float] = {}
    for raw_key, value in analysis.items():
        key = str(raw_key).strip().lower().replace(" ", "_")
        if isinstance(value, bool):
            continue
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(coerced):
            metrics[key] = coerced
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "stage0",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--safemap-batch", type=int, default=8)
    parser.add_argument("--dedup-k-bucket", type=int, default=8)
    args = parser.parse_args(argv)

    programs, xs = _build_programs(
        args.rows, args.cols, args.safemap_batch, args.dedup_k_bucket
    )
    written: list[Path] = []
    for strategy, program in programs.items():
        # Cost analysis on the LOWERED program only -- never executed here.
        lowered = jax.jit(program).lower(xs)
        analysis = lowered.compile().cost_analysis()
        assert isinstance(analysis, dict), type(analysis)
        metrics = _numeric_cost_metrics(analysis)
        if not metrics:
            raise SystemExit(
                f"cost_analysis yielded no numeric fields for {strategy}: "
                f"{sorted(map(str, analysis))}"
            )

        path = args.out_dir / f"stage0_tiling_{strategy}.json"
        record = emit_probe_record(
            path=path,
            probe_id=f"stage0_tiling_{strategy}",
            stage=0,
            # Scale axis is the padded leading batch (structural rows), per
            # D8: field name stays n_atoms; semantics documented in config.
            n_atoms=args.rows,
            platform="cpu",
            metrics=metrics,
            config={
                "strategy": strategy,
                "kernel": "sin_mul_rowsum",
                "n_rows": str(args.rows),
                "n_cols": str(args.cols),
                "safemap_batch": str(args.safemap_batch),
                "dedup_k_bucket": str(args.dedup_k_bucket),
                "axis_note": "n_atoms == padded leading batch rows",
            },
        )
        written.append(path)
        print(
            f"wrote {path.name}: {sorted(record.metrics)} "
            f"(contract {record.contract_version}, git {record.git_sha[:12]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

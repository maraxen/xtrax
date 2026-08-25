#!/usr/bin/env python3
"""Stage 0 one-hot probe: XLA cost analysis over categorical encoding variants. Never executes.

P1 of .praxia/docs/specs/260825_jax-optimizing-skill-scope.md (Tier-3 exemplar).
Compares materialized vs on-the-fly categorical encoding over a representative
gather-classify kernel:

  - "materialized": host builds a dense f32[N,K] one-hot matrix; the jitted
    program multiplies it through.
  - "onthefly": host passes int32 class indices [N]; the program computes the
    one-hot inside jit via jax.nn.one_hot and multiplies through.

Both programs are lowered+compiled for cost_analysis() only -- never executed
here, the same never-execute pattern as prof_stage0_tiling_cost.py.

Stage 0 records support STRUCTURAL claims only; they cannot back a term
ranking (that needs an executed, GPU-measured Stage-2 trace).

Usage:
    XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage0_onehot_cost.py \
        [--out-dir outputs/profiling/stage0] [--rows 256] [--classes 32] [--cols 16]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from xtrax.profiling.emitters import emit_probe_record

ROOT = Path(__file__).resolve().parents[1]


def _kernel(one_hot_f32: jax.Array, w: jax.Array) -> jax.Array:
    """Representative gather-classify kernel: one-hot @ weights + row reduce."""
    return ((one_hot_f32 @ w) ** 2).sum(axis=-1)


def _build_programs(rows: int, classes: int, cols: int):
    """One jittable program per encoding variant over the shared kernel."""
    w = jnp.ones((classes, cols))
    # Materialized variant's input IS the dense one-hot; the on-the-fly
    # variant takes compact int32 indices and encodes in-graph.
    oh = jnp.ones((rows, classes))
    idx = jnp.zeros((rows,), dtype=jnp.int32)

    def materialized(oh_arg):
        return _kernel(oh_arg, w)

    def onthefly(idx_arg):
        return _kernel(jax.nn.one_hot(idx_arg, classes), w)

    return {"materialized": materialized, "onthefly": onthefly}, oh, idx


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
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--cols", type=int, default=16)
    args = parser.parse_args(argv)

    programs, oh, idx = _build_programs(args.rows, args.classes, args.cols)
    inputs = {"materialized": oh, "onthefly": idx}
    written: list[Path] = []
    for variant, program in programs.items():
        # Cost analysis on the LOWERED program only -- never executed here.
        lowered = jax.jit(program).lower(inputs[variant])
        analysis = lowered.compile().cost_analysis()
        assert isinstance(analysis, dict), type(analysis)
        metrics = _numeric_cost_metrics(analysis)
        if not metrics:
            raise SystemExit(
                f"cost_analysis yielded no numeric fields for {variant}: "
                f"{sorted(map(str, analysis))}"
            )

        path = args.out_dir / f"stage0_onehot_{variant}.json"
        record = emit_probe_record(
            path=path,
            probe_id=f"stage0_onehot_{variant}",
            stage=0,
            # Scale axis is the padded leading batch rows, per D8 convention:
            # field name stays n_atoms; semantics documented in config.
            n_atoms=args.rows,
            platform="cpu",
            metrics=metrics,
            config={
                "variant": variant,
                "kernel": "onehot_matmul_sq_rowsum",
                "encoding": (
                    "dense_f32_host_built"
                    if variant == "materialized"
                    else "int32_indices_in_graph"
                ),
                "n_rows": str(args.rows),
                "n_classes": str(args.classes),
                "n_cols": str(args.cols),
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

#!/usr/bin/env python3
"""Stage 1 feed-overlap probe: sequential vs async-prefetched batch feeding.

P2 of .praxia/docs/specs/260825_jax-optimizing-skill-scope.md (Tier-2
exemplar). Quantifies whether host-side prefetch actually hides data-feed
latency behind device compute, or merely shifts it. The pipeline: an
artificially slowed blocking iterator (configurable per-batch sleep) feeds a
jitted step; the comparison is

  sequential : next(iter) -> run(batch), strictly alternating, no overlap
  overlapped : xtrax.engine.io.async_indexed_stream(iter, buffer_size=B)
               consumed as (index, batch) pairs driving the same steps

Both arms consume the SAME batches in the SAME order through the SAME jitted
step; the only difference is the feed discipline. Walls are untraced
steady-state (tracing would distort exactly the overlap being measured).

Honesty guard: if the jitted step is so fast that host-side loop overhead
dominates, the measured "benefit" is not transfer hiding -- the record carries
step_seconds so a reader can check the regime before citing anything.

Stage-1 records support STRUCTURAL and DISPATCH_COUNT claims only.

Usage:
    XTRAX_GIT_SHA=$(git rev-parse HEAD) uv run python scripts/prof_stage1_feed_overlap.py \
        [--out-dir outputs/profiling/stage1] [--batches 32] [--rows 256] \
        [--feed-sleep-ms 1.0] [--buffer-size 4] [--trials 3]
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from xtrax.engine.io import async_indexed_stream
from xtrax.profiling.claims import permitted_claims
from xtrax.profiling.emitters import emit_probe_record

ROOT = Path(__file__).resolve().parents[1]


def _slow_iterable(n_batches: int, rows: int, cols: int, sleep_s: float):
    """Deterministic batches from a deliberately slow blocking producer."""
    for i in range(n_batches):
        time.sleep(sleep_s)
        yield jnp.full((rows, cols), float(i))


def _make_step(rows: int, cols: int):
    """Representative jitted step with enough work to be worth overlapping."""
    w = jax.random.normal(jax.random.key(0), (cols, cols))

    def step(x):
        y = x
        for _ in range(20):
            y = jnp.tanh(y @ w) + y
        return y.sum()

    compiled = jax.jit(step).lower(jnp.ones((rows, cols))).compile()
    return compiled


def _run_sequential(compiled, batches, trials: int) -> float:
    best = float("inf")
    for _ in range(trials):
        start = time.perf_counter()
        for batch in batches:
            out = compiled(batch)
            jax.block_until_ready(out)
        best = min(best, time.perf_counter() - start)
    return best


async def _run_overlapped(compiled, batches, buffer_size: int, trials: int) -> float:
    best = float("inf")
    for _ in range(trials):
        start = time.perf_counter()
        async for _index, batch in async_indexed_stream(
            iter(list(batches)),  # materialize once; producer sleep dominates cost
            buffer_size=buffer_size,
        ):
            out = compiled(batch)
            await asyncio.to_thread(jax.block_until_ready, out)
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "stage1",
    )
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--feed-sleep-ms", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args(argv)

    # --- Compile + warm up OUTSIDE every timed region -----------------------
    compiled = _make_step(args.rows, args.cols)
    warm_batch = jnp.ones((args.rows, args.cols))
    for _ in range(max(args.warmup_steps, 1)):
        jax.block_until_ready(compiled(warm_batch))

    # --- Isolated step wall (the thing prefetch must hide under) ------------
    step_seconds = None
    big_batch = jnp.ones((args.rows, args.cols))
    for _ in range(5):
        s = time.perf_counter()
        jax.block_until_ready(compiled(big_batch))
        step_seconds = min(step_seconds or float("inf"), time.perf_counter() - s)

    # --- Arms ---------------------------------------------------------------
    def make_batches():
        return list(_slow_iterable(args.batches, args.rows, args.cols, args.feed_sleep_ms / 1000))

    seq_seconds = _run_sequential(compiled, make_batches(), args.trials)
    ovl_seconds = asyncio.run(
        _run_overlapped(compiled, make_batches(), args.buffer_size, args.trials)
    )

    metrics: dict[str, float | int | str] = {
        "sequential_seconds": seq_seconds,
        "overlapped_seconds": ovl_seconds,
        "speedup_ratio": seq_seconds / ovl_seconds,
        "isolated_step_seconds": step_seconds,
        "feed_seconds_lower_bound": args.batches * args.feed_sleep_ms / 1000,
    }
    path = args.out_dir / "stage1_feed_overlap.json"
    record = emit_probe_record(
        path=path,
        probe_id="stage1_feed_overlap",
        stage=1,
        # Scale axis = number of fed batches in one pass.
        n_atoms=args.batches,
        platform="cpu",
        metrics=metrics,
        config={
            "pipeline": "slow_producer_to_jitted_step",
            "n_batches": str(args.batches),
            "n_rows": str(args.rows),
            "n_cols": str(args.cols),
            "feed_sleep_ms": str(args.feed_sleep_ms),
            "buffer_size": str(args.buffer_size),
            "n_trials_best_of": str(args.trials),
            "axis_note": "n_atoms == batches fed per measured pass",
            "regime_note": (
                "check isolated_step_seconds vs feed cost before citing: "
                "host-loop-dominated regimes do not evidence transfer hiding"
            ),
        },
    )

    print(f"wrote {path}")
    print(f"  isolated step: {step_seconds * 1000:.3f} ms")
    print(f"  sequential: {seq_seconds:.4f}s  overlapped: {ovl_seconds:.4f}s")
    print(f"  speedup ratio: {metrics['speedup_ratio']:.3f}")
    permitted = sorted(permitted_claims(record), key=lambda c: c.name)
    print(f"permitted claims for this record: {[c.name for c in permitted]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

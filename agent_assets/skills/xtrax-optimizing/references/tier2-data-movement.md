# Tier 2: Data Movement

Change surface: when/how arrays reach the device -- still not the math.
Measured exemplar: `scripts/prof_stage1_feed_overlap.py` (slow blocking
producer -> jitted step; sequential vs `async_indexed_stream` overlapped).

## First measured results (CPU, jax 0.10.2, 32 batches, 256x64, 1 ms producer sleep, buffer=4)

- isolated step: 0.84 ms
- sequential: 53.5 ms/pass (~1.67 ms/batch)
- overlapped: 76.2 ms/pass -- speedup ratio **0.70x (SLOWER)**

Reading: on this regime the async machinery (asyncio.to_thread hop per item +
queue operations) costs MORE than it hides. Prefetch is not free; its value
case requires hidden latency > machinery overhead. This negative result is
the point of the probe: it converts "prefetching is good practice" into a
measured decision per regime. Re-run with a genuinely slow feed (real H2D
under GPU jaxlib, network storage, decode-heavy producers) before concluding
anything about those regimes -- the driver takes `--feed-sleep-ms`,
`--buffer-size`, and shape flags precisely for that sweep.

## Regime check before citing anything

The record carries `isolated_step_seconds` and `feed_seconds_lower_bound`
(batches * per-batch feed cost). A speedup number is only meaningful when:

1. feed cost is a real fraction of the sequential pass (otherwise nothing to
   hide), AND
2. step work is large enough that host loop overhead is negligible (see the
   config `regime_note`; if steps are sub-millisecond you are measuring the
   async machinery, not transfer hiding).

## Tools and knobs (verify-paths)

- `xtrax.engine.io.async_indexed_stream(iterable, buffer_size=N)`:
  thread-pool producer, bounded queue, exceptions re-raised at consumption
  (`src/xtrax/engine/io.py`). buffer_size bounds memory AND lookahead.
- Shape bucketing to bound jit recompiles for variable-length inputs:
  `src/xtrax/tiling/bucket.py` (`select_bucket` / `bucketize`). Prefer over
  unbounded re-tracing whenever shapes vary.
- dtype coercion belongs at load, not in-graph, WHEN the graph would
  otherwise widen then shrink repeatedly -- but verify with a stage-0 cost
  probe first; XLA often fuses naive casts away.

## Evidence protocol

Untraced steady-state walls ONLY (tracing distorts exactly the overlap being
measured), best-of-N trials to cut scheduler jitter, identical batches and
step executable across arms, compile+warmup outside timed regions. The
stage-1 driver implements all four; sweep knobs through its flags rather
than editing kernels between measurements.

---
title: 'XLA hoisting of invariant work: CPU vs CUDA, vmap vs chunked lax.map'
description: 'Tracked bathos probe: XLA hoists truly invariant encoder work out of vmap on CPU and CUDA, and out of chunked lax.map on CPU, but not on CUDA (4.5x); false-dependence control 15-71x'
status: draft
task_id: 261020_hoist-consumer-migrations
date: '261020'
confidence: 'measured, single run per platform'
sources: 'bathos runs 53c35140 (CPU), c2849876 (CUDA); sidecar 1d72251; script f06ccda'
---
# XLA hoisting of invariant work: CPU vs CUDA, vmap vs chunked lax.map


## Question

Does XLA (jax 0.11.1) remove truly invariant work from a mapped function, so that
`make_hoisted` only pays off for false dependence? The 261020 throwaway probe said yes on
CPU; GPU was unverified (spec `.praxia/docs/specs/261020_hoisted-map.md`).

## Method

Pre-registered: `scripts/experiments/hoist_invariant_xla_probe.bth.toml` (commit 1d72251,
before the script existed); script `hoist_invariant_xla_probe.py` (f06ccda). N=64 elements,
encoder `(L,3)->(L,L,C)` with L=256, C=64 and 32 chained sin/cos passes, per-element work
one einsum. For each path (`jax.vmap`, xtrax `chunked_map(batch_size=4)`) a naive arm and a
hand-hoisted twin; median wall over 20 calls after 3 warm-ups. Control: the encoder made
falsely dependent on a per-element key (noise scale 0.0 passed as an array) must be >= 4x
slower than its hoisted twin, or the run concludes nothing.

## Results

| run | platform | inv vmap | inv chunked | fd vmap | fd chunked | outcome (pre-registered conditions) |
|---|---|---|---|---|---|---|
| `53c35140` | CPU (this box) | 1.00x | 1.03x | 70.8x | 51.8x | `hoisted` |
| `c2849876` | CUDA, NVIDIA TITAN RTX (titanix) | 1.00x | 4.51x | 47.7x | 15.6x | `not_hoisted` |

Both runs: `n_failed = 0`, git clean at f06ccda, sidecar sha256 `cd189ed2...` bound.
bathos records exploration-stage runs as `observed`; the outcome column applies the
committed `[outcomes]` conditions to the recorded payload. Result payloads:
`data/261020_hoist_invariant_cpu.json`, `data/261020_hoist_invariant_cuda.json`.

Descriptive (not gated): on CUDA the chunked naive arm costs 15.3 ms vs 3.4 ms hoisted;
one false-dependent encode costs about (53.0 - 3.4) / 64 = 0.78 ms, so the 11.9 ms gap
is about 15 encodes, close to one per `lax.map` iteration (16 chunks). XLA hoists out of
the vmap within a chunk but not out of the loop.

## Consequences

- On GPU, set `batch_size` only with a hand hoist or `make_hoisted`; invariant work is
  otherwise recomputed per chunk.
- False dependence is replicated on every platform and path (15-71x here).
- Not measured: IREE/StableHLO, other GPUs, other jax versions, repeat runs (one run per
  platform; ratios far from the 1.25 / 2.0 bands).

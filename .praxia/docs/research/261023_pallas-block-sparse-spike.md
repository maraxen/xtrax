---
title: Pallas block-sparse matmul feasibility spike
description: 'Interpret-mode Pallas kernel: parity, custom_vjp grads, jit/vmap, dataclass config, export; Mosaic GPU export blocked on jax 0.11.1'
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261023'
confidence: ''
sources: ''
---
# Pallas block-sparse matmul feasibility spike

Exploratory feasibility spike for Part 2 of the equinox/Pallas plan (spec
`specs/261023_equinox-decoupling.md`). **Not pre-registered** (no bathos sidecar): these are
pass/fail feasibility checks to steer a design, not findings to cite. No timing numbers (CPU
interpret timings mean nothing). Scripts: `scripts/experiments/pallas_block_sparse_spike_261023/`
(`common.py`, `q1`–`q5*.py`), run with `OMP_NUM_THREADS=4 uv run python <script>` from that directory. Environment: jax 0.11.1,
equinox 0.13.8, CPU only (WSL2). Kernel: y = x @ W with W as block mask + dense blocks,
M,K,N = 128,256,256, 32-blocks, ~50% density, masked blocks skipped with `pl.when`.

## Results (written by a Haiku agent; re-run and reproduced by the orchestrator 261023)

| Q | Result | Detail |
|---|---|---|
| q1 parity | PASS | matches `x @ (W*mask)` at atol=rtol=1e-5; a **control** that corrupts an unmasked block turns the check red |
| q2 grads | PASS | `custom_vjp` with a dense-rule backward matches the reference; raw `pallas_call` autodiff raises `NotImplementedError`, so every kernel needs `custom_vjp` |
| q3 transforms | PASS | `jit` and `vmap` match; 1 trace across 3 same-shape calls |
| q4 config pytree | PASS | `register_dataclass` config with static block sizes: a value change does not retrace, under `jax.jit` and inside an `eqx.Module` via `eqx.filter_jit`. Tolerance was scale-relative (1e-5·max(1,max\|ref\|)); elementwise 1e-5 fails at \|ref\|≈150 from f32 accumulation order |
| q5 export | PARTIAL | interpret-mode kernel exports to StableHLO with no custom_call, for both `cpu` and `cuda` platforms, and round-trips on CPU. **Non-interpret CUDA export fails** on jax 0.11.1 Mosaic GPU: `Scalar prefetch not supported in Mosaic GPU lowering` (mask via scalar prefetch) and `Unimplemented primitive … dot_general for lowering semantics Warpgroup` (mask as blocked input; 7 tile/dtype variants) |

Gotcha: with a `None`-squeezed `BlockSpec` dimension the kernel ref is 2-D, so `w_ref[0, 0]`
silently indexes the wrong element. q1 was red until fixed.

Pallas API as read in the installed source: `pallas_call` (`jax/_src/pallas/pallas_call.py:1136`),
`BlockSpec` (`core.py:558`), `GridSpec` (`core.py:1217`), `PrefetchScalarGridSpec`
(`mosaic/core.py:312`, exported from `jax.experimental.pallas.tpu`), `pl.when` (`helpers.py:68`),
`pl.program_id` (`primitives.py:62`).

## What this means for Part 2

- The design holds on CPU: reference + Pallas variant behind one op, `custom_vjp` mandatory,
  dataclass config with static block sizes, parity checked under `interpret=True`.
- Export stays on the reference path; interpret-mode export works but lowers to a loop, not a kernel.
- **Open:** a real GPU kernel. Not tried: Pallas's Triton backend and a Mosaic-GPU-dialect rewrite
  (warpgroup matmul). The next spike runs on titanix's GPU and is the first place a timing could be
  cited, so it gets a bathos sidecar before it runs.


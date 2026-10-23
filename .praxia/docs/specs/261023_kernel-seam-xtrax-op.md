---
title: 'Kernel seam (xtrax.op): reference + kernel behind one op, first target demistify pairwise histogram'
description: 'Part 2 of the equinox-decouple-pallas epic: xtrax.op(reference, kernels) as a custom_jvp whose primal is a backend kernel and whose derivative is the reference''s JVP; consumers write reference+kernel pairs now; first GPU target is demistify''s pairwise joint histogram'
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261023'
backlog_ids: ''
adversarial_review: ''
---
# Kernel seam (xtrax.op): reference + kernel behind one op, first target demistify pairwise histogram

Part 2 of the epic. Part 1a is `.praxia/docs/superpowers/specs/261023_intent-resolution-design.md`
(the intent layer: `loop`, `map`, `compile`), and Part 1b is `261023_equinox-decoupling.md`. Spike
evidence: `research/261023_pallas-block-sparse-spike.md` (q1–q5), plus probe q6 below. User
decisions, 261023: "xtrax owns the seam" (kernels live with the models or in a sibling package);
first GPU target is demistify's pairwise histogram; consumer work is filed as backlog now, with no
consumer code changes until it is picked up.

## 1. Intent

A model author writes the computation once, in plain JAX (the *reference*), and may add one or more
*kernels* with the same signature for particular backends. Callers get one function. It runs the
kernel where the kernel exists, applies, and is allowed, and otherwise runs the reference.
Derivatives, `vmap` and export behave as they do for the reference. Kernels never need a
hand-written backward pass.

Why consumers care (recon 261023, cited lines read at those paths):

- demistify's pairwise mutual information runs through **denxity's** `graph/mi_kernel.py`.
  `fused_mi_matrix` (`:172`) chunks `_pair_mi` (gather → composite `bincount` → entropy) over the
  pairs axis with `chunked_e_step`. demistify calls it on every path (`mist.py:658,748,768`). The
  older `_compute_all_pairwise_mi_vmapped` (`mist.py:2229`, flat `vmap(bincount)`) has no live
  caller. `mist.py:1598-1636` documents why it was retired: above about 2.45e9 elements, flat
  `vmap(bincount)` returns **all zeros, silently** (jax 0.10.2).
- denxity `distances/metrics.py`, `distances/fused_bhattacharyya.py`: pairwise distance matrices.
- aminx `host/logit_aggregation.py:244` (pairwise JSD), `families/protonpotts_mpnn/ph_potentials.py:88,168`
  (scatter-add and edge sums). aminx's large runs are CPU-only today
  (`scripts/analysis/laser_sample_chunk_width_cost.bth.toml:7`), so kernels pay off only after its
  GPU move.

## 2. Contract

```python
xtrax.op(reference, *, kernels: Mapping[str, Callable] = {}, name: str | None = None) -> Callable
```

- `kernels` maps a backend name (`"gpu"`, `"tpu"`, `"cpu"`) to a function with the reference's
  signature and output pytree structure, shapes and dtypes.
- The returned function is a `jax.custom_jvp`:
  - **primal**: the kernel for the resolved backend when one is selected (rules below), otherwise
    the reference;
  - **JVP**: `jax.jvp(reference, primals, tangents)`, with `float0` tangents for integer and bool
    inputs. Reverse mode comes from JAX transposing the reference's linear JVP, so forward mode,
    reverse mode, Hessians and `vmap` all work, and no kernel needs a `custom_vjp`.
  - Recorded tradeoff: under differentiation the reference's JVP does the tangent work at reference
    speed. Kernels accelerate the primal, the computation that dominates inference and analysis.
- **Backward-kernel opt-in (first cut).** `xtrax.op(reference, *, kernels, vjp_kernels=None)`, where
  `vjp_kernels` maps a backend to a `VjpKernel(fwd, bwd)`:
  - `fwd(*args) -> (out, residuals)`;
  - `bwd(residuals, cotangent) -> input cotangents`.

  When one is selected (same rules as `kernels`), the op is a `jax.custom_vjp` for that call.
  Training workloads need this. plegadx reports that on L40S fp32 a sublayer's forward+backward is
  about 3.5× its forward (6.3 ms vs 22.3 ms, bathos `482c77d6`). With per-block remat, a forward-only
  kernel reaches at most about 35-40% of sublayer time (plegadx LOOP-E1, 261023).
  - Cost: forward-mode (`jvp`, `jacfwd`, Hessians built on them) through a VJP-kernel op raises
    `IntentError` naming the op, the backend and "vjp kernel selected; forward mode needs the JVP
    form". It does not raise JAX's generic `custom_vjp` error. Backends with no VJP kernel keep the
    `custom_jvp` form, so the same model is fully differentiable on CPU and reverse-only on a
    kernel backend. `Plan.table()` reports which form each op site used.
  - Parity gate for a VJP kernel: the plegadx harness shape (`tests/loop_e1/kernel_parity.py`,
    commit `4871f7fc`) checks against `jax.vjp(reference)` with a **random** output cotangent (an
    all-ones cotangent from `sum(out)` hides sign and permutation errors) and per-argument relative
    L2. Three negative controls must each turn it red: a 0.1% scale error, a permuted weight, and a
    dropped weight gradient. The §4b trajectory gate applies on top.
  - Unverified alternative, to probe before implementing: keep `custom_jvp` and route the *tangent
    map* through a `custom_vjp` whose `bwd` is the backward kernel. Forward mode would then use the
    reference JVP and reverse mode the kernel. The cotangent with respect to the primals (second
    order) must come from reference autodiff, never zeros. If this probes clean on CPU interpret
    mode with controls, it replaces the opt-in's forward-mode error.
- **Selection rules**, in order:
  1. `Context.target == "export"` → reference. Spike q5: non-interpret Mosaic-GPU export fails on
     jax 0.11.1, and interpret-mode export lowers to a loop.
  2. `Context.kernels is False` → reference. This field already exists in the MVP `Context` and is
     inert until this spec lands.
  3. A kernel exists for `Context.resolved_backend()` → that kernel.
  4. Otherwise → reference.
- Inside `xtrax.compile`, an op is one more intent site, of kind `"op"`. Its `SiteDecision.primal`
  records `kernel:<backend>` or `reference` with a reason, so `Plan.table()` shows which path ran.
  Outside `compile`, an op applies the same rules to a default `Context`. That standalone form
  depends only on Task 1 (`Context`) and can ship before the resolver.
- An op whose outputs are all integer or bool (histograms, counts) has no derivative. The JVP
  returns `float0` output tangents. This keeps it usable inside differentiated programs that only
  read the counts.

### What consumers do now, before `xtrax.op` ships

Each kernel lands in the consumer repo as a **reference + kernel pair**:

1. `reference`: the existing JAX code, moved into a function unchanged.
2. `kernel`: the same signature, Pallas, written in the consumer.
3. Parity tests under `interpret=True` on CPU. They check exact equality for integer outputs, and
   the scale-relative tolerance `1e-5·max(1, max|ref|)` for floats (spike q4: an elementwise 1e-5
   fails at |ref|≈150 from f32 accumulation order). Each parity test has a control that corrupts
   one kernel block and must turn red (spike q1 pattern).
4. A local `_op` shim of about 15 lines that implements the `custom_jvp` pattern from probe q6,
   with backend selection.

When `xtrax.op` ships, the shim import is replaced, and nothing else changes.

## 3. First target: pairwise joint histogram + MI (denxity `fused_mi_matrix`, used by demistify)

**Reference.** denxity `graph/mi_kernel.py:_pair_mi`: for a pair `(i, j)`,
`counts = bincount(s_i * K_max + s_j, length=K_max**2)`, then
`MI = max(H_i + H_j − H(counts / sum), 0)`, with the entropy computed in **float64**. It is
chunked over pairs by `chunked_e_step` under an xtrax `MemoryBudget` (`plan_pairs_axis`). The kernel
fuses the histogram and the entropy reduction per pair tile and emits `(n_pairs,)` MI values. The
inputs are integer states, so the op has no derivative.

**Precision decision for the sidecar.** Counts are exact integers in every form. The entropy
reduction is float64 in the reference. demistify's `scripts/perf/x64_wall_ratio.bth.toml:10`
cites hardware FP64:FP32 throughput ratios of about 1:2 on H200 and about 1:64 on Blackwell (vendor
ratios, not a measurement; that sidecar measured CPU only). Each form is run with a
float64 entropy and with a float32 entropy, and float32 is accepted only if
`max|MI_f32 − MI_ref| ≤ 1e-6` nats on the production shape. That threshold is fixed in the sidecar
before the run.

**Candidates, all compared at production shapes in one pre-registered run:**

| | Form | Notes |
|---|---|---|
| A | current `fused_mi_matrix` (chunked per-pair `bincount` + entropy) | baseline (the retired flat `vmap(bincount)` is not a candidate) |
| B | chunked one-hot matmul in plain XLA: `einsum("pfk,pfl->pkl", onehot(s_i), onehot(s_j))`, chunked over pairs and frames with `xtrax` `ChunkedMap` | no kernel at all. A joint histogram of two discrete sequences is `onehot(s_i)ᵀ @ onehot(s_j)`, a dense matmul. The kernel has to beat this. |
| C | Pallas kernel, grid over (pair tile, frame tile), one-hot matmul per tile, accumulating into the `(K, K)` output block | the block-matmul pattern the spike proved (q1–q4); no atomics |
| D | Pallas Triton-backend kernel with per-block histogram and `pl.triton.atomic_add` | only if C is limited by small K |

**Pre-registration** (bathos sidecar committed before the run, on titanix's GPU):

- Shapes: demistify's production run, 423 residues (89,253 pairs) × 28,007 frames
  (`mist.py:730,1601`), plus the 422-edge tree subset, plus one shape above the 2.45e9-element
  boundary.
- K: the production `K_max`, read from the demistify run config before writing the sidecar.
- Outcomes:
  - (1) joint counts equal a host NumPy reference exactly on a sampled pair subset, for every form,
    and MI is within the precision threshold above;
  - (2) every count row sums to `n_frames`, the post-condition `mist.py:1627` already checks;
  - (3) wall time per form, with the median of 5 after warm-up and compile time reported separately;
  - (4) recompile count, which must be 0 at steady state.
- The result is a decision: adopt the fastest form that passes (1) and (2). If B is within 1.5× of
  the best kernel, ship B and no kernel (the threshold is fixed in the sidecar before the run).
- Preemption: each (form, shape) cell is its own process with its own timeout and writes its own
  stamped artifact (the global long-computation rule).

## 4. Assumptions

| Assumption | Status |
|---|---|
| A `custom_jvp` with a Pallas primal and the reference's `jax.jvp` gives correct `grad`, `jvp`, `jacfwd(jacrev)`, `vmap(grad)` and `jit(grad)`, and the forward jaxpr keeps `pallas_call` | **verified** by probe q6 (`scripts/experiments/pallas_block_sparse_spike_261023/q6_custom_jvp.py`, CPU, interpret mode, jax 0.11.1). Control: scaling the JVP by 2 turns all five derivative checks red. |
| Raw `pallas_call` has no autodiff | verified (spike q2; plegadx confirms on jax 0.10.2: "Linearization failed") |
| custom_jvp with the tangent map routed through a custom_vjp (kernel bwd) keeps forward mode and gives kernel-speed reverse mode | unverified; probe before implementing the VJP opt-in |
| Pallas kernel plus dataclass config: a value change does not retrace | verified (spike q4) |
| Non-interpret GPU export of a Pallas kernel works | **false** on jax 0.11.1 Mosaic GPU (spike q5), hence selection rule 1 |
| `pl.triton.atomic_add` and `pl.mosaic_gpu.atomic_add` exist in jax 0.11.1 | verified (import probe 261023) |
| A compiled (non-interpret) Pallas kernel runs and beats XLA on titanix's GPU | **unverified**; this is the §3 run |
| An integer-output `custom_jvp` with `float0` tangents works inside `grad` of a program that reads the counts | unverified; probe before implementing |
| demistify's production K_max | unverified; read the run config first (frame and pair counts come from `mist.py:730,1601` comments, not a run record) |
| denxity `fused_mi_matrix` is demistify's live MI path and `_compute_all_pairwise_mi_vmapped` is dead | verified 261023 (`mist.py:658,748,768` call sites; `rg` finds no live call of the latter) |
| Plain-XLA one-hot matmul (form B) avoids the silent-zero failure | unverified; covered by outcome (1) at the boundary shape |

## 4b. Kernel declaration, measurement and parity (input from plegadx LOOP-E1, 261023)

Reported by the plegadx microfold session (task 261023_microfold-pallas; branch
`feat/loop-e1-of3-grain-loader`; commits `6b3fee3f`, the per-piece split, and `78fb4ee0`, the phase
heartbeat, both verified to exist). The numbers below are that session's bathos runs, not
re-measured here.

**What an op declares, beyond `reference` and `kernels`:**
- *coverage*: which pieces of the reference the kernel replaces;
- *analytic bytes and FLOPs*: lowered `cost_analysis` returned no FLOPs on GPU, named-scope
  attribution in GPU traces found 0 of 5 scopes (xtrax debt #2084), and XLA fusion names do not
  map back to model code without the HLO;
- *parity config*: randomised weights, because reference modules zero-init their output weights
  and gates, so parity on a fresh module proves nothing; a "plain" control variant at the same
  precision; and the deepest-gradient configuration (for example, backprop through every
  weight-tied pass).

With coverage and analytic cost declared, split-and-time is automatic.

Two more declaration fields (from the plegadx ColabFold-family prior-art survey, plegadx
`.praxia/docs/research/261023_pallas-prior-art-colabfold-af3open.md`, commit `64f1a720`):
- *`dot_precision`*: Pallas Triton lowers `Precision.DEFAULT` and `HIGH` to TF32
  (`_TF32_PRECISIONS`, jax `_src/pallas/triton/lowering.py:2315`, verified in plegadx/.venv, jax
  0.10.2). An "fp32" kernel with default dots is a TF32 kernel. The declaration states the
  precision, and the parity tolerance follows from it.
- *`deterministic: bool`*: the only upstream Pallas attention backward (colabfold `tri_flash.py`)
  accumulates dQ and dbias with fp32 `atomic_add`, so its results vary run to run. A
  non-deterministic kernel is never selected when `Context` asks for reproducibility, and its
  parity test compares within tolerance across repeated runs.

**Reporting where kernels did not apply.** `Plan.table()` lists, for each op site, *why* the
reference ran: export target, `kernels=False`, no kernel for the backend, dtype not supported, or a
non-deterministic kernel refused. This follows alphafold3-open's `kernel_sites.py` refusal census.
Every op test also asserts that the kernel actually ran (`pallas_call` is in the jaxpr, as q6
checks), because a kernel whose output exactly equals the reference is often a silent fallback.
Timings run twice, so autotuning is excluded.

**Measure at the level a kernel replaces.** Sublayer timing could not pick a kernel: all five
sublayers came in at 22-23 ms each, at about 70% of DRAM bandwidth (plegadx bathos `482c77d6`).
Measure each fusible piece, forward and forward+backward, against a floor:
- floor = max(bytes / bandwidth, FLOPs / GEMM rate), with both rates measured in-process (a copy
  kernel for bandwidth, a GEMM for compute), never taken from datasheets;
- excess = time − floor, with the forward pass counted twice under remat, times the per-block
  multiplicity;
- controls: the doubled-work sensitivity ratio is ≥ 1.6, the sum of pieces reconciles with the
  whole sublayer within [0.7, 2], and timing CV is reported;
- the selection rule and its tie-break order are fixed in the sidecar before the run;
- **backward timing uses one form**: `jax.vjp(fn, *args)[1](ct)`, with `ct` a random array passed in
  as a jit *input*. The byte floor is: read the inputs and `ct`, write the input cotangents. Wholes
  and the sensitivity control use the same form. `grad(sum(out**2))` lets XLA fuse a synthetic
  cotangent the real model never has, so it is not used (plegadx commit `05ee8d4f`). This form is
  necessary but **not sufficient**: see the next item.
- **the sum of pieces must reconcile with the whole, as a hard gate**: a run whose piece-sum ÷ whole
  falls outside [0.7, 2] is `invalid_measurement`, and its ranking is not used. In two plegadx runs
  on L40S, every other control passed (copy bandwidth 651-658 GB/s, tf32 GEMM 145-149 TFLOP/s,
  doubled-work sensitivity about 2.0×, CV < 1.2%), yet the pieces summed to well under the whole:
  - run 1 (bathos `79c50485`, `grad(sum(out**2))`): 0.59, 0.76, 0.94;
  - run 2 (bathos `6f5d881d`, vjp with an input cotangent): 0.53, 0.73, 0.89.

  The gate caught it both times. **The cause is not established.** One untested hypothesis (plegadx,
  261023): an isolated vjp never writes the piece's forward output to HBM, because nothing consumes
  it, while the composed whole writes every intermediate in the forward pass and reads it back in
  the backward. That would hit wide hidden layers hardest, which fits the transition (8c hidden)
  reconciling worst. Two candidate instrument forms, both unvalidated:
  - (a) the timed function returns `(out, vjp(ct))`, which forces the forward output to be
    materialised;
  - (b) leave-one-out: time the composed whole with one piece swapped for a cheaper stand-in,
    instead of timing pieces in isolation.

  #6109 adopts a form only after it reconciles on a known case.

The xtrax tool for this is backlog #6109. Debt #2084 is not a regression. plegadx's 0-of-5 result
came from a copied xtrax pin, `5a6ab77` (0.4.0a7, 2026-09-29), and the #2084 fix `b5afea1`
(2026-10-07) is not an ancestor of it (`git merge-base --is-ancestor`, checked 261023). A re-run on
current xtrax settles it.

**Parity gates for kernels:**
- CPU interpret-mode parity first, then GPU.
- Gate on the training *trajectory* against a seed-noise yardstick, not only on per-step gradients.
  In plegadx, a per-step gradient cosine ≥ 0.9997 still gave 23.5% final-loss divergence against a
  1.6% seed gap (bathos `acc54607`, `2814fc77`). A fused path passed single-block and 6-step parity
  but produced gradient errors of 1e12-1e15 only when backpropagating through all weight-tied
  passes.

**Fusion trade-off.** A fused kernel whose backward recomputes GEMMs trades bytes for FLOPs.
plegadx's fused SwiGLU MLP adds about 50% FLOPs in the backward pass, so on fp32/tf32 hardware the
gain is about 1.5×, not the byte ratio. Budget the gain from the floor model, not from byte counts
alone.

**Observability.** `bth run` swallows the child's logging, so a job killed at walltime leaves no
trace. Profiling hooks write an fsynced phase-heartbeat file outside the timed loops (plegadx
`scripts/loop_e1_gpu_profile.py` `_beat()`).

**Prior art to reuse before writing a kernel** (Apache-2.0; present in plegadx/.venv): tokamax
`gated_linear_unit`, `layer_norm` and attention (Triton backends with VJPs), plus tokamax
`benchmarking.py` and `autotuning.py`, which must be evaluated before #6109 builds its own; and
`jax/experimental/pallas/ops/gpu/{layer_norm, rms_norm, softmax, attention}`. No Pallas
triangle-multiplication kernel exists. The Mosaic GPU backend needs Hopper or newer, so sm89
(L40S) and earlier mean the Triton backend.

From the plegadx survey (`64f1a720`; upstream line citations are that session's, not re-read here):
- No upstream ColabFold-family kernel runs fp32. colabfold-kernels 0.9.0 gates dispatch to
  bf16/f16, and alphafold3-open returns `None` for non-half dtypes.
- Only attention has a real Pallas backward upstream. Every LN, projection, GLU and transition
  fusion is forward-only with an XLA-recompute backward, and alphafold3-open measures GLU *losing*
  1.4-1.5× under grad. That is direct evidence for the §2 backward-kernel opt-in.
- tokamax 0.0.10's Triton attention has a VJP including dbias, has no dtype gate, and beat
  colabfold's Pallas attention for forward+backward at N=384 on A10 (7.1 vs 12.6 ms, per
  alphafold3-open's notebooks). It is the first candidate to wrap as an op rather than write.

## 5. Non-goals

Automatic kernel generation; exporting kernels (export always takes the reference); TPU kernels;
second-order derivatives through a backward kernel (those fall back to the reference).

## 6. Delivery

1. Probe the integer-output `float0` assumption; implement standalone `xtrax.op` with tests:
   selection rules, parity harness helpers, JVP and transform tests ported from q6.
2. titanix GPU run for §3 (pre-registered). This doubles as the epic's GPU Pallas spike.
3. denxity adopts the winning form inside `fused_mi_matrix` behind its `_op` shim (backlog), and
   demistify picks it up by bumping its denxity pin. denxity's distance kernels and aminx's pairs
   follow the same pattern afterwards (backlog).
4. Once Tasks 3–7 of the intent MVP land: `op` becomes an intent site with `Plan` reporting.

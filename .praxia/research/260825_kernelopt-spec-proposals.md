# Spec-Optimization Proposals for xtrax-optimizing (complete list)

Date: 2026-08-25 | Branch: feat/profiling-stage2-evidence
Grounding: `.praxia/research/260825_kernelopt-grounding-report.md` (citations use sections f1-f15, d1-d9, e1-e5)
Scope doc: `.praxia/docs/specs/260825_jax-optimizing-skill-scope.md`
Status: proposal round only — no files changed except this list. 20 items: 8 refinements, 4 new capabilities, 3 cuts, 5 questions for Marielle. TOP 5 marked **DO NEXT**. House rules audited per item; nothing below widens a guard or breaks profiling leaf purity (no first-party imports in `xtrax.profiling`; fail-closed guards never widened; verify-paths cite real code).

## A. REFINEMENTS (land in existing files)

### A1. [DO NEXT] Scan-body remat recipe -> references/tier3-composition.md (new section)

Rationale: f7 — XLA auto-remat is weak across lax.scan; manually applying `jax.checkpoint` to scan bodies plus name-based policies (`save_only_these_names`, `save_and_offload_only_these_names`, incl. residual offload to pinned host memory) is the standard defense [JAX 0.11 ref]. xtrax pipelines are scan-heavy (`xtrax.training.grad.accumulate_grads` microbatches via lax.scan), so the report calls this the single highest-value documentation addition.
Content: decision tree (auto-remat suffices iff stage-0 bytes curve is flat; else manual checkpoint on the scan body; offload last), sharp-edge notes.
Effort S. Risk: none (doc-only). GPU-unblock: PARTIAL — defines exactly the candidate family Stage-2 should rank.

### A2. [DO NEXT] Measurement-protocol bundle -> references/measurement-protocol.md (three edits, one commit)

Rationale:
(i) f5/d5 — make "baseline = strongest currently-shipped configuration" explicit in §2; a winning record against a weaker-than-shipped baseline is INVALID for keep claims (MARLIN pairs against four specialized kernels + roofline ideal; Halide refuses naive-C++ strawmen).
(ii) d6 — micro/macro dissonance clause in §5: when micro and macro disagree, both records persist and neither is averaged away (vLLM publishes its own kernel losing micro by 20–26% yet winning end-to-end 2–4x). Extends our winner-and-loser policy for the TERM_RANKING-vs-END_TO_END conflicts GPU runs will produce.
(iii) d7 — one-line ablation preregistration convention (config keys `candidate_mechanism=` / `ablated_from=`) formalizing FA-3 Table-2-style single-removal attribution inside the existing one-tier-per-candidate rule.
Effort S. Risk: none. GPU-unblock: YES — protects the whole Stage-2 matrix from strawman baselines and misattributed wins.

### A3. [DO NEXT] Opaque-kernel cliff rule -> references/measurement-protocol.md (new short §6)

Rationale: f9/e3 — any Pallas/CustomCall escape hatch becomes opaque to XLA: no cross-operator fusion, no global memory planning, no async scheduling across that boundary; a "fast kernel" record can hide a slower program.
Rule to write NOW (no custom-op path exists yet, so it is pure insurance): any candidate introducing a non-XLA-lowered region owes BOTH dispatch counts AND a fused-thunk-count delta (stage-0 `cost_analysis()` + named-scope attribution already produce the ingredients — verify-paths: `scripts/prof_stage0_onehot_cost.py`, `src/xtrax/profiling/trace.py`) proving fusion loss was priced in, not hidden.
Effort S. Risk: none. GPU-unblock: INDIRECT — prevents laundering bad future GPU records.

### A4. Async-overlap crossover framing -> references/tier2-data-movement.md

Rationale: f6 — literature publishes crossovers, not binary verdicts (vLLM recomputation-vs-swapping curves across block sizes); our honest 0.70x CPU negative result is ONE POINT on a regime curve, not a verdict.
Content: expected-benefit inequality (benefit requires hidden latency > per-item thread-hop + queue overhead) and the sweep knobs the driver already exposes (`--feed-sleep-ms`, `--buffer-size`).
Effort S. Risk: none. GPU-unblock: PARTIAL — tells re-run operators which ladder to climb.

### A5. Donation sharp-edges box -> references/tier3-composition.md donate/remat section

Rationale: f8 — donated buffers are invalidated after the call; keyword-passed args are NEVER donated; PyTree donation sweeps every leaf; unmatched donations are silently dropped with only a warning [JAX 0.11 ref].
Content: codify that the single audited surface (`src/xtrax/sparse/inference.py::sparse_filter_jit`) is policy, not accident; add verification tip — confirm donation actually took effect before crediting any memory saving.
Effort S. Risk: none. GPU-unblock: PARTIAL (memory-side Stage-2 reads).

### A6. Mixed-precision safety recipe -> references/tier2-data-movement.md dtype section

Rationale: f13 — FP16 safety requires FP32 master weights + FP32 accumulation + FP32 reductions in sensitive ops [Micikevicius]; FP8 splits E4M3 (weights/activations) vs E5M2 (range-critical/gradients) [FP8 formats]. CPU-only today, so ship as documentation now with a parity-hook note; activation waits for GPU.
Effort S/M. Risk: low (could be misread as license to coerce dtypes — gate the text behind the parity obligation). GPU-unblock: PARTIAL — pre-registers the numerics gate dtype candidates will owe.

### A7. Regime-shift bucket boundaries -> references/tier2-data-movement.md bucketing paragraph

Rationale: f11/d8 — ADEPT's multi-length sweeps show bottleneck identity shifts with size; vLLM places block-size edges at measured crossovers. Our `tiling.bucket` edges should sit at measured Stage-1 shift points, never round numbers; every record declares its shape axis (benches already stamp `xtrax_n_atoms` per scope-doc §1).
Effort S. Risk: none. GPU-unblock: YES — comparability-within-regime is a precondition for clean GPU rankings.

### A8. Dual-path boundary costing -> references/tier1-host-boundary.md

Rationale: f15 — GPUDirect Storage's direct-fast-path-with-compatibility-fallback design legitimizes our tap/sink staging pattern; document staged drain (`ZarrStagingSink`) vs blocking fallback costs side-by-side, using the measured ~1040–1450x per-step callback tax as the motivating constant.
Effort S. Risk: none. GPU-unblock: NO (CPU-side facts already stable).

## B. NEW CAPABILITIES (P4+)

### B1. [DO NEXT — highest priority overall] ProbeRecord variance + locked-clock fields, BEFORE any GPU re-run

Rationale: f3/d1/d3 — FA-3 locks clocks (1830 MHz, n=100); MARLIN shows unlocked boost clocks DISTORT relative rankings; Ansor (n=5 medians ± std) and FlashAttention (n=10 mean ± std) report central tendency AND spread. Our records lean on point estimates.
Leaf-pure design:
(a) flat metric keys `wall_median_seconds` / `wall_dispersion_seconds` / `n_runs` — `metrics` is already `dict[str, float | int | str]` (`src/xtrax/profiling/record.py`), so NO schema surgery;
(b) `clock_locked` / `gpu_clock_mhz` as DRIVER-DECLARED config keys (string-valued identity per `src/xtrax/profiling/emitters.py`) rather than auto-captured env, avoiding a pynvml dependency in the leaf package;
(c) OPTIONAL floor: TERM_RANKING sources require `n_runs>=3` + finite dispersion — STRICTER gate, fail-closed-compatible (tightening never widens a guard), but it changes claim floors and retroactive rankability of existing records => Marielle decision (see D1).
Effort M. Risk: medium (shared infra pinned by tests/scripts/test_prof_optimizing_drivers.py + claims tests). GPU-unblock: YES — GATING; rankings without this are not trustworthy per MARLIN.

### B2. [DO NEXT] Stage-0 arithmetic-intensity emitter + device-ceilings table

Rationale: f4/d4 — roofline-ceiling overlays are how MARLIN/Triton/Nsight prove proximity to physical limits; our stage-0 probes never execute, so an intensity estimate (FLOPs/byte per variant scope) compared against a published-ceiling table is the same trick at ZERO runtime cost.
Artifacts: extend `prof_stage0_onehot_cost.py` pattern; new `references/device-ceilings.md` keyed by device_kind, L40S row filled first from the dogfood run. Turns every GPU result into a predicted-vs-achieved gap statement (bandwidth-bound confirmed vs compute-limited surprise).
Effort M. Risk: low (estimates are STRUCTURAL-only claims; label them so). GPU-unblock: YES — improves attribution of every Stage-2 number.

### B3. P4 headline: scan-remat probe-driver PAIR

Artifacts: `scripts/prof_stage0_scan_remat_cost.py` + `scripts/prof_stage1_scan_remat_micro.py`. Variants: XLA auto-remat default vs manual `jax.checkpoint(scan_body)` vs name-based save/offload policy; memory + walls + parity gate. Gives A1 its measured exemplar, honoring house style (every reference page ships WITH its driver — cf. one-hot pair). Reuses one-hot driver scaffolding.
Effort M/L. Risk: medium (new scripts + tests; scan construction has known static-carry constraints — verify-path `src/xtrax/tiling/dispatch.py::make_axis_dispatch`). GPU-unblock: YES — creates THE memory-bound candidate that GPU re-runs exist to test. Sequence AFTER B1.

### B4. Crossover-sweep mode for scripts/prof_stage1_feed_overlap.py

Design: sleep-ms x buffer-size ladder emitting one record per rung; report renders the crossover per platform (d8 regime-sweep pattern). Converts the 0.70x CPU negative result into a regime map; on GPU, real H2D makes the beneficial regime reachable for the first time.
Effort S/M. Risk: low. GPU-unblock: PARTIAL.

## C. SCOPE CUTS

### C1. Cut "pinned-memory placement if ever added" from Tier-2 contents

Rationale: scope doc §2 lists it; the corpus offers zero direct support for host-pinned staging in our pipeline; nearest analogue (GDS, f15) concerns storage paths, not pinned host buffers; it invites speculative library surface the probes have not justified. Keep at most a one-line mention.
Effort S (doc edit). Risk: none. GPU-unblock: neutral.

### C2. Demote CPU micro-wall precision work

Rationale: our own SKILL.md concedes CPU walls fluctuated ~0.5x–1.9x across runs; the literature's rigor bar (locked clocks, n>=5 medians, d1/d3) implies uncontrolled-clock CPU micro benchmarks cannot back directional claims AT ALL.
Action: stop investing in further CPU jitter reduction (extra trial logic, warm-up schemes); repurpose CPU stage-1 records as smoke + regime-classification + DISPATCH_COUNT evidence only until GPU lands. Redirects effort to B1/B2 where it pays. Phrase as prospective so existing CPU citations are not retroactively invalidated.
Effort S (policy sentence in measurement-protocol §4). Risk: low. GPU-unblock: YES by redirection.

### C3. Defer the Foldcomp follow-up query

Rationale: coverage ledger (g) — Foldcomp was never cited by any of 12 answers; ensemble-compression is adjacent to staging/sink concerns but nothing in the current P4 plan consumes it. Revisit only if T1/T2 staging work resumes.
Effort S (decision only). Risk: none.

## D. OPEN QUESTIONS FOR MARIELLE

D1. Tighten the TERM_RANKING floor to require `n_runs>=3` + finite dispersion (and/or `clock_locked` for GPU sources)? Stricter-than-current is fail-closed-compatible, but it changes claim floors RETROACTIVELY (some old records may lose rankability). Approve tightening + retroactivity stance? Gates B1 final shape.

D2. Can the L40S dogfood box lock application clocks (`nvidia-smi -lgc` needs root)? If NOT: do we accept unlocked-clock Stage-2 rankings provided `gpu_clock_mhz` is recorded per run AND dispersion requirements widen? This sets the minimum credible protocol for the entire re-run campaign.

D3. GPU budget & priority order for the Stage-2 matrix: proposed order B3 (scan-remat) > one-hot pair > feed-overlap ladder (B4) > host-boundary (lowest; CPU-stable facts already strong). How many L40S-hours exist, and does the order hold?

D4. Performance-gate tripwire wiring (existing user-owned remainder): does the repo stay pinned to NO dispatch config (`tests/audit/test_performance_gate.py::test_repo_targets_have_no_dispatch_config` policy) with ceilings living only in opt-in CI profiles? A3's future opaque-kernel rule will want teeth eventually — same policy call.

D5. Who owns parity-tolerance VALUES for the coming dtype era (A6)? `PARITY_TOLERANCE` today lives per-driver (`prof_stage1_onehot_micro.py`); FP16/FP8 candidates will need a tolerance policy (per-experiment declared? central table?) — numerics sign-off is a human call.

## HOUSE-RULE AUDIT

B1 touches the profiling leaf but adds stdlib-only flat keys + driver-declared config strings (no first-party imports; no guard widened — option (c) only TIGHTENS). A2/A3/C2 add rules, never remove them. Every verify-path cites real code: `src/xtrax/profiling/record.py` (metrics typing, stage>=2 platform/device_kind enforcement), `src/xtrax/profiling/claims.py` (REQUIRED_METRICS, paired_configs unanimity set), `src/xtrax/profiling/emitters.py`, `src/xtrax/profiling/trace.py`, `src/xtrax/tiling/dispatch.py::make_axis_dispatch`, `src/xtrax/sparse/inference.py::sparse_filter_jit`, `src/xtrax/stages/executor.py` module docstring, `src/xtrax/stages/topology.py::validate_plan_topology`, `src/xtrax/tiling/bucket.py`, `src/xtrax/engine/io.py::async_indexed_stream`, `src/xtrax/run/zarr_sink.py::ZarrStagingSink`.

## SEQUENCING SUMMARY

B1 (+ D1/D2 answers) gates everything GPU. A1+A2+A3 are one cheap doc commit doable today. B2 fills in as soon as L40S ceilings are known. B3+B4 are the P4 build.

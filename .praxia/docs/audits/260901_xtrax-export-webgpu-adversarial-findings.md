---
title: Adversarial review findings — xtrax.export WebGPU spec
description: Consolidated challenger + defender findings with orchestrator empirical verification
task_id: 260901_xtrax-export-webgpu
status: open
---

# Consolidated adversarial findings

Two Opus reviewers audited `.praxia/docs/specs/260901_xtrax-export-webgpu.md` independently.

- **Challenger verdict:** `not_ready` — 14 BLOCKER, 18 MAJOR, 10 MINOR
- **Defender verdict:** `needs_revision` — implementable after bounded fixes; 4 blocking, 9 warnings

The naming defect (spec invented API names) is already recorded in the spec's Appendix A and is
NOT repeated here.

---

## Part 0 — Orchestrator empirical verification (authoritative, measured after review)

Three testable challenger claims were checked directly. **These results override any
contradicting statement in either review or in the spec body.**

### V1. `metal-spirv` does NOT emit SPIR-V — challenger B7 CONFIRMED

Compiling a matmul+tanh pipeline with `--iree-hal-dump-executable-binaries-to`:

```
vulkan-spirv -> module_pipe_dispatch_0_..._matmul_128x32x64_f32.spv   16836 B  magic 0x07230203 (SPIR-V)
metal-spirv  -> module_pipe_dispatch_0_metal_msl_fb0.metal             7652 B  magic 0x636e6923 ("#inc")
```

`0x636e6923` is ASCII `#inc` — Metal Shading Language **source**. IREE lowers SPIR-V to MSL
internally and dumps MSL. **AC-8 cannot pass for `METAL_SPIRV`.** Metal must be demoted to
`VerificationLevel.CODEGEN_ONLY`, and the wgpu/naga gate applies to `vulkan-spirv` only.

### V2. wgpu DOES reject invalid SPIR-V synchronously — challenger B10 REFUTED on mechanism, UPHELD on contract

```
valid IREE .spv            -> ACCEPTED (no exception)
valid magic, garbage body  -> RAISED GPUValidationError: "unknown instruction 44510"
random bytes               -> RAISED ValueError: "Given shader data does not look like a SpirV module"
```

The naga gate is REAL, not a false positive; no `push_error_scope` bracketing is required for
these cases. **But the spec's stated contract is backwards:** spec:257-264 says `validate_webgpu`
"never raises on naga rejection -- returns `valid=False`". It demonstrably DOES raise, so the
implementation MUST catch `GPUValidationError` and `ValueError` and convert them to
`valid=False, error=<msg>`. The challenger's demand for a negative-case AC stands, and the two
payloads above are the ready-made fixtures for it.

### V3. Dump is not necessarily plural — challenger B8 PARTIALLY REFUTED

A matmul+tanh pipeline produced exactly **one** `.spv`; IREE fused the two ops into a single
dispatch. Plurality is therefore not automatic and B8's urgency is lower than stated. The
structural point still holds: nothing guarantees one file, so the field should be a mapping of
`executable_name -> bytes` with `valid` as the conjunction, and the directory scan must filter
by SPIR-V magic (which V1 shows is necessary anyway, since metal dumps `.metal` files).

---

## Part 1 — Blockers BOTH reviewers found independently (highest confidence)

| # | Defect | Challenger | Defender |
|---|---|---|---|
| 1 | **AC-14 requires an ordered `Sink` that AC-3 must reject.** Physically ill-posed: a `Sink` is `io_callback`-backed and cannot exist inside a compiled vmfb. | B1, B2 | C1 |
| 2 | **`scan_init` dropped** — spike's `compose_single_axis` raises `ComposerError` without it, so AC-2's `Scan` fixture is unconstructible. | B5 | C3 |
| 3 | **CI has no Vulkan ICD.** All `ci.yml` jobs are bare `ubuntu-latest`; no `mesa-vulkan-drivers`. AC-8 green locally, red in CI; conflicts with AC-7's zero-skip. | B12 | C2 |
| 4 | **`export_pipeline` placed in `__init__.py`**, which BOTH `pyproject.toml:101` and `distribution/coverage_dag.toml:30` omit → AC-6 is vacuous. | B14 | C5 |
| 5 | **`export` extra names unresolvable distributions and omits two required ones.** Spike uses `iree-base-compiler`/`iree-base-runtime`; spec pins `iree-compiler`/`iree-runtime>=3.11,<4` (legacy, date-versioned, matches nothing). `huggingface_hub` + `safetensors` missing entirely. | B13 | C7 |
| 6 | **bytes-vs-Path mismatch.** Only proven executor is `run_native_vmfb(vmfb_path: Path)` via `VmModule.mmap`, which needs a file; spec passes `vmfb_bytes: bytes`. | B4.2 | C10 |
| 7 | **`DedupGather` dropped on a false premise.** Spec claims it needs a runtime-index contract; the spike routes it today with host-computed static indices (`composer.py:87-94`, `EXPORTABLE_STRATEGIES` includes it). This is a regression vs the spike. | M6 | C12 |
| 8 | **Task 20's "certified recipe exactly" is not achievable** — `TestBatchedShapeVmapOfScanPreservesOrder` calls bare `jax.lax.scan`, never `execute_scan_axis`. | M9 | C9 |

---

## Part 2 — Challenger-only findings that matter most

**B3 (deepest finding in either review) — the parity oracle is self-referential.**
`verify_native_parity` compares `jax.jit(composed)` against `IREE(composed)` — the SAME composed
callable. That detects XLA-vs-IREE lowering divergence but CANNOT detect a composition error
(wrong axis nesting, fuse applied per-step, dropped boundary, mis-shaped carry) because both
sides change identically and yield max|diff| ~= 0. The spike knew this and used an INDEPENDENT
oracle in its tests: `want = jnp.stack([model(x) for x in xs])` (`test_iree_export_spike.py:100`).
The spec dropped it. `.verified is True` therefore certifies far less than AC-2/AC-14 claim.
**Fix:** define the parity reference explicitly as an independent recomputation, and state in
the `EXECUTED` docstring that it bounds lowering fidelity, not composition correctness.

**B4.1 — `verify_native_parity` receives the wrong `fn`.** `export_pipeline`'s `fn` is the
per-element step function; calling `fn(*concrete_inputs)` yields a per-element result, and
`compare()` short-circuits shape mismatch to `passed=False, max_abs_diff=inf` (`parity.py:54-62`),
so EVERY EXECUTED export fails parity. Parity must take the COMPOSED callable.

**B11 — wrong wgpu API.** Spec says `wgpu.request_adapter()`. In wgpu-py 0.32 it is
`wgpu.gpu.request_adapter_sync(...)` then `adapter.request_device_sync(...)`. Orchestrator
confirms: every working probe in this session used `request_adapter_sync`.

**B6** — two sources of truth for the scan transition/carry: `Scan` strategy already carries
`transition` and `init` fields (`tiling/strategy.py:73-78`) AND `export_pipeline` supplies a
callable; precedence unspecified. `Scan.ordered_sinks: bool = True` interacts with Rule 3 unaddressed.

**B9** — WebGPU has no push-constant equivalent and caps `maxStorageBuffersPerShaderStage` at 8;
IREE's vulkan HAL uses push constants. `create_shader_module` validates the module only — binding
layout and workgroup limits are enforced at pipeline creation, which this design never reaches.
AC-8 must name the exact kernel (the AC-2 pipeline, not a synthetic one) and state the disposition
if it is rejected.

**M3** — the StableHLO portable-artifact downgrade fallback (`compile_iree.py:65-80,114-138`,
tested at `test:318-342`) and its `CompileResult.downgraded_stablehlo` flag are silently dropped.
xtrax pins `jax>=0.10.2,<0.12` against `iree>=3.11` — this is a live skew path.

**M4** — the three-record merge loses: `CompileResult.path`/`.downgraded_stablehlo`;
`ParityResult.rtol`/`.shape_expected`/`.shape_actual` (the shape guard is deliberate:
*"a silently broadcast comparison is how a real regression gets missed"*, `parity.py:47-49`);
all of `WeightReport`. Default `atol` also silently moves 1e-5 -> 1e-6 with no justification.

**M5** — export-safety collapses two entry points (`check_plan_export_safety` returns ALL blockers;
`validate_plan_topology` raises on the FIRST). Multi-blocker reporting is lost. Worse, the same
condition can raise three different exception types, so a caller cannot write a correct `except`.

**M10 / M11** — `load_hf_weights` is never wired into `export_pipeline`, so AC-10's "same cast
weights" is unenforceable; and it drops `filename`, three mandatory shape params, and `dtype`.
The spike is shape-driven not name-driven, fabricates `b1`/`b2` as zeros, and truncates
`dtypes_cast` to two entries (`hf_weights.py:140`) — contradicting "one diagnostics string per
cast leaf".

**M16 — the spike's disposition is entirely unspecified.** No task deletes/migrates
`scripts/iree_export_spike/`, its 409-line test file, the `export-spike` dependency group, or
closes draft PR #111. Two maintained copies with divergent error strings (`match="export-spike"`
vs `"pip install xtrax[export]"`).

**M13** — AC-11 writes `request_features={"shader-f16"}` (a `set`) against a `frozenset[str]`
signature, under a beartype hook the spec itself installs → `BeartypeCallHintParamViolation`.

**M15** — docs tasks rest on false claims: `docs/conf.py:97` sets `nitpicky = False`,
`warn_is_error` is absent, and `just audit-docs-build` does not build Sphinx. The real omission
(no `api/export` in `docs/index.md`'s toctree) is unaddressed.

**M17 / M18** — `.verified` undefined for `CODEGEN_ONLY` and `VALIDATED`; multi-target partial
failure semantics unspecified (`dict[str, ExportResult]` has no failure slot).

---

## Part 3 — Defender-only findings

**C4 — the dtype gate has a closure-shaped hole.** `validate_export_safe` inspects
`abstract_inputs` only. Because weights are closure constants (which is exactly WHY AC-9 and
AC-10 are consistent), a user-supplied Equinox module holding bf16/f64 leaves passes every gate
and bakes forbidden dtypes into SPIR-V-bound StableHLO. The spike had the precedent for closing
this — `find_bcoo_leaves` (`export_safety.py:141-163`) — and the spec drops it (also challenger m9).

**C8** — PR1 must define `SpirvValidationResult` (a PR2 type) because `ExportResult` is a frozen
dataclass with a field typed by it, and the field-freeze rule forbids adding it later.

---

## Part 4 — Where the reviewers DISAGREE (resolve deliberately)

**bf16 cast point.** The defender REBUTS this objection with strong evidence: AC-9 (runtime
inputs -> raise) and AC-10 (closure weights -> upcast) apply to DISJOINT populations, because the
spike proves weights are closure constants and never appear in `abstract_inputs`
(`test_iree_export_spike.py:158-160`). The cast point IS specified twice (spec:143, Task 13) and
the mechanism already exists (`hf_weights.py:107-115`). **Accept the defender here.**
The challenger's M12 is a DIFFERENT and still-valid point: bf16->f32 is exact, so parity between
an f32 export and an f32 reference can never surface divergence from the original bf16 model's
intended numerics. That is a missing test, not a missing cast point. Keep both conclusions.

**`ExportResult` merge.** Defender partially rebuts: the record IS field-specified, so it is a
new record rather than an underspecified merge. What is genuinely unspecified is only where the
DROPPED fields go (M4). **Accept the defender's framing, act on the challenger's field list.**

---

## Part 5 — Remediation priority

1. **Correctness-of-claim first** (these make green CI meaningless): B3 parity oracle, V1 metal
   demotion, V2 raise-vs-return contract + negative AC, B10's missing negative case.
2. **Contradictions**: blocker 1 (AC-14/AC-3), M5 exception-type collision, M9/C9 recipe.
3. **Unimplementable signatures**: B4.1, B5, blocker 6, B11, B6.
4. **Mechanical/CI**: blocker 3 (Vulkan ICD), 4 (`__init__.py`), 5 (extra), M1 fake-shadowing,
   M2 zero-skip gate, M14.
5. **Dropped capability**: blocker 7 (DedupGather), M3 (downgrade fallback), C4/m9 (`find_bcoo_leaves`).
6. **Scope honesty**: M16 (spike disposition), M10/M11 (`load_hf_weights` scope), M17/M18.

---

## Regenerating the spike snapshot

`.praxia/spike_snapshot/` is a working aid and is deliberately NOT committed — it duplicates
code that already lives on `origin/spike/iree-wasm-export` (draft PR #111). Read-only agents
cannot run `git show`, so materialise it before dispatching any:

```bash
mkdir -p .praxia/spike_snapshot/scripts/iree_export_spike .praxia/spike_snapshot/tests/scripts
for f in scripts/iree_export_spike/{__init__,__main__,composer,compile_iree,export_safety,hf_weights,parity}.py \
         tests/scripts/test_iree_export_spike.py; do
  git show "origin/spike/iree-wasm-export:$f" > ".praxia/spike_snapshot/$f"
done
```

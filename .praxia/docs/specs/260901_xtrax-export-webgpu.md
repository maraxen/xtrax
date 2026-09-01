---
category: specs
title: "xtrax.export: promote the IREE/WASM/SPIR-V feasibility spike to a shipped subpackage"
description: "Spec for src/xtrax/export/ — native/wasm32/vulkan-spirv/metal-spirv targets, wgpu/naga WebGPU-validity gate, three-PR rollout"
task_id: 260901_xtrax-export-webgpu
status: draft
---

# Specification: `xtrax.export` — promote the IREE/WASM/SPIR-V spike to a shipped subpackage

## Changelog vs v1

This revision was produced against `.praxia/docs/audits/260901_xtrax-export-webgpu-adversarial-findings.md`
(challenger `not_ready` 14 BLOCKER/18 MAJOR/10 MINOR, defender `needs_revision`) with the
spike source now readable at `.praxia/spike_snapshot/scripts/iree_export_spike/`. AC numbers are
preserved; where an AC's *content* changed, the AC itself says so inline (search "CHANGED").

**Addressed:**

- **V1** (metal-spirv doesn't emit SPIR-V) — `METAL_SPIRV` demoted to `CODEGEN_ONLY`; the wgpu/naga
  gate now applies to `vulkan-spirv` only. AC-8 split accordingly.
- **V2** (wgpu raises, doesn't return) — `validate_webgpu()`'s contract corrected: it still never
  raises *to its caller* for a naga-rejection reason, but only because it now explicitly catches
  `GPUValidationError`/`ValueError` around `create_shader_module` and converts them to
  `valid=False, error=<msg>`. New negative-case AC-8b with the two orchestrator-supplied fixtures.
- **V3** (dump not necessarily plural) — `spirv_bytes` becomes a `dict[str, bytes]` mapping
  (`executable_name -> bytes`), `valid` is the conjunction across entries; the dump-directory scan
  filters by SPIR-V magic `0x07230203` (necessary regardless, since metal dumps `.metal` files).
- **Blocker 1 / B1/B2/C1** (AC-14 requires a Sink that AC-3 forbids) — AC-14 split into AC-14a
  (ordering certified in pure JAX, Sink permitted, no `ExportResult`/`.verified` claim) and AC-14b
  (export/parity asserted on a separately-composed sink-free variant of the same shape).
- **Blocker 2 / B5/C3** (`scan_init` dropped) — restored as a parameter on both
  `build_traceable_callable` and `export_pipeline`, matching the spike's `compose_exportable`.
- **Blocker 3 / B12/C2** (no Vulkan ICD in CI) — Task 8 now installs `mesa-vulkan-drivers` +
  `libvulkan1` before the real-toolchain job runs, so `llvmpipe` is actually reachable.
- **Blocker 4 / B14/C5** (`export_pipeline` lands in `__init__.py`, which coverage/packaging
  configs both treat as opaque) — moved to a new `pipeline.py` module; `__init__.py` becomes a
  thin re-export shim (matching the repo's own per-subpackage convention), so it is not silently
  excluded by `coverage_omit`'s blanket `*/__init__.py` rule.
- **Blocker 5 / B13/C7** (unresolvable/incomplete `export` extra) — repinned to
  `iree-base-compiler`/`iree-base-runtime` (the spike's real install target, confirmed by
  `compile_iree.py`'s own error strings) plus `huggingface_hub` and `safetensors`.
- **Blocker 6 / B4.2/C10** (bytes vs. Path) — `CompileResult` keeps a real `path: Path` (matching
  the only proven executor, `run_native_vmfb(vmfb_path: Path)`); `ExportResult.vmfb_bytes` is
  populated by reading that path once, for API convenience only.
- **Blocker 7 / M6/C12** (`DedupGather` dropped on a false premise) — restored as a routed,
  exportable strategy (host-computed static indices, no runtime-index contract needed), matching
  the spike's real `compose_single_axis` and `xtrax.tiling.dispatch.axis_dispatch`.
- **Blocker 8 / M9/C9** (Task 20 can't follow the "certified recipe" — the certifying test never
  calls `execute_scan_axis`) — Task 20 now calls bare `jax.lax.scan` directly for the multi-axis
  batched-shape path, matching `TestBatchedShapeVmapOfScanPreservesOrder`'s actual code, with the
  boundary's sink applied inline (not delegated to `execute_scan_axis`).
- **B3** (self-referential parity oracle, "the deepest finding in either review") —
  `verify_native_parity` no longer computes its own expected value from the composed callable.
  `export_pipeline` gains a `reference_fn` parameter: an independently-computed oracle the caller
  supplies (mirroring the spike test's `want = jnp.stack([model(x) for x in xs])`), required
  whenever any target is `EXECUTED`. `ExportResult`'s EXECUTED docstring now says it bounds
  lowering fidelity, not composition correctness.
- **B4.1** (parity received the wrong `fn`) — resolved as part of B3: parity now compares
  `reference_fn(concrete_inputs)` (full composed-shape) against `run_native_vmfb(path,
  *concrete_inputs)` (also full composed-shape) — no more per-element-vs-full-batch shape
  mismatch.
- **B11** (wrong wgpu API) — `validate_webgpu` now calls `wgpu.gpu.request_adapter_sync(...)` then
  `adapter.request_device_sync(required_features=...)`, never a module-level `wgpu.request_adapter`.
- **B6** (two sources of truth for Scan transition/carry) — documented explicitly: the composer's
  `fn` parameter is *always* the transition used for export; `Scan.transition` is read only by
  the unrelated eager `axis_dispatch` path and is never consulted by the export composer.
  `Scan.init` participates only as a `scan_init` fallback. `Scan.ordered_sinks` is unrelated to
  `AxisBoundary.tap`/`.sink` and is not consulted by Rule 3 — documented as an inert field for
  export purposes, not silently reinterpreted.
- **B9** (AC-8 needs the exact kernel + a disposition statement) — AC-8 now names the AC-2 fixture
  kernel explicitly and states that binding-layout/workgroup/push-constant limits are structurally
  unreachable in this spec's scope (module validation only, never pipeline creation — see
  Non-goals).
- **M3** (StableHLO portable-artifact downgrade fallback silently dropped) — `CompileResult`
  keeps `downgraded_stablehlo: bool`; Task 4's gate now includes the downgrade-retry case.
- **M4** (three-record merge drops fields) — `CompileResult.path`/`.downgraded_stablehlo`,
  `ParityResult.rtol`/`.shape_expected`/`.shape_actual`, and `WeightReport` are all given explicit
  homes (see Public API signatures). Default `atol` reverted `1e-6` -> `1e-5` (the spike's real
  default in `parity.py::compare`); no justification for the tighter value was ever given.
- **M5** (export-safety collapses two entry points into one, and the same condition can raise
  three exception types) — both spike entry points are promoted: `check_export_safety()`
  (list-returning, promoted from `check_plan_export_safety`) and `validate_export_safe()`
  (raising, promoted from `assert_plan_export_safe`, reporting every blocker). Both let
  `xtrax.stages.topology.PlanTopologyError` propagate unwrapped; a caller can `except
  (PlanTopologyError, ExportSafetyError)`.
- **M10/M11** (`load_hf_weights` never wired in; dropped params; truncated diagnostics) —
  documented as a caller-driven pre-step (the caller builds `fn` *and* `reference_fn` from the
  same loaded model instance, so the cast naturally flows into both); `dtypes_cast` truncation
  bug (`[:2]`) is called out as a fix, not carried over; genericizing beyond `TinyMLP` remains
  explicitly out of scope (Non-goals), matching Appendix A note 4.
- **M13** (beartype violation: `set` vs `frozenset[str]`) — AC-11's example corrected to
  `frozenset({"shader-f16"})`.
- **M15** (docs tasks rest on false claims) — Task 9/19/22 now say plainly that
  `just audit-docs-build` does **not** build Sphinx (verified: it runs
  `scripts/audit_docs_plumbing.py` + a plumbing test, nothing else) and add the real missing
  piece: an `api/export` entry in `docs/index.md`'s toctree.
- **M16** (spike disposition unspecified) — new Task 9b: retire `scripts/iree_export_spike/`,
  its test file, and any `export-spike` dependency group; close draft PR #111 without merging,
  once PR1's promoted coverage supersedes it.
- **M17/M18** (`.verified` undefined for `CODEGEN_ONLY`/`VALIDATED`; no multi-target failure slot)
  — `.verified` is now defined per level (`False` unconditionally for `CODEGEN_ONLY`; `parity.passed`
  for `EXECUTED`; `spirv_validation.valid` for `VALIDATED`); `export_pipeline` is documented as
  all-or-nothing across `targets=` — the first target-level exception aborts the whole call, no
  partial `dict[str, ExportResult]` is ever returned.
- **C4/m9** (`find_bcoo_leaves` dropped; closure-held dtype hole) — `validate_export_safe` now
  additionally scans `fn`'s closure-reachable leaves (generalizing `find_bcoo_leaves`'s
  `tree_flatten_with_path` pattern to dtype checking, not just BCOO detection) for forbidden
  dtypes, closing the gap where a closure-held bf16/f64 leaf passed every gate because
  `abstract_inputs` never saw it.
- **C8** (`SpirvValidationResult` needed a PR1 home because `ExportResult`'s fields are frozen) —
  `spirv.py` is now created in PR1 (Task 1) containing only `SpirvValidationResult`; PR2's Task 11
  *extends* the same file with `validate_webgpu`/`WebGPUValidationError`.
- **M12** (bf16->f32 is exact; f32-vs-f32 parity can't see bf16 divergence) — kept as a *separate*
  gap from the cast-point question (which the defender correctly defends, see below): new AC-16
  plus Task 13b add a bf16-precision-tolerance parity test between the original bf16 forward pass
  and the exported f32 artifact.

**Deliberately not "fixed" (defender's framing accepted verbatim):**

- **bf16 cast point** (AC-9/AC-10) — the defender is right: AC-9 (runtime inputs) and AC-10
  (closure weights) govern disjoint populations, and the cast point is specified twice already.
  No change to the cast point itself; only M12 above is new work.
- **`ExportResult` merge** — the defender is right that this is a new, fully field-specified
  record rather than an underspecified merge. The action item was always the *dropped-field*
  list (M4, addressed above), not the merge concept.

**Named but not resolved here (flagged rather than invented):**

- **M1** (fake-shadowing) — inferred concern: `tests/export/conftest.py`'s `sys.modules`-fake
  fixtures must be function-scoped (`monkeypatch.setitem`, auto-reverting) so a fake `iree`/`wgpu`
  injected for one test cannot leak into a same-session real-toolchain test guarded by
  `pytest.importorskip`, which would silently exercise the fake instead of the real toolchain.
  Task 18 now says this explicitly. This is this author's best reconstruction of M1's intent from
  the one-line Part 5 reference — the findings doc gives no further detail, so treat this
  disposition as provisional and re-open if a fixer finds a different fake-shadowing mechanism.
- **M2** (zero-skip gate) — tied to Blocker 3: once the Vulkan ICD is actually installed (Task 8),
  `pytest.importorskip("wgpu")`-guarded real tests should stop skipping, but this was not
  independently re-measured in this revision pass (no shell/CI access here). Task 8's gate now
  explicitly asserts "0 skipped" is visible in the job's own pytest summary line, not just green
  exit — the fixer must verify this against a real CI run, not assume it from the text alone.
  **Flagged as needing a measurement**, not resolved by spec text.
  - The exact wgpu-py 0.32 exception import path for `GPUValidationError` (e.g.
    `wgpu.GPUValidationError` vs. a nested `wgpu.classes.*` location) was not independently
    re-verified in this revision pass either — Task 11's fixer must confirm the real import path
    against an installed `wgpu-py==0.32.*` before writing the `except` clause, rather than
    trusting this document's guess.
- **M14** — the findings doc names "M14" only in Part 5's remediation list
  (`4. Mechanical/CI: ... M14.`) with no defining prose anywhere else in the document. This
  revision could not determine what M14 refers to and has **not** invented a fix for it. Flagging
  for the next reviewer to either supply M14's content or drop the reference.

## Note on sourcing

This spec was written from three inputs, in this priority order:

1. **The verified environment facts and approved design supplied directly in the dispatch
   prompt** (measured 2026-08-31/09-01, and explicitly approved by the user) — these are
   authoritative and this spec does not second-guess them, **except where Part 0 of the
   adversarial-findings doc supersedes them with newer, orchestrator-measured facts** (the
   metal-spirv/SPIR-V-magic and wgpu-raises findings above). Where the two conflict, the
   findings doc's Part 0 wins, per that doc's own framing.
2. **The spike source itself**, now readable at `.praxia/spike_snapshot/scripts/iree_export_spike/`
   and its test file at `.praxia/spike_snapshot/tests/scripts/test_iree_export_spike.py`. Every
   real symbol name is in Appendix A below (verified this revision, by direct `Read`, not by
   AST-parsing a remote ref this agent couldn't reach). Where this spec's body names something
   differently from Appendix A, that is a **stated rename-on-promote**, never new construction.
3. **Current `main`-branch code**, read directly: `src/xtrax/stages/executor.py`,
   `src/xtrax/stages/topology.py`, `src/xtrax/stages/boundaries.py`, `src/xtrax/tiling/strategy.py`,
   `src/xtrax/tiling/dispatch.py`, `src/xtrax/run/zarr_sink.py`, `pyproject.toml`,
   `distribution/coverage_dag.toml`, `.github/workflows/ci.yml`, `docs/conf.py`, `docs/index.md`,
   `Justfile`, `tests/stages/test_nested_ordering.py`, `tests/cli/test_export.py`,
   `tests/conftest.py`, `tests/test_import_isolation.py`.

## Overview

Promote the proven `scripts/iree_export_spike/` feasibility spike (native vmfb parity
1.02e-10 measured by the spike's own driver, wasm32 codegen, SPIR-V extraction, wgpu/naga
WebGPU-validity gate — all verified 2026-08-31, with the caveat below about what that parity
number actually bounds) into a shipped `src/xtrax/export/` subpackage with four compile targets
(`native`, `wasm32`, `vulkan-spirv`, `metal-spirv`), landed as three independently-green PRs.

**Caveat carried over from the changelog (B3):** the spike's driver (`__main__.py`) computed its
1.02e-10 parity number by comparing `jax.jit(forward)(xs)` against the IREE-executed vmfb of that
*same* `forward` — a real, valid check of XLA-vs-IREE lowering fidelity, but not, by itself, proof
that `forward`'s composition (axis nesting, boundary wiring) matches the original per-element
semantics. The spike's own *test suite* used a genuinely independent oracle for the composer
(`want = jnp.stack([model(x) for x in xs])`, `test_iree_export_spike.py:100`) — this spec adopts
that independent-oracle pattern for the promoted `export_pipeline`'s parity check too (AC-2,
Task 5, `reference_fn`).

## Non-goals (explicit)

- **Executing SPIR-V kernels under `wgpu`** (real WebGPU numeric execution). This needs
  reconstructing IREE's undocumented dispatch ABI (descriptor sets, push constants, workgroup
  counts) and is deferred past this spec. `spirv.py`'s job is validation only
  (`wgpu.create_shader_module` succeeding), never dispatch. (Unchanged from v1; restated because
  it directly bears on B9's disposition below — binding-layout and workgroup-limit enforcement,
  which only happens at pipeline creation, is therefore never reached by this spec at all.)
- **Browser execution** of any exported artifact (no JS/WASM glue, no npm package, no
  `emsdk`-built IREE runtime — IREE's own browser/web runtime is experimental with no npm
  package per the 260831 close entry).
- **IREE's own `webgpu-spirv` compiler backend.** It does not exist in IREE 3.11.0 (verified
  fact #1) and is not a target of this spec — the WebGPU story here is entirely `vulkan-spirv`
  SPIR-V bytes validated by `wgpu`/naga, not IREE-native WebGPU. (`metal-spirv` no longer has a
  WebGPU story at all — see the V1 changelog entry: it dumps MSL source, not SPIR-V, so it is
  `CODEGEN_ONLY` and never touches `spirv.py`.)
- **`rocm`/AMD NPU targets.** `rocm` fails to compile without an explicit `--iree-hip-target`
  chip (verified fact #2) and there is no hardware or chip target list to pin against; not
  attempted.
- **jax-js interop.**
- **`Bucket`, `WhileCarry` axis strategies in the composer.** `Bucket` is host-side by
  construction (never traced); `WhileCarry` has an unbounded compile-time trip count. Both raise a
  named error from the composer (Task 3) rather than silently mis-compiling. Revisit in a
  follow-up spec if a caller needs them.
  **CHANGED from v1:** `DedupGather` is **removed from this non-goal** (Blocker 7 /
  M6/C12) — the spike already routes it via host-computed static indices
  (`export_safety.py::EXPORTABLE_STRATEGIES` includes it; `composer.py::compose_single_axis`
  dispatches it through `xtrax.tiling.dispatch.axis_dispatch`), so dropping it in v1 was a
  regression relative to the spike, not a deliberate scope cut. It is now a supported, routed
  strategy (Task 3, AC-4).
- **A CLI verb.** `xtrax export` (the existing StableHLO-text/flatbuffers CLI verb) is
  untouched; `xtrax.export` is a library subpackage only in this spec.
- **Run-ledger/telemetry provenance recording** for `export_pipeline()` calls (unlike
  `cli/export.py`'s `_record_export`). Not requested by the approved design; can be a
  follow-up.
- **Genericizing HF-weight loading beyond the spike's `TinyMLP` shape.** (Restated from
  Appendix A note 4, elevated to an explicit non-goal per the M10/M11 disposition above.) The
  spike's `mlp_from_hf` is shape-driven, not name-driven, and only proves "HF weights ->
  `TinyMLP`", not "HF weights -> arbitrary Equinox module". The promoted `load_hf_weights` keeps
  this scope; a real architecture port (mapping by parameter name into an arbitrary module) is
  separate, unstarted work.
- **Recovering a partial multi-target result on failure.** `export_pipeline(targets=(...))` is
  all-or-nothing: the first target whose compilation/validation/parity step raises aborts the
  whole call. `dict[str, ExportResult]` is only ever returned in full, never partially populated.
  (New non-goal, closing M17/M18 — see AC-2's note and Task 5.)

## Acceptance Criteria

Each AC lists its landing PR. All are independently verifiable (a test, a CI job, or a named
command), never "looks good." Where an AC's *content* changed vs. v1, it is marked **CHANGED**
with a one-line reason; unmarked ACs are unchanged.

- **AC-1 (PR1):** `import xtrax.export` succeeds in a fresh interpreter with only the `dev`
  and `io` extras installed (no `export` extra, no `iree`, no `wgpu`).
- **AC-2 (PR1, CHANGED — independent parity oracle, B3/B4.1):**
  `export_pipeline(fn, plan, abstract_inputs, concrete_inputs, targets=(NATIVE, WASM32),
  reference_fn=...)` on a fixture pipeline (single `SafeMap` or `Scan` axis, fuse-only boundary)
  returns a `dict[str, ExportResult]` keyed `"native"`/`"wasm32"`; `result["native"].verification_level
  is VerificationLevel.EXECUTED` and `.verified is True`; `result["wasm32"].verification_level
  is VerificationLevel.CODEGEN_ONLY` and `.verified is False` (CODEGEN_ONLY never claims
  verification — see the `.verified` semantics note under Task 5). `reference_fn` must be an
  independently-computed oracle (e.g. `lambda xs: jnp.stack([step_fn(x) for x in xs])` for a
  `SafeMap`/`Vmap` fixture) — a test that passes `jax.jit(build_traceable_callable(...))` as
  `reference_fn` is testing nothing and must be rejected in review. **Note (closes M17/M18):**
  `export_pipeline` is all-or-nothing across `targets=` — add a test asserting that when a later
  target in the tuple raises, no partial `dict` is returned (the whole call raises).
- **AC-3 (PR1):** a plan whose any `AxisDecision`'s `AxisBoundary` has a non-`None` `tap` or
  `sink` raises before any `jax.jit`/compile call — verified by a call-counting fake compiler
  recording zero invocations. **Unchanged, but see AC-14a/AC-14b below**: this rule is *global*
  to `export_pipeline`/`validate_export_safe`; AC-14a's ordering certification deliberately
  operates one layer below this gate (directly against the composer/executor, never through
  `export_pipeline`) so it does not contradict AC-3.
- **AC-4 (PR1, CHANGED — DedupGather restored, Blocker 7/M6/C12):** a plan containing a `Bucket`
  or `WhileCarry` strategy raises a named error identifying the unsupported strategy and pointing
  at `Vmap`/`SafeMap`/`Scan`/`DedupGather`. (v1 also listed `DedupGather` here as rejected; it is
  now routed and exportable — see Task 3 and the Non-goals changelog entry.)
- **AC-5 (PR1):** `xtrax.export` is added to both existing import-linter `source_modules`
  lists (`xtrax.eda`-forbidden, `xtrax.devtools`-forbidden) and a new forbidden contract
  rejects `xtrax.tiling`/`xtrax.stages`/`xtrax.transforms` importing `xtrax.export`; `uv run
  --extra dev lint-imports` passes.
- **AC-6 (PR1, CHANGED — `export_pipeline` moved out of `__init__.py`, Blocker 4/B14/C5):**
  `xtrax/export/` code is inside the `tier1_core` coverage surface (`coverage_packages =
  ["xtrax"]`, no new omit entry) and `just audit-coverage-tier1` still clears 90% line / 80%
  branch **without** the `export` extra installed. Because `coverage_omit` already contains a
  blanket `*/__init__.py` entry (`distribution/coverage_dag.toml:26-31`), the actual pipeline
  logic must live in `src/xtrax/export/pipeline.py` (new, PR1 Task 1/5), with `__init__.py`
  reduced to a thin re-export shim — otherwise this AC is measuring nothing, which is exactly
  what v1 did.
- **AC-7 (PR1):** a new `export-toolchain-tests` CI job (installs `--extra export` **and** the
  Vulkan ICD, see AC-8/Task 8) runs `tests/export/` for real against IREE's `llvm-cpu` backend;
  zero tests in that run are skipped by `pytest.importorskip`.
- **AC-8 (PR2, CHANGED — split by target, V1/B9):** `Target.VULKAN_SPIRV` compiles the same
  fixture kernel used by AC-2 (not a synthetic one) and `xtrax.export.spirv.validate_webgpu()`
  returns `valid=True` against a real CPU (`llvmpipe`) `wgpu` adapter — i.e. `wgpu.create_shader_module`
  on the extracted `.spv` bytes succeeds, with no GPU present. `Target.METAL_SPIRV` is
  `CODEGEN_ONLY` (V1: it dumps Metal Shading Language source, not SPIR-V — magic `0x636e6923`
  "#inc", not `0x07230203`) and is **not** subject to this AC at all; it is only required to
  compile (see AC-2-style codegen-only assertions). Disposition (B9): this spec validates shader
  *module* acceptance only — `create_shader_module` — never pipeline creation, so
  `maxStorageBuffersPerShaderStage` (capped at 8 on WebGPU) and IREE's vulkan-HAL push-constant
  usage are never exercised or enforced by this AC (see Non-goals).
- **AC-8b (PR2, NEW — V2/B10 negative case):** `validate_webgpu()` returns `valid=False` (never
  raises to its caller) for exactly the two orchestrator-measured fixtures: (1) valid SPIR-V
  magic with a garbage instruction body (`GPUValidationError: "unknown instruction 44510"`
  internally), and (2) non-SPIR-V random bytes (`ValueError: "Given shader data does not look
  like a SpirV module"` internally). Both cases populate `SpirvValidationResult.error` with the
  caught exception's message text.
- **AC-9 (PR2):** a plan whose abstract inputs include an `f64` or `bf16` leaf, exported with
  `targets=(VULKAN_SPIRV,)`, raises `DtypeNotSupportedError` at `export_pipeline` call time
  (before `jax.export.export`), naming the offending dtype and the target.
- **AC-9b (PR2, NEW — closes C4/m9):** the same rule applies to a **closure-held** leaf — e.g. an
  Equinox module passed as `fn` whose weights include a bf16/f64 array — even though that leaf
  never appears in `abstract_inputs`. `validate_export_safe` scans `fn`'s closure-reachable
  pytree leaves (generalizing `find_bcoo_leaves`'s `tree_flatten_with_path` scan to dtype
  checking) and raises `DtypeNotSupportedError` naming the leaf's keypath and the target.
- **AC-10 (PR2):** the same `bf16` leaf, loaded via `load_hf_weights(..., target=VULKAN_SPIRV)`,
  is upcast to `f32` before tracing; `ExportResult.diagnostics` records the cast (one string per
  cast leaf — the spike's `dtypes_cast[:2]` truncation is a bug, not a spec, and must not be
  carried over); the `native` target's EXECUTED parity check runs against the **same cast
  weights**, not the original bf16 values. **Clarification (M10/M11):** `export_pipeline` never
  calls `load_hf_weights` internally — this is enforced by the *caller* building both `fn` and
  `reference_fn` (AC-2) from the same `load_hf_weights(...)` return value, so the cast flows into
  both sides of the parity check by construction, not by any wiring inside `export_pipeline`.
- **AC-11 (PR2, CHANGED — beartype-safe example, M13):** requesting `f16` on a SPIR-V target
  without `request_features=frozenset({"shader-f16"})` raises `DtypeNotSupportedError` naming the
  missing feature; passing it succeeds against a fake adapter reporting the feature available.
  (v1's prose example used a `set` literal `{"shader-f16"}` against a `frozenset[str]`-typed
  parameter under a beartype hook this spec itself installs — that call would raise
  `BeartypeCallHintParamViolation` before reaching the dtype logic at all.)
- **AC-12 (PR2):** `ExportResult.size_bytes` for every target is checked in
  `tests/export/test_size_budget.py` against a recorded budget; a regression beyond budget
  fails the test with actual-vs-budget numbers in the message.
- **AC-13 (PR2):** `tests/export/` still passes in full with neither `iree` nor `wgpu` importable
  (re-verifies AC-1/AC-2 after PR2 lands).
- **AC-14a (PR3, NEW — split from v1's AC-14, Blocker 1/B1/B2/C1):** a 2-axis `BatchPlan` (outer
  `Vmap`-strategy axis, inner `Scan`-strategy axis, ordered `Sink` on the inner axis whose sunk
  value depends on the outer lane) is composed via the **composer/executor layer directly** —
  never through `export_pipeline`, never producing an `ExportResult`, and no `.verified` claim is
  made. The batched-shape recipe (Task 20) bakes the outer axis into a bare `jax.lax.scan` call
  (not `execute_scan_axis` — see Blocker 8 below); a test-double sink records host-call order
  matching the expected `(lane, step)` sequence, proving no literal `jax.vmap` ran internally.
  This certifies ordering only. A `Sink` is legal here precisely because this path never crosses
  the export boundary that AC-3 protects.
- **AC-14b (PR3, NEW — split from v1's AC-14):** the *same 2-axis shape*, but composed as a
  separately-built, **sink-free** variant (e.g. a `Fuse` in place of the `Sink`, or no boundary at
  all) IS run through `export_pipeline` using `targets=(NATIVE,)`, and produces a `native`
  `ExportResult` with `.verification_level is VerificationLevel.EXECUTED` and `.verified is True`
  (parity against an independent `reference_fn`, per AC-2). This is what actually exercises
  multi-axis codegen + parity; AC-14a alone never reaches `jax.export.export` at all.
- **AC-15 (PR3):** the same plan shape, constructed so the composer would have to nest a
  literal `jax.vmap` around the lane-dependent ordered inner axis, raises a composer-level
  error whose message contains the certified guidance text (matches the existing
  `ExecutorError`'s `"Vmap axis's `fn`"` substring) instead of a raw `ValueError` or silent
  misordering. (Unchanged from v1 — this test also operates at the composer/executor layer, like
  AC-14a, and never claims an `ExportResult`, so it was never actually in tension with AC-3.)
- **AC-16 (PR2, NEW — closes M12):** parity between the *original* bf16 model's forward pass
  (plain JAX, bf16, no export) and the exported-and-native-executed f32 artifact is checked with a
  bf16-appropriate tolerance (e.g. `atol=1e-2`, not the `f32` default `atol=1e-5`) and must pass;
  this is a *different* comparison from AC-10's cast-weight parity (which is f32-vs-f32 by
  construction and therefore cannot see bf16-precision divergence at all — bf16->f32 upcasting is
  exact, so an f32-vs-f32 check alone can never surface the original model's intended numerics
  drifting from its bf16 baseline).

## Design decisions carried over verbatim from the approved design

(Restated here for fixer convenience; do not re-derive or re-litigate. Items marked **CHANGED**
were altered by this revision — see the Changelog.)

- Package layout: `src/xtrax/export/{__init__,targets,safety,composer,compile,spirv,pipeline,parity}.py`.
  **CHANGED:** `pipeline.py` added (Blocker 4) — `ExportResult`/`export_pipeline` live there, not
  in `__init__.py`.
- Four targets: `native` (llvm-cpu, `--iree-llvmcpu-target-cpu=host`, EXECUTED), `wasm32`
  (llvm-cpu, `wasm32-unknown-emscripten` triple, `+simd128,+atomics,+bulk-memory`,
  CODEGEN_ONLY), `vulkan-spirv` (VALIDATED). **CHANGED:** `metal-spirv` is now **CODEGEN_ONLY**,
  not VALIDATED (V1 — it dumps MSL, not SPIR-V).
- Verification depth = option A: `wgpu`/naga validation is the WebGPU gate, **now for
  `vulkan-spirv` only** (CHANGED, V1); numerics are verified against JAX only through IREE's
  native runtime, and — CHANGED per B3 — that verification's *reference* value must be an
  independently-computed oracle, never the composed-and-jitted callable itself. No SPIR-V kernel
  is ever executed under `wgpu` in this spec.
- Dtype table: `f32`/`i32`/`bool` on all four targets; `f16` on SPIR-V targets only (in practice,
  `vulkan-spirv` only, since `metal-spirv` no longer participates in the SPIR-V/wgpu story) behind
  the `shader-f16` feature gate; `bf16` never valid on SPIR-V targets (cast to `f32` at
  weight-load time — HF ships bf16); `f64` never valid on SPIR-V targets (rejected, never
  auto-cast). **CHANGED (AC-9b):** this rejection now also applies to closure-held leaves, not
  only `abstract_inputs` leaves.
- Composer promotes single-axis routing now (PR1/PR2) — **CHANGED (Blocker 7):** routes
  `Vmap`/`SafeMap`/`Scan`/`DedupGather` (DedupGather via host-computed static indices, matching
  the spike exactly); refuses only `Bucket`/`WhileCarry`. Multi-axis (PR3) uses the certified
  batched-shape recipe — **CHANGED (Blocker 8):** bake the outer axis directly into a bare
  `jax.lax.scan` call with the boundary's sink applied inline (matching
  `TestBatchedShapeVmapOfScanPreservesOrder`'s actual code, which never calls
  `execute_scan_axis`) — and refuses lane-dependent literal-`vmap` nesting.
- Packaging: `xtrax.export` ships in the wheel; IREE + `wgpu` go in a new `export` extra;
  importing without the extra raises a clear install message; `xtrax.export` added to both
  existing forbidden import-linter contracts; nothing in `tiling`/`stages`/`transforms` may
  import `xtrax.export`. **CHANGED (Blocker 5):** the extra's exact package names are now
  `iree-base-compiler`/`iree-base-runtime` (not `iree-compiler`/`iree-runtime`), plus
  `huggingface_hub`/`safetensors` (both were missing entirely in v1).
- Testing: full suite passes with neither `iree` nor `wgpu` installed via `sys.modules`
  injection (spike precedent, 33 tests, no network); real-toolchain tests skip when the
  toolchain is absent; CI validates codegen + naga on every PR with no GPU (**CHANGED**: this
  requires the Vulkan ICD driver, Blocker 3 — a bare `ubuntu-latest` cannot get an `llvmpipe`
  adapter at all); a recorded bundle-size budget guards against vmfb regressions.
- Rollout: PR1 (native + wasm32 + import-linter + extra), PR2 (SPIR-V + wgpu/naga gate), PR3
  (multi-axis composer). Unchanged. **New (M16):** PR1 also retires the spike (Task 9b).

## Public API signatures

```python
# src/xtrax/export/targets.py  (NEW)
class VerificationLevel(str, Enum):
    EXECUTED = "executed"        # numerics verified vs an independent JAX oracle, via IREE's
                                  # native runtime. Bounds LOWERING fidelity (XLA vs IREE), not
                                  # composition correctness -- see the Overview caveat (B3).
    CODEGEN_ONLY = "codegen_only"  # compiles; never executed or otherwise validated. `.verified`
                                    # is unconditionally False for this level (M17/M18).
    VALIDATED = "validated"      # SPIR-V accepted by wgpu/naga; never executed. `.verified`
                                  # mirrors spirv_validation.valid.

@dataclass(frozen=True)
class Target:
    name: str                                   # "native" | "wasm32" | "vulkan-spirv" | "metal-spirv"
    iree_backend: str                           # "llvm-cpu" | "vulkan-spirv" | "metal-spirv"
    verification_level: VerificationLevel
    supported_dtypes: frozenset[str]            # dtype names as in xtrax.cli.shapes._DTYPE_MAP style: "f32","i32","bool","f64","bf16","f16"
    optional_dtypes: frozenset[str] = frozenset()          # e.g. {"f16"} for SPIR-V targets
    optional_dtype_features: Mapping[str, str] = field(default_factory=dict)  # {"f16": "shader-f16"}
    extra_compiler_flags: tuple[str, ...] = ()

NATIVE: Target        # PR1
WASM32: Target         # PR1
VULKAN_SPIRV: Target   # PR2, verification_level=VALIDATED
METAL_SPIRV: Target    # PR2, verification_level=CODEGEN_ONLY  -- CHANGED from v1 (was VALIDATED)
ALL_TARGETS: tuple[Target, ...]

# src/xtrax/export/safety.py  (NEW)
class ExportSafetyError(Exception):
    """Base for xtrax.export's own plan-time gate failures (dtype/feature/closure gating).
    Distinct from, and does not replace, xtrax.stages.topology.PlanTopologyError,
    which both entry points below let propagate unchanged."""

class DtypeNotSupportedError(ExportSafetyError):
    """A leaf's dtype is not in target.supported_dtypes, and (if applicable) not unlocked
    by a requested optional feature. Fires for abstract_inputs leaves (AC-9) AND for
    closure-reachable leaves of `fn` itself (AC-9b, closes C4/m9)."""

@dataclass(frozen=True)
class ExportBlocker:
    """One reason a plan/leaf cannot cross the export boundary. Promoted verbatim from the
    spike's ExportBlocker (Appendix A)."""
    axis: str
    rule: str
    detail: str

def check_export_safety(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    abstract_inputs: Sequence[Any],
    fn: Callable[..., Any],
    target: Target,
    *,
    request_features: frozenset[str] = frozenset(),
) -> list[ExportBlocker]: ...
    # List-returning twin of validate_export_safe, promoted from the spike's
    # check_plan_export_safety (closes M5's "collapsed two entry points" complaint).
    # Does NOT call xtrax.stages.topology.validate_plan_topology -- that one always raises
    # PlanTopologyError directly and is never converted into an ExportBlocker list, by design
    # (M5: PlanTopologyError propagates unwrapped from BOTH entry points below).

def validate_export_safe(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    abstract_inputs: Sequence[Any],
    fn: Callable[..., Any],
    target: Target,
    *,
    request_features: frozenset[str] = frozenset(),
) -> None: ...
    # 1. xtrax.stages.topology.validate_plan_topology(decisions, axis_boundaries, export_safe=True)
    #    -- PlanTopologyError propagates unchanged (rule 3: fuse-only; rule 4: strategy allow-list)
    # 2. per-leaf dtype check against target.supported_dtypes / optional_dtypes+features,
    #    over BOTH abstract_inputs leaves (AC-9) AND fn's closure-reachable leaves (AC-9b) --
    #    raises DtypeNotSupportedError naming every ExportBlocker from check_export_safety(),
    #    matching the spike's assert_plan_export_safe's multi-blocker reporting.

# src/xtrax/stages/topology.py  (MODIFIED — additive, default-False kwarg)
def validate_plan_topology(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    *,
    export_safe: bool = False,
) -> None: ...
    # export_safe=True adds, both raising the EXISTING PlanTopologyError:
    #   Rule 3: every axis's tap AND sink must be None (fuse-only)
    #   Rule 4: strategy must be Vmap, SafeMap, Scan, or DedupGather (not Bucket/WhileCarry)
    #           -- CHANGED: DedupGather moved from the reject-list to the allow-list (Blocker 7)

# src/xtrax/export/composer.py  (promoted)
class UnsupportedStrategyError(Exception):
    """Bucket / WhileCarry is not routed by this composer.
    CHANGED: DedupGather removed from this error's scope -- it is now routed (Blocker 7)."""

class MultiAxisCompositionError(Exception):  # PR3
    """Composing this plan would require nesting a literal jax.vmap around an axis
    whose ordered Tap/Sink depends on the outer lane. Wraps the underlying
    xtrax.stages.executor.ExecutorError; see its 'Nesting: vmap-of-scan' docstring."""

def build_traceable_callable(
    fn: Callable[..., Any],
    plan: BatchPlan,
    axis_boundaries: Mapping[str, AxisBoundary] | None = None,
    *,
    scan_init: Any = None,   # CHANGED (Blocker 2): restored, matches spike's compose_exportable.
                              # Falls back to the axis's Scan.init if not given (spike precedent);
                              # only consulted for a Scan-strategy axis.
) -> Callable[..., Any]: ...
    # Design note (B6, documented not "fixed" -- it's a real, decidable precedence, not an
    # ambiguity): `fn` is ALWAYS the transition used for a Scan axis. `Scan.transition` (the
    # strategy field) is read only by the unrelated eager xtrax.tiling.dispatch.axis_dispatch
    # path and is NEVER consulted here. A caller who sets both Scan(transition=g) and passes
    # fn=h to build_traceable_callable/export_pipeline gets `h` exported, which can silently
    # differ from what axis_dispatch would eagerly run for the same plan object -- this is a
    # real footgun to document in docs/api/export.md (Task 9/19), not a bug to "fix" by
    # inventing a merge rule neither spike entry point specifies.
    # `Scan.ordered_sinks` is unrelated to AxisBoundary.tap/.sink and is not read anywhere in
    # this module or in validate_export_safe's Rule 3 -- an inert field for export purposes.

# src/xtrax/export/compile.py  (promoted, generalised over targets)
class CompileError(Exception):
    """Wraps an IREE compiler failure with target name, backend, and stderr."""

@dataclass(frozen=True)
class CompileResult:
    path: Path                    # CHANGED (Blocker 6): real file, matching the only proven
                                   # executor, run_native_vmfb(vmfb_path: Path) via VmModule.mmap.
    size_bytes: int
    spirv_bytes: dict[str, bytes] | None   # CHANGED (V3): mapping executable_name -> bytes,
                                            # populated only for vulkan-spirv (VALIDATED); always
                                            # None for metal-spirv (CODEGEN_ONLY, dumps .metal not
                                            # .spv -- see V1).
    downgraded_stablehlo: bool     # CHANGED (M3/M4): restored -- StableHLO portable-artifact
                                    # retry-on-failure flag, dropped in v1's merge.
    stderr: str

def compile_for_target(
    mlir_text: str,
    target: Target,
    *,
    out_path: Path | None = None,   # defaults to a fresh tempfile path when None; spike's
                                     # compile_stablehlo required an explicit out_path -- this
                                     # keeps that real-file requirement while adding the
                                     # Target-object-based dispatch that generalizes it.
) -> CompileResult: ...
    # lazy `import iree.compiler`; ImportError -> CompileError("... pip install xtrax[export]")
    # retries once through a version-pinned StableHLO portable artifact on first failure
    # (promoted from the spike's _downgrade_to_portable; sets downgraded_stablehlo=True on retry)

def run_native_vmfb(vmfb_path: Path, *args: Any, function: str = "main") -> Any: ...
    # promoted verbatim from the spike (Appendix A) -- the ONLY execution path that exists.
    # Native only; wasm32/vulkan-spirv/metal-spirv vmfbs are never executed by this function
    # or anywhere else in this spec.

# src/xtrax/export/spirv.py  (stub created PR1 Task 1 -- C8; extended PR2 Task 11)
@dataclass(frozen=True)
class SpirvValidationResult:
    valid: bool
    adapter_type: str
    backend: str
    device_name: str
    error: str | None

class WebGPUValidationError(Exception):
    """Raised for INFRASTRUCTURE failures in validate_webgpu's own setup (e.g. no adapter
    found at all, or the adapter cannot satisfy a requested device feature) -- distinct from
    a naga-rejected shader, which never raises (see below); it is reported as
    SpirvValidationResult(valid=False, error=...) instead."""

def validate_webgpu(
    spirv_bytes: bytes,
    *,
    request_features: frozenset[str] = frozenset(),
) -> SpirvValidationResult: ...
    # lazy `import wgpu`; ImportError -> raise ImportError("... pip install xtrax[export]")
    # CHANGED (B11): wgpu.gpu.request_adapter_sync(...) then
    # adapter.request_device_sync(required_features=request_features) -- there is no
    # module-level wgpu.request_adapter()/request_device() in wgpu-py 0.32.
    # CHANGED (V2): wraps create_shader_module(code=spirv_bytes) in try/except catching BOTH
    # GPUValidationError (valid magic, invalid body) and ValueError (non-SPIR-V bytes) and
    # converts either to valid=False, error=str(exc) -- it does NOT naturally "return" this,
    # it must catch. The exact import path for GPUValidationError was not independently
    # re-verified in this revision (see Changelog "named but not resolved"); confirm against a
    # real wgpu-py==0.32.* install before writing the except clause.

def validate_all_webgpu(
    spirv_dumps: Mapping[str, bytes],
    *,
    request_features: frozenset[str] = frozenset(),
) -> SpirvValidationResult: ...
    # NEW (closes V3's "make valid the conjunction" requirement): validates every entry in a
    # multi-executable dump, returns the aggregate result (valid = all(...) across entries;
    # error = joined messages from any failing entry, or None if all pass).

# src/xtrax/export/parity.py  (promoted)
@dataclass(frozen=True)
class ParityResult:
    passed: bool
    max_abs_diff: float
    atol: float                    # CHANGED (M4): default reverted 1e-6 -> 1e-5 (spike default,
                                    # no justification was ever given for tightening it)
    rtol: float                    # CHANGED (M4): restored -- dropped in v1's merge
    shape_expected: tuple[int, ...]  # CHANGED (M4): restored -- the shape guard is deliberate
    shape_actual: tuple[int, ...]    # ("a silently broadcast comparison is how a real
                                      #   regression gets missed", spike parity.py:47-49)

def compare(
    expected: object,
    actual: object,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ParityResult: ...
    # promoted verbatim from the spike (Appendix A) -- the comparison primitive.
    # Shape mismatch short-circuits to a failure rather than broadcasting.

def verify_native_parity(
    expected: Any,               # CHANGED (B3/B4.1): an INDEPENDENTLY-computed reference value
                                  # (e.g. export_pipeline's caller-supplied reference_fn(inputs)),
                                  # NEVER jax.jit(build_traceable_callable(...))(inputs) -- that
                                  # was v1's bug: comparing the composed callable against itself
                                  # under two backends detects lowering divergence only, not
                                  # composition errors (wrong nesting, dropped boundary, mis-
                                  # shaped carry), because both sides change identically.
    vmfb_path: Path,             # CHANGED (Blocker 6): Path, not bytes -- run_native_vmfb needs
                                  # a real mmap-able file.
    concrete_inputs: Sequence[Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    function: str = "main",
) -> ParityResult: ...
    # actual = run_native_vmfb(vmfb_path, *concrete_inputs, function=function)
    # return compare(expected, actual, atol=atol, rtol=rtol)

# src/xtrax/export/hf_weights.py  (promoted, PR2)
@dataclass(frozen=True)
class WeightReport:                # CHANGED (M4): restored, promoted verbatim from the spike
    source: str                    # (Appendix A) as its own record -- not folded/lost.
    tensors_seen: int
    tensors_used: int
    dtypes_cast: tuple[str, ...]   # CHANGED (M10/M11): NO truncation -- one string per cast
                                    # leaf. The spike's `[:2]` slice (hf_weights.py:140) is a bug
                                    # in the spike itself and must not be carried into the
                                    # promoted version.

@dataclass(frozen=True)
class LoadedWeights:
    tree: Any                      # pytree of jax.Array -- narrow, TinyMLP-shaped (Non-goals:
                                    # genericizing beyond this is explicitly out of scope)
    report: WeightReport
    diagnostics: tuple[str, ...]   # e.g. "cast layer.0.weight bf16 -> f32 for vulkan-spirv"

def load_hf_weights(model_id: str, *, target: Target, **kwargs: Any) -> LoadedWeights: ...
    # promotes the spike's mlp_from_hf (rename -- Appendix A) with its narrow filename/
    # in_dim/hidden/out_dim/dtype signature preserved via **kwargs, not dropped (M10/M11).
    # Casts bf16 leaves to f32 when the target's supported_dtypes+optional_dtypes lack "bf16"
    # (true for vulkan-spirv). Never casts f64 -- that is rejected loudly by validate_export_safe
    # instead (AC-9), never silently downcast. Caller-driven wiring (M10/M11, AC-10): build BOTH
    # `fn` for export_pipeline AND `reference_fn` from this SAME LoadedWeights.tree instance, so
    # the cast is reflected on both sides of the parity check by construction.

# src/xtrax/export/pipeline.py  (NEW -- was src/xtrax/export/__init__.py in v1; moved per
# Blocker 4/B14/C5, since coverage_dag.toml's coverage_omit blankets every "*/__init__.py")
@dataclass(frozen=True)
class ExportResult:
    target: Target
    vmfb_bytes: bytes              # read once from CompileResult.path -- convenience only;
                                    # execution/parity always go through .path internally.
    size_bytes: int
    spirv_bytes: dict[str, bytes] | None   # CHANGED (V3): mapping, not a single blob.
    verification_level: VerificationLevel
    verified: bool                 # CHANGED (M17/M18), defined per level:
                                    #   EXECUTED     -> parity.passed
                                    #   VALIDATED    -> spirv_validation.valid
                                    #   CODEGEN_ONLY -> ALWAYS False (nothing was verified beyond
                                    #                   successful compilation -- inspect
                                    #                   verification_level to distinguish this
                                    #                   from a genuine EXECUTED/VALIDATED failure)
    parity: ParityResult | None              # EXECUTED targets only
    spirv_validation: SpirvValidationResult | None  # VALIDATED targets only (vulkan-spirv);
                                                      # always None for metal-spirv (CODEGEN_ONLY)
    diagnostics: tuple[str, ...]

def export_pipeline(
    fn: Callable[..., Any],
    plan: BatchPlan,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any] | None = None,
    *,
    axis_boundaries: Mapping[str, AxisBoundary] | None = None,
    targets: Sequence[Target] = (NATIVE, WASM32),
    request_features: frozenset[str] = frozenset(),
    scan_init: Any = None,           # CHANGED (Blocker 2): restored, threaded to
                                      # build_traceable_callable.
    reference_fn: Callable[[Sequence[Any]], Any] | None = None,  # CHANGED (B3): required
                                      # whenever any target is EXECUTED. Must be an
                                      # independently-computed oracle over concrete_inputs --
                                      # see verify_native_parity's docstring for what this
                                      # excludes.
) -> dict[str, ExportResult]: ...
    # ValueError up front if concrete_inputs is None and any target is EXECUTED
    # ValueError up front if reference_fn is None and any target is EXECUTED (NEW, B3)
    # All-or-nothing across targets (NEW, M17/M18): targets are compiled/validated in the given
    # order; the first target-level exception (CompileError, ExportSafetyError,
    # DtypeNotSupportedError, WebGPUValidationError) propagates immediately and aborts the whole
    # call. No partial dict[str, ExportResult] is ever returned.

# src/xtrax/export/__init__.py  (thin re-export shim, per-subpackage-init convention)
# Re-exports the public names above (Target/VerificationLevel/ExportResult/export_pipeline/
# etc.) with an explicit __all__. Contains NO logic of its own -- that is the whole point of
# Blocker 4's fix: coverage_dag.toml's blanket "*/__init__.py" omit rule can safely skip this
# file precisely because nothing load-bearing lives in it anymore.
```

## Fixer Tasks

### PR1 — `xtrax.export` at spike parity: native + wasm32, import-linter, extra

#### Task 1: Package skeleton + Target registry (native, wasm32) + spirv.py stub
Create `src/xtrax/export/__init__.py`, `targets.py`, and `spirv.py` (stub — **CHANGED, closes
C8**: this file exists from PR1 so `ExportResult` can reference `SpirvValidationResult` in its
frozen-dataclass field before PR2 adds any validation logic). Define `VerificationLevel`,
`Target`, `NATIVE`, `WASM32`, `ALL_TARGETS` per the signatures above; `spirv.py` contains only
`SpirvValidationResult` in PR1. `__init__.py` re-exports the public names with an explicit
`__all__` (per-subpackage-init convention, e.g. `checkpoint/__init__.py`) and contains no other
logic.
**Files**: `src/xtrax/export/__init__.py`, `src/xtrax/export/targets.py`,
`src/xtrax/export/spirv.py` (create)
**Gate**: `uv run python -c "from xtrax.export.targets import NATIVE, WASM32, VerificationLevel; from xtrax.export.spirv import SpirvValidationResult; assert NATIVE.verification_level is VerificationLevel.EXECUTED"`
**Scope estimate**: ~90 LOC
**Verifies**: AC-1 (partial)

#### Task 2: Export-safety gate
Extend `src/xtrax/stages/topology.py::validate_plan_topology` with the additive
`export_safe: bool = False` kwarg and Rules 3/4 (both `PlanTopologyError`). **CHANGED (Blocker
7)**: Rule 4's allow-list is `Vmap`/`SafeMap`/`Scan`/`DedupGather` (not just the first three — v1
wrongly excluded `DedupGather`). Add `src/xtrax/export/safety.py` with `ExportSafetyError`,
`DtypeNotSupportedError`, `ExportBlocker`, `check_export_safety()` (list-returning, **NEW**, closes
M5), and `validate_export_safe()` (raising, per the signature above — dtype check against
`target.supported_dtypes` only in PR1, over `abstract_inputs` leaves only; the closure-leaf scan
(AC-9b) and `optional_dtypes`/`request_features` plumbing land in PR2, but define both parameters
now so PR2 doesn't change either call signature).
**Files**: `src/xtrax/stages/topology.py` (modify), `src/xtrax/export/safety.py` (create),
`tests/stages/test_topology.py` (modify — add export_safe=True cases, including a DedupGather-
passes case)
**Gate**: `uv run pytest tests/stages/test_topology.py -q`
**Scope estimate**: ~140 LOC + tests
**Verifies**: AC-3, AC-4

#### Task 3: Composer — single-axis promotion (Vmap/SafeMap/Scan/DedupGather)
Promote the spike's composer into `src/xtrax/export/composer.py`:
`build_traceable_callable()` iterating `plan.decisions`, routing `Vmap`/`SafeMap` →
`xtrax.stages.executor.execute_map_axis`, `Scan` → `execute_scan_axis` (with `scan_init` falling
back to `Scan.init`, per the promoted signature's design note — **CHANGED, Blocker 2**),
**and `DedupGather` → `xtrax.tiling.dispatch.axis_dispatch(strategy, fn, xs)`** (**CHANGED,
Blocker 7** — matches the spike's `compose_single_axis` exactly: the dedup/gather indices are
host-computed NumPy arrays baked in as closure constants at their static `k_bucket` shape, so no
runtime-index contract is needed, contrary to v1's premise for excluding it). `Bucket`/`WhileCarry`
raise `UnsupportedStrategyError` naming the strategy and pointing at the four supported ones.
**Files**: `src/xtrax/export/composer.py` (create)
**Gate**: unit test asserting `UnsupportedStrategyError` for `Bucket`/`WhileCarry`, plus a
positive single-`SafeMap`-axis composition round-trip AND a positive `DedupGather` round-trip
(fixture: small `unique_indices`/`index_map`, verify `dedup_fn`→map→`gather_fn` order).
**Scope estimate**: ~180 LOC
**Verifies**: AC-4

#### Task 4: Compile + parity (native, wasm32)
Promote `compile_iree.py` → `src/xtrax/export/compile.py`: `compile_for_target()`,
`CompileResult` (**CHANGED — Blocker 6/M3/M4**: `path: Path` not `vmfb_bytes: bytes`;
`downgraded_stablehlo: bool` restored), `CompileError`, generalised so `iree_backend`+
`extra_compiler_flags` come from `Target` rather than being hardcoded to `"native"`/`"wasm32"`
string literals (the spike hardcoded exactly two string targets — verify the generalization
against `.praxia/spike_snapshot/scripts/iree_export_spike/compile_iree.py` directly, it is
readable now). Include the portable-artifact-downgrade retry path (promoted from
`_downgrade_to_portable`) in the gate. Promote `parity.py` → `src/xtrax/export/parity.py`:
`compare()` (verbatim), `ParityResult` (**CHANGED — M4**: `rtol`/`shape_expected`/`shape_actual`
restored, default `atol` reverted to `1e-5`), and **NEW** `verify_native_parity()` (per the
signature above — takes an independently-supplied `expected`, never re-derives it from the
composed callable; **CHANGED, closes B3/B4.1**).
**Files**: `src/xtrax/export/compile.py`, `src/xtrax/export/parity.py` (create)
**Gate**: `uv run pytest tests/export/test_compile_native_wasm32.py -q` (fake-injected `iree`,
no real toolchain) — asserts `wasm32` never calls the parity path, and asserts the
portable-artifact downgrade path sets `downgraded_stablehlo=True` on a first-compile failure.
**Scope estimate**: ~220 LOC
**Verifies**: AC-2

#### Task 5: Wire `export_pipeline()` end-to-end (moved to `pipeline.py`)
Implement `ExportResult` and `export_pipeline()` in **`src/xtrax/export/pipeline.py`**
(**CHANGED — Blocker 4**: not `__init__.py`), composing Tasks 1-4:
`build_traceable_callable` (with `scan_init`) → `validate_export_safe` per target →
`jax.export.export(jax.jit(callable))(*abstract_inputs)` → `compile_for_target` → (`EXECUTED`
only) `verify_native_parity(reference_fn(concrete_inputs), compile_result.path,
concrete_inputs, ...)`. Set `ExportResult.verified` per the per-level rule in the signatures
section above (**CHANGED, M17/M18**: unconditionally `False` for `CODEGEN_ONLY`). Enforce
all-or-nothing across `targets=` (no partial dict on a mid-loop exception — **NEW, M17/M18**).
Raise `ValueError` up front if `reference_fn is None` and any target is `EXECUTED` (**NEW, B3**).
`__init__.py` becomes a thin re-export of `pipeline.py`'s public names plus the other modules'.
**Files**: `src/xtrax/export/pipeline.py` (create), `src/xtrax/export/__init__.py` (modify —
re-export only)
**Gate**: `uv run pytest tests/export/test_pipeline_native_wasm32.py -q` — must include a test
using a real independent `reference_fn` (not `jax.jit(build_traceable_callable(...))`) and a test
asserting all-or-nothing behavior when a second target raises.
**Scope estimate**: ~150 LOC
**Verifies**: AC-2, AC-3

#### Task 6: Packaging + import-linter + isolation wiring
Add `export = ["iree-base-compiler>=3.11,<4", "iree-base-runtime>=3.11,<4", "huggingface_hub",
"safetensors"]` to `[project.optional-dependencies]` (**CHANGED — Blocker 5**: v1's
`iree-compiler`/`iree-runtime>=3.11,<4` names an unresolvable legacy/date-versioned distribution
that matches nothing on PyPI for this pin range; the spike's own `_require_compiler`/
`_load_safetensors` error strings confirm `iree-base-compiler`/`iree-base-runtime` and
`huggingface_hub`/`safetensors` are the real installs — verify exact version pins against what
the 260831 spike session actually used before merging). Add `"xtrax.export"` to both existing
`source_modules` lists (eda-forbidden at `pyproject.toml:130-146`, devtools-forbidden at
`:159-183`). Add a new forbidden contract: `source_modules = ["xtrax.tiling", "xtrax.stages",
"xtrax.transforms"]`, `forbidden_modules = ["xtrax.export"]`. Add `"xtrax.export"` to
`tests/test_import_isolation.py::_PACKAGES` and `tests/conftest.py::XTRAX_BEARTYPE_PACKAGES`.
**Files**: `pyproject.toml`, `tests/test_import_isolation.py`, `tests/conftest.py` (modify)
**Gate**: `uv run --extra dev lint-imports` and `uv run pytest tests/test_import_isolation.py -k export -q`
**Scope estimate**: ~40 LOC (config + list entries)
**Verifies**: AC-1, AC-5

#### Task 7: Fake-injection test suite (no toolchain)
Build `tests/export/` mirroring the spike's proven `sys.modules`-injection pattern for
`iree.compiler`/`iree.runtime` (33-test precedent) covering: `Target` registry contents,
`validate_export_safe`/`check_export_safety` (topology delegation + dtype rejection over both
`abstract_inputs` and closure leaves — AC-9b stub cases using a fake abstract input only in PR1,
real closure-leaf cases land in Task 14), `composer` routing + rejections (including the new
`DedupGather` positive case), `compile_for_target` (fake vmfb bytes written to a real tmp `Path`,
size accounting, downgrade retry), `verify_native_parity` (fake numeric arrays, independent
`expected` supplied directly — never derived from the callable under test), `export_pipeline`
end-to-end for native+wasm32 (including the all-or-nothing case), and the missing-extra
`ImportError` message.
**Files**: `tests/export/__init__.py`, `tests/export/conftest.py` (fake `iree` module
factory — **NOTE (M1)**: use function-scoped `monkeypatch.setitem`, never a session-scoped
`sys.modules` mutation, so this fake can never leak into a real-toolchain test in the same
pytest session), `tests/export/test_*.py` (create)
**Gate**: `uv run pytest tests/export/ -q` with `iree` **not** installed
**Scope estimate**: ~380 LOC
**Verifies**: AC-1, AC-2, AC-3, AC-4, AC-6

#### Task 8: CI — `export-toolchain-tests` job (with a real Vulkan ICD)
Add a new unconditional job to `.github/workflows/ci.yml` (pattern: `lint-format-type-test`,
not path-filtered — the approved design says "on every PR"): install `mesa-vulkan-drivers` and
`libvulkan1` via `apt-get` (**NEW, closes Blocker 3/B12/C2** — a bare `ubuntu-latest` runner has
no Vulkan ICD at all, so `wgpu` can never get an `llvmpipe` adapter without this; AC-8 is green
locally and silently red-or-skipped in CI without it), then `uv sync --extra dev --extra io
--extra export`, then `uv run pytest tests/export/ -q`. Real-toolchain test modules use
`pytest.importorskip("iree.compiler")` (the existing precedent at `tests/cli/test_export.py:76`
for `flatbuffers`) so the same files run for real here and skip in Task 7's job (which has no
toolchain installed at all, real or fake-shadowed — see Task 7's conftest note). **Gate must
assert 0 skips are visible in this job's own pytest summary line** (not just green exit — this is
the M2 zero-skip contradiction flagged in the Changelog as needing a live-CI measurement, not
just a text change).
**Files**: `.github/workflows/ci.yml` (modify)
**Gate**: CI run shows `export-toolchain-tests` green with 0 skips inside its own step log
**Scope estimate**: ~30 LOC
**Verifies**: AC-7

#### Task 9: Docs
Add `docs/api/export.md` (narrative page) **and** add `api/export` to `docs/index.md`'s
`toctree` list (**NEW, closes the real M15 gap** — `docs/index.md` currently lists
`api/overview`, `api/engine`, ... `api/eda` but no `api/export`). **Note (M15):** `just
audit-docs-build` does **not** build Sphinx — verified: it runs `uv run ruff check
scripts/audit_docs_plumbing.py tests/distribution/test_docs_plumbing.py`, `uv run pytest
tests/distribution/test_docs_plumbing.py`, and `uv run python scripts/audit_docs_plumbing.py`;
none of those invoke `sphinx-build`. `docs/conf.py` also sets `nitpicky = False` and has no
`warn_is_error`, so a missing cross-ref would not fail a real Sphinx build either even if one
ran. The actual gate for this task is therefore narrower than v1 implied: it verifies the
plumbing/toctree entry exists and passes `audit_docs_plumbing.py`'s checks, **not** that a
Sphinx build succeeds — if a real `sphinx-build -W` gate is wanted, that is separate,
unstarted work, not something this task can honestly claim.
**Files**: `docs/api/export.md` (create), `docs/index.md` (modify — toctree entry)
**Gate**: `just audit-docs-build`
**Scope estimate**: ~70 lines of prose + 1 toctree line

#### Task 9b: Retire the spike (NEW — closes M16)
Once Task 7's fake-injection suite and Task 8's real-toolchain job give `tests/export/`
equivalent-or-greater coverage of everything `tests/scripts/test_iree_export_spike.py` exercises:
delete `scripts/iree_export_spike/` and `tests/scripts/test_iree_export_spike.py`; remove any
`export-spike` `dependency-groups` entry from `pyproject.toml` if one has been added on the
integration branch (none exists on `main` today — it lives only on the unmerged
`origin/spike/iree-wasm-export` branch); close draft PR #111 without merging it, referencing this
PR1 as its supersession. This avoids shipping two divergent copies of the same logic with
different error-message conventions (`match="export-spike"` vs `"pip install xtrax[export]"`).
**Files**: `scripts/iree_export_spike/` (delete), `tests/scripts/test_iree_export_spike.py`
(delete), `pyproject.toml` (modify, if applicable)
**Gate**: `git log --oneline -- scripts/iree_export_spike` shows the deletion commit;
`uv run pytest tests/export/ -q` (Task 7+8) still passes with the spike tree gone
**Scope estimate**: ~0 net LOC (deletion) + PR description note

---

### PR2 — SPIR-V targets + the wgpu/naga validation gate

#### Task 10: `Target.VULKAN_SPIRV` / `Target.METAL_SPIRV`
Add both constants to `targets.py`. `VULKAN_SPIRV`: `verification_level=VALIDATED`,
`supported_dtypes=frozenset({"f32","i32","bool"})`, `optional_dtypes=frozenset({"f16"})`,
`optional_dtype_features={"f16": "shader-f16"}`. `METAL_SPIRV` (**CHANGED, V1**):
`verification_level=CODEGEN_ONLY`, same dtype table (the dtype restrictions come from WebGPU's
own numeric model, which the spec's Non-goals still forbid trying to execute under, independent
of whether SPIR-V validation applies) — metal-spirv never populates `spirv_bytes` or
`spirv_validation` (both always `None` for it).
**Files**: `src/xtrax/export/targets.py` (modify)
**Gate**: unit test asserting both are in `ALL_TARGETS`, `VULKAN_SPIRV.verification_level is
VerificationLevel.VALIDATED`, `METAL_SPIRV.verification_level is VerificationLevel.CODEGEN_ONLY`.
**Scope estimate**: ~30 LOC
**Verifies**: AC-8 (partial)

#### Task 11: `spirv.py` — extraction + wgpu/naga validation (vulkan-spirv only)
Extend `src/xtrax/export/spirv.py` (created as a stub in Task 1): `extract_spirv_from_vmfb()`
(recompile with `--iree-hal-dump-executable-binaries-to=<dir>` per verified fact #3, **filtering
the dump directory for files whose first 4 bytes are the SPIR-V magic `0x07230203`** — **NEW,
closes V1/V3**: `metal-spirv` dumps `.metal` MSL source with magic `0x636e6923`, which this
filter must reject rather than mistake for SPIR-V; a matmul+tanh pipeline was measured to fuse to
exactly ONE `.spv` for `vulkan-spirv`, so plurality is not automatic but the return type stays a
mapping regardless) — returns `dict[str, bytes]`. Add `validate_webgpu()` and `validate_all_webgpu()`
per the signatures above (`wgpu.gpu.request_adapter_sync(...)` + `adapter.request_device_sync(
required_features=...)` + `create_shader_module(code=spirv_bytes)` — **CHANGED, B11**: never a
module-level `wgpu.request_adapter()`; verified facts #4-#6 establish this works on a CPU-only
machine and is authoritative). **CHANGED (V2)**: wrap `create_shader_module` in
`try/except (GPUValidationError, ValueError)`, converting either to
`SpirvValidationResult(valid=False, error=str(exc))` — confirm the exact `GPUValidationError`
import path against a real `wgpu-py==0.32.*` install first (not independently re-verified in this
revision — see Changelog). This function is called for `vulkan-spirv` only; `metal-spirv` never
reaches it (Task 15).
**Files**: `src/xtrax/export/spirv.py` (modify)
**Gate**: `uv run pytest tests/export/test_spirv.py -q` (fake-injected `wgpu`) — must include
AC-8b's exact two negative fixtures (valid-magic garbage body; random non-SPIR-V bytes) asserting
`valid=False` with the corresponding caught-exception message, never a raised exception.
**Scope estimate**: ~170 LOC
**Verifies**: AC-8, AC-8b

#### Task 12: Wire SPIR-V backends into `compile.py`
Extend `compile_for_target` to map `vulkan-spirv`/`metal-spirv` `Target.iree_backend` to the
matching IREE compiler target, and populate `CompileResult.spirv_bytes` via Task 11's
extraction **for `vulkan-spirv` only** (**CHANGED, V1** — `metal-spirv` never calls
`extract_spirv_from_vmfb`; its `CompileResult.spirv_bytes` is always `None`).
**Files**: `src/xtrax/export/compile.py` (modify)
**Gate**: `uv run pytest tests/export/test_compile_spirv.py -q` — asserts `metal-spirv`'s
`CompileResult.spirv_bytes is None` even on a successful compile.
**Scope estimate**: ~70 LOC
**Verifies**: AC-8

#### Task 13: `hf_weights.py` — bf16-at-load-time cast
Promote `hf_weights.py` → `src/xtrax/export/hf_weights.py`: `load_hf_weights()` (rename from
`mlp_from_hf`, Appendix A), `LoadedWeights`, `WeightReport` (**CHANGED — M4/M10/M11**: both
restored as their own records, not folded away; `WeightReport` gets its own field on
`LoadedWeights`). Casts every `bf16` leaf to `f32` when `target.supported_dtypes` (nor
`optional_dtypes`) contains `"bf16"` (true for both SPIR-V targets — the WebGPU numeric model
itself has no bf16, independent of which target's SPIR-V gets validated), recording one
diagnostics string **per cast leaf, with no truncation** (**CHANGED, M10/M11**: the spike's
`dtypes_cast[:2]` slice at `hf_weights.py:140` is a bug in the spike, do not carry it forward).
Never casts `f64` — that is rejected loudly by `validate_export_safe` (Task 14) instead, never
silently downcast. Keeps the spike's narrow `TinyMLP`-shaped, shape-driven-not-name-driven scope
(Non-goals) — genericizing to arbitrary Equinox modules is explicitly deferred.
**Files**: `src/xtrax/export/hf_weights.py` (create)
**Gate**: `uv run pytest tests/export/test_hf_weights.py -q` (fake HF weight fixture — no
network; follow the existing `tests/scripts/test_smoke_outlines_constrained_decode.py`
`types.ModuleType`+`monkeypatch.setitem(sys.modules, ...)` fake-package precedent for any HF
hub import) — must assert `dtypes_cast` has one entry per actually-cast leaf, not `<= 2`.
**Scope estimate**: ~160 LOC
**Verifies**: AC-10

#### Task 13b: bf16-exactness parity test (NEW — closes M12)
Create `tests/export/test_bf16_exactness.py`: compute the original bf16 model's plain-JAX
forward pass (no export at all) as one reference, and separately export+native-execute the
f32-cast version via `export_pipeline`; compare the two with a bf16-appropriate tolerance (e.g.
`atol=1e-2`). Document in the test's docstring that this is deliberately a *different* comparison
from AC-10's parity check (which is f32-cast-vs-f32-cast and therefore structurally cannot detect
this divergence, since bf16->f32 upcasting is exact).
**Files**: `tests/export/test_bf16_exactness.py` (create)
**Gate**: `uv run pytest tests/export/test_bf16_exactness.py -q`
**Scope estimate**: ~60 LOC
**Verifies**: AC-16

#### Task 14: `safety.py` — dtype/feature gating extension + closure-leaf scan
Extend `validate_export_safe()`/`check_export_safety()` to honor `target.optional_dtypes`/
`optional_dtype_features` against the caller's `request_features`: a leaf whose dtype is in
`optional_dtypes` and whose required feature is in `request_features` passes; otherwise
`DtypeNotSupportedError` naming the missing feature. `f64`/`bf16` leaves on a SPIR-V target
always raise (never gated by any feature). **NEW (closes C4/m9)**: additionally scan `fn`'s
closure-reachable pytree leaves (generalizing the spike's `find_bcoo_leaves`
`tree_flatten_with_path(fn, is_leaf=...)` pattern from BCOO-detection to dtype-checking) and
apply the identical dtype rule to them — closing the gap where a user-supplied Equinox module
holding a bf16/f64 leaf passes every gate today because `abstract_inputs` never contains it.
**Files**: `src/xtrax/export/safety.py` (modify)
**Gate**: `uv run pytest tests/export/test_safety_dtype_gating.py -q` — include a closure-only
(never in `abstract_inputs`) bf16 leaf case asserting `DtypeNotSupportedError` naming the leaf's
keypath.
**Scope estimate**: ~90 LOC
**Verifies**: AC-9, AC-9b, AC-11

#### Task 15: Wire VALIDATED targets into `export_pipeline()`
Extend `export_pipeline()`: for `vulkan-spirv` specifically (**CHANGED, V1**: not "every
VALIDATED target" — `metal-spirv` is no longer `VALIDATED`), after `compile_for_target`, call
`spirv.validate_all_webgpu(compile_result.spirv_bytes, request_features=request_features)` and
populate `ExportResult.spirv_validation`/`.verified = spirv_validation.valid`. For `metal-spirv`,
`spirv_validation` stays `None` and `.verified` is `False` (`CODEGEN_ONLY`'s unconditional rule).
Thread `request_features` through to `validate_export_safe` (Task 14) as well.
**Files**: `src/xtrax/export/pipeline.py` (modify)
**Gate**: `uv run pytest tests/export/test_pipeline_spirv.py -q` — asserts `metal-spirv`'s
`ExportResult.verified is False` and `.spirv_validation is None` even on a successful compile.
**Scope estimate**: ~60 LOC
**Verifies**: AC-8, AC-9, AC-9b, AC-10, AC-11

#### Task 16: Packaging — add `wgpu` to the `export` extra
Add `"wgpu>=0.32,<0.33"` to the existing `export` extra list from Task 6 (verify exact pin
against the version actually installed during the 260831 spike session — 0.32.0 per verified
fact #4).
**Files**: `pyproject.toml` (modify)
**Gate**: `uv sync --extra export` resolves cleanly (CI's `export-toolchain-tests` job import
step succeeds)
**Scope estimate**: ~5 LOC
**Verifies**: AC-8

#### Task 17: Size-budget test
Create `tests/export/test_size_budget.py` with a `SIZE_BUDGET_BYTES` dict keyed by target
name. Seed with the verified spike measurements plus headroom (`vulkan-spirv` ≈6.2KB measured →
budget e.g. 16KB; `metal-spirv` ≈6.5KB measured on its own compiled-bytes size (its `.metal`
dump is a *separate*, non-SPIR-V artifact not counted here — see Task 10/12) → budget e.g. 16KB;
`wasm32` ≈14.5KB measured → budget e.g. 32KB; `native` budget e.g. 32KB pending its own
measurement) and a separate raw-`.spv` budget for `vulkan-spirv` only (measured 812B → e.g. 4KB;
`metal-spirv` has no `.spv` budget at all, since `spirv_bytes` is always `None` for it).
**Replace these illustrative placeholder numbers with the real sizes your own CI run measures
before merging — do not ship guessed budgets.**
**Files**: `tests/export/test_size_budget.py` (create)
**Gate**: `uv run pytest tests/export/test_size_budget.py -q`
**Scope estimate**: ~70 LOC
**Verifies**: AC-12

#### Task 18: Fake-injection + real-toolchain tests for SPIR-V
Extend Task 7's fake-injection pattern to `wgpu` (mirroring the IREE precedent) so `tests/export/`
in full still passes with neither `iree` nor `wgpu` installed. Add
`pytest.importorskip("wgpu")`-guarded real-toolchain tests exercising actual naga validation
of a compiled `vulkan-spirv` kernel (AC-8; **not** `metal-spirv` — it has no SPIR-V to validate),
running in the PR1 CI job (Task 8) — no CI file change needed since that job already installs the
full `export` extra and the Vulkan ICD driver (Task 8). **NOTE (M1, provisional disposition — see
Changelog)**: this task's fake-`wgpu` fixture must be function-scoped
(`monkeypatch.setitem(sys.modules, ...)`, reverting automatically per-test) so a fake-injected
`wgpu` from one test can never remain in `sys.modules` for a later `importorskip("wgpu")`-guarded
real-toolchain test in the same job/session — an already-populated `sys.modules` entry makes
`importorskip` succeed against the fake instead of skipping or exercising the real package,
silently invalidating the "real toolchain" claim.
**Files**: `tests/export/conftest.py` (modify — add fake `wgpu` factory), `tests/export/test_*.py` (create/modify)
**Gate**: `uv run pytest tests/export/ -q` with neither toolchain installed (Task 7's job),
and again with both installed (Task 8's job, 0 skips per that task's gate)
**Scope estimate**: ~210 LOC
**Verifies**: AC-8, AC-8b, AC-13

#### Task 19: Docs update
Extend `docs/api/export.md` with the SPIR-V targets (noting the vulkan-spirv/metal-spirv
verification-level asymmetry, V1) and the dtype-gating rules (including the closure-leaf scan,
AC-9b). Same M15 caveat as Task 9: `just audit-docs-build` verifies plumbing/toctree presence,
not a real Sphinx build.
**Files**: `docs/api/export.md` (modify)
**Gate**: `just audit-docs-build`
**Scope estimate**: ~50 lines of prose

---

### PR3 — Multi-axis composer (certified batched-shape recipe)

#### Task 20: Multi-axis composition in `composer.py` (CHANGED — Blocker 8/M9/C9)
Extend `build_traceable_callable()` to handle a `BatchPlan` with more than one axis decision
where an outer `Vmap`-strategy axis wraps an inner `Scan`-strategy axis. **CHANGED from v1**: bake
the outer axis's cardinality directly into a **bare `jax.lax.scan(batched_transition, init, xs)`
call built by this function itself** — **not** a call to `execute_scan_axis` — with any
`boundary.sink`/`.tap` applied **inline inside `batched_transition`**, mirroring
`execute_scan_axis`'s own `_wrapped_transition` pattern by hand. This is a deliberate, literal
match to what `tests/stages/test_nested_ordering.py::TestBatchedShapeVmapOfScanPreservesOrder`
actually exercises — that test's own `batched_transition` calls `boundary.sink(carry)` manually
and is scanned via bare `jax.lax.scan`, **never** `execute_scan_axis` (verified by reading the
test file directly: `run = jax.jit(lambda init, xs, ...: jax.lax.scan(batched_transition, init,
xs), ...)`). Building the multi-axis composer around `execute_scan_axis` instead, as v1 specified,
would not be "following the certified recipe exactly," because that helper is never exercised by
the certification harness in this batched-shape configuration — it would be an *uncertified*
extrapolation dressed up as a certified one. If the plan's shape instead requires nesting a
literal `jax.vmap` around an axis with a lane-dependent ordered `Tap`/`Sink` (the counter-example
in `TestLiteralVmapOfScanOrdering::test_lane_dependent_ordering_fails_loud`), catch the resulting
`xtrax.stages.executor.ExecutorError` and re-raise `MultiAxisCompositionError`, preserving its
message (which already contains the `"Vmap axis's `fn`"` guidance text).
**Files**: `src/xtrax/export/composer.py` (modify)
**Gate**: see Task 21
**Scope estimate**: ~170 LOC
**Verifies**: AC-14a, AC-14b, AC-15

#### Task 21: Multi-axis certification tests (CHANGED — split per AC-14a/AC-14b)
Create `tests/export/test_multi_axis.py` with three classes mirroring
`tests/stages/test_nested_ordering.py`'s structure:
1. **AC-14a — ordering only**: a positive stress test (batched-shape recipe, varying
   batch/steps like `TestBatchedShapeVmapOfScanPreservesOrder`, `N_TRIALS=20`) calling the
   composer/executor layer directly with an ordered `Sink` attached, asserting a test-double
   sink's recorded call order matches the expected `(lane, step)` sequence. **Never calls
   `export_pipeline`, never constructs an `ExportResult`.**
2. **AC-14b — export/parity, sink-free**: the same 2-axis shape, rebuilt with a `Fuse` (or no
   boundary) instead of the `Sink`, run through `export_pipeline(targets=(NATIVE,))` with an
   independent `reference_fn`, asserting `.verification_level is VerificationLevel.EXECUTED` and
   `.verified is True`.
3. **AC-15 — negative**: the lane-dependent counter-example shape, asserting
   `MultiAxisCompositionError` is raised with the certified message substring, at the
   composer/executor layer (no `ExportResult` claim here either).
**Files**: `tests/export/test_multi_axis.py` (create)
**Gate**: `uv run pytest tests/export/test_multi_axis.py -q`
**Scope estimate**: ~220 LOC
**Verifies**: AC-14a, AC-14b, AC-15

#### Task 22: Docs addendum
Document the multi-axis composition contract (including why it's split into an ordering
certification and a separate sink-free export leg, AC-14a/AC-14b) and the lane-dependent-nesting
refusal in `docs/api/export.md`. Same M15 caveat as Tasks 9/19.
**Files**: `docs/api/export.md` (modify)
**Gate**: `just audit-docs-build`
**Scope estimate**: ~40 lines of prose

## Risks

| Risk | Mitigation |
|------|-----------|
| This spec's promoted-API names were reconciled against the spike source this time (Appendix A, read directly from `.praxia/spike_snapshot/`), but a fixer working from a future, drifted spike branch could still diverge | Every promoted function/class in Appendix A's rename table is cross-checked against this revision's Public API signatures section; if the two disagree at implementation time, the appendix (real spike inventory) wins for "what exists today," and any deliberate rename must still be stated explicitly, not silently substituted |
| Coverage regression: `xtrax/export/` lands inside `tier1_core`'s 90%/80% gate but the toolchain isn't installed there | Every branch must be reachable via the `sys.modules`-fake-injection pattern (Task 7/18) — the spike already proved this works for IREE at 33 tests; extend the same proof to `wgpu`; **and** `export_pipeline`'s logic must live in `pipeline.py`, not `__init__.py` (Blocker 4), or the blanket `*/__init__.py` coverage omit makes this gate vacuous regardless of test coverage |
| `iree-base-compiler`/`iree-base-runtime`/`wgpu`/`huggingface_hub`/`safetensors` exact PyPI package names/pins were reconciled against the spike's error strings and driver code in this revision, but not against a live `pip index`/resolver run (no shell access in this authoring session either) | Tasks 6 and 16 explicitly require the fixer to run a real `uv sync --extra export` and confirm resolution before merging, not trust this document's pins as final |
| Adding `export_safe: bool = False` to `xtrax.stages.topology.validate_plan_topology` could regress existing callers | Default is `False`, preserving current behavior byte-for-byte; existing `tests/stages/test_topology.py` cases must still pass unmodified after Task 2; the Rule 4 allow-list change (adding `DedupGather`) only *widens* what passes, so it cannot regress an existing caller relying on rejection |
| Import-linter contract additions (Task 6) could surface a pre-existing accidental `xtrax.export`-adjacent import xtrax doesn't know about yet | `uv run --extra dev lint-imports` fails loud in `audit-deterministic` CI, not silently — this is the intended outcome, not a regression to route around |
| `bf16`→`f32` auto-cast (Task 13) silently changes numerics for SPIR-V targets without anyone noticing | `LoadedWeights.diagnostics`/`ExportResult.diagnostics` records every cast (one string per leaf, no truncation); AC-16/Task 13b adds the bf16-precision-appropriate parity check that AC-10 alone (f32-vs-f32) cannot provide |
| `wasm32` (CODEGEN_ONLY) has zero execution verification — a wrong triple/flag combination compiles successfully but produces a broken artifact, undetected | Documented as a known gap (non-goal: browser/emsdk execution); size-budget test (Task 17) is the only affordable regression signal in this scope |
| PR2/PR3 branch from PR1's merged `Target`/`ExportResult` API; a rename during PR1 review forces rebases | Freeze field names once PR1 merges; PR2/PR3 only *add* new `Target` constants and dtype-table entries, never rename existing fields. `spirv.py`'s stub existing from PR1 (C8) means PR2 only *extends* a file, never renames `SpirvValidationResult` out from under `ExportResult`'s frozen field |
| A test author supplies `reference_fn=lambda xs: jax.jit(build_traceable_callable(fn, plan))(xs)` — technically type-correct, but reintroduces exactly the B3 self-referential-oracle bug this revision fixed | Called out explicitly in AC-2's note and `verify_native_parity`'s docstring; this is a review-discipline risk, not something the type system can catch — reviewers of Task 5/21's tests must check the `reference_fn` body's independence by eye |
| Size-budget numbers in Task 17 are illustrative placeholders derived from single trivial-kernel measurements, not this repo's real fixture sizes | Task 17 explicitly instructs the fixer to replace them with real CI-measured numbers before merging, not ship the placeholders |
| Retiring the spike (Task 9b) before `tests/export/`'s coverage is actually verified equivalent could silently lose a case the spike's 33 tests covered | Task 9b is ordered *after* Tasks 7/8 in PR1, and its own gate requires Task 7+8's suite to still pass with the spike tree already deleted, not just "coverage looks similar" |
| M1/M2/M14's dispositions in the Changelog are this revision's best reconstruction (M1) or an explicit non-fix pending live measurement (M2), and M14 is entirely unaddressed for lack of any defining prose in the findings doc | Flagged prominently in the Changelog's "named but not resolved" section; a future reviewer with either live CI access or the missing M14 definition should revisit before treating PR1/PR2 as fully closing Part 5's mechanical/CI bucket |

## References

- Dispatch prompt's VERIFIED environment facts (measured 2026-08-31/09-01): IREE 3.11.0 has
  no `webgpu-spirv` backend (`target backend 'webgpu-spirv' not registered; registered
  backends: [cuda, llvm-cpu, metal-spirv, rocm, vmvx, vmvx-inline, vulkan-spirv]`); runtime
  drivers `['cuda','hip','local-sync','local-task','vulkan']`; `vulkan-spirv`/`metal-spirv`
  compile OK (6470/6191 bytes trivial kernel); `rocm` needs an explicit `--iree-hip-target`;
  raw SPIR-V extraction via `--iree-hal-dump-executable-binaries-to=<dir>` (812 bytes, magic
  `0x07230203`); `wgpu-py` 0.32.0 gets a CPU (`llvmpipe`) adapter with no GPU; naga
  parses/validates SPIR-V (no `spirv-shader-passthrough`), making `create_shader_module`
  success an authoritative WebGPU-validity proof; `shader-f16` is optional/gated; no bf16/f64
  in WebGPU at all; local WSL2 box has no usable GPU (only `wgpu`-via-`llvmpipe` works
  locally; Engaging H200 is the real-hardware option).
- **`.praxia/docs/audits/260901_xtrax-export-webgpu-adversarial-findings.md`** — this revision's
  primary input: Part 0's orchestrator-measured facts (metal-spirv dumps MSL not SPIR-V; wgpu
  raises synchronously on invalid SPIR-V; dump plurality is not automatic) override any
  contradicting v1 claim, per that document's own framing; Parts 1-4's challenger/defender
  findings and Part 5's remediation order are what this revision's Changelog addresses
  point-by-point.
- **`.praxia/spike_snapshot/scripts/iree_export_spike/*.py`** and
  **`.praxia/spike_snapshot/tests/scripts/test_iree_export_spike.py`** — the actual spike source
  and its test suite, read directly for this revision (Appendix A below is verified against
  these files, not AST-parsed from a remote ref this agent couldn't reach in the prior pass).
- `.praxia/docs/specs/260831...` daily-log close entry `260831_iree-wasm-webgpu-export_close`
  (`.praxia/daily.jsonl:19`): spike verified end-to-end, native parity max|diff|=1.02e-10 (see
  the Overview's B3 caveat about what this number actually bounds), wasm32 vmfb 14.5KB
  uncompiled-only, 33 tests pass with no IREE/network, draft PR #111 open.
- `src/xtrax/stages/executor.py` module docstring, "Nesting: vmap-of-scan" section — the
  certified multi-axis recipe PR3 implements; also the source of the `ExecutorError`
  `"Vmap axis's `fn`"` guidance text AC-15 matches against.
- `tests/stages/test_nested_ordering.py` — certification harness PR3's tests mirror; read
  directly this revision to confirm `TestBatchedShapeVmapOfScanPreservesOrder` calls bare
  `jax.lax.scan`, never `execute_scan_axis` (Blocker 8/M9/C9).
- `src/xtrax/tiling/dispatch.py::axis_dispatch` — the real `DedupGather` routing mechanism
  (host-computed `unique_indices`/`index_map`, static `k_bucket` shape) this revision restores
  as a supported composer path (Blocker 7/M6/C12).
- `src/xtrax/tiling/strategy.py` — real `Scan(transition, init, ordered_sinks)` and
  `DedupGather(unique_indices, index_map, k, k_bucket, dedup_fn, gather_fn)` field shapes,
  confirming both the B6 precedence note and the Blocker 7 restoration are grounded in real
  code, not invented.
- `distribution/coverage_dag.toml:26-31` — the blanket `*/__init__.py` `coverage_omit` entry
  that makes Blocker 4's fix (`pipeline.py` instead of `__init__.py`) necessary, confirmed by
  direct read this revision.
- `docs/conf.py:97` (`nitpicky = False`) and `Justfile:207-210` (`audit-docs-build`'s real
  contents) — confirmed by direct read this revision; grounds the M15 disposition in Tasks
  9/19/22.
- `docs/index.md` — confirmed by direct read this revision to omit `api/export` from its
  toctree; the real, actionable M15 fix.
- `src/xtrax/cli/export.py`, `src/xtrax/run/zarr_sink.py:159-166` — existing
  lazy-import-and-reraise + StableHLO-export precedent this spec's `compile.py`/`spirv.py`
  follow.
- `tests/cli/test_export.py:76` — the real `pytest.importorskip("flatbuffers")` precedent Task 8
  cites, confirmed by direct read this revision.
- `tests/conftest.py::XTRAX_BEARTYPE_PACKAGES`, `tests/test_import_isolation.py::_PACKAGES` —
  confirmed real by direct read this revision (Task 6).


---

# Appendix A — Verified spike API inventory (read directly from `.praxia/spike_snapshot/`)

**Status: authoritative.** This revision read the spike's actual source at
`.praxia/spike_snapshot/scripts/iree_export_spike/{composer,compile_iree,export_safety,
hf_weights,parity}.py` and its test file directly (not AST-parsed from a remote ref, which the
prior authoring pass could not reach). The inventory below is unchanged in substance from v1's
appendix, since it was itself independently correct — the new information this revision adds is
in the "Notes reviewers should weigh" section and the body's Public API signatures, both updated
against the real code.

Where the body of this spec and this appendix disagree on a name, **this appendix is correct
about what exists today**. The body may still be correct about what the *promoted public API
should be called* — renaming on promotion is legitimate — but any such rename must be stated
as an explicit rename, not presented as new construction.

## Real symbols today

### `compile_iree.py`
```python
NATIVE_TARGET = "native"
WASM32_TARGET = "wasm32"
NATIVE_FLAGS  = ("--iree-llvmcpu-target-cpu=host",)
WASM32_FLAGS  = ("--iree-llvmcpu-target-triple=wasm32-unknown-emscripten",
                 "--iree-llvmcpu-target-cpu=generic",
                 "--iree-llvmcpu-target-cpu-features=+simd128,+atomics,+bulk-memory")

class IREECompileError(Exception)

@dataclass(frozen=True)
class CompileResult:
    target: str
    path: Path
    size_bytes: int
    downgraded_stablehlo: bool

def compile_stablehlo(mlir_text: str, out_path: Path, *, target: str = NATIVE_TARGET) -> CompileResult
def run_native_vmfb(vmfb_path: Path, *args: Any, function: str = "main") -> Any
```

### `composer.py`
```python
class ComposerError(Exception)

def compose_single_axis(step_fn, decision, boundary=None, *, scan_init=None) -> Callable[[Any], Any]
def compose_exportable(step_fn, plan, axis_boundaries=None, *, scan_init=None) -> Callable[[Any], Any]
```
**Confirmed this revision**: `compose_single_axis` routes `Vmap`/`SafeMap` via
`execute_map_axis`, `Scan` via `execute_scan_axis` (reading `scan_init` or `strategy.init` as the
carry — `strategy.transition` is never read here), and **`DedupGather` via
`xtrax.tiling.dispatch.axis_dispatch(strategy, step_fn, xs)`** — this is the real mechanism
Blocker 7/M6/C12 restores. Only `Bucket`/`WhileCarry` raise `ComposerError`.

### `export_safety.py`
```python
EXPORTABLE_STRATEGIES = frozenset({"Vmap", "SafeMap", "Scan", "DedupGather"})
HOST_TIER_STRATEGIES  = frozenset({"Bucket"})
UNBOUNDED_STRATEGIES  = frozenset({"WhileCarry"})

class ExportUnsafeError(Exception)

@dataclass(frozen=True)
class ExportBlocker:
    axis: str
    rule: str
    detail: str

def check_plan_export_safety(decisions, axis_boundaries=None) -> list[ExportBlocker]
def assert_plan_export_safe(decisions, axis_boundaries=None) -> None
def find_bcoo_leaves(model) -> list[str]
```
**Confirmed this revision**: `EXPORTABLE_STRATEGIES` already includes `DedupGather` — v1's
Non-goals exclusion of it was never grounded in this file, which is exactly the "false premise"
Blocker 7/M6/C12 names. `find_bcoo_leaves` uses `jax.tree_util.tree_flatten_with_path(model,
is_leaf=lambda x: isinstance(x, BCOO))` — the pattern this revision's AC-9b generalizes from
BCOO-detection to dtype-checking.

### `hf_weights.py`
```python
class HFWeightsError(Exception)
class TinyMLP(eqx.Module):        # w1, b1, w2, b2
@dataclass(frozen=True)
class WeightReport:               # source, tensors_seen, tensors_used, dtypes_cast
def random_mlp(in_dim, hidden, out_dim, seed=0) -> TinyMLP
def mlp_from_hf(repo_id, *, filename="model.safetensors", in_dim, hidden, out_dim, dtype=jnp.float32) -> tuple[TinyMLP, WeightReport]
```
**Confirmed this revision**: `mlp_from_hf` is shape-driven (first two 2-D tensors by sorted
name), fabricates `b1`/`b2` as zeros (never loaded from the checkpoint), and truncates
`dtypes_cast` to its first two entries (`report.dtypes_cast = tuple(cast[:2])`, line 140) — this
is a real bug in the spike, confirmed by direct read, and Task 13 explicitly does not carry it
forward.

### `parity.py`
```python
@dataclass(frozen=True)
class ParityResult:
    passed: bool
    max_abs_diff: float
    atol: float
    rtol: float
    shape_expected: tuple[int, ...]
    shape_actual: tuple[int, ...]

def compare(expected, actual, *, atol=1e-5, rtol=1e-5) -> ParityResult
```
**Confirmed this revision**: the real `ParityResult` already has `rtol`/`shape_expected`/
`shape_actual` — v1's invented `ExportResult`-adjacent `ParityResult` (which had only
`max_abs_diff`/`passed`/`atol`) dropped three fields that exist in the spike today. `compare`'s
real default is `atol=1e-5, rtol=1e-5` — confirming M4's "no justification was given" complaint
about v1's silent `1e-6` change.

### `__main__.py` (driver — not previously inventoried; read this revision)
**New finding this revision, feeding directly into the B3 fix**: the driver's own parity check
(`main()`, stage `[6/7]`) computes `reference = jax.jit(forward)(xs)` — the SAME composed
`forward` callable — and compares it against the native-executed vmfb of that same `forward`.
This is exactly the self-referential comparison the challenger's B3 identified, and it is where
the "1.02e-10" headline number actually came from. The *test suite*'s `TestComposer` class, by
contrast, uses a genuinely independent oracle (`want = jnp.stack([model(x) for x in xs])`,
`test_iree_export_spike.py:100`) — but only to certify the composer's own correctness in isolation
(no compile/execute step involved at all), never as the reference for a compile+execute parity
check. **Neither existing code path in the spike does what AC-2/AC-14b now require**: an
independent oracle used specifically as the reference for a real compile-and-execute parity
check. This is genuinely new composing logic this spec must specify (the `reference_fn`
parameter), not a promotion of anything that already exists.

## Name mapping the body of this spec implies

| Spec body (promoted name) | Exists in spike as | Action taken |
|---|---|---|
| `compile_for_target()` | `compile_stablehlo()` | rename-on-promote (generalizes `target: str` to `target: Target`, adds `Target`-driven flag/backend dispatch — a genuine interface change, stated as such, not hidden) |
| `check_export_safety()` / `validate_export_safe()` | `check_plan_export_safety()` + `assert_plan_export_safe()` | rename-on-promote, **both** entry points kept distinct (v1 collapsed them into one `validate_export_safe`, which is the M5 defect this revision fixes) |
| `build_traceable_callable()` | `compose_exportable()` (plan-level) + `compose_single_axis()` (axis-level) | rename-on-promote for the plan-level entry point; the axis-level helper is promoted too but not part of the public surface exposed in this spec's signatures section |
| `load_hf_weights()` | `mlp_from_hf()` | rename-on-promote; **kept `TinyMLP`-specific and shape-driven** (Non-goals) — genericizing is real, unspecified, deferred work, not claimed as done |
| `ExportSafetyError` | `ExportUnsafeError` | rename-on-promote |
| `CompileError` | `IREECompileError` | rename-on-promote |
| `ExportResult` | `CompileResult` (+ `ParityResult`, `WeightReport`) | **CHANGED this revision**: the merge itself is accepted per the defender's framing (a new, fully field-specified record), but every field the merge previously dropped (`.path`, `.downgraded_stablehlo`, `.rtol`, `.shape_expected`, `.shape_actual`, all of `WeightReport`) now has an explicit home — see Public API signatures |
| `verify_native_parity()` | *(does not exist in the spike)* | genuinely new, composing logic (independent-oracle parity check) — see the `__main__.py` finding above; not a rename of anything |

## Notes reviewers should weigh

1. `compile_stablehlo` takes `target: str`, not a `Target` object. Introducing a `Target`
   dataclass is a genuine interface change, not a promotion.
2. There is **no** SPIR-V dump plumbing in the spike at all. `--iree-hal-dump-executable-binaries-to`
   is verified working by the orchestrator (Part 0 of the findings doc) but is entirely new code,
   and now must additionally filter by SPIR-V magic (V1) since a naive directory scan would also
   pick up `metal-spirv`'s `.metal` dumps under the same flag.
3. `run_native_vmfb` is the only execution path that exists. Nothing executes wasm32 or SPIR-V.
4. `mlp_from_hf` being `TinyMLP`-shaped means "HF weights → arbitrary Equinox module" is
   **not** proven by the spike, despite the spike's 64-tensor result. This spec keeps that scope
   narrow rather than silently widening it (Non-goals).
5. **New this revision**: the spike's driver (`__main__.py`) is the source of the "1.02e-10"
   parity headline, and it computes that number via the self-referential comparison B3 identifies
   as insufficient for detecting composition errors. The spike's test suite's independent-oracle
   pattern (`TestComposer`) is real and sound, but was never wired into a compile+execute parity
   check anywhere in the spike — `reference_fn` (this spec's fix) has no direct precedent to
   promote from; it is new design work, stated as such.

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

## Changelog vs v2

This pass adds a fourth boundary kind, `materialize`, on top of v2's blanket rejection of any
axis whose `AxisBoundary` has a non-`None` `tap` or `sink` (topology.py's Rule 3, the "fuse-only"
rule AC-3 enforces). Driven by task `260901_xtrax-export-webgpu`, with ground-truth facts about
`src/xtrax/stages/{executor,boundaries,topology}.py` supplied directly and verified against
current `main` (not re-derived, not contradicted).

**Note on rule numbering** (avoiding a real ambiguity rather than silently picking a side): this
pass's brief refers to the tap/sink rejection loosely as "Rule 2 / Rule 3." This spec's own
Public API signatures section already establishes two *distinct* rules in
`validate_plan_topology`: **Rule 2** is the pre-existing, unconditional ordered-Tap/Sink-vs-`Vmap`
check (always active, `export_safe` irrelevant); **Rule 3** is the `export_safe=True`-only
fuse-only/tap-sink check AC-3 relies on. This pass revises **Rule 3 only**. Rule 2 is untouched —
it governs eager semantics that `materialize` never changes (see below).

**Added:**

- **`AxisBoundary.materialize: bool`** (new field, `boundaries.py`, `eqx.field(static=True,
  default=False)`) — declares that this axis's `sink` (if set) is a *materializing* sink.
  Ground truth: `executor.py`'s `_apply_fuse`/`_wrapped_transition`/`execute_map_axis`/
  `execute_scan_axis` (lines 136-140, 220-248) already return exactly the per-step values a sink
  receives, whenever `fuse is None` — `ys` **is** the stack of the same `y` values passed to
  `boundary.sink(y)`. `Sink.__call__` is `T -> None` (its return is discarded, `boundaries.py:81`),
  so removing the sink call changes neither the traced dataflow nor the returned values — it only
  removes an `io_callback` (→ `custom_call`) that would otherwise pierce the export boundary.
  **CORRECTED this pass (C2, see Changelog vs v3): this sentence is false as stated for a
  `SafeMap` axis whose sink is `ordered=True`.** `execute_map_axis`'s `SafeMap` branch chooses
  its lowering by reading `_has_ordered_op(boundary)` (`executor.py:107-113`), which inspects
  `boundary.sink.ordered` — replacing `sink` with bare `None` (the v3 design) makes that read
  return `False` even when the original sink was `ordered=True`, flipping the branch from
  `jax.lax.map(wrapped, xs)` (ordered path, ignores `batch_size` entirely) to
  `safe_map(wrapped, xs, batch_size=strategy.batch_size)` — a *different lowering* than the
  eager run took, and one that raises `ValueError` outright when the axis's cardinality isn't
  divisible by `batch_size` (`src/xtrax/transforms/map.py:33-37`). See the Changelog vs v3 (C2)
  for the fix: the strip preserves `.ordered` via a no-op sentinel instead of `None`, restoring
  the truth of this sentence.
  **`materialize` has zero effect outside `xtrax.export`** (an eager, non-export run of the same
  `AxisBoundary` still fires the sink exactly as before). Default `False` is fully backward
  compatible: no existing `AxisBoundary(...)` call site changes behavior, and an undeclared
  (`materialize=False`) sink is still rejected by Rule 3 exactly as before this pass (AC-17b).
  **`materialize` never applies to `tap`** — `Tap.__call__` is `T -> T` and participates in
  dataflow (its return replaces the step value, `executor.py:127-128`), so it can never be
  dropped without changing the exported program's semantics, on *any* target. This is also a
  **permanent** limit for `vulkan-spirv`/`metal-spirv` (WebGPU/SPIR-V has no host-callback
  mechanism at all — a compute shader cannot call host code mid-dispatch) and for `wasm32`'s
  `CODEGEN_ONLY` scope (no emsdk/browser runtime is promised, per Non-goals) — not a deferral,
  a structural fact about the target, restated here because `materialize` is the one boundary
  kind that sidesteps needing any callback mechanism at all, which is exactly why it is valuable
  for the callback-incapable targets, not despite them.
- **Rule 3 becomes kind-based, not slot-based** (`topology.py::validate_plan_topology`,
  `export_safe=True`), in this order: (1) a non-`None` `tap` always raises `PlanTopologyError` —
  unconditional, `materialize` irrelevant. (2) `materialize=True` with `sink is None` raises the
  new `MaterializeWithoutSinkError` — nothing to materialize is a caller error, not a silent
  no-op. (3) a non-`None` `sink` with `materialize=False` (the default) raises `PlanTopologyError`
  — **unchanged, an undeclared sink is rejected exactly as before** (AC-17b re-proves this as a
  regression check). (4) a non-`None` `sink` with `materialize=True` **and** a non-`None` `fuse`
  on the same axis raises the new `MaterializeFuseConflictError` — `fuse` collapses the very
  per-step stacked array `materialize` needs to expose as the axis's own output (GT #6), so the
  two cannot coexist on one axis's export. (5) otherwise (`materialize=True`, `sink` set, `fuse`
  `None`): **passes.** Both new exceptions subclass `PlanTopologyError`, so every existing
  `except PlanTopologyError` call site — including `validate_export_safe`'s M5-documented
  unwrapped-propagation contract — keeps working with zero changes.
- **`export_pipeline`'s new boundary pre-strip step** (`pipeline.py`, new private helper,
  ~15 LOC) — before calling `build_traceable_callable`, builds a view of `axis_boundaries` where
  every `materialize=True` axis's boundary has `sink` replaced with `None`. **`composer.py`'s
  `build_traceable_callable` itself is UNCHANGED by this pass** — it stays a "compose whatever
  boundary dict you're given" primitive, which AC-14a already depends on (it calls the
  composer/executor layer *directly*, bypassing `export_pipeline`/Rule 3 entirely, and needs the
  real, un-stripped sink to fire so its test-double can record call order). Confining the strip
  to `export_pipeline`'s entry point means PR3's multi-axis composer (Task 20) needs **no code
  changes at all** to support `materialize` — it inherits the behavior transparently through the
  same boundary dict, since `export_pipeline` calls `build_traceable_callable` exactly once
  regardless of axis count.
- **AC-17/17b/17c/17d (PR1, NEW)**: the positive case (a materialized sink survives export; the
  exported artifact's own executed output matches what the sink would have recorded on an eager
  run of the same boundary) and three negative/regression cases (tap always rejected; undeclared
  sink still rejected — proving Rule 3 was not weakened; the new fuse-conflict and no-sink error
  paths).
- **AC-14a/AC-14b (PR3, CHANGED again this pass)** — AC-14b's `Sink` no longer needs swapping for
  a separately-built `Fuse`/no-boundary variant; the *same* `Sink`, now declared
  `materialize=True` (with no `fuse` on that axis), runs directly through `export_pipeline`, and
  ordering is now asserted on the **exported artifact's own native-executed output** — strictly
  stronger than a pre-export test-double's recorded call order. AC-14a is *kept*, not superseded:
  it certifies a different failure surface (pure composer/JAX composition correctness — wrong
  axis nesting, dropped boundary — independent of whatever IREE's own lowering/execution might do
  to array order), which only a pre-export, non-IREE check can isolate; AC-14b's now-artifact-level
  check can only see the *combination* of composition-correctness and lowering-fidelity, not
  disentangle them if both were somehow wrong at once.
- **Ordering-cost note**: `ordered=True` on a `materialize=True` sink is fully free *within the
  exported artifact* — the sink call, and therefore its `io_callback` token-threading cost (see
  `executor.py`'s module docstring), is stripped before `jax.export.export` ever traces it, so the
  exported program contains no `io_callback` for that axis at all. This is unrelated to Rule 2
  (ordered-Tap/Sink-vs-`Vmap`), which still applies unconditionally at plan-construction time,
  governing the *eager* semantics `materialize` never touches: an axis with `ordered=True` still
  cannot use `Vmap` strategy, `materialize` or not, for any eager run of the same boundary.
- **Memory tradeoff, stated honestly**: materialization allocates `[steps, ...]` (or
  `[steps, lanes, ...]` for the batched multi-axis shape — **CORRECTED this pass, C10**: v3 had
  this inverted; `jax.lax.scan` stacks its `ys` output on the LEADING axis, which is the *step*
  axis for Task 20's batched-shape recipe (`init`/lanes is the carry, `xs`/steps is the scanned
  sequence), so the stacked shape is `[steps, lanes, ...]`, not `[lanes, steps, ...]`) where a
  live `io_callback` sink would
  have streamed each step's value to the host and discarded it. For a long scan over large
  per-step values this is real device-memory pressure with no streaming escape valve — exactly
  why `materialize` is opt-in (default `False`), never automatic for an export-time sink.

**Explicitly out of scope for this pass (flagged, not invented):**

- **Two independently `materialize=True` axes composing in the same multi-axis plan.** The
  certified batched-shape recipe (Task 20) bakes the outer `Vmap`-strategy axis's cardinality
  directly into scan structure rather than giving it its own independent `AxisBoundary` output
  slot, so there is no existing shape for a second materialized array to occupy alongside the
  inner axis's. A caller needing this is unstarted follow-up work — not attempted here.
  **Now detected, not just documented (this revision, closes C6):** this Non-goal previously had
  no enforcement — such a plan would silently reach Task 20's composer with an unspecified second
  output slot instead of being rejected. `MultipleMaterializeAxesError` (Task 2, AC-17h) now
  raises before any `jax.jit`/compile call whenever more than one axis has `materialize=True`,
  converting this from a silent-wrong outcome into a named, caught, still-unimplemented
  supported-non-goal.
- **Any change to target-specific behavior.** `materialize` only changes whether a plan *passes
  Rule 3's gate at all*; it does not change `VerificationLevel`, dtype gating, or any existing
  Non-goal (wasm32/SPIR-V execution restrictions in particular). It is orthogonal to target
  choice — the same `materialize=True` axis is equally eligible for `native`, `wasm32`,
  `vulkan-spirv`, and `metal-spirv`, subject to those targets' own, unchanged rules.

## Changelog vs v3

This pass is a **narrow adversarial review of the `materialize` boundary-kind delta only**
(everything the "Changelog vs v2" section above covers) — nothing outside that delta was
re-litigated, and the v1/v2 changelogs and everything they cover are preserved unchanged above.
Findings are labeled **C1-C15** per this pass's dispatch prompt (orchestrator adjudication of a
review of `.praxia/spike_snapshot/scripts/iree_export_spike/composer.py` — the previous reviewer
did not have this source available and drew one wrong conclusion as a result, see C1). **Label
collision warning:** C1-C15 in this section are this pass's own IDs and are **unrelated** to the
identically-shaped "C4"/"C5"/"C6"/"C7"/"C8"/"C9"/"C10"/"C12" labels already used elsewhere in this
document's v1 Changelog and Public API signatures (e.g. "Blocker 4/B14/C5", "C4/m9", "Blocker
6/B4.2/C10", "Blocker 7/M6/C12") — those came from an earlier, full-spec adversarial review round
and numbered their own findings independently. Every reference below and inline in the body is
written as "this revision's CN" or "this pass" wherever ambiguity was possible; treat any bare
"CN" appearing *inside a block already discussing `materialize`* as this section's ID, and any
bare "CN" elsewhere in the document as the earlier round's.

**Refuted (no fix implemented for the speculated bug; a related documentation gap fixed
instead):**

- **C1** — the reviewer speculated the composed callable might return a `Scan` axis's
  `final_carry` instead of `ys`, which would let XLA's dead-code elimination silently drop the
  materialized stream. Verified false against `composer.py::compose_single_axis`'s `Scan`
  branch (`_run_scan`): it returns `ys` and explicitly discards `_final_carry`, with a comment
  saying so. `ys` is a live output; DCE does not apply; "no composer changes needed" stands. The
  real gap the reviewer surfaced — this spec never *stated* the return contract `materialize`
  depends on — is fixed by the new "Composed callable return contract" section (above the Public
  API signatures), which states per-strategy what the composed callable returns and what future
  change would silently break `materialize`.

**Confirmed and fixed:**

- **C2 (the real blocker)** — stripping a `materialize=True` axis's `sink` to bare `None` (v3's
  design) is observable to `execute_map_axis`'s `SafeMap` branch: `_has_ordered_op` reads
  `boundary.sink.ordered`, so a `None` sink reads as `ordered=False` regardless of what the
  original sink declared, flipping the branch from the ordered `jax.lax.map(wrapped, xs)` path
  (ignores `batch_size`) to `safe_map(wrapped, xs, batch_size=strategy.batch_size)` — a different
  lowering, and one that raises `ValueError` outright when cardinality isn't divisible by
  `batch_size` (`src/xtrax/transforms/map.py:33-37`). **Fixed** (Task 5): the strip now replaces
  `sink` with a no-op sentinel that preserves the original `.ordered` value instead of `None`,
  so `_has_ordered_op` reads identically to the un-stripped boundary and the exported program
  takes the same branch the eager run would have taken, with the `io_callback` itself still
  fully removed. New regression AC: AC-17e (a non-divisible `SafeMap(batch_size=4)` + ordered
  `materialize=True` sink over 10 elements now passes, exercising exactly the case that would
  have raised under v3's design). The false "changes neither the traced dataflow nor the returned
  values" sentence in the Changelog vs v2 (Added, first bullet) is corrected in place.
- **C3** — AC-17 and AC-14b asserted against data no record actually exposed: `ExportResult` had
  no `.path`/output-array field (only `parity: ParityResult | None`, and `ParityResult` carries
  only `passed`/`max_abs_diff`/`atol`/`rtol`/`shape_expected`/`shape_actual`, never an array), and
  `ExportResult.vmfb_bytes`'s own comment already claimed "execution/parity always go through
  `.path` internally" despite `.path` not existing on the class. **Fixed:** added
  `ExportResult.path: Path` (populated from `compile_result.path` in Task 5); AC-17/AC-14b now
  call `run_native_vmfb(result.path, *concrete_inputs)` directly to obtain the real output array,
  rather than claiming a route through `.parity` that never exposed one.
- **C4** — Rule 3 must read `getattr(boundary, "materialize", False)`, matching Rule 2's existing
  duck-typed `getattr(boundary.tap, "ordered", False)` pattern, so a foreign plan's boundary
  object (e.g. `aminx.tiling`, per `topology.py`'s own structural-compatibility promise) that
  predates this field is treated as `materialize=False` rather than raising `AttributeError`.
  **Fixed** (Task 2); new AC-17f (regression proof: a boundary object with no `materialize`
  attribute and a bare sink is rejected exactly as `materialize=False`, no `AttributeError`).
- **C5** — the sink-stripping helper must build its output via `dataclasses.replace(b,
  sink=...)`, never by re-listing `AxisBoundary(fuse=..., tap=..., sink=..., materialize=...)`
  field-by-field (which silently drops any future field and downcasts a boundary subclass).
  **Fixed** (Task 5); `eqx.tree_at` does not work here since every `AxisBoundary` field is
  `static=True` (aux_data, not a pytree leaf) — `dataclasses.replace` is the correct tool since
  `eqx.Module` is a frozen dataclass regardless of which fields are static. New AC-17g: the
  helper touches only `materialize=True` axes' `sink`, preserves `fuse`/`tap` on that axis, and
  passes every other axis through by identity (`is`, not just `==`).
- **C6** — the "two independently `materialize=True` axes" Non-goal had no detector; such a plan
  would silently reach Task 20's composer with an unspecified second output slot. **Fixed**
  (Task 2): a new whole-plan Rule 3 step 6 raises `MultipleMaterializeAxesError` (a
  `PlanTopologyError` subclass) naming every axis, once more than one axis passes the per-axis
  steps with `materialize=True`. New AC-17h. Both "Explicitly out of scope"/"Non-goals" mentions
  of this Non-goal are updated to cross-reference the new detector — the Non-goal itself is
  unchanged (still unimplemented), only now it fails loud instead of silent.

**Confirmed and fixed, cheap items:**

- **C7** — `tests/stages/test_nested_ordering.py:112-117`'s hand-written `batched_transition`
  sinks the pre-update `carry` and also returns that same pre-update `carry` as its `y`; sink-value
  == `y`-value holds there only because that specific transition happens to do both, not because
  `jax.lax.scan` or Task 20's composer enforces it structurally (unlike the flat single-axis path,
  where `_wrap_step`/`_wrapped_transition` compute `y` once and pass exactly that value to both
  `tap` and `sink` by construction). **Fixed:** Task 20 now states this as an explicit
  precondition on any `batched_transition` written for the batched-shape recipe when combined
  with `materialize`, rather than silently relying on it. The "no changes needed in Task 20"
  claim is **not withdrawn** — this is a caller-obligation/documentation fix, not a composer code
  change.
- **C8** — reframed the "no new executor plumbing" safety argument as a **precondition** on the
  assigned `sink`, not a proof about it: the slot's type is `Sink | BoundaryCallable | None` (any
  callable), the `T -> None` return contract is enforced by nothing but a docstring, and
  `tests/stages/test_executor.py`'s own `ReturningSink` fixture is a real, currently-tolerated
  non-conforming implementation. **Fixed:** the precondition is now stated in
  `AxisBoundary.materialize`'s own field docstring (Task 2) and in Task 9's docs subsection.
- **C9** — AC-14a composes the **un-stripped** program directly (never through
  `export_pipeline`) while AC-14b exports the **STRIPPED** one; a green AC-14a + red AC-14b left
  three undistinguished suspects (a composer bug specific to the un-stripped path, a composer
  bug specific to the stripped path, or an IREE lowering bug). **Fixed:** AC-14b/Task 21 gain a
  middle leg — run the same STRIPPED composed callable in pure JAX against `reference_fn` and
  assert exact agreement, before ever calling `compile_for_target`/IREE — isolating the stripped
  composer bug as its own diagnosable failure mode.
- **C10** — the stacked shape for the batched multi-axis recipe was stated inverted:
  `jax.lax.scan` stacks its output on the **leading** (step) axis, so the shape is `[steps,
  lanes, ...]`, not `[lanes, steps, ...]`. **Fixed** in both the Changelog vs v2 (Added,
  "Memory tradeoff" bullet) and the Risks table row, in place.
- **C11/C13/C14/C15** — wording/scoping fixes, all applied in place: `MaterializeWithoutSinkError`
  now states explicitly it is an export-time validation error only, never fired by any eager
  path; `MaterializeFuseConflictError`'s docstring now names the real cause (this pass's own "no
  executor changes" scope constraint, not a structural impossibility) and states the workaround
  (drop `fuse`, reduce outside the exported function); AC-17 now states explicitly why it is
  `Scan`/`SafeMap`-only (`Vmap` host-call order is not a documented sequence, and any
  `Vmap`+`materialize=True` axis reaching `export_pipeline` at all necessarily has an *unordered*
  sink, since an ordered one already dies at Rule 2 — noted as deliberate conservatism, not an
  oversight); Task 9's docs gain one explicit sentence that `build_traceable_callable` called
  directly with a caller's original (un-stripped) `axis_boundaries` does **not** return what
  `export_pipeline` actually exports for the same arguments, mirroring the existing B6
  `Scan.transition`-vs-`fn` footgun note.
- **C12** — qualified "fully backward compatible" in `AxisBoundary.materialize`'s field
  docstring: adding a new `static=True` field changes the pytree treedef's aux_data, so (a) any
  already-pickled pre-pass `AxisBoundary` instance has no `materialize` entry in its serialized
  state, and whether it unpickles correctly is an open question this spec does not resolve
  (flagged, not asserted fixed by `default=False` alone); (b) any `jax.jit` cache keyed on the
  old (3-field) treedef invalidates once this field exists (one recompile — expected, but not
  literally "zero change"). C4's `getattr(..., default=False)` read still covers the *foreign,
  never-pickled* boundary-object case described in C4 itself; C12 is the narrower, additional
  persisted-pickle/jit-cache caveat.

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

- **Two independently `materialize=True` axes in the same multi-axis plan.** (New this pass.)
  The certified batched-shape recipe (Task 20) bakes the outer `Vmap`-strategy axis's cardinality
  directly into scan structure rather than giving it its own independent `AxisBoundary` output
  slot, so there is no existing shape for a second materialized array to occupy alongside the
  inner axis's. Unstarted follow-up work if a caller needs it. **Now detected, not just
  documented (this revision, closes C6):** `MultipleMaterializeAxesError` (Task 2, AC-17h) raises
  before any `jax.jit`/compile call for exactly this shape, so this Non-goal is a supported
  refusal, not a silent accident, without being implemented.
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
- **AC-3 (PR1, CHANGED — kind-based precision for `materialize`, this pass):** a plan whose any
  `AxisDecision`'s `AxisBoundary` has a non-`None` `tap`, OR a non-`None` `sink` that is either
  not declared `materialize=True` or conflicts with a non-`None` `fuse` on the same axis, raises
  before any `jax.jit`/compile call — verified by a call-counting fake compiler recording zero
  invocations. **Every case this AC covered before this pass still raises identically** (AC-17b
  re-proves the plain-tap and undeclared-sink cases as a regression check, so this precision
  change did not widen the gate); the only *new* passing case — a `materialize=True` sink with no
  conflicting `fuse` — is certified separately by AC-17, in PR1. See AC-14a/AC-14b below: this
  rule is *global* to `export_pipeline`/`validate_export_safe`; AC-14a's ordering certification
  deliberately operates one layer below this gate (directly against the composer/executor, never
  through `export_pipeline`) so it does not contradict AC-3. **Note (this revision):** the
  whole-plan "more than one `materialize=True` axis" check (Rule 3 step 6,
  `MultipleMaterializeAxesError`) is a separate tier from the per-axis checks this AC covers —
  see AC-17h.
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
- **AC-14a (PR3, NEW — split from v1's AC-14, Blocker 1/B1/B2/C1; kept unchanged this pass, see
  note):** a 2-axis `BatchPlan` (outer `Vmap`-strategy axis, inner `Scan`-strategy axis, ordered
  `Sink` on the inner axis whose sunk value depends on the outer lane) is composed via the
  **composer/executor layer directly** — never through `export_pipeline`, never producing an
  `ExportResult`, and no `.verified` claim is made. The batched-shape recipe (Task 20) bakes the
  outer axis into a bare `jax.lax.scan` call (not `execute_scan_axis` — see Blocker 8 below); a
  test-double sink records host-call order matching the expected `(lane, step)` sequence, proving
  no literal `jax.vmap` ran internally. This certifies ordering only. A `Sink` is legal here
  precisely because this path never crosses the export boundary that AC-3 protects, and
  `build_traceable_callable` never strips it (the pre-strip transform lives in `export_pipeline`
  only, per this pass's Changelog) — the sink genuinely fires and is genuinely observed here.
  **Note (`materialize`, this pass):** kept, not superseded by AC-14b's strengthening below — it
  certifies pure composer/JAX composition correctness (wrong axis nesting, dropped boundary),
  which is a different failure surface from AC-14b's post-lowering, artifact-level check and
  cannot be recovered from AC-14b alone if IREE's own lowering introduced a *compensating*
  reordering bug.
- **AC-14b (PR3, CHANGED this pass — strengthened via `materialize`, split from v1's AC-14;
  CHANGED again this pass, closes C3 and this revision's C9 -- distinct from the earlier
  "Blocker 8/M9/C9" about Task 20's recipe):** the *same 2-axis shape*, but the inner axis's `Sink`
  is declared `materialize=True` (no `fuse` on that axis) and the **same boundary, sink
  included**, is run directly through `export_pipeline(targets=(NATIVE,))` with an independent
  `reference_fn` (per AC-2) — no separate sink-free variant needs to be built, since Rule 3
  (revised this pass) now accepts this configuration and `export_pipeline` strips the sink
  internally before tracing. Produces a `native` `ExportResult` with `.verification_level is
  VerificationLevel.EXECUTED` and `.verified is True` (parity against `reference_fn`), **and, new
  this pass**: the sunk/materialized value at each step encodes its own `(lane, step)` identity
  (e.g. `y = lane * STEPS + step`, or an equivalent structured value), and the test asserts —
  **by calling `run_native_vmfb(result.path, *concrete_inputs)` directly** (**CORRECTED, C3**:
  v3 said this was "surfaced via the `ExportResult`/`verify_native_parity` path," which exposes
  no output array; see AC-17's identical correction and the new `ExportResult.path` field) —
  that the decoded array reconstructs the expected `(lane, step)` sequence directly from its
  content/order. Ordering is now certified on **the exported artifact's own output**, not merely
  on a pre-export test-double's recorded call order. This is what actually exercises multi-axis
  codegen + parity; AC-14a alone never reaches `jax.export.export` at all.
  **New middle leg (this revision's C9, unrelated to the earlier "Blocker 8/M9/C9"):** as
  written, AC-14a certifies the **un-stripped** composed callable
  (called directly, never through `export_pipeline`) and AC-14b certifies the **STRIPPED**
  callable's compiled-and-executed artifact — there is no test of the STRIPPED callable's pure-JAX
  composition correctness in between. If AC-14a is green and AC-14b is red, that leaves three
  candidate causes (a composer bug specific to the un-stripped path that AC-14a happens not to
  exercise; a composer bug specific to the stripped path; or an IREE lowering bug) with no way to
  tell which. Add the missing middle leg: before compiling, run the **same STRIPPED** composed
  callable (i.e. `jax.jit(build_traceable_callable(fn, plan, stripped_boundaries))`, using
  `export_pipeline`'s own private stripping helper or an equivalent direct call) in pure JAX
  against `concrete_inputs` and assert it matches `reference_fn(concrete_inputs)` exactly (float
  tolerance), before ever invoking `compile_for_target`/IREE. A failure here isolates the bug to
  the stripped composition itself, independent of both AC-14a's un-stripped path and IREE's
  lowering.
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
- **AC-17 (PR1, NEW — `materialize` positive case, this pass; CHANGED again this pass, closes
  C3):** a single-axis plan (`Scan` or `SafeMap` — **never `Vmap`**, see note below) whose
  `AxisBoundary` has `materialize=True`, a non-`None` `sink`, no `tap`, no `fuse` passes
  `check_export_safety`/`validate_export_safe` (this configuration was rejected before this
  pass) and `export_pipeline(..., targets=(NATIVE,))` produces a `native` `ExportResult`. The
  test first runs the **same, un-stripped** boundary eagerly through
  `xtrax.stages.executor.execute_scan_axis` (or `execute_map_axis`) directly, recording every
  value the sink actually received into a plain list; it then asserts that calling
  **`run_native_vmfb(result.path, *concrete_inputs)` directly** — not through `ParityResult`,
  which carries only `passed`/`max_abs_diff`/`atol`/`rtol`/`shape_expected`/`shape_actual` and
  never the arrays themselves — reproduces that same recorded list, up to parity tolerance. This
  is the end-to-end proof of the Changelog's "no new executor plumbing" claim: the exported
  artifact's own output *is* what the sink would have recorded, not a re-derivation of it.
  **CORRECTED, C3:** v3's text said this comparison was "surfaced via `ExportResult`/
  `verify_native_parity`'s comparison" — no such route existed: `ExportResult` had no `.path`
  field and `.parity`'s `ParityResult` exposes no output array. Fixed by adding `ExportResult.path`
  (see Public API signatures) so the test has a real, specified way to call `run_native_vmfb`
  itself and inspect the raw output array, independently of `.parity`'s pass/fail verdict (which
  the test still also checks, via `.verified`/`.parity.passed`, as a second, independent
  assertion).
  **Note (scope, C11):** this AC is `Scan`/`SafeMap` only because the eager-recording technique
  ("record every value the sink actually received into a plain list", then compare against the
  exported array **in order**) requires deterministic step order, which only `Scan`'s
  `jax.lax.scan` and `SafeMap`'s non-`vmap` paths give. A `Vmap`-strategy axis's host-call order
  under `jax.vmap` is not a documented sequence at all (it is effectively a multiset for an
  unordered sink); comparing a recorded list positionally would be testing an implementation
  accident, not a guarantee. This is consistent with, not in tension with, Rule 2's unconditional
  rejection of any `Vmap` axis with an *ordered* `tap`/`sink` (`materialize` or not) — the only
  `Vmap`+`materialize=True` axes that can even reach `export_pipeline` already have an unordered
  sink (an ordered one dies at Rule 2 first), and this AC simply does not attempt to test that
  narrower, order-agnostic remaining case; that is deliberate conservatism, not an oversight.
- **AC-17b (PR1, NEW — regression proof, this pass):** two cases, each still raising exactly as
  before this pass: (1) a non-`None` `tap` on any axis (`materialize` set or not, since
  `materialize` never applies to `tap`) raises `PlanTopologyError` before any `jax.jit`/compile
  call; (2) a non-`None` `sink` with `materialize=False` (the default — including a boundary
  built by a caller who never heard of `materialize`) raises `PlanTopologyError` identically to
  pre-this-pass behavior. Both verified via the same call-counting fake compiler as AC-3, proving
  Rule 3's kind-based revision did not widen the gate for either case.
- **AC-17c (PR1, NEW — fuse/materialize conflict, this pass):** an axis whose `AxisBoundary` has
  both a non-`None` `fuse` and `materialize=True` with a non-`None` `sink` raises
  `MaterializeFuseConflictError` (a `PlanTopologyError` subclass) before any `jax.jit`/compile
  call, naming the axis.
- **AC-17d (PR1, NEW — no-sink-to-materialize, this pass):** an axis whose `AxisBoundary` has
  `materialize=True` and `sink=None` raises `MaterializeWithoutSinkError` (a `PlanTopologyError`
  subclass) before any `jax.jit`/compile call, naming the axis.
- **AC-17e (PR1, NEW this revision — closes C2):** an ordered `SafeMap` axis (`sink.ordered is
  True`) with `materialize=True`, `batch_size=4`, run over a **10-element** input (not divisible
  by 4) passes `export_pipeline(..., targets=(NATIVE,))` without raising, and the exported
  artifact's own executed output (via `ExportResult.path` + `run_native_vmfb`, see AC-17)
  matches the same boundary's eager-recorded values. This is the regression proof for the fix in
  Task 5: `export_pipeline`'s sink-stripping helper must preserve `.ordered=True` on its
  replacement sentinel (not strip to bare `None`), so `_has_ordered_op` still selects the ordered
  `jax.lax.map(wrapped, xs)` branch — which unconditionally ignores `batch_size` — inside the
  exported trace, exactly as the eager run does. Without this fix, stripping to `None` flips the
  branch to `safe_map(wrapped, xs, batch_size=4)`, which raises `ValueError: n=10 is not
  divisible by batch_size=4` for this exact fixture (`src/xtrax/transforms/map.py:33-37`) — this
  AC's fixture is chosen specifically to be a non-divisible case, so it fails loudly under the
  v3 (bare-`None`) design and passes under this revision's sentinel design.
- **AC-17f (PR1, NEW this revision — closes C4):** a boundary-like object that has `fuse`/`tap`/
  `sink` but **no `materialize` attribute at all** (simulating a foreign, pre-this-pass plan
  object, e.g. a stand-in for `aminx.tiling`) and a non-`None` `sink` is rejected by Rule 3
  exactly as an explicit `materialize=False` boundary would be (plain `PlanTopologyError`, not
  `AttributeError`) — proving Rule 3 reads the field via `getattr(boundary, "materialize",
  False)`, not bare attribute access.
- **AC-17g (PR1, NEW this revision — closes C5):** `export_pipeline`'s private
  `_boundaries_for_export` helper, given a multi-entry `axis_boundaries` mapping where exactly
  one axis has `materialize=True`: (1) that axis's returned `AxisBoundary` has its `sink`
  replaced but its `fuse` and `tap` identical (`is`) to the input's; (2) every other axis's
  `AxisBoundary` in the returned mapping is the exact same object (`is`) as in the input, not a
  reconstructed equal-but-different instance.
- **AC-17h (PR1, NEW this revision — closes C6):** a plan with two (or more) axes both declared
  `materialize=True` (each independently satisfying Rule 3's per-axis steps 1-5) raises
  `MultipleMaterializeAxesError` (a `PlanTopologyError` subclass) naming every offending axis,
  before any `jax.jit`/compile call — converting the "two independently `materialize=True` axes"
  Non-goal from a silent, unspecified-output-shape outcome into a named, caught error.

## Design decisions carried over verbatim from the approved design

(Restated here for fixer convenience; do not re-derive or re-litigate. Items marked **CHANGED**
were altered by this revision — see the Changelog.)

- Package layout: `src/xtrax/export/{__init__,targets,safety,composer,compile,spirv,pipeline,parity}.py`.
  **CHANGED:** `pipeline.py` added (Blocker 4) — `ExportResult`/`export_pipeline` live there, not
  in `__init__.py`.
- **NEW, this pass: a fourth boundary kind, `materialize`.** `AxisBoundary` (in the *base*
  `xtrax.stages.boundaries` module, not `xtrax.export`) gains a `materialize: bool = False`
  field. Rule 3 (`topology.py`, `export_safe=True`) becomes kind-based: `tap` always rejected
  (permanent, not a deferral — no host-callback mechanism exists on `vulkan-spirv`/`metal-spirv`
  to run it against, and it participates in dataflow so it can never be dropped even on `native`);
  an undeclared `sink` still rejected exactly as before; a `sink` declared `materialize=True`
  with no conflicting `fuse` on the same axis now passes, and `export_pipeline` (not
  `build_traceable_callable`) strips that sink before tracing — relying on `executor.py`'s
  existing `_apply_fuse`/per-step-return behavior, so no executor or composer code changes at
  all. See AC-3/AC-14a/AC-14b/AC-17/17b/17c/17d and the Changelog vs v2.
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
  **New, this pass:** `materialize`'s foundation (the `AxisBoundary` field, Rule 3's kind-based
  revision, `export_pipeline`'s boundary pre-strip, AC-17/17b/17c/17d) lands in **PR1** — Task 2
  and Task 5 are where Rule 3 and `export_pipeline` are *first implemented* (neither exists on
  `main` yet), so extending them in place, rather than opening a new PR4, avoids landing a
  slot-based Rule 3 and then immediately replacing it. **PR3 gets a small, test-only follow-on**
  (Task 21's AC-14b, item 2) with **no composer code changes** (Task 20 is unmodified) — because
  the pre-strip lives in `export_pipeline` (PR1), PR3's multi-axis composer inherits `materialize`
  support transparently through the same boundary dict it already receives.

## Composed callable return contract (NEW this revision, closes C1)

**This is new, load-bearing documentation, not a design change** — the reviewer speculated that
the composed callable might return a `Scan` axis's `final_carry` (in which case XLA's dead-code
elimination could drop the materialized `ys` entirely, since nothing downstream would reference
it, silently defeating `materialize`). Verified directly against
`.praxia/spike_snapshot/scripts/iree_export_spike/composer.py::compose_single_axis`'s `Scan`
branch: it returns `ys`, explicitly discarding `_final_carry`, with a comment saying so
(`_final_carry, ys = execute_scan_axis(...); return ys`). `ys` **is** a live output of the
composed callable, so DCE cannot drop it, and the reviewer's suggested fix ("no composer changes
needed") does not apply — but the reviewer was right that this spec never *stated* the contract
its own `materialize=True` design depends on, so a future change to any of these branches could
silently break materialization without violating anything this spec's text actually says. Stated
explicitly here, per strategy (`build_traceable_callable`/`compose_single_axis`, promoted
unchanged from the spike per Task 3):

- **`Vmap`/`SafeMap`**: returns `execute_map_axis`'s own return value — the (optionally fused)
  stacked per-step output. For a `materialize=True` axis (`fuse` is `None` by Rule 3's conflict
  check), this is exactly the stack of values the (stripped) sink would have received.
- **`Scan`**: returns `ys` **only** — `execute_scan_axis`'s `final_carry` is discarded by
  `compose_single_axis`'s `_run_scan` closure and is never part of the composed callable's
  output at all, at any point in this spec's scope. A `materialize=True` `Scan` axis therefore
  always has a live, non-DCE-able output to expose.
- **`DedupGather`**: returns `axis_dispatch(strategy, step_fn, xs)`'s result directly.
  `materialize` is not applicable to `DedupGather` in this spec's scope (its dispatch has no
  `Sink`-shaped per-step call to strip in the first place) — restated here to avoid a reader
  assuming `materialize` generalizes to every strategy this composer routes.

**`materialize=True`'s "the sink's stripped value is genuinely the axis's exported output"
claim depends on this contract holding.** If a future change to `composer.py` ever returns
`final_carry` instead of `ys` for `Scan` (or otherwise stops returning the per-step stacked
value for any strategy), `materialize` on that axis would silently stop meaning what this spec
says it means — the exported program would still compile and run, just without exposing the
values a `materialize=True` sink was declared to expose. This section, plus AC-17/AC-17e's
tests actually reading the exported output, is the regression net for that risk; there is no
runtime assertion inside the composer itself that checks it (out of scope — see Task 3's "no
composer changes" note, which this section does not revise).

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

# src/xtrax/stages/boundaries.py  (MODIFIED — additive field, this pass)
class AxisBoundary(eqx.Module):
    fuse: Fuse | BoundaryCallable | None = eqx.field(static=True, default=None)
    tap: Tap | BoundaryCallable | None = eqx.field(static=True, default=None)
    sink: Sink | BoundaryCallable | None = eqx.field(static=True, default=None)
    materialize: bool = eqx.field(static=True, default=False)
    # NEW (this pass): declares `sink` (if set) as a *materializing* sink -- the executor's
    # existing per-step return value (ys, when fuse is None) already equals what `sink` receives
    # (GT #1-4), so `xtrax.export.pipeline.export_pipeline` may strip the sink call before tracing
    # instead of rejecting the whole plan (topology.py Rule 3, revised below). Zero effect outside
    # `xtrax.export` -- an eager run of the same boundary still fires `sink` exactly as before.
    # Never applies to `tap` (T -> T, participates in dataflow, never droppable on any target).
    # Default False: fully backward compatible in the sense that matters for this pass -- no
    # existing `AxisBoundary(...)` call site's traced behavior changes, and Rule 3 rejects an
    # undeclared sink exactly as before (AC-17b). **Qualified, C12:** this field is `static=True`,
    # so it is part of the eqx pytree's *aux_data*, i.e. part of the treedef, not the leaves. Any
    # already-pickled `AxisBoundary` instance persisted before this pass has no `materialize`
    # entry in its serialized state at all -- unpickling such an object is a genuine compatibility
    # question this spec does not resolve (untested here; flag for the fixer, don't assume
    # `default=False` alone makes old pickles load correctly). Separately, any `jax.jit` cache
    # keyed on the old (3-field) `AxisBoundary` treedef invalidates once this field exists --
    # expected and harmless (one recompile), but not literally "zero change," so this parenthetical
    # is scoped to "behavior," not "treedef identity."
    # **CAUTION (this revision's C8 -- distinct from the earlier, unrelated "C8" in the v1
    # Changelog about SpirvValidationResult's PR1 home): this is a PRECONDITION on `sink`, not
    # a proof about it.** The slot's type is
    # `Sink | BoundaryCallable | None` -- any callable can be assigned, and the `T -> None` return
    # contract above is enforced by nothing but this docstring (`Sink` is a `Protocol`, not a
    # runtime-checked base class beyond `runtime_checkable`'s method-presence check).
    # `tests/stages/test_executor.py`'s own `ReturningSink` fixture is a real, currently-tolerated
    # sink that returns a non-`None` value and is simply ignored by `_wrap_step`/
    # `_wrapped_transition` (both discard the call's return unconditionally) -- so GT #1-4's
    # "ys already equals what sink receives" claim is about the *argument* the sink is called
    # with, never about anything the sink itself does or returns. `materialize=True` verifies
    # only that the stripped call is behaviorally absent from the exported trace; it says nothing
    # about, and provides no guarantee for, whatever side effect a misbehaving sink implementation
    # would have performed had it not been stripped.
    # **Rule 3 must read this field via `getattr(boundary, "materialize", False)` (C4)** --
    # matching Rule 2's own duck-typed reads of `boundary.tap`/`boundary.sink` (`topology.py:150,
    # 158`, `getattr(..., "ordered", False)`). `topology.py`'s module docstring explicitly promises
    # structural compatibility with any library's plan objects with matching field names (e.g. a
    # parallel `aminx.tiling` `BatchPlanner`); a bare `boundary.materialize` attribute access
    # raises `AttributeError` on exactly a foreign boundary object that predates this field,
    # which is worse than the "false confidence" failure mode that docstring already warns
    # against for nominal `isinstance` checks. See AC-17f.

# src/xtrax/stages/topology.py  (MODIFIED — additive, default-False kwarg; Rule 3 is kind-based
# as of this pass, was slot-based before)
class MaterializeFuseConflictError(PlanTopologyError):
    """materialize=True + a non-None fuse on the same axis (this pass, NEW). fuse collapses the
    per-step stacked array materialize needs to expose as the axis's own output -- the two
    cannot coexist on one axis's export. Subclasses PlanTopologyError: every existing
    `except PlanTopologyError` call site (e.g. validate_export_safe's M5-documented unwrapped
    propagation) keeps working unmodified.

    **Real cause, stated honestly (C11):** this is a consequence of this pass's own "no executor
    changes" scope constraint, not a structural impossibility -- `execute_map_axis`/
    `execute_scan_axis` always call `_apply_fuse` on the stacked `ys` and return only the fused
    result (`executor.py:136-140`); exposing BOTH the pre-fuse per-step stream (what materialize
    needs) AND the fused value from the same axis would require the executor to return a second
    value, which this pass deliberately does not implement. **Workaround for a caller who wants
    both:** drop `fuse` from the `AxisBoundary`, keep `materialize=True`, and perform the
    equivalent reduction yourself, outside the exported function, on the materialized array
    `export_pipeline` returns."""

class MaterializeWithoutSinkError(PlanTopologyError):
    """materialize=True with sink=None (this pass, NEW) -- nothing to materialize; a caller
    error, not a silent no-op. Also a PlanTopologyError subclass.

    **Scope, stated explicitly (C11):** this fires from `validate_plan_topology`/
    `validate_export_safe`, i.e. at export-time plan validation, before any `jax.jit`/compile
    call -- it says nothing about, and does not fire during, an eager (non-export) run of the
    same `AxisBoundary`, where a `sink=None`+`materialize=True` combination is simply never
    reached by any eager code path in the first place (materialize is read only by
    `export_pipeline`)."""

class MultipleMaterializeAxesError(PlanTopologyError):
    """More than one axis has materialize=True in the same plan (NEW this revision, closes
    C6). Two independently-materialized axes have no defined output shape in this spec (see
    the "Two independently materialize=True axes" Non-goal) -- this converts what was
    previously a silent, unspecified-shape outcome into a named, caught, supported-non-goal
    error instead of an implementation accident. Also a PlanTopologyError subclass."""

def validate_plan_topology(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    *,
    export_safe: bool = False,
) -> None: ...
    # export_safe=True adds, in this order (all raising PlanTopologyError or a subclass):
    #   Rule 3 (CHANGED this pass -- kind-based, not slot-based). Per axis, reading
    #   `getattr(boundary, "materialize", False)` -- NEW, C4 -- not bare `boundary.materialize`,
    #   matching Rule 2's own `getattr(boundary.tap/.sink, "ordered", False)` duck-typing so a
    #   foreign (e.g. aminx.tiling) boundary object that predates this field is treated as
    #   materialize=False rather than raising AttributeError (AC-17f):
    #     1. tap is not None                                -> PlanTopologyError (unconditional;
    #        materialize never applies to tap)
    #     2. materialize and sink is None                   -> MaterializeWithoutSinkError
    #     3. sink is not None and not materialize            -> PlanTopologyError (UNCHANGED:
    #        an undeclared sink is rejected exactly as before this pass -- AC-17b)
    #     4. sink is not None and materialize and fuse is not None -> MaterializeFuseConflictError
    #     5. otherwise (materialize, sink set, fuse is None) -> passes (AC-17, NEW this pass)
    #   Rule 3, step 6 (whole-PLAN, not per-axis; NEW this revision, closes C6): after the
    #   per-axis loop above has run for every decision without raising, count how many axes
    #   passed step 5 with materialize=True. If more than one -> MultipleMaterializeAxesError,
    #   naming every offending axis. This is the detector the "two independently materialize=True
    #   axes" Non-goal needed and did not have before this revision -- previously such a plan
    #   would silently reach Task 20's batched-shape composer with an unspecified second output
    #   slot instead of being rejected at validation time. See AC-17h.
    #   Rule 4: strategy must be Vmap, SafeMap, Scan, or DedupGather (not Bucket/WhileCarry)
    #           -- CHANGED: DedupGather moved from the reject-list to the allow-list (Blocker 7)
    # Rule 2 (pre-existing, unconditional, export_safe-independent -- ordered Tap/Sink vs Vmap)
    # is UNCHANGED by this pass: it governs eager semantics that materialize never touches.

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
    path: Path                     # NEW this revision (closes C3): the real compiled-artifact
                                    # file (== CompileResult.path). Before this pass, the
                                    # `vmfb_bytes` comment below already CLAIMED "execution/parity
                                    # always go through .path internally" while no `.path` field
                                    # existed anywhere on `ExportResult` -- neither AC-17 nor
                                    # AC-14b had any specified route to the actual executed output
                                    # array (`ParityResult` carries only pass/fail + max_abs_diff,
                                    # never the arrays themselves). This field is that route: a
                                    # test wanting the real executed array calls
                                    # `run_native_vmfb(result.path, *concrete_inputs)` directly,
                                    # the same function `verify_native_parity` already calls
                                    # internally to build `.parity`.
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
    # NEW (this pass, materialize): after validate_export_safe passes (so every materialize=True
    # axis is already known to have no conflicting fuse), export_pipeline builds a *view* of
    # axis_boundaries where every materialize=True axis's boundary has sink replaced with None,
    # and passes THAT view -- not the caller's original axis_boundaries -- to
    # build_traceable_callable. build_traceable_callable/composer.py itself does not change and
    # does not know about materialize; the strip is entirely local to this function.

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
wrongly excluded `DedupGather`). **CHANGED, this pass (`materialize`):** Rule 3 is kind-based, not
slot-based — implement the five-step per-axis order from the Public API signatures section above
(`tap` always raises; `materialize=True`+`sink is None` raises `MaterializeWithoutSinkError`; an
undeclared `sink` (`materialize=False`) still raises `PlanTopologyError` unchanged;
`materialize=True`+`sink` set+a non-`None` `fuse` raises `MaterializeFuseConflictError`;
otherwise passes) **plus the whole-plan step 6 (NEW this revision, closes C6): after the
per-axis loop completes without raising, if more than one axis passed step 5 with
`materialize=True`, raise `MultipleMaterializeAxesError` naming every offending axis** — this is
the detector the "two independently `materialize=True` axes" Non-goal needed and did not have
before this revision. **Read `materialize` via `getattr(boundary, "materialize", False)`, never
bare `boundary.materialize` (NEW this revision, closes C4)** — matches Rule 2's own existing
duck-typed `getattr(boundary.tap, "ordered", False)` pattern (this file, Rule 2) so a foreign
plan's boundary object (e.g. `aminx.tiling`, per this module's own docstring's structural-
compatibility promise) that predates this field is treated as `materialize=False` rather than
raising `AttributeError`. Add all three new exception classes (`MaterializeFuseConflictError`,
`MaterializeWithoutSinkError`, `MultipleMaterializeAxesError`) to `topology.py` (subclassing
`PlanTopologyError`) and export them from its `__all__`. Add `AxisBoundary.materialize: bool =
eqx.field(static=True, default=False)` to `src/xtrax/stages/boundaries.py` per the Public API
signatures section (small, same-PR change — `boundaries.py` is the one existing base module this
new field belongs to). Add `src/xtrax/export/safety.py` with `ExportSafetyError`,
`DtypeNotSupportedError`, `ExportBlocker`, `check_export_safety()` (list-returning, **NEW**, closes
M5), and `validate_export_safe()` (raising, per the signature above — dtype check against
`target.supported_dtypes` only in PR1, over `abstract_inputs` leaves only; the closure-leaf scan
(AC-9b) and `optional_dtypes`/`request_features` plumbing land in PR2, but define both parameters
now so PR2 doesn't change either call signature). Rule 3's `materialize` cases raise
`PlanTopologyError`/its subclasses directly, unwrapped, exactly like Rule 3's other cases
today — they do **not** go through `ExportBlocker`/`check_export_safety`'s list (matching the
existing M5 design note: only the dtype rules use `ExportBlocker`).
**Files**: `src/xtrax/stages/topology.py` (modify), `src/xtrax/stages/boundaries.py` (modify —
add `materialize` field), `src/xtrax/export/safety.py` (create), `tests/stages/test_topology.py`
(modify — add `export_safe=True` cases, including a DedupGather-passes case and six `materialize`
cases: AC-17's positive pass, AC-17b's two regression raises, AC-17c's
`MaterializeFuseConflictError`, AC-17d's `MaterializeWithoutSinkError`, AC-17f's
missing-attribute duck-typing case, and AC-17h's `MultipleMaterializeAxesError` case)
**Gate**: `uv run pytest tests/stages/test_topology.py -q`
**Scope estimate**: ~195 LOC + tests
**Verifies**: AC-3, AC-4, AC-17b, AC-17c, AC-17d, AC-17f, AC-17h

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
**Note (`materialize`, this pass): no change needed here.** `build_traceable_callable` composes
whatever `axis_boundaries` dict it is given, unaware of `materialize` — the sink-stripping
transform lives entirely in `pipeline.py::export_pipeline` (Task 5), one layer above this
function, precisely so that AC-14a's direct composer/executor call (which never goes through
`export_pipeline`) keeps getting the real, un-stripped sink.
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
`validate_export_safe` per target → **(NEW, this pass) build a `materialize`-stripped view of
`axis_boundaries`** → `build_traceable_callable` (with `scan_init`, using the stripped view) →
`jax.export.export(jax.jit(callable))(*abstract_inputs)` → `compile_for_target` → (`EXECUTED`
only) `verify_native_parity(reference_fn(concrete_inputs), compile_result.path,
concrete_inputs, ...)`. Populate `ExportResult.path = compile_result.path` (**NEW this revision,
closes C3** — see the new field in Public API signatures; before this pass no field on
`ExportResult` gave a caller/test any route to the real executed output array). Set
`ExportResult.verified` per the per-level rule in the signatures section above (**CHANGED,
M17/M18**: unconditionally `False` for `CODEGEN_ONLY`). Enforce all-or-nothing across `targets=`
(no partial dict on a mid-loop exception — **NEW, M17/M18**). Raise `ValueError` up front if
`reference_fn is None` and any target is `EXECUTED` (**NEW, B3**).
**CHANGED (this pass, `materialize`; corrected again this revision, closes C2/C5):** add a
private helper (e.g. `_boundaries_for_export`, ~20 LOC) that returns `axis_boundaries` unchanged
when `None`, and otherwise returns a new mapping where every axis whose
`getattr(boundary, "materialize", False)` is `True` gets `sink` replaced with a **no-op sentinel
that preserves `.ordered`**, and every other axis is passed through **by identity** (not
reconstructed). **CORRECTED, C2:** v3 specified `sink=None`, which is observably different from
the original sink for an `ordered=True` `SafeMap` sink — `execute_map_axis`'s `_has_ordered_op`
check reads `boundary.sink.ordered`, so a bare `None` flips the branch from the ordered
`jax.lax.map(wrapped, xs)` path (ignores `batch_size`) to `safe_map(wrapped, xs,
batch_size=strategy.batch_size)` (a different lowering, and one that raises `ValueError` when
the axis's cardinality isn't divisible by `batch_size`). Fix: define a small private
`_StrippedSink` (or equivalent) with `ordered: bool` copied from the original sink's own
`.ordered` (`getattr(original_sink, "ordered", False)`) and a `__call__` that unconditionally
returns `None` without invoking any `io_callback` — this makes `_has_ordered_op` read
identically to the un-stripped boundary (so `execute_map_axis`/`execute_scan_axis` select
exactly the same branch the eager run would have taken), while still removing the actual
`io_callback` call from the trace, preserving the "no io_callback in the exported program"
property. See AC-17e. **`Scan` and unordered `SafeMap`/`Vmap` axes are unaffected by this
correction** — `execute_scan_axis` never reads `.ordered` at all, and an unordered sink's
branch selection was already identical between `None` and any no-op replacement.
**CORRECTED, C5:** build the stripped copy via `dataclasses.replace(b, sink=_StrippedSink(...))`,
never by re-listing `AxisBoundary(fuse=b.fuse, tap=b.tap, sink=..., materialize=b.materialize)`
field-by-field — the latter silently drops any future `AxisBoundary` field a fixer adds later
and downcasts a boundary subclass to the base `AxisBoundary` type. **Note: `eqx.tree_at` does
NOT work here** — every `AxisBoundary` field is `eqx.field(static=True)`, i.e. aux_data, not a
pytree leaf, and `eqx.tree_at` only rewrites leaves. `dataclasses.replace` works directly on any
frozen dataclass (which `eqx.Module` is) regardless of which fields are static. Call the helper
once, after `validate_export_safe` has already confirmed no
`MaterializeFuseConflictError`/`MaterializeWithoutSinkError`/`MultipleMaterializeAxesError`
applies, and pass its result — not the caller's original `axis_boundaries` — into
`build_traceable_callable`. `build_traceable_callable`/`composer.py` itself is not modified (see
Task 3's note). `__init__.py` becomes a thin re-export of `pipeline.py`'s public names plus the
other modules'.
**Files**: `src/xtrax/export/pipeline.py` (create), `src/xtrax/export/__init__.py` (modify —
re-export only)
**Gate**: `uv run pytest tests/export/test_pipeline_native_wasm32.py -q` — must include a test
using a real independent `reference_fn` (not `jax.jit(build_traceable_callable(...))`), a test
asserting all-or-nothing behavior when a second target raises, AC-17's test (a `materialize=True`
single-axis fixture whose `run_native_vmfb(result.path, *concrete_inputs)` output equals the same
boundary's recorded eager-sink values), AC-17e's test (an ordered `SafeMap` axis, `batch_size=4`,
`materialize=True` sink, over a **10-element** axis — deliberately not divisible by 4 — passes
export/execution without raising, because the stripped sentinel preserves `.ordered=True` and
the exported program takes the same one-element-at-a-time `jax.lax.map` path the eager run would
have; `strategy.batch_size` is silently ignored either way, matching the pre-existing documented
`SafeMap`+`ordered=True` behavior in `executor.py`'s module docstring, not a new behavior this
pass introduces), and AC-17g's test (`_boundaries_for_export` on a multi-axis boundary dict
touches only the `materialize=True` axis's `sink`, leaves that axis's `fuse`/`tap` untouched, and
returns every other axis's `AxisBoundary` as the exact same object — `is`, not just `==`).
**Scope estimate**: ~185 LOC
**Verifies**: AC-2, AC-3, AC-17, AC-17e, AC-17g

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
unstarted work, not something this task can honestly claim. **NEW, this pass:** include a
`materialize` subsection — the boundary kind, its backward-compatible default, the kind-based
Rule 3 it enables, the fuse-mutual-exclusion restriction, the caller-precondition framing (not a
theorem — this revision's C8 (unrelated to the earlier v1-Changelog "C8" about
`SpirvValidationResult`'s PR1 home): a non-conforming `Sink` implementation is not detected or fixed by `materialize`,
only its call is removed), and the memory-vs-streaming tradeoff (see the Changelog vs v2) — so
Task 22's PR3 addendum can cross-reference it rather than re-explaining it. **NEW this revision
(closes C11):** add one explicit sentence alongside the existing B6 footgun note: **for a
`materialize=True` axis, `build_traceable_callable(fn, plan, axis_boundaries)` called directly
with the caller's original (un-stripped) `axis_boundaries` does NOT return the same callable
`export_pipeline(fn, plan, ..., axis_boundaries=axis_boundaries)` actually exports for those same
arguments** — `export_pipeline` internally substitutes a stripped view before calling
`build_traceable_callable` (Task 5), so the direct-call callable still contains the sink's
`io_callback` while the exported one does not; a caller inspecting/testing
`build_traceable_callable`'s output directly is not looking at what got exported.
**Files**: `docs/api/export.md` (create), `docs/index.md` (modify — toctree entry)
**Gate**: `just audit-docs-build`
**Scope estimate**: ~85 lines of prose + 1 toctree line

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
**Note (`materialize`, this pass): no change needed here either**, for the same reason as Task
3 — `export_pipeline` (Task 5) already strips a `materialize=True` axis's `sink` to a no-op
sentinel (Task 5's `_StrippedSink`, C2) *before* calling this function, so `batched_transition`'s
inline `boundary.sink(carry)` call simply invokes the sentinel's no-op `__call__` for that axis,
with zero code changes here. AC-14a's direct composer/executor call (never through
`export_pipeline`) still gets the real, un-stripped sink, exactly as its own text requires.
**Precondition on `batched_transition` for `materialize` (NEW this revision, closes C7 — a
caller obligation, not a composer change; this Task's own code is still unmodified by this
note):** for the exported, `materialize`-stripped array to equal what the (un-stripped) sink
would have received, `batched_transition` must call `boundary.sink(v)` with the **exact same
value `v`** it returns as its own scanned `y` (i.e. `boundary.sink(v); return new_carry, v`).
`TestBatchedShapeVmapOfScanPreservesOrder`'s own `batched_transition`
(`tests/stages/test_nested_ordering.py:112-117`) happens to satisfy this — it calls
`boundary.sink(carry)` then `return carry + x, carry`, sinking and returning the *same*
pre-update `carry` — but this is a property of that specific hand-written transition, not
something `jax.lax.scan` or this composer enforces structurally, unlike the flat single-axis
path: `execute_map_axis`/`execute_scan_axis`'s own `_wrap_step`/`_wrapped_transition`
(`executor.py:125-131,238-245`) compute `y` once and pass that exact `y` to both `tap` and
`sink`, so the invariant holds by construction there. Task 20's hand-written inline sink call has
no such structural guarantee — a `batched_transition` that sinks one value but returns a
different value as `y` would silently break `materialize`'s "no new plumbing" claim for the
batched-shape recipe specifically. Document this precondition in this Task's own docstring/
comment for `batched_transition`, and in Task 21's AC-14b fixture.
**Files**: `src/xtrax/export/composer.py` (modify)
**Gate**: see Task 21
**Scope estimate**: ~175 LOC
**Verifies**: AC-14a, AC-14b, AC-15

#### Task 21: Multi-axis certification tests (CHANGED — split per AC-14a/AC-14b; CHANGED again
this pass — AC-14b strengthened via `materialize`)
Create `tests/export/test_multi_axis.py` with three classes mirroring
`tests/stages/test_nested_ordering.py`'s structure:
1. **AC-14a — ordering only**: a positive stress test (batched-shape recipe, varying
   batch/steps like `TestBatchedShapeVmapOfScanPreservesOrder`, `N_TRIALS=20`) calling the
   composer/executor layer directly with an ordered `Sink` attached, asserting a test-double
   sink's recorded call order matches the expected `(lane, step)` sequence. **Never calls
   `export_pipeline`, never constructs an `ExportResult`.** Unchanged this pass.
2. **AC-14b — export/parity, materialize-based (CHANGED this pass, was "sink-free"; CHANGED
   again this revision, closes C3 and this revision's C9):** the same 2-axis shape's `batched_transition` (subject to
   Task 20's new sink-equals-`y` precondition, C7) has its inner-axis `Sink` declared
   `materialize=True` (no `fuse` on that axis), and the value each step sinks encodes its own
   `(lane, step)` identity (e.g. `y = lane * STEPS + step`). Run this **same, sink-attached**
   boundary directly through `export_pipeline(targets=(NATIVE,))` with an independent
   `reference_fn` that computes the same `(lane, step)`-encoded array without going through the
   composer.
   - **New middle leg (closes this revision's C9):** before compiling, run the same `materialize`-stripped
     composed callable directly in pure JAX (`jax.jit(build_traceable_callable(fn, plan,
     stripped_boundaries))(concrete_inputs)`, using `export_pipeline`'s own stripping helper or
     an equivalent call) and assert it matches `reference_fn(concrete_inputs)` exactly, before
     ever calling `compile_for_target`/IREE. This isolates "stripped composition is wrong" from
     "IREE lowering is wrong" — previously AC-14a-green+AC-14b-red left three undistinguished
     suspects (a composer bug AC-14a's un-stripped path doesn't exercise, a composer bug specific
     to the stripped path, or an IREE bug); this leg removes the middle one as a candidate on its
     own.
   - Then assert `.verification_level is VerificationLevel.EXECUTED`, `.verified is True`
     (parity against `reference_fn`), **and** — **corrected, closes C3**: call
     `run_native_vmfb(result.path, *concrete_inputs)` **directly** (not "via the `ExportResult`/
     `verify_native_parity` path", which exposes only `ParityResult`'s pass/fail + scalar diff,
     never the array) — decodes back to the exact expected `(lane, step)` sequence in array
     order. This no longer needs a separately-built `Fuse`/no-boundary variant, and it certifies
     ordering on the exported artifact's own output, not only on a pre-export test-double's
     recorded calls (AC-14a's job).
3. **AC-15 — negative**: the lane-dependent counter-example shape, asserting
   `MultiAxisCompositionError` is raised with the certified message substring, at the
   composer/executor layer (no `ExportResult` claim here either). Unchanged this pass.
**Files**: `tests/export/test_multi_axis.py` (create)
**Gate**: `uv run pytest tests/export/test_multi_axis.py -q`
**Scope estimate**: ~260 LOC
**Verifies**: AC-14a, AC-14b, AC-15

#### Task 22: Docs addendum
Document the multi-axis composition contract (including why it's split into an ordering
certification and a separate exported-and-verified leg, AC-14a/AC-14b) and the
lane-dependent-nesting refusal in `docs/api/export.md`. **CHANGED, this pass:** AC-14b's leg is
now `materialize`-based (the same `Sink`, declared `materialize=True`, not a separately-built
sink-free variant) — update the prose accordingly, and cross-reference the `materialize` section
Task 9/19 already documents rather than re-explaining it. **CHANGED again this revision (closes
C7 and this revision's C9):** document Task 20's sink-equals-`y` precondition on `batched_transition` for
`materialize` to work correctly in the batched-shape recipe, and mention AC-14b's three-leg
structure (un-stripped composition (AC-14a) / stripped composition in pure JAX / stripped
composition compiled-and-executed) so a reader understands what each leg isolates. Same M15
caveat as Tasks 9/19.
**Files**: `docs/api/export.md` (modify)
**Gate**: `just audit-docs-build`
**Scope estimate**: ~45 lines of prose

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
| `materialize=True` allocates the full `[steps, ...]` (or `[steps, lanes, ...]` — **CORRECTED, C10**: v3 said `[lanes, steps, ...]`, inverted; `jax.lax.scan` stacks on the leading step axis) stacked array where a live `io_callback` sink would have streamed and discarded each step — a caller who declares `materialize=True` on a long scan over large per-step values can hit real device-memory pressure they didn't have before | Opt-in, default `False`; documented explicitly in the `materialize` field's docstring and Task 9's docs subsection, not left as a surprise a fixer or caller has to discover empirically |
| A caller declares `materialize=True` believing it also drops the sink's *eager* (non-export) execution cost, or that it changes Rule 2's ordered-Vmap restriction | `materialize`'s docstring and the Changelog vs v2 state explicitly that it has zero effect outside `xtrax.export`, and that Rule 2 is completely unmodified and unconditional — this is a documentation discipline risk (like the `reference_fn`-independence risk above), not something the type system enforces |
| PR1 lands Rule 3's kind-based revision and the `AxisBoundary.materialize` field in `src/xtrax/stages/{topology,boundaries}.py` — both *base* modules outside `xtrax.export` — so a bug here has blast radius beyond the export subpackage, unlike every other PR1 change | Both changes are strictly additive (new field default `False`, new exception subclasses of the existing `PlanTopologyError`); `tests/stages/test_topology.py`'s pre-existing cases (Rule 1/2/4, and Rule 3's plain-tap/undeclared-sink cases) must still pass unmodified after Task 2, which is the direct regression check for "outside xtrax.export, nothing changed" |
| M1/M2/M14's dispositions in the Changelog are this revision's best reconstruction (M1) or an explicit non-fix pending live measurement (M2), and M14 is entirely unaddressed for lack of any defining prose in the findings doc | Flagged prominently in the Changelog's "named but not resolved" section; a future reviewer with either live CI access or the missing M14 definition should revisit before treating PR1/PR2 as fully closing Part 5's mechanical/CI bucket |

## References

- **This pass's dispatch prompt** (task `260901_xtrax-export-webgpu`) supplied
  orchestrator-verified ground truth against current `main`'s `src/xtrax/stages/{executor,
  boundaries,topology}.py`, treated as authoritative and not re-derived: `execute_scan_axis`/
  `execute_map_axis`'s return-value equivalence to sink-received values when `fuse is None`
  (`executor.py:136-140,220-248`); `Sink`'s `T -> None` contract discarding its return
  (`boundaries.py:66-81`) versus `Tap`'s `T -> T` dataflow participation (`boundaries.py:47-64`);
  `AxisBoundary.sink`'s untyped `Sink | BoundaryCallable | None` slot and Rule 3's current
  slot-presence (not kind) rejection; WebGPU/SPIR-V's total absence of a host-callback mechanism;
  wasm32's `io_callback`→JS-import path remaining a deferred non-goal (no published npm package
  for IREE's browser runtime). Read directly this pass to confirm before building on them.
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

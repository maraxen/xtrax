---
category: specs
title: "WebGPU export: route selection, honest verification levels, and the tech-debt register"
description: "Re-specification of WebGPU for xtrax.export after AC-8 was falsified — four mutually-exclusive routes, a recommendation (R2″) that adds structural SPIR-V validation without moving any verification level, and a TD-WGPU debt register"
task_id: 260910_webgpu-respec
status: draft
---

# Specification: WebGPU export route

## Overview

WebGPU is not one decision. This spec enumerates the mutually-exclusive routes by which
`xtrax.export` could reach a browser, recommends the one that ships no false claim, and
records what each remaining route would cost.

## The falsified premise

The prior spec, `.praxia/docs/specs/260901_xtrax-export-webgpu.md`, made WebGPU shader
validity an acceptance criterion. Measurement falsified it before implementation
(`.praxia/docs/research/260901_webgpu-export-measurement-pass.md`). AC-8 is marked
**VOID** at that spec's line 551:

> IREE's Vulkan HAL passes dispatch parameters through push constants, which are not a
> WebGPU capability, so naga rejects every SPIR-V module IREE emits; IREE 3.11 registers
> no webgpu backend and no flag removes them. The B9 disposition below is **inverted**: it
> scoped the push-constant risk out on the grounds that `create_shader_module` never
> exercises it, but that call is exactly where naga validates and exactly where the
> rejection happens.

AC-8b is marked **VOID** at line 571, as it depended on the `validate_webgpu()` that AC-8's
disposition removed from the plan.

The measured error, reproduced identically on a local WSL2 box and on `ubuntu-latest`
(`.praxia/docs/research/260901_webgpu-export-measurement-pass.md:22-26`, `:158-165`):

```
Shader validation error: Global variable [1] '__push_constant_var__' is invalid
  = Capability Capabilities(IMMEDIATES) is not supported
```

Two properties of that finding govern everything below.

1. **It is an IREE ABI property, not an xtrax defect.** No change under `src/xtrax/`
   affects it. Eight flag combinations were tried; the only partial result,
   `--iree-scheduling-optimize-bindings=false`, removed push constants from one dispatch of
   two and is data-dependent, so it cannot underpin a gate
   (`260901_webgpu-export-measurement-pass.md:79-94`).
2. **The criterion was dishonestly satisfiable.** wgpu exposes push constants as a
   native-only `immediates` feature. A device requested with no features validates 0/2 of
   IREE's modules; the same device with `required_features=["immediates"]` validates 2/2
   (`:55-59`). The green light was reachable without moving one step closer to a browser.

The current tree already encodes this correction. `VULKAN_SPIRV` and `METAL_SPIRV` are both
`CODEGEN_ONLY` (`src/xtrax/export/targets.py:212-230`), no target is registered at
`VALIDATED` (asserted at `tests/export/test_targets.py:33-36`), `xtrax.export.spirv` ships
extraction with no validator and says why (`src/xtrax/export/spirv.py:1-16`), and
`docs/api/export.md:240-246` states the limitation in public docs.

## Constraint governing every route

**No artifact, test, docstring, CHANGELOG entry, or CI badge produced under this spec may
imply browser compatibility without browser-execution evidence.** A gate that can pass
vacuously is a defect. This is the failure mode the prior sprint hit and the one the active
release-trail sprint exists to remove.

## Routes

The four routes are mutually exclusive *as ways of delivering WebGPU*. Two further items, R2′
and R2″, are listed with them because they are the things most likely to be confused with a
WebGPU route, and the distinction is the substance of this spec: both deliberately deliver no
WebGPU claim. R2′ and R2″ differ from each other in exactly one respect — whether
`VULKAN_SPIRV`'s verification level changes.

### R1 — Wait for IREE

The WebGPU epic is [IREE #13702](https://github.com/iree-org/iree/issues/13702). Its
end-to-end path landed upstream in
[#24463](https://github.com/iree-org/iree/issues/24463), closed 2026-05-18: a compiler
backend emitting WGSL executables, plus a JavaScript-hosted HAL driver that submits them
through the browser/Node WebGPU API from a freestanding wasm32 runtime -- explicitly
*not* an Emscripten port and not a native Dawn HAL. The route is therefore no longer
"nobody has written a WebGPU HAL for IREE"; one exists and is merged.

What it is blocked on instead is
[#24650](https://github.com/iree-org/iree/issues/24650), open since 2026-06-29:
`webgpu-spirv` compilation fails on any dispatch carrying push constants, with Tint
reporting ``use of variable address space 'immediate' requires the
immediate_address_space language feature, which is not allowed in the current
environment``. **That independently confirms this spec's central finding** -- push
constants are the blocker -- while reclassifying it from "no backend exists" to "a live
bug in a backend that does". Engineering cost zero; agency zero; latency bounded by
someone else's bug rather than unbounded.

**Trigger condition.** `iree-compile --iree-hal-list-target-backends` lists a `webgpu`
or `webgpu-spirv` backend. **Not met on the pinned toolchain, re-measured 260910:** the
installed `iree-base-compiler 3.11.0rc20260316` lists exactly
`cuda  llvm-cpu  metal-spirv  rocm  vmvx  vmvx-inline  vulkan-spirv`, reproducing
`260901_webgpu-export-measurement-pass.md:69-73` exactly.

**A newer build does not help, and this is the load-bearing measurement.** Today's
nightly -- `iree-base-compiler 3.12.0rc20260910`, resolved from
`--find-links https://iree.dev/pip-release-links.html` -- lists the *same seven*
backends, four months after #24463 closed. The reason is not that the code is missing:
`compiler/plugins/target/WebGPUSPIRV/` exists in the IREE tree, and
`compiler/plugins/iree_compiler_plugin.cmake:47-48` adds it only under
`if(IREE_TARGET_BACKEND_WEBGPU_SPIRV)`. It is a build-time plugin, and **no published
wheel enables it** -- not the pinned rc, not stable `3.11.0` (which is stable PyPI's
ceiling against the extra's `>=3.11,<4`), not the 3.12 nightly.

So the honest statement of R1's latency is neither "nobody has written it" nor "wait for
the next release". It is: *the backend exists upstream, no installable artifact carries
it, and even once one does, #24650 rejects the push constants this package's dispatches
carry.* **Three conditions must hold**, not one -- a wheel built with
`IREE_TARGET_BACKEND_WEBGPU_SPIRV`, #24650 fixed, and the browser glue of TD-WGPU-07 --
and only the first two are upstream's to grant. Building IREE from source with the
option enabled is possible and is *not* proposed here; it would put a bespoke toolchain
on the critical path of a package whose export story is otherwise pip-installable.

**What becomes cheap on trigger.** Nearly everything. `Target` is plain data and
`compile_for_target` reads the backend and flags off the object rather than branching
(`src/xtrax/export/compile.py:145-148`), so a new backend is one registry entry plus the
mechanical checklist in "What a new target costs". The dump-and-filter path already works
(`src/xtrax/export/spirv.py:53-74`), and `--iree-hal-dump-executable-binaries-to` produced
two `.spv` files with nothing else needing filtering (`:120-122`).

**What stays expensive on trigger.** Execution. A registered backend produces a shader;
running it in a browser still needs the glue layer of TD-WGPU-07 and a headless-browser CI
lane. R1 unblocks codegen, not verification.

**Cost of adoption:** a watch item. No code.

### R2 — `VALIDATED_NATIVE` via wgpu with `immediates` enabled

Re-add `wgpu`, request a device with `required_features=["immediates"]`, and validate IREE's
SPIR-V through naga. Measured to turn 0/2 into 2/2 (`:55-59`).

**What it would legitimately establish.** naga is a shader front end, not a bit checker: it
builds a typed IR and must be able to emit WGSL/HLSL/MSL from the module. It would therefore
be expected to reject a class of malformed module a structural validator accepts — a codegen
regression that emitted a truncated dump, a dangling type reference, or a mis-decorated
binding. **Unmeasured: no comparison run between naga and `spirv-val` exists.** The relation
is non-subsumption in both directions. `spirv-val` also rejects specification violations that
naga's front end normalizes away, so neither tool's verdict implies the other's, and R2 is
not a strict superset of R2″.

**What it cannot establish.** Anything about browsers. `immediates` is precisely the
non-web extension — absent from the W3C WebGPU spec, so no browser offers it
(`:51-53`). Enabling it makes the one capability that a browser refuses invisible to the
check.

**Why this spec does not take R2.** The cost is a `wgpu` dependency (absent from
`pyproject.toml` today), a Vulkan ICD apt step in CI, and a permanent naming hazard: the
package is named for WebGPU, the level would be named for WebGPU-adjacency, and every
future reader must re-derive that `immediates` voids the association. The prior sprint
proves that re-derivation fails. R2″ buys structural signal with none of the hazard, and the
non-subsumption above means R2 was never a superset of it in the first place.

**Cost of adoption:** ~200 LOC, one new extra, one apt step, and a standing misreading risk.

### R2′ — Structural SPIR-V validation, promoted to `VALIDATED`

Validate the extracted modules with `spirv-val` from SPIRV-Tools: a specification
conformance checker with no device, no adapter, no feature negotiation, and no WebGPU
semantics. Register `VULKAN_SPIRV` at `VerificationLevel.VALIDATED`.

**Provenance.** R2′ is not one of the four options the research pass enumerated
(`.praxia/docs/research/260901_webgpu-export-measurement-pass.md:175-193`). It was derived
afterward, from that pass's own finding that the extracted bytes are checked for a magic
number and nothing else.

**This is not a WebGPU route and must never be described as one.** It establishes that the
module IREE emitted is a well-formed SPIR-V module. It says nothing about which devices
accept it, and by construction cannot be misread as saying so — there is no browser-shaped
device in the loop to be mistaken for a browser.

**Why it is worth doing anyway.** Today the extracted bytes are checked only for the
`0x07230203` magic (`src/xtrax/export/spirv.py:36-50`) and a size window
(`tests/export/test_size_budget.py:117-122`). A dump that regressed to a magic-carrying
but structurally broken module passes both. `spirv-val` closes that, and it closes it for
the artifact that actually ships rather than a transformed one.

**Two measured preconditions, either of which cancels this route** (Task 0):

- **M1 — availability.** `spirv-val --version` must succeed on `ubuntu-latest` from a
  package the CI job can install, and be reachable from the test process. Unmeasured.
- **M2 — verdict.** `spirv-val --target-env vulkan1.3 <module>` must *accept* every SPIR-V
  module IREE emits for the `tests/export/test_size_budget.py:67` fixture — **4780 B of
  extracted SPIR-V in total** for that fixture (`tests/export/test_size_budget.py:16`). The
  target environment is named because `spirv-val`'s accepted ruleset differs materially
  between `--target-env` values, so an unpinned environment makes M2 unfalsifiable. If it
  rejects a module, the route either found a real IREE bug or is measuring something other
  than what this spec assumes; either way it stops and the finding is recorded before any
  code lands.

The `6264 B` and `3344 B` pair recorded at
`260901_webgpu-export-measurement-pass.md:161-163` belongs to that pass's own larger probe
program, not to this fixture. M2 is stated against the fixture's 4780 B so a fixer extracting
that figure does not read it as a failed extraction.

M1 and M2 gate R2″ identically: they are properties of `spirv-val`, not of the verification
level.

**Cost of adoption:** ~150 LOC plus one CI package install, gated on M1 and M2.

**Why this spec does not take R2′.** The level promotion is the whole difference between R2′
and R2″, and it buys nothing measurable while removing two invariants. The measurement is in
R2″ below.

### R2″ — Structural validation without a level change

R2′'s validator, R2′'s M1 and M2, and R2′'s `spirv_validation` payload, with `VULKAN_SPIRV`
left at `VerificationLevel.CODEGEN_ONLY` (`src/xtrax/export/targets.py:212-218`).
`ExportResult.verified` stays unconditionally False for it; the validator's verdict is read
off `result.spirv_validation.valid`.

**What the promotion would have bought: nothing, measured.** `verified` has no programmatic
consumer anywhere in the tree. Grepping `\.verified` across `src/`, `tests/`, `controller/`,
`scripts/`, `docs/`, and `agent_assets/`: every hit under `src/xtrax/export/` is the field's
own definition or docstring (`src/xtrax/export/pipeline.py:41-44`, `:56`, `:256`;
`src/xtrax/export/targets.py:50-56`). Nothing branches on it. The only consumer-facing use is
one documentation line, `docs/api/export.md:126`. A consumer reading
`result.spirv_validation.valid` instead gets identical information content plus the
validator's name, its version, and its error string — strictly more than the bool.

**The "one field to poll across targets" defense self-destructs.** That defense says a
consumer wants a single uniform bool rather than a per-level payload. But this spec's own
quoted contract (`src/xtrax/export/pipeline.py:41-44`) tells the caller to read
`verification_level` in order to interpret `verified` at all. A consumer polling `verified`
alone is precisely the misreader this document identifies as the prior sprint's failure mode.
The defense therefore argues for serving that consumer the field they must not read alone.

**What the promotion would have cost: two invariants, replaced by a weaker one.** It deletes
`test_no_target_is_registered_as_validated` (`tests/export/test_targets.py:33-36`) and
narrows the runtime guard at `src/xtrax/export/pipeline.py:208-217`, leaving AC-1 as the
remaining barrier. AC-1 checks target *names*. The guard is a different and stronger safety
property: under it, nobody can register any target at `VALIDATED` without wiring validation
for that target first. AC-1 is an addition to that guarantee, not a substitute for it, and
under R2″ both the test and the guard stay exactly as they are.

**What R2″ still gains.** Everything R2′ was worth doing for. `spirv_validation` closes the
magic-number-only gap on the artifact that actually ships, and it does so with no level
change to argue about, no test deleted, and no guard narrowed.

**Cost of adoption:** ~130 LOC plus one CI package install, gated on M1 and M2 — R2′'s cost
minus the promotion and the two invariants it would have spent.

### R3 — Lower push constants post-hoc

Rewrite the emitted SPIR-V, replacing the `__push_constant_var__` global with a uniform or
storage buffer binding, then validate the rewritten module.

**Assessment: reject.** Three independent reasons, in increasing order of severity.

1. **The shader is half of an ABI.** Push constants are written host-side by
   `vkCmdPushConstants` from IREE's Vulkan HAL. Moving them into a buffer requires the host
   to allocate, bind, and fill that buffer at the matching set/binding. IREE has no WebGPU
   HAL to change (R1's trigger condition is exactly its absence), so the rewritten shader
   has no host that can drive it.
2. **The binding budget was never reached.** WebGPU caps
   `maxStorageBuffersPerShaderStage` at 8. The prior spec scoped that out on the grounds
   that module validation never reaches it — true, and true for the same reason the push
   constants blocked first (`260901_xtrax-export-webgpu.md:567-570`). Adding a binding per
   dispatch spends from a budget that has never been measured against a real xtrax pipeline.
3. **It produces the worst false green available.** A rewritten module that validates looks
   like progress toward execution while being strictly further from it than the artifact
   xtrax actually ships. The research pass reached the same disposition:
   "Validates a *transformed* artifact rather than the shippable one, and needs matching
   host-side binding changes to ever execute. Weak evidence, real cost."
   (`260901_webgpu-export-measurement-pass.md:183-186`).

**Cost of adoption:** a SPIR-V rewriter, an IREE HAL fork, and a verification story that
gets weaker as the engineering gets larger.

### R4 — Leave IREE for the WebGPU leg

Emit WGSL by a non-IREE route, keeping IREE for `native`, `native-portable`, and `wasm32`.

**What xtrax gives up.** The entire shared compile path. `compile_for_target` takes
StableHLO text and a `Target` and returns a `CompileResult`
(`src/xtrax/export/compile.py:115-161`); everything downstream — `SIZE_BUDGET_BYTES`, the
SPIR-V dump, `ExportResult`, the parity oracle via `run_native_vmfb`
(`src/xtrax/export/compile.py:227-264`) — is shaped by that return. A WGSL emitter shares
the `jax.export.export` front half (`src/xtrax/export/pipeline.py:236`) and nothing after
it. `Target` would need a second, non-IREE variant, or a sibling registry.

**What xtrax gains.** An artifact that can actually run in a browser, and therefore an
honest browser claim — the only route here that can produce one.

**What is unmeasured.** Whether a StableHLO-to-WGSL or JAX-to-WGSL path exists that xtrax
could depend on rather than write. This spec does not assert one does. The measurement that
settles it is a survey with a concrete pass/fail: compile the `test_size_budget.py:67`
fixture model (`jnp.tanh(x @ W1) @ W2`) to WGSL through a candidate toolchain, and run it
under a headless browser against the same independent oracle
(`tests/export/conftest.py:69-82` shows the oracle shape). Nothing short of that
distinguishes R4 from R3.

**Cost of adoption:** its own epic. Out of scope here.

## Recommendation

**Take R1 as the standing position on WebGPU, and R2″ as the only engineering, conditional
on M1 and M2.**

Stated as claims rather than routes:

- xtrax ships **no** WebGPU claim, and a gate makes shipping one accidentally fail loudly
  (AC-5).
- `VULKAN_SPIRV` stays `CODEGEN_ONLY` and gains a structural-conformance attachment.
  `spirv_validation` reports whether the modules that actually ship are well-formed SPIR-V;
  `verified` stays False, because nothing was executed. The larger true statement is made in
  the field that can carry the validator's name, version, and error, rather than in a bool
  with no consumer.
- No test is deleted and no guard is narrowed to buy that. `VerificationLevel.VALIDATED`
  stays unused and `export_pipeline` keeps refusing it outright.
- The debt that would block or degrade a future WebGPU target is written down with owners
  and resolutions instead of being rediscovered (TD-WGPU register).

R2 is rejected for its naming hazard rather than its technique. R2′ is rejected because the
promotion to `VALIDATED` buys no consumer signal that `spirv_validation` does not already
carry, while spending two invariants. R3 is rejected on feasibility. R4 is deferred pending a
measurement that this spec names but does not fund.

**If M1 or M2 fails, R2″ is cancelled and the spec degrades to R1 alone.** Tasks 4 through 8
are cancelled with it, so the surviving acceptance criteria are exactly those of the three
Task-0-independent tasks: **AC-5, AC-5a, AC-5b, AC-5c, AC-5d** (Task 1), **AC-7** (Task 3),
and **AC-8** (Task 2). Nothing else survives. AC-6b does not: it constrains the validator
wiring Task 6 lands, which is cancelled with the rest. In particular AC-11 does not: it mandates docs
saying `vulkan-spirv` is checked for structural conformance, which would be false in the
branch where the validator was never built. That degradation is a planned outcome, not a
failure of the spec.

## The verification-level question

`VerificationLevel.VALIDATED` exists (`src/xtrax/export/targets.py:55-56`), no target uses
it, and `export_pipeline` refuses it outright (`src/xtrax/export/pipeline.py:208-217`).
Three questions have to be settled, and the third is the one that matters.

### Does R2″ reuse `VALIDATED` or add a level?

**Neither. No level changes.** `VULKAN_SPIRV` stays `CODEGEN_ONLY`, `VALIDATED` stays unused,
and the refusal guard stays as written. The question is recorded because it was live under
R2′ and because a future reader will ask it again.

`VALIDATED`'s docstring reads "SPIR-V accepted by a shader validator; never executed", which
describes `spirv-val` exactly — that fit is what made R2′ tempting. It is not sufficient: a
level's meaning is only as strong as the invariants guarding who may claim it, and the
measurement in R2″ shows the promotion buys no consumer signal to pay for weakening them.

For the record, the research pass floated `VALIDATED_NATIVE`
(`260901_webgpu-export-measurement-pass.md:182-184`); this spec rejects that name on two
grounds, and they survive R2″ unchanged: it was coined for R2, whose distinguishing property
was a *native-only device feature* that structural validation does not use, and "native"
already names two targets (`src/xtrax/export/targets.py:125`, `:175`), so a level called
`VALIDATED_NATIVE` on a target called `vulkan-spirv` reads as a contradiction.

The **fields** do need to change, and this is the one part of R2′ that R2″ keeps in full.
`SpirvValidationResult` carries `adapter_type`, `backend`, and `device_name`
(`src/xtrax/export/spirv.py:89-93`), shaped for a wgpu adapter that neither route
constructs. Populating `device_name="llvmpipe"` would be false and `device_name="N/A"` would
be noise. They are replaced by `validator` and `validator_version`. A third field, `module_count`, is
added in the same change: `valid=False` alone cannot distinguish "a module was rejected" from
"there was no module to validate", and AC-4's `claim` must not report the first sentence for
the second state — which AC-3 makes the *default* state on the fake path.

This is a public-API break — the class is in `__all__` (`src/xtrax/export/spirv.py:21-27`) —
and **no deterministic gate observes it**, measured. `audit-added-types-diff` collects only
`FunctionDef` nodes (`src/xtrax/devtools/gates/added_types_diff.py:47-55`) and diffs
signatures (`_signature_changed`, `:67-79`); a dataclass field is neither.
`scripts/audit_public_api.py` reads only `src/xtrax/__init__.py`'s `__all__`, `_LAZY`, and
`tier1_exports` (`:199-262`), and for subpackages asserts only that `__all__` is non-empty
(`:164-179`) — and `SpirvValidationResult` is not a root export. A CHANGELOG entry is
therefore mandatory *and* must be asserted by a literal grep, because nothing else will catch
the rename. AC-10 does that; TD-WGPU-11 records the general gap.

### Is wiring `VALIDATED` justified by dead weight?

**No, and the argument must not be made.** The level is already fully exercised:
`_verified_for` is tested at `VALIDATED` in three cases
(`tests/export/test_pipeline_native_wasm32.py:111`, `:117`, `:127`), the refusal guard in
three (`:73-91`), and `SpirvValidationResult` is constructed three times across two files
(`tests/export/test_spirv.py:73`, `tests/export/test_pipeline_native_wasm32.py:112`, `:118`).
Nothing here is uncovered and nothing is unreached. The case for structural validation rests
entirely on the new signal it produces; the level is not part of that signal, which is why
R2″ leaves it unused indefinitely — the disposition this subsection recommends.

### Does a never-executed artifact deserve `verified=True`?

**No.** This is the false-green question, and it is settled by measurement rather than by
taste.

**The case against `verified=True` is that the field has no reader.** `verified` has no
programmatic consumer anywhere in the tree. Grepping `\.verified` across `src/`, `tests/`,
`controller/`, `scripts/`, `docs/`, and `agent_assets/`: every hit under `src/xtrax/export/`
is the field's own definition or docstring (`src/xtrax/export/pipeline.py:41-44`, `:56`,
`:256`; `src/xtrax/export/targets.py:50-56`). Nothing branches on it. The single
consumer-facing use is one documentation line, `docs/api/export.md:126`. Flipping a bool that
nothing reads is not a feature; the *only* thing the flip reliably produces is a new way to
be misread.

**The defense that it is "one field to poll across targets" defeats itself.** The package's
own contract tells the caller to "Read `verification_level` to tell that apart from a genuine
failure" (`src/xtrax/export/pipeline.py:41-44`). A consumer who polls `verified` alone is
exactly the misreader identified as the prior sprint's failure mode. An argument that the
field must stay uniform *for that consumer's benefit* is an argument for serving them the
field they must not read alone.

**What the field's meaning still is.** `verified` has never meant "this artifact is correct";
it means "this level's own check passed". That meaning is unchanged here, and
`_verified_for`'s contract at `src/xtrax/export/pipeline.py:120-130` is untouched. Under
`CODEGEN_ONLY` the level's own check is compilation, so `verified` is False unconditionally
(`:130`) even when a structural validation is attached — and that is the honest reading, not
a limitation to work around.

**Resolution.** The validator's verdict is reported in the field built to carry it,
`spirv_validation`, which gives a consumer strictly more than the bool would: `valid`, the
validator's name, its version, and its error string. `verified` stays False. The residual
misreading risk — a result carrying `verified=False` beside `spirv_validation.valid=True`,
which invites the opposite misreading — is met with a mechanism instead of a convention.
`ExportResult` gains a **`claim` property**, derived from `verification_level`, `verified`,
and `spirv_validation`, returning one sentence stating exactly what was established. A
property rather than a field, so no constructor signature changes and no existing call site
breaks. The six claims:

| level | `verified` | `spirv_validation` | `claim` |
|---|---|---|---|
| `EXECUTED` | True | — | "executed against an independent oracle; numerics matched" |
| `EXECUTED` | False | — | "executed against an independent oracle; numerics did not match" |
| `CODEGEN_ONLY` | False | `valid=True` | "compiled; the SPIR-V is structurally valid; not executed on any device" |
| `CODEGEN_ONLY` | False | `valid=False`, `module_count > 0` | "compiled; the SPIR-V was rejected by the structural validator" |
| `CODEGEN_ONLY` | False | `valid=False`, `module_count == 0` | "compiled; no SPIR-V module was extracted, so nothing was validated" |
| `CODEGEN_ONLY` | False | None | "compiled only; nothing further was established" |

**The last two rows must not be collapsed, and `module_count` is why they can be told apart.**
AC-3 makes an empty extraction `valid=False`, so keying the rejection sentence on `valid=False`
alone reports *"the SPIR-V was rejected by the structural validator"* for an artifact where
nothing was ever handed to a validator. That is not an edge case: the fake compiler writes no
dump, so `spirv_bytes` is exactly `{}` (`tests/export/test_spirv.py:136-145`), and Task 7 makes
that the primary fake path — the mechanism introduced here specifically to make claims
unmisreadable would assert a false sentence on **every** fake-path `vulkan-spirv` export. The
discriminator is a count rather than the `.error` string, because matching on error prose is
exactly the kind of convention this section exists to replace with a mechanism.

`CODEGEN_ONLY` has no `verified=True` row because `_verified_for` returns False
unconditionally for it (`src/xtrax/export/pipeline.py:130`); AC-4 asserts that
unreachability rather than inventing a string for it. There are no `VALIDATED` rows because
no target is registered at that level and `export_pipeline` refuses one (`:208-217`), so no
`VALIDATED` result can be produced.

`claim` is where the two-field reading becomes a single sentence a human cannot get wrong,
and AC-4 makes it testable: every claim must be distinct, the two validation-bearing claims
must contain "not executed" or "rejected", and no claim may match
`(?i)(webgpu|browser|wgsl)`. That is the point at which "this isn't a browser claim" stops
being a docstring convention and becomes a failing test.

## What a new target costs, mechanically

Adding a `Target` is data, not a branch (`src/xtrax/export/compile.py:145-148`), but it
triggers a fixed checklist. Every item below is enforced by something that already exists.

1. **Registry entry.** A `Target` in `src/xtrax/export/targets.py`, appended to
   `ALL_TARGETS` (`:232-238`). `tests/export/test_targets.py:24-31` asserts the exact tuple,
   so it fails until updated.
2. **Dtype policy.** `test_the_envelope_splits_by_level_not_by_backend`
   (`tests/export/test_targets.py:62-68`) asserts the envelope is a pure function of the
   verification level: `_EXECUTABLE_DTYPES` for `EXECUTED`, `| {"bf16"}` otherwise. A new
   target inherits that rule automatically — including bf16 if it is not `EXECUTED`, whether
   or not anyone examined bf16 on its backend. See TD-WGPU-06.
3. **Size budget.** `SIZE_BUDGET_BYTES` needs a row, forced by
   `test_every_registered_target_has_a_budget` (`tests/export/test_size_budget.py:113-115`),
   which asserts set equality against `ALL_TARGETS`. Both a ceiling and a floor are then
   checked (`:96-111`). **The value must be measured, not guessed** — the flat 32 KiB
   ceiling in the file today was chosen against four recorded measurements
   (`tests/export/test_size_budget.py:12-25`), and a target whose budget was invented has a
   gate that means nothing. A target that emits SPIR-V additionally needs a
   `SPIRV_BUDGET_BYTES` row (`:53`), which AC-7 makes mandatory.
4. **CI coverage.** `export-toolchain-tests` (`.github/workflows/ci.yml:125-176`) runs
   `tests/export/` against real IREE 3.11 and asserts **zero skips** by grepping
   `^SKIPPED \[[0-9]+\] ` (`:166-175`). A new target whose test skips for a missing
   dependency turns that job red, by design.
5. **Fake-toolchain path.** `lint-format-type-test` runs `tests/export/` with no export
   extra, against fakes injected into `sys.modules`
   (`tests/export/conftest.py:85-181`). A target exercising a new toolchain surface needs a
   fake for it — function-scoped and `monkeypatch`-installed, for the reason the conftest
   docstring gives (`:1-8`): a session-scoped `sys.modules` mutation leaks a fake into the
   real-toolchain job and the failure reads as an artifact bug.
6. **Base-install purity.** `test_importing_the_package_does_not_pull_in_iree`
   (`tests/export/test_targets.py:143-158`) runs a fresh interpreter and asserts no `iree`
   module is imported by `import xtrax.export`. Any new toolchain must be imported lazily
   inside a function, as IREE is (`src/xtrax/export/compile.py:74`, `:248`).
7. **Public API and CHANGELOG.** New exported symbols trip `just audit-public-api` and
   `audit-added-types-diff` (`Justfile:328`). Both gates have a known blind spot recorded at
   TD-WGPU-11: neither observes a *renamed field* on an already-exported dataclass.

## Relationship to the compilable-boundaries spec

A parallel spec at `.praxia/docs/specs/260910_compilable-boundaries.md` covers making
taps, sinks, and hooks compilable.

The interaction is one-directional and narrow: **no GPU target can call a host callback.**
`Tap` and `Sink` implementations are required to use `io_callback`
(`src/xtrax/stages/boundaries.py:54`, `:71`), which dispatches back into the Python host. A
SPIR-V dispatch has no host to dispatch to. `export_pipeline` handles this today by
stripping declared materializing sinks before tracing
(`src/xtrax/export/pipeline.py:86-117`) and letting the topology gate reject the rest
(`src/xtrax/export/safety.py:275`), so nothing reaches a GPU backend that could not.

Consequence for that spec: whatever it makes compilable must be compilable **without a host
callback** to be usable on `VULKAN_SPIRV` or `METAL_SPIRV`. A boundary made compilable by
routing it through `io_callback` more efficiently remains a CPU-target-only capability. This
spec asserts nothing further about that spec's internals.

**Edit-collision surface.** Both specs modify `src/xtrax/export/pipeline.py`. This spec's
edits there are confined to `ExportResult`'s docstring (`:29-48`), the new `claim` property,
the guard's justifying comment (`:205-207`), and the `spirv_validation` argument in the
`ExportResult` construction (`:249-260`). It does not touch `_boundaries_for_export`
(`:86-117`) or the tracing path (`:234-237`), which is where the sibling spec works.

**R2″ preserves that spec's out-of-scope premise.** `260910_compilable-boundaries.md:900-901`
scopes SPIR-V targets out by citing `export_pipeline`'s blanket refusal of `VALIDATED`. Under
R2′ that premise would have gone stale on landing. Under R2″ it survives intact: the guard at
`src/xtrax/export/pipeline.py:208-217` stays verbatim and no target moves to `VALIDATED`.

## Acceptance criteria

Each is independently testable and names the file it lives in. AC-5 is the negative
criterion.

- **AC-1 — no target names WebGPU.** For every `t in ALL_TARGETS`, neither `t.name` nor
  `t.iree_backend` matches `(?i)(webgpu|wgpu|browser)`. Lives in
  `tests/export/test_targets.py`, **alongside** the existing
  `test_no_target_is_registered_as_validated` (`:33-36`), which R2″ leaves in place. AC-1 is
  an addition to that test, not a replacement for it: the two assert different things, and
  AC-1 exists as the stated mitigation for the WebGPU-naming risk row below.
- **AC-2 — the validator is not named for the web.** `xtrax.export.spirv.__all__` contains
  no symbol matching `(?i)(webgpu|wgpu|browser|wgsl)`, and the validator entry point is
  named `validate_spirv_structure`. Lives in `tests/export/test_spirv.py`.
- **AC-3 — `VULKAN_SPIRV` stays `CODEGEN_ONLY` and carries the validator's verdict.**
  `VULKAN_SPIRV.verification_level is VerificationLevel.CODEGEN_ONLY`; `export_pipeline` on
  the `tests/export/test_size_budget.py:67` fixture returns
  `result["vulkan-spirv"].spirv_validation` with `valid=True`; a module mutated to be
  structurally invalid returns `valid=False` with a non-empty `.error`; and
  `result["vulkan-spirv"].verified is False` in **both** cases. Two aggregation rules are
  asserted with them, because `spirv_bytes` is a dict of N modules, not one:
  - **Zero extracted modules is a failure, not a pass.** An empty `spirv_bytes` yields
    `valid=False` with `.error` naming the empty extraction **and `module_count == 0`**, which
    is what AC-4's `claim` keys on so it does not report a rejection that never happened.
    `all([])` is `True`, so the
    natural implementation is a vacuous green — and the fake-toolchain path reaches it: the
    fake compiler writes no dump (`tests/export/conftest.py:93-101`),
    `src/xtrax/export/compile.py:212-214` sets `spirv_bytes = spirv_binaries_in(dump_dir)`,
    and `tests/export/test_spirv.py:136-145` asserts exactly `{}` for `VULKAN_SPIRV` there.
  - **N modules aggregate explicitly.** `valid = all(...)` over every module, and on failure
    `.error` names every rejecting module by its `spirv_bytes` dict key.

  Lives in `tests/export/test_spirv_validation.py` (new).
- **AC-4 — `claim` says what was established and never implies a browser.**
  `ExportResult.claim` returns the six distinct strings tabled above — distinctness itself is
  asserted, so the two `valid=False` rows cannot silently collapse into one; the two
  validation-bearing strings contain "not executed" or "rejected"; no claim matches
  `(?i)(webgpu|browser|wgsl)`; and `_verified_for(CODEGEN_ONLY, <passing parity>, <valid
  validation>)` is False, so no `CODEGEN_ONLY`-with-`verified=True` claim is reachable
  through `export_pipeline`. That last assertion is live rather than hypothetical, because
  Task 6 threads the real validation value into `_verified_for`.

  `ExportResult` is a public frozen dataclass and hand-constructible, so `claim` must be
  total over instances `export_pipeline` would never produce. It raises `ValueError` naming
  the contradiction when `verification_level is CODEGEN_ONLY and verified`, and likewise for
  a `VALIDATED` instance, which `export_pipeline` refuses to emit
  (`src/xtrax/export/pipeline.py:208-217`). AC-4 asserts both raises.

  Lives in `tests/export/test_pipeline_native_wasm32.py`, extending `TestVerifiedFor`
  (`:94-140`).
- **AC-5 — a WebGPU-implying claim without browser evidence fails the build.** A new
  `tests/audit/test_webgpu_claim_gate.py` scans tracked text under `src/`, `docs/`,
  `agent_assets/`, `.github/workflows/`, `README.md`, and `CHANGELOG.md` for claim patterns
  and fails naming every `path:line` it finds. `tests/` is deliberately **excluded**: the
  gate's own positive fixtures and error strings live there and would false-positive on
  themselves.

  Every pattern carries a negative-lookbehind prefix, written `NEG` below, so a *denial* of
  a claim is not itself flagged:

  ```
  NEG = (?<!not )(?<!never )(?<!no )(?<!cannot )
  ```

  The patterns, at minimum:

  ```
  (?i)NEG webgpu[- ](compatible|ready|valid|support|verified)
  (?i)NEG (runs?|running|executes?) in (a |the )?browser
  (?i)NEG browser[- ](compatible|ready|support|tested|verified)
  ```

  The third pattern deliberately mirrors the first's alternation. Narrowed to
  `(tested|verified)` it would let the most natural false claim straight through: a CHANGELOG or
  docs line reading *"wasm32 artifacts are browser-ready"* or *"browser-compatible"* would pass
  green while the WebGPU-worded equivalent failed — an asymmetry with no basis, since the risk
  this gate exists for is a browser claim shipping by accident, whatever noun it is spelled with.

  (with `NEG` spliced in immediately before the claim, no space). Without the lookbehind the
  gate flags true statements: `"vulkan-spirv is not WebGPU-valid"` matches the first pattern
  as written before this revision, and the tree is clean under those patterns today — so the
  gate would have manufactured the only defect it ever found. Each lookbehind alternative is
  individually fixed-width, which Python's `re` requires.

  **The scan runs over a whitespace-normalized buffer, not line by line.** A lookbehind is
  line-local, and this repo wraps prose near 88 columns — `docs/api/export.md:242-243` already
  wraps mid-sentence as `…which are not` / `part of the WebGPU feature set…`. A denial that
  happened to wrap as `…is not` / `WebGPU-valid` would put its negation on the previous line,
  out of the lookbehind's reach, and the gate would redden the build on a true statement. That
  is the *same* false-positive class the `NEG` prefix was added to fix, merely displaced from
  "no lookbehind" to "the lookbehind cannot see across the wrap". So the scanner collapses every
  run of whitespace in a file to a single space before matching, and keeps an offset→line map
  built during that collapse so a finding still reports `path:line` against the original file,
  using the line the match *starts* on. AC-11's negation-free wording for two files stays a
  belt-and-braces convention; it is not the mechanism.

  One escape hatch: a marker comment `webgpu-claim-evidence: <path>` on the line above, whose
  `<path>` the gate resolves relative to the repo root and **fails if it does not exist**.
  Three sub-assertions make the gate non-vacuous, and all three are required:
  - **AC-5a — the gate fires.** Given a synthetic in-memory document containing
    `"xtrax emits WebGPU-compatible kernels"`, the scanner returns exactly one finding. A
    gate that matches nothing is green forever and establishes nothing; this is the
    assertion that distinguishes the two.
  - **AC-5b — the escape hatch is not a bypass.** Given the same document with an evidence
    marker naming a nonexistent path, the scanner still returns a finding, and its message
    names the unresolvable path.
  - **AC-5c — a denial is not a claim.** Given the synthetic document
    `"xtrax artifacts are not WebGPU-compatible"`, the scanner returns **zero** findings,
    while AC-5a's positive case still returns exactly one. Both halves are asserted in the
    same test, so widening the lookbehind until it swallows AC-5a fails rather than passes.
  - **AC-5d — a denial survives a line wrap.** Given a synthetic document whose text is
    `"xtrax artifacts are not\nWebGPU-compatible"`, the scanner returns **zero** findings, while
    AC-5a's positive case, run through the same normalization, still returns exactly one. This is
    the criterion that fails if the scanner is ever implemented line-by-line, and it is the
    reason AC-5's patterns are specified against a normalized buffer rather than a file's lines.

  Picked up automatically by `just audit-deterministic`, which runs
  `uv run pytest tests/audit/ -v` (`Justfile:329`). No new `just` recipe, so
  `audit-orphan-recipes` (`Justfile:328`) is unaffected.
- **AC-6 — the validator cannot skip itself green, and cannot break the no-toolchain job.**
  `validate_spirv_structure` resolves the `spirv-val` binary **lazily, at call time**, and
  raises when it is absent rather than returning a skip-shaped result. Import-time
  resolution is forbidden: `lint-format-type-test` (`.github/workflows/ci.yml:64-89`)
  installs no export extra and has no `spirv-val`, yet runs `tests/export/` through
  `just audit-coverage-tier1`, so an import-time raise is a collection error in the one job
  AC-9 requires to pass. Task 7's fake is function-scoped `monkeypatch`
  (`tests/export/conftest.py:1-8`) and structurally cannot intercept an import-time raise.

  Call-time resolution splits the two jobs correctly: `export-toolchain-tests` goes red on a
  runner without SPIRV-Tools, while `lint-format-type-test` passes through the fake.

  **The third environment — real IREE, no `spirv-val` — must degrade, not raise.** Those two
  jobs are not exhaustive. A developer or downstream consumer with the `export` extra installed
  and SPIRV-Tools absent is the ordinary case off CI, and it is this author's own box (see "What
  is not established"): `spirv-val` is a *system* binary, so no Python extra can carry it. Left
  unhandled, `export_pipeline(..., targets=(VULKAN_SPIRV,))` would begin raising where it
  previously returned an artifact with `verified=False`, and `tests/export/test_size_budget.py`'s
  `exported` fixture — guarded only by `importorskip("iree.compiler")` (`:73-74`) — would go from
  passing to erroring there.

  So the raise lives in `validate_spirv_structure` and **stops at `export_pipeline`**: Task 6
  catches the missing-binary error, leaves `spirv_validation` at `None`, and appends an
  `ExportResult.diagnostics` entry naming `spirv-val`. That lands the result on the existing
  "compiled only; nothing further was established" row — the same outcome as before this spec —
  rather than on a new failure mode. `export-toolchain-tests` is unaffected: its own install step
  (Task 6) makes the binary present, and the zero-skips grep makes its absence red. The new
  optional system dependency is named in `docs/api/export.md` and in the `CHANGELOG.md` entry
  AC-10 requires, because nothing else tells a consumer it exists.

  Asserted three ways: `tests/export/test_spirv_validation.py` contains none of
  `importorskip`, `skipif`, `pytest.skip`, `mark.skip`, or `mark.xfail` (a source-level
  assertion in `tests/audit/test_webgpu_claim_gate.py`); a call with the binary absent and no
  fake installed raises; and CI's existing `^SKIPPED \[[0-9]+\] ` grep
  (`.github/workflows/ci.yml:166-175`) still reports zero. A bare early `return` remains
  invisible to the source-level grep — TD-WGPU-10 owns that general form and this criterion
  does not claim to close it.
- **AC-6b — a missing `spirv-val` degrades the export, and only the export.** With the binary
  absent and no fake installed, both halves are asserted in the **same** test, so collapsing the
  raise into the degrade (or the degrade into the raise) fails: a direct
  `validate_spirv_structure(...)` call raises (AC-6), **and**
  `export_pipeline(..., targets=(VULKAN_SPIRV,))` returns an `ExportResult` with
  `spirv_validation is None`, `verified is False`,
  `claim == "compiled only; nothing further was established"`, and a `diagnostics` entry naming
  `spirv-val`. `tests/export/test_size_budget.py`'s `exported` fixture still passes under the
  same conditions. Lives in `tests/export/test_spirv_validation.py`.
- **AC-7 — every SPIR-V-emitting target has a measured SPIR-V budget.**
  `set(SPIRV_BUDGET_BYTES) == {t.name for t in ALL_TARGETS if t.emits_spirv}`, mirroring
  `test_every_registered_target_has_a_budget` (`tests/export/test_size_budget.py:113-115`).
  `SPIRV_BUDGET_BYTES["vulkan-spirv"]` keeps its 16 KiB ceiling and gains a floor check
  against `SIZE_FLOOR_BYTES`, which `test_extracted_spirv_is_within_its_own_budget` already
  applies (`:117-122`) but which no completeness test enforces for a future target. Lives in
  `tests/export/test_size_budget.py`.
- **AC-8 — the Vulkan ICD step has a live consumer or is gone.**
  `.github/workflows/ci.yml` installs `mesa-vulkan-drivers`/`libvulkan1` (`:142-145`) for a
  wgpu adapter that no longer exists in the tree. Either the step is removed, or a tracked
  file imports `wgpu` and the step's comment names it. Asserted by a workflow-parsing test in
  `tests/audit/test_webgpu_claim_gate.py`: if the workflow installs a Vulkan ICD, some file
  under `src/` or `tests/` must import `wgpu`.
- **AC-9 — the suite still passes with no toolchain, without mass-skipping to get there.**
  `uv run --extra dev --extra io pytest tests/export/ -q` passes with neither `iree` nor
  `spirv-val` present, and `tests/export/test_targets.py:143-158` still reports `clean`. The
  same no-toolchain run under `--cov=xtrax.export.spirv --cov-report=term-missing` reports
  **≥90% line coverage of `validate_spirv_structure`**, so "the suite still passes" cannot be
  satisfied by skipping the validator's logic. The SPIR-V-dump and validator fakes needed for
  AC-3's fake-path coverage are function-scoped and `monkeypatch`-installed, per
  `tests/export/conftest.py:1-8`.
- **AC-10 — the public-API break is declared, because nothing else catches it.** The
  `SpirvValidationResult` field rename has a `CHANGELOG.md` entry under `Changed` naming all
  three removed fields (`adapter_type`, `backend`, `device_name`) and both added ones
  (`validator`, `validator_version`), asserted by a **literal grep for those five names** in
  `tests/audit/test_webgpu_claim_gate.py`. The grep is the whole criterion: neither
  `audit-public-api` nor `audit-added-types-diff` observes a dataclass field rename
  (measured; see TD-WGPU-11), so an assertion that `just audit-deterministic` passes would
  establish nothing here.
- **AC-11 — public docs and docstrings state the boundary, without relying on negation.**
  Five locations are updated, and AC-5's gate passes over every one:
  - `docs/api/export.md:240-246` — wording that carries no negated claim, so it does not
    depend on AC-5's lookbehind firing: "`vulkan-spirv` is checked for structural SPIR-V
    conformance only. Browser compatibility is out of reach through IREE — see the
    measurement pass." Its citation of
    `.praxia/docs/research/260901_webgpu-export-measurement-pass.md` is kept. "Structurally
    validated" is dropped as a *level* claim: the target's level does not change.
  - `agent_assets/skills/using-xtrax/references/export.md` — the same boundary in the same
    negation-free wording.
  - `src/xtrax/export/spirv.py:1-16` — rewritten to describe the structural validator this
    module now ships, **keeping** its statement that no WebGPU validator exists here and why.
    Landed by Task 5.
  - `src/xtrax/export/pipeline.py:46` — "The shader validation, for VALIDATED targets only"
    becomes "for targets that emit SPIR-V; present regardless of verification level"; and
    `:205-207`, the guard's justifying comment, is restated: the guard exists because no
    target is registered at `VALIDATED` and nothing wires validation for one that would be,
    not because validation is unimplemented. Landed by Task 6.
  - `src/xtrax/export/targets.py:53-54` — "never executed or otherwise validated" becomes
    "never executed; `ExportResult.verified` is unconditionally False even when a structural
    validation is attached". The module docstring's trailing clause at `:22-23`,
    "`export_pipeline` refuses one, since it has nothing to populate the result with", is
    restated to the surviving reason — no target is registered at that level — because the
    populate-nothing rationale stops being true once `spirv_validation` is wired. Landed by
    Task 6.

  `src/xtrax/export/targets.py:17-20`'s "compiled only" description, the first clause of
  `:22` ("No target is registered at `VALIDATED`"), and `tests/export/test_targets.py:57-60`
  all stay true under R2″ and need no edit. Only `:22-23`'s trailing rationale changes.
- **AC-12 — a tracked debt row points somewhere real, as far as a test can see.** Every debt
  row in this document whose Resolution text begins with the literal `Unresolved, tracked at `
  names either a praxia backlog id matching `#\d+` or a tree path that itself contains the
  literal row id. Rows whose honest disposition is "accepted, unfunded" are out of scope by
  construction; the criterion does not force an id onto them.

  **The two branches are not equally strong, and this criterion must not claim they are.** Only
  the tree-path branch is bidirectional: the gate opens the path and checks the row id appears
  inside it, so a pointer and its target cannot drift apart. The backlog-id branch is a **shape
  check only** — a test in this repo cannot reach the praxia backlog DB, so nothing verifies that
  the item exists, is open, or mentions the row, and `Unresolved, tracked at #1` passes. The sole
  row in that form today is TD-WGPU-11 → `#5090`. That residual gap is stated rather than papered
  over; closing it needs a gate with DB access, which this spec does not fund.

  **Row ids parse as `TD-WGPU-\d{2}(?:-[A-Z])?`, not `TD-WGPU-\d{2}`.** The narrower shape
  matches the `01` prefix inside `TD-WGPU-01-M` and maps two rows onto one key, so a parser keyed
  by id silently drops a row while a "found every row" assertion still passes green.

  Asserted by a parser over this file in `tests/audit/test_webgpu_claim_gate.py`, with three
  non-vacuity assertions:
  - the parser finds every debt row in the table (a parser matching nothing fails), and at least
    one row uses the `Unresolved, tracked at ` form — TD-WGPU-11 does today;
  - the number of parsed ids equals the number of **distinct** parsed ids — the assertion that
    fails if the `-M` suffix is ever dropped from the pattern;
  - over a synthetic table fragment whose Resolution reads
    `Unresolved, tracked at nowhere/at/all.md`, the parser returns exactly one finding.

## Fixer tasks

Dependency-ordered. Tasks 4 through 8 are cancelled if Task 0 returns a negative on M1 or M2.

### Task 0 — Measure `spirv-val` (blocks 4-8)

Establish M1 and M2. On `ubuntu-latest` and locally: install SPIRV-Tools, run
`spirv-val --version`, then run `spirv-val --target-env vulkan1.3 <module>` over every SPIR-V
module extracted from the `tests/export/test_size_budget.py:67` fixture compiled for
`vulkan-spirv` — 4780 B in total for that fixture
(`tests/export/test_size_budget.py:16`). Record each module's size, exit code, and stderr.

**Files**: `.praxia/docs/research/260910_spirv-val-availability.md` (create)
**Gate**: the research doc records the installing package's name, the version string, the
exact `--target-env` value used, and one exit code per module. The `--target-env` value is
recorded because `spirv-val`'s ruleset differs between environments, so a verdict without it
is unfalsifiable. A nonzero exit on a module IREE emitted is a **negative** result that
cancels Tasks 4-8 and is reported as such, not worked around.
**Scope estimate**: ~0 LOC, one probe branch.

### Task 1 — The claim gate (independent of Task 0)

Implement AC-5, AC-5a, AC-5b, AC-5c.

**Files**: `tests/audit/test_webgpu_claim_gate.py` (create)
**Gate**: `uv run --extra dev pytest tests/audit/test_webgpu_claim_gate.py -q` passes, and
temporarily inserting `WebGPU-compatible` into `docs/api/export.md` makes it fail naming
that file and line. The tree is clean under these patterns before the insertion, so a
failure at that point means the patterns are too broad, not that a claim exists.
**Scope estimate**: ~140 LOC.

### Task 2 — Dispose of the Vulkan ICD step (independent of Task 0)

Implement AC-8. `wgpu` is absent from `pyproject.toml`, `src/`, and `tests/`, so the apt step
at `.github/workflows/ci.yml:142-145` and its comment (`:137-141`) describe a validation
that no longer happens.

**Files**: `.github/workflows/ci.yml` (modify), `tests/audit/test_webgpu_claim_gate.py`
(modify)
**Gate**: `uv run --extra dev pytest tests/audit/test_webgpu_claim_gate.py -q` passes; the
`export-toolchain-tests` job still reports zero skips.
**Scope estimate**: ~40 LOC.

### Task 3 — SPIR-V budget completeness (independent of Task 0)

Implement AC-7.

**Files**: `tests/export/test_size_budget.py` (modify)
**Gate**: `uv run --extra dev --extra io pytest tests/export/test_size_budget.py -q` passes;
adding a second `emits_spirv=True` target to a local copy of `ALL_TARGETS` makes it fail.
**Scope estimate**: ~25 LOC.

### Task 4 — `ExportResult.claim` (needs Task 0 positive)

Implement AC-4 as a property, not a field.

**Files**: `src/xtrax/export/pipeline.py` (modify),
`tests/export/test_pipeline_native_wasm32.py` (modify)
**Gate**: `uv run --extra dev --extra io pytest tests/export/test_pipeline_native_wasm32.py -q`
passes with no export extra installed.
**Scope estimate**: ~50 LOC.

### Task 5 — `validate_spirv_structure` (needs Task 0 positive)

Implement AC-2 and the `SpirvValidationResult` field change — the
`adapter_type`/`backend`/`device_name` → `validator`/`validator_version` rename, plus the added
`module_count` AC-4's claim table keys on. The `spirv-val` binary is
resolved **lazily, inside `validate_spirv_structure`** — not at import — for AC-6's reason,
which also keeps `import xtrax.export` clean (AC-9).

`SpirvValidationResult`'s docstring (`src/xtrax/export/spirv.py:78-87`) is rewritten for the
N-module contract it actually has: its first line becomes "Outcome of validating the SPIR-V
modules extracted from one artifact", replacing "one SPIR-V module", and the attribute list
documents how `valid` aggregates and what `error` names. The module docstring (`:1-16`) is
refreshed per AC-11.

**Files**: `src/xtrax/export/spirv.py` (modify), `tests/export/test_spirv.py` (modify)
**Gate**: `uv run --extra dev --extra io pytest tests/export/test_spirv.py -q` passes; the
fresh-interpreter check at `tests/export/test_targets.py:143-158` still prints `clean`.
**Scope estimate**: ~90 LOC.

### Task 6 — Wire validation into `export_pipeline` (needs Task 5)

Implement AC-1, AC-3, AC-6, AC-6b. `VULKAN_SPIRV` **stays `CODEGEN_ONLY`**
(`src/xtrax/export/targets.py:212-218` is unchanged), `test_no_target_is_registered_as_validated`
(`tests/export/test_targets.py:33-36`) is kept, and the guard at
`src/xtrax/export/pipeline.py:208-217` is kept verbatim. AC-1's assertion is **added** to
`test_targets.py` beside the retained test.

The wiring is a short block. Populate `spirv_validation` at
`src/xtrax/export/pipeline.py:258` for targets with `target.emits_spirv`, and thread that
same value into `_verified_for` at `:256`, which today hardcodes
`_verified_for(target.verification_level, parity, None)`. Wrap the populate step in a
`try/except` on the missing-binary error, leaving `spirv_validation` at `None` and appending a
`diagnostics` entry naming `spirv-val` (AC-6b) — raising is the *validator's* contract, not the
pipeline's. Threading it cannot produce a
contradiction — `_verified_for` returns False for `CODEGEN_ONLY` unconditionally (`:130`) —
and it turns AC-4's `_verified_for(CODEGEN_ONLY, <passing parity>, <valid validation>) is
False` assertion into a live safety test over the value the pipeline really passes.

Also lands the `pipeline.py:46`, `:205-207`, `targets.py:22-23`, and `targets.py:53-54`
docstring edits AC-11 specifies, and adds the CI install step: an `Install SPIRV-Tools` step
in `export-toolchain-tests` **before** the `uv sync` at `.github/workflows/ci.yml:148`, using
the package name Task 0's M1 records. Without it this task turns that job red by design
(AC-6), and no other task touches that file in a way that would add it — Task 2 only removes
a step.

**Files**: `src/xtrax/export/pipeline.py` (modify), `src/xtrax/export/targets.py` (modify),
`tests/export/test_targets.py` (modify), `.github/workflows/ci.yml` (modify),
`tests/export/test_spirv_validation.py` (create)
**Gate**: `uv run --extra dev --extra io pytest tests/export/ -q` passes; the
`export-toolchain-tests` job reports zero skips and its `Install SPIRV-Tools` step succeeds.
**Scope estimate**: ~150 LOC.

### Task 7 — Fake-toolchain path (needs Task 6)

Implement AC-9. Add a function-scoped fake for the validator subprocess so AC-3's logic is
covered without the binary installed.

The fake must exercise **both** aggregation cases AC-3 names, because the default fake
compiler path reaches only the degenerate one: it writes no dump
(`tests/export/conftest.py:93-101`), so `spirv_bytes` is exactly `{}`
(`tests/export/test_spirv.py:136-145`) and an unguarded `all([])` would return True. Cover
the zero-module case (expecting `valid=False`) and a multi-module case with one rejecting
module (expecting `valid=False` and an `.error` naming that module's key).

**Files**: `tests/export/conftest.py` (modify), `tests/export/test_spirv_validation.py`
(modify)
**Gate**: `uv run --extra dev --extra io pytest tests/export/ -q` passes with no export
extra; the same run under `--cov=xtrax.export.spirv --cov-report=term-missing` shows ≥90%
line coverage of `validate_spirv_structure`; `just audit-coverage-tier1` clears its
thresholds.
**Scope estimate**: ~80 LOC.

### Task 8 — Docs, CHANGELOG, debt references (needs Task 6)

The CHANGELOG entry covers two things, not one: the `SpirvValidationResult` field change
(AC-10) **and** the new *optional* system dependency on SPIRV-Tools, which no Python extra can
declare and which AC-6b makes soft — absent, an export degrades to "compiled only" rather than
failing. `docs/api/export.md` states the same in prose.


Implement AC-10, AC-11's two documentation files, and AC-12. The source-docstring half of
AC-11 lands in Tasks 5 and 6; this task adds `docs/api/export.md`,
`agent_assets/skills/using-xtrax/references/export.md`, the CHANGELOG entry, and the tests
that assert all of it.

**Files**: `docs/api/export.md` (modify), `CHANGELOG.md` (modify),
`agent_assets/skills/using-xtrax/references/export.md` (modify),
`tests/audit/test_webgpu_claim_gate.py` (modify)
**Gate**: both commands, because the first is not sufficient on its own —
`just audit-deterministic` halts at the first hard failure and its coverage-DAG step has been
observed printing PASS over failing tests, so exit 0 does not prove the suite is green:

```
just audit-deterministic
uv run --extra dev pytest tests/audit/ -q
```

**Scope estimate**: ~70 LOC of prose plus ~60 LOC of test.

## Gate commands

Every `uv run` carries its `--extra` flags explicitly, to guarantee the extra is present in
that invocation's environment — not because extras are pruned between invocations. CI relies
on the persistence: `.github/workflows/ci.yml:163` runs a bare `uv run pytest` after
`uv sync --extra export` at `:148`.

```
uv run --extra dev --extra io pytest tests/export/ -q
uv run --extra dev --extra io pytest tests/export/ -q --cov=xtrax.export --cov-report=term-missing
uv run --extra dev --extra io pytest tests/export/ -q --cov=xtrax.export.spirv --cov-report=term-missing
just audit-coverage-tier1
uv run --extra dev ruff check . && uv run --extra dev ty check src/
just audit-deterministic
uv run --extra dev pytest tests/audit/ -q
```

`just audit-deterministic` is listed with an explicit `pytest tests/audit/` run beside it
because it halts at the first hard failure and its coverage-DAG step has been observed
printing PASS over failing tests. Its exit code alone is not evidence the suite is green.

## Tech debt

| ID | Item | Resolution |
|----|------|------------|
| TD-WGPU-01 | Pallas kernels. Absent today — `grep -rniE "pallas\|mosaic\|triton\|custom_call\|\bffi\b" src/ tests/` returns zero hits, re-confirmed 260910, so this is forward-looking, not a current blocker. Adopting one would lower to a platform-specific custom call, which IREE's StableHLO importer has no rule for; the failure would land in `compile_for_target` (`src/xtrax/export/compile.py:179-207`) on **every** target including `native`, not only GPU ones | Accepted, unfunded. The blast radius is asserted, not measured: settle it by tracing a one-line `pallas_call` through `jax.export.export` and passing the StableHLO to `compile_for_target` for `NATIVE` and `VULKAN_SPIRV`, recording both diagnostics. TD-WGPU-01-M records the gate that would name the failure |
| TD-WGPU-01-M | No plan-time gate would catch TD-WGPU-01. `validate_export_safe` checks dtypes and topology only (`src/xtrax/export/safety.py:275-288`), so a Pallas kernel would surface as a raw IREE legalization error rather than a named blocker | Accepted, unfunded. A custom-call blocker in `check_export_safety` is the fix; there is no Pallas kernel to gate, so it is not built here |
| TD-WGPU-02 | `METAL_SPIRV` is named for its input dialect and emits Metal Shading Language source, magic `0x636e6923` / ASCII `"#inc"` (`src/xtrax/export/targets.py:220-230`). The name misdescribes the artifact | Accepted permanently. The name matches IREE's own backend string, which `Target.iree_backend` passes through verbatim; `emits_spirv=False` plus the magic filter (`src/xtrax/export/spirv.py:38-42`) carry the correction, and `test_only_vulkan_emits_spirv` (`tests/export/test_targets.py:38-41`) enforces it |
| TD-WGPU-03 | `SPIRV_BUDGET_BYTES` (`tests/export/test_size_budget.py:53`) covers one target with no completeness check, and its 16 KiB ceiling sits ~3.3x over the measured 4780 B (`:16`) | Resolved by AC-7 / Task 3 for completeness. The ceiling's generosity is deliberate and matches the flat vmfb ceiling's rationale (`:1-10`): it catches an order-of-magnitude blowup, not drift |
| TD-WGPU-04 | Symbolic shapes work only for reshape-free programs, and every measurement was taken on `NATIVE` (`tests/export/test_symbolic_shapes.py:1-46`). Nothing establishes the boundary on `vulkan-spirv`, where the failing op is a different legalization | Accepted, unfunded. No symbolic-shape claim may be made for a SPIR-V target until the same positive/negative/control triple is run there. Cheap: the fixtures exist, only the target changes |
| TD-WGPU-05 | Stochastic ops are unmeasured on every target. Every `jax.random` call in `tests/export/` runs at fixture-construction time (`tests/export/conftest.py:44-47`, `test_size_budget.py:58-59`, `test_parity_multi_size.py:88-91`); none is traced through `export_pipeline` | Accepted, unfunded. A threefry key threaded through the exported program is the measurement; it bears on GPU targets specifically because counter-based PRNG lowers to integer ops whose SPIR-V support is unverified |
| TD-WGPU-06 | The dtype envelope is a pure function of verification level, asserted at `tests/export/test_targets.py:62-68`. `VULKAN_SPIRV` is non-`EXECUTED`, so it carries bf16 automatically — a dtype nobody examined on that backend | Accepted for every non-`EXECUTED` level, because nothing is executed and bf16's only measured failure is in IREE's *runtime* numpy mapping (`src/xtrax/export/targets.py:113-119`). R2″ changes nothing here: the level does not move. Any promotion to `EXECUTED` must re-derive the envelope and change that test, which will fail loudly if skipped |
| TD-WGPU-07 | No browser or JS glue layer exists. A WebGPU-valid shader still needs a host to allocate buffers, bind them, and dispatch — IREE's absent WebGPU HAL — plus a headless-browser CI lane xtrax does not have | Accepted, out of scope. R4 is the only route that would retire it, and R4 is unfunded here. This is the precondition for any browser claim, and AC-5's evidence marker has nothing to point at until it exists |
| TD-WGPU-08 | Host callbacks are impossible on GPU targets. `Tap`/`Sink` must use `io_callback` (`src/xtrax/stages/boundaries.py:54`, `:71`); a SPIR-V dispatch has no host to call | Accepted. `export_pipeline` strips declared materializing sinks (`src/xtrax/export/pipeline.py:86-117`) and the topology gate rejects the rest (`src/xtrax/export/safety.py:275`), so nothing unrepresentable reaches a backend. Cross-referenced to `.praxia/docs/specs/260910_compilable-boundaries.md` |
| TD-WGPU-09 | CI installs `mesa-vulkan-drivers`/`libvulkan1` (`.github/workflows/ci.yml:142-145`) for a wgpu adapter that no tracked file constructs, and the step's comment (`:137-141`) explains it in terms of the voided AC-8 | Resolved by AC-8 / Task 2 |
| TD-WGPU-10 | The zero-skips assertion greps `^SKIPPED \[[0-9]+\] ` (`.github/workflows/ci.yml:166-175`), which matches pytest's `-rs` short-summary shape. A whole-module collection-level skip has been observed in this repo to evade a line-shaped skip check | Resolved by AC-6 for the new validator test, which cannot skip at all. Accepted, unfunded for the grep itself: the general fix is to assert on `--junitxml` skip counts rather than on stdout text, and no task here funds it. This row also owns the bare-early-`return` form that AC-6's source-level assertion cannot see |
| TD-WGPU-11 | No deterministic gate observes a public dataclass field rename — `audit-added-types-diff` collects only `FunctionDef` nodes and diffs signatures (`src/xtrax/devtools/gates/added_types_diff.py:47-55`, `:67-79`); `audit_public_api.py` inspects only root `__all__`/`_LAZY`/`tier1_exports` (`scripts/audit_public_api.py:199-262`) and merely asserts subpackage `__all__` is non-empty (`:164-179`). `SpirvValidationResult` is neither a function nor a root export, so the `adapter_type`/`backend`/`device_name` → `validator`/`validator_version`/`module_count` field change fires nothing | Unresolved, tracked at #5090. Interim mitigation in this spec is AC-10's literal `CHANGELOG.md` grep, which is per-change and does not generalize |

## Risks

| Risk | Mitigation |
|---|---|
| `spirv-val` is unavailable or rejects IREE's modules (M1/M2 negative) | Task 0 runs before any code. A negative cancels Tasks 4-8 and the spec degrades to R1 plus AC-5, AC-5a, AC-5b, AC-5c, AC-7, and AC-8. Rollback: none needed — nothing has landed |
| AC-5's claim gate is written so broadly it flags ordinary prose, and gets weakened until vacuous | The pattern list is fixed in AC-5 and is claim-shaped, not topic-shaped: `docs/api/export.md:240-246` says "WebGPU" four times and must pass both before and after AC-11 rewrites it. AC-5a pins that the gate still fires on a real claim and AC-5c pins that it stays silent on a denial, both in one test, so weakening either direction fails rather than passing quietly |
| `SpirvValidationResult`'s field rename breaks a downstream consumer | The class is exported (`src/xtrax/export/spirv.py:21-27`) but its only construction sites in-tree are two tests (`tests/export/test_spirv.py:73-75`, `tests/export/test_pipeline_native_wasm32.py:112-124`), and no target has ever produced one, so no shipped artifact carries the old shape. AC-10 forces the CHANGELOG entry and greps for it, because no deterministic gate observes the rename (TD-WGPU-11). Rollback: revert `src/xtrax/export/spirv.py` |
| A consumer reads `verified=False` beside `spirv_validation.valid=True` as a validation failure | This is R2″'s residual misreading risk, and it is the one `claim` exists for: AC-4's property returns "compiled; the SPIR-V is structurally valid; not executed on any device" as one sentence, so the two-field reading never has to be performed. AC-11 puts the same sentence in the public docs. The inverse risk R2′ carried — `verified=True` on a never-executed artifact — does not arise, because the level does not move |
| Task 6 turns `export-toolchain-tests` red on a runner without SPIRV-Tools | That is AC-6 working as designed — a red job, not a silent skip. Task 6 lands the `Install SPIRV-Tools` step into `.github/workflows/ci.yml` before the `uv sync` at `:148`, in the same commit and using the package name Task 0's M1 recorded. No other task adds it: Task 2 is the only other task touching that file and it removes a step |
| AC-6's raise fires at import and breaks `lint-format-type-test`, which has no `spirv-val` and no export extra | AC-6 mandates call-time resolution for exactly this reason. `lint-format-type-test` (`.github/workflows/ci.yml:64-89`) runs `tests/export/` through `just audit-coverage-tier1`, and Task 7's fake is function-scoped `monkeypatch` (`tests/export/conftest.py:1-8`), which cannot intercept an import-time raise. AC-9 asserts that job's suite still passes, so an import-time implementation fails a criterion rather than reaching CI |
| Removing the Vulkan ICD step (Task 2) breaks a job that quietly depended on it | Nothing in `src/` or `tests/` imports `wgpu`, and `spirv-val` needs no ICD. AC-8's test makes the dependency explicit in both directions. Rollback: revert `.github/workflows/ci.yml` |
| A future contributor adds a `webgpu` target once IREE ships one and reuses `VALIDATED` for it, inheriting a browser claim it has not earned | Three independent barriers, all retained by R2″: AC-1 forbids a target named for WebGPU outright, `test_no_target_is_registered_as_validated` (`tests/export/test_targets.py:33-36`) forbids any target at that level, and `export_pipeline` refuses one at runtime (`src/xtrax/export/pipeline.py:208-217`). Such a target cannot land without deliberately changing all three — which is the review conversation this spec wants to force |
| `just audit-coverage-tier1` regresses because the new `spirv.py` branches are only covered under the export extra | Task 7's fake makes the validator path reachable with no toolchain, matching how `compile.py` is already covered (`tests/export/conftest.py:110-116`). AC-9's ≥90% line-coverage floor on `validate_spirv_structure` is asserted on the no-toolchain run specifically, so covering it only under the export extra fails |
| The validator returns a vacuous green because the fake path extracts zero modules and `all([])` is True | AC-3 makes zero modules an explicit `valid=False` with an `.error` naming the empty extraction, and Task 7's fake exercises that case directly — it is the default shape there, since the fake compiler writes no dump (`tests/export/conftest.py:93-101`) and `spirv_bytes` is exactly `{}` (`tests/export/test_spirv.py:136-145`) |

## Out of scope

- **Any browser claim.** Nothing here executes a shader in a browser, and AC-5 makes
  claiming otherwise a build failure.
- **R3's SPIR-V rewriter** and any fork of IREE's Vulkan HAL.
- **R4's non-IREE WGSL path.** Named, costed at the level of "its own epic", and left
  unfunded.
- **A `webgpu` target registry entry.** Blocked on R1's trigger condition, which is an
  upstream event.
- **Executing `wasm32` or either SPIR-V target.** Unchanged from
  `src/xtrax/export/targets.py:15-20`.
- **Adopting Pallas.** TD-WGPU-01 records what it would cost; this spec does not adopt it
  and does not build the gate that would catch it.
- **Changing `_verified_for`'s contract** (`src/xtrax/export/pipeline.py:120-130`). Task 6
  passes it a real validation value instead of a hardcoded `None`, which the existing
  contract already handles; the contract itself is untouched. The false-green risk is met
  with `claim` instead, for the reasons argued above.
- **Promoting any target to `VerificationLevel.VALIDATED`.** That is R2′, rejected above.
  `test_no_target_is_registered_as_validated` and the runtime guard both stay.
- **Widening `audit-added-types-diff` to observe dataclass fields.** TD-WGPU-11 and #5090
  own it; AC-10's grep is this spec's compensating control and does not generalize.

## Open questions

- **Which package provides `spirv-val` on `ubuntu-latest`, and is the binary on `PATH` after
  install?** Settled by Task 0's M1. The answer is what Task 6's `Install SPIRV-Tools` step
  installs, so M1 is a prerequisite for that step, not just for the verdict.
- **Does `spirv-val --target-env vulkan1.3` accept IREE 3.11's `vulkan-spirv` output?**
  Settled by Task 0's M2. A rejection is a finding about IREE, not a reason to relax the
  check or to loosen `--target-env` until it passes.
- **Does naga reject any module `spirv-val` accepts, or the reverse?** Unmeasured; no
  comparison run exists. It does not gate anything here — R2 is rejected on its naming
  hazard, independently of relative strictness — but the answer would tell a future reader
  what R2 would add on top of R2″ rather than leaving it asserted.
- **Does a StableHLO-to-WGSL or JAX-to-WGSL toolchain exist that xtrax could depend on?**
  Unmeasured. Settled by compiling the `tests/export/test_size_budget.py:67` fixture through
  a candidate and running it headless against an independent oracle. Until then R4 is a
  route, not a plan.
- **What is `maxStorageBuffersPerShaderStage` consumption for a real xtrax pipeline?** Never
  reached, because the push-constant rejection fires first
  (`260901_webgpu-export-measurement-pass.md:47-48`). It becomes the next question the day
  R1's trigger fires.
- **Does a Pallas kernel fail on `native` as well as on GPU targets?** Asserted in
  TD-WGPU-01 from IREE's importer having no custom-call rule; unmeasured. The measurement is
  in that row.

## References

- `.praxia/docs/specs/260901_xtrax-export-webgpu.md` — the amended prior spec; AC-8 VOID at
  line 551, AC-8b VOID at line 571.
- `.praxia/docs/research/260901_webgpu-export-measurement-pass.md` — the measurement that
  falsified AC-8, including the flag matrix (`:79-94`) and the CI reproduction (`:124-167`).
- `.praxia/docs/specs/260910_compilable-boundaries.md` — parallel spec; interaction stated
  above.
- `.praxia/docs/plans/260909_runnable-artifact-and-wasm-price.md` — Phase A2's symbolic-shape
  boundary table, cited by `tests/export/test_symbolic_shapes.py:27-29`.
- [IREE issue #13702](https://github.com/iree-org/iree/issues/13702) — the WebGPU epic;
  R1's watch item. [#24463](https://github.com/iree-org/iree/issues/24463) (closed
  2026-05-18) landed the WGSL target and JS-hosted HAL driver;
  [#24650](https://github.com/iree-org/iree/issues/24650) (open 2026-06-29) is the
  push-constant bug that still gates it, and is upstream's independent confirmation of
  this spec's central finding.
- [IREE issue #8327](https://github.com/iree-org/iree/issues/8327) — *not* a WebGPU
  issue, despite being cited as R1's watch item in this spec's first revision. It is
  "Port the IREE runtime to WebAssembly+JavaScript without Emscripten", and it bears on
  the `wasm32` target's execution story (`src/xtrax/export/targets.py:15-17`), not on
  WebGPU. #24463's freestanding-wasm32 runtime is the same territory.
- `.praxia/docs/specs/260618_hmw-design-unified-implementation-valida.md:292` — the
  `## Tech debt` table format this document follows.

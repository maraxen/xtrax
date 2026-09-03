# xtrax.export — ahead-of-time export

TIER-2 reference for `xtrax.export`. The public narrative version is `docs/api/export.md`;
this file is the agent-facing summary. Read the live source at the cited paths when in doubt.

`xtrax.export` compiles a `BatchPlan`-shaped computation ahead of time through IREE. It is an
optional extra: `uv sync --extra export` (`iree-base-compiler`, `iree-base-runtime`,
`huggingface_hub`, `safetensors`). Importing `xtrax.export` without the toolchain works; only
compilation needs it.

## Targets, and what each one actually proves

`verify: src/xtrax/export/targets.py:119-174`

| Target | IREE backend | VerificationLevel | Emits SPIR-V |
|---|---|---|---|
| `NATIVE` | `llvm-cpu` | `EXECUTED` | no |
| `WASM32` | `llvm-cpu` (wasm32 triple) | `CODEGEN_ONLY` | no |
| `VULKAN_SPIRV` | `vulkan-spirv` | `CODEGEN_ONLY` | yes |
| `METAL_SPIRV` | `metal-spirv` | `CODEGEN_ONLY` | **no** — it dumps MSL, not SPIR-V, despite the name |

`VerificationLevel` is the whole point of the type, so read it literally:

- `EXECUTED` — the artifact was compiled **and run**, and its output compared against a
  caller-supplied `reference_fn`. Only `NATIVE` reaches this.
- `CODEGEN_ONLY` — the artifact compiled. Nothing ran it. A green `CODEGEN_ONLY` export says
  the compiler accepted the program, and says nothing whatsoever about numerics.
- `VALIDATED` exists in the enum but no target registers it; `export_pipeline` raises
  `NotImplementedError` for one. That is deliberate, not unfinished — see WebGPU below.

## The parity oracle must be independent

`export_pipeline(..., reference_fn=...)` is **required** whenever a target is `EXECUTED`, and
it must be computed independently of the composed callable. A parity check that derives its
expected value from the same callable it is checking proves only that a function equals
itself. `verify_native_parity` compares `reference_fn`'s output against
`run_native_vmfb(result.path, *concrete_inputs)`.

`ParityResult` exposes a scalar diff only. If you need to assert *ordering* (which element ran
when), call `run_native_vmfb` directly and decode the array — the parity scalar cannot show it.

## Dtypes

The envelope is measured, not assumed (`verify: src/xtrax/export/targets.py`). `f64` is
rejected on **every** target including `NATIVE`; `bf16` splits by verification level. Casting
happens through `load_hf_weights`, which reports every cast leaf in its `WeightReport` — one
entry per leaf, deliberately untruncated.

Dtype gating scans closure leaves, not just `abstract_inputs`: a `bf16`/`f64` array captured in
`fn`'s closure is caught. That was a real escape before it was fixed.

## What can cross the boundary

`check_export_safety` / `validate_export_safe` reject what cannot be traced. A `Sink` declared
`materialize=True` is replaced by `_StrippedSink` before composition, so the exported artifact
returns the value the sink would have consumed instead of performing a host callback. That
means **the exported program and the un-stripped one are only equal if the transition sinks the
exact value it returns** — hold that precondition in mind when writing one.

## Multi-axis plans

`compose_vmap_of_scan` handles an outer `Vmap`-strategy axis wrapping an inner `Scan`. A
lane-dependent ordered `Tap`/`Sink` under a literal `vmap` raises `MultiAxisCompositionError`.
`Bucket` (host-tier — pad with `bucketize()` before the boundary) and `WhileCarry` (unbounded
trip count — convert to `Scan` with a static length) raise `UnsupportedStrategyError`.

## WebGPU: there is no gate, on purpose

IREE's Vulkan HAL passes dispatch parameters through push constants, which are not a WebGPU
capability, so naga rejects every SPIR-V module IREE emits. IREE 3.11 has no webgpu backend and
no flag removes them. A gate *could* be made green by enabling wgpu's native-only `immediates`
feature — a passing check that establishes nothing about browsers, which is why none was built.
`VULKAN_SPIRV` therefore ships as `CODEGEN_ONLY`. Do not read SPIR-V emission as browser
readiness. Tracked as backlog #4856.

## Entry points

`export_pipeline` is the one to reach for. `build_traceable_callable`/`compose_single_axis`/
`compose_vmap_of_scan` are the composer layer beneath it; `compile_for_target`,
`run_native_vmfb`, `verify_native_parity`, `load_hf_weights`, `spirv_binaries_in`/`is_spirv`
are the pieces it orchestrates. Full list: `src/xtrax/export/__init__.py:54`.

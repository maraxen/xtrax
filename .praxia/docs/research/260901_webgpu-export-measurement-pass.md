---
category: research
title: "WebGPU export measurement pass: IREE SPIR-V is not WebGPU-valid"
description: "Empirical de-risking of the xtrax.export spec before implementation — pins resolve, but AC-8's WebGPU-validity gate is falsified by IREE's push-constant ABI"
task_id: 260901_xtrax-export-webgpu
status: complete
---

# WebGPU export measurement pass

De-risking run against `.praxia/docs/specs/260901_xtrax-export-webgpu.md` before
starting its three-PR rollout. Every number here was measured, not reasoned about.

## Verdict

**PR1 is de-risked and safe to start. PR2's central acceptance criterion (AC-8) is
falsified as written and needs a decision before it is implemented.**

Real IREE-emitted SPIR-V from a genuine composed xtrax pipeline is **rejected by
naga** on a WebGPU-shaped device:

```
Shader validation error: Global variable [1] '__push_constant_var__' is invalid
  = Capability Capabilities(IMMEDIATES) is not supported
```

IREE's Vulkan HAL passes dispatch parameters through **push constants**. Push
constants are not part of the W3C WebGPU feature set, so naga refuses the module.
The rejection is correct: the artifact genuinely is not WebGPU-compatible.

This is an IREE ABI property, not a defect in the pipeline, the plan, or the
composer. No xtrax-side change affects it.

## What the spec got wrong, precisely

The spec's B9 disposition (spec line ~561) reads:

> this spec validates shader *module* acceptance only — `create_shader_module` —
> never pipeline creation, so `maxStorageBuffersPerShaderStage` (capped at 8 on
> WebGPU) and IREE's vulkan-HAL push-constant usage are never exercised or
> enforced by this AC

That reasoning is inverted. `create_shader_module` is exactly where naga runs
validation, and exactly where the push constants fail. The disposition scoped the
risk out on the grounds that made it unavoidable.

The storage-buffer half of that disposition was never reached — the module is
rejected before binding counts matter.

## The trap: AC-8 can be made green dishonestly

wgpu renamed push constants to `immediates`, and exposes it as a **native-only**
feature (it is absent from the W3C WebGPU spec, so no browser offers it).

| device configuration | real IREE SPIR-V |
|---|---|
| `request_device_sync()` — no features, browser-WebGPU-shaped | **0 / 2 valid** |
| `request_device_sync(required_features=["immediates"])` | **2 / 2 valid** |

AC-8 says `validate_webgpu()` must return `valid=True` "against a real CPU
(`llvmpipe`) `wgpu` adapter". That is satisfiable — by enabling `immediates`. The
gate would be green, the CI badge would be honest-looking, and the claim
"downstream packages get tested WebGPU kernels" would be false. Any implementation
of PR2 must construct its validation device with **no** required features, and that
must be stated in the AC rather than left to a fixer's discretion.

## What was ruled out

IREE 3.11.0 registered target backends, from `iree-compile --iree-hal-list-target-backends`:

```
cuda  llvm-cpu  metal-spirv  rocm  vmvx  vmvx-inline  vulkan-spirv
```

There is **no webgpu backend**. Both `--iree-hal-target-backends=webgpu-spirv` and
`--iree-hal-target-device=webgpu` fail to compile, and `--help-hidden` contains zero
flags matching `webgpu` or `push-constant`.

Flag combinations tried against a feature-free device, all still rejected:

| flags | result |
|---|---|
| `--iree-hal-target-backends=vulkan-spirv` (baseline) | 0/2 valid, both carry push constants |
| `+ --iree-vulkan-target=vp_android_baseline_2022` | 0/2 valid |
| `+ --iree-hal-indirect-command-buffers=true` | 0/2 valid |
| `+ --iree-spirv-index-bits=32` | 0/2 valid |
| `+ --iree-stream-resource-alias-mutable-bindings` | 0/2 valid, 2/2 still carry push constants |
| `+ --iree-scheduling-optimize-bindings=false` | **1/2 valid**, 1/2 still carry push constants |
| `--iree-vulkan-experimental-indirect-bindings` | compile fails |
| `+ --iree-hal-indirect-command-buffers=true` | compile fails |

`--iree-scheduling-optimize-bindings=false` is the only partial result: it removed
push constants from one of two dispatches. That is data-dependent, not a toggle —
it cannot underpin a gate.

## Confirmed good (no spec change needed)

- **Every dependency pin resolves**, exit 0, on Python 3.13, both standalone and
  combined with xtrax's core pins. Resolved: `iree-base-compiler==3.11.0`,
  `iree-base-runtime==3.11.0`, `wgpu==0.32.0`, `safetensors==0.8.0`,
  `huggingface-hub==1.29.0`, against `jax==0.10.2` / `jaxlib==0.10.2` /
  `numpy==2.5.2`. This closes the spec's "pins not reconciled against a live
  resolver" risk row.
- **`wgpu.GPUValidationError` resolves at the top level.** The spec flagged this
  import path as unverified. It is available as all of `wgpu.GPUValidationError`,
  `wgpu.classes.GPUValidationError`, `wgpu._classes.GPUValidationError`, and
  `wgpu.backends.wgpu_native.GPUValidationError`. Raised instances report
  `__module__ == "wgpu._classes"`; catching `wgpu.GPUValidationError` works, as it
  is the same class object.
- **Both AC-8b negative fixtures reproduce verbatim.** Valid SPIR-V magic with a
  garbage body raises `GPUValidationError` containing `unknown instruction <n>`;
  non-SPIR-V bytes raise `ValueError: Given shader data does not look like a SpirV
  module`. Both are raised synchronously from `create_shader_module` (confirms V2/B10).
- **The metal magic claim holds.** `metal-spirv` dumps `.metal` MSL source with
  leading bytes `23 70 72 61`, correctly rejected by a `0x07230203` magic filter.
  `METAL_SPIRV`'s demotion to `CODEGEN_ONLY` is right.
- **`--iree-hal-target-backends=<backend>` is still accepted in 3.11**, so the
  spec's `Target.iree_backend` field shape needs no change.
  (`--iree-hal-target-device=<device>` also works and produces a byte-identical vmfb.)
- **SPIR-V extraction works.** `--iree-hal-dump-executable-binaries-to=<dir>` on a
  matmul pipeline produced 2 `.spv` files (6264 B, 3344 B), plus nothing else needing
  filtering. Confirms verified fact #3 and the plurality caveat.

## CI measurements (`ubuntu-latest`, run 33557090199)

Three jobs on branch `probe/webgpu-ci-adapter`. All three completed.

**The Vulkan ICD install is genuinely required.** On a bare runner:

```
ICD dir exists: False
0 adapter(s)
ADAPTER/DEVICE FAILED: RuntimeError Request adapter failed (3): Validation Error
  No suitable graphics adapter found; noop support not compiled in, vulkan
  drivers/libraries could not be loaded, metal support not compiled in, dx12
  support not compiled in, gl drivers/libraries could not be loaded, webgpu
  support not compiled in
```

After `apt-get install -y mesa-vulkan-drivers libvulkan1` (exactly what the spec's
Task 8 prescribes), the same runner reports:

```
ICD dir exists: True
1 adapter(s)
adapter: llvmpipe (LLVM 20.1.2, 256 bits) / CPU
'immediates' on adapter: True
'shader-f16' on adapter: True
```

So **Task 8's apt step is load-bearing and correct**, and AC-7/AC-8's "real CPU
llvmpipe adapter in CI" premise holds. `shader-f16` being present also means AC-11's
optional-dtype path is exercisable in CI.

**The blocker reproduces identically in CI**, ruling out any local-environment
explanation:

```
2 SPIR-V module(s) extracted
  - ..._matmul_8x16x8_f32.spv (6264 B), push-constants: True
  - ..._matmul_8x4x16_f32.spv (3344 B), push-constants: True
[no features (browser-WebGPU-shaped)] 0/2 VALID
      = Capability Capabilities(IMMEDIATES) is not supported
[required_features=['immediates']] 2/2 VALID
```

Byte-for-byte the same two dispatches, the same sizes, the same verdict as locally.

## New finding not in the spec

`huggingface_hub` is left **unpinned** in the `export` extra and floats to
**1.29.0** — a major version. The spike was written against whatever was current on
260831. Pin it explicitly rather than discovering a 0.x→1.x API break inside PR2.

## Options

1. **Ship PR1 only; reclassify WebGPU as research.** PR1 (native + wasm32) is
   entirely unaffected by this finding — its pins resolve and it touches no SPIR-V.
   WebGPU becomes an open question rather than a deliverable.
2. **Redefine the gate honestly as native-wgpu.** Keep PR2's machinery, enable
   `immediates`, and rename the verification level to something that does not say
   WebGPU (e.g. `VALIDATED_NATIVE`). Cheap, real, and does not deliver the original
   goal.
3. **Post-process the SPIR-V** to lower push constants into a uniform buffer before
   validation. Validates a *transformed* artifact rather than the shippable one, and
   needs matching host-side binding changes to ever execute. Weak evidence, real cost.
4. **Find a non-IREE path to WGSL/WebGPU.** Out of scope for this spec; would be its
   own investigation.

**Recommendation: option 1, with option 2 available later.** The user's stated goal
was tested WebGPU kernels for downstream packages. Options 2 and 3 both produce a
green gate that does not establish that. PR1 is genuinely ready and delivers real
value (native parity + wasm32 codegen) without depending on any of this.

## Reproduction

Probes are throwaway; the CI workflow lives on branch `probe/webgpu-ci-adapter`
(`scripts/probe_wgpu_ci.py` + `.github/workflows/probe-wgpu.yml`), which is not for
merge. The local measurements used a scratch venv:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e . \
  "iree-base-compiler>=3.11,<4" "iree-base-runtime>=3.11,<4" \
  "wgpu>=0.32,<0.33" huggingface_hub safetensors
```

then a real `BatchPlanner().plan([AxisSpec("batch", 32, 8)])` (resolves to `SafeMap`)
folded through the spike's `compose_exportable`, exported via `jax.export.export`,
compiled with `--iree-hal-target-backends=vulkan-spirv
--iree-hal-dump-executable-binaries-to=<dir>`, magic-filtered, and pushed through
`device.create_shader_module(code=...)`.

Local host: WSL2, Mesa 25.2.8, `mesa-vulkan-drivers` + `libvulkan1` installed;
adapter `llvmpipe (LLVM 20.1.2, 256 bits)`, type CPU, backend Vulkan. Notably this
box exposes **no** hardware Vulkan adapter, so it is already CI-shaped.

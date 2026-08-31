"""Feasibility spike: xtrax-composed pipeline -> StableHLO -> IREE -> WASM.

NOT shipped. This package lives under ``scripts/`` deliberately: it stays out of
the wheel (``pyproject.toml`` ships only ``src/xtrax``), out of both import-linter
``source_modules`` lists, and out of the tier1_core coverage gate (pytest measures
``--cov=xtrax`` only).

Scope of the spike, per the approved plan (task 260831_iree-wasm-webgpu-export):

- Target **serverless WASM** via IREE's ``llvm-cpu`` backend. WebGPU is deliberately
  excluded: ``webgpu-spirv`` is marked *Experimental* upstream and is absent from
  IREE's six stable deployment configurations.
- Export the model forward plus **pure-JAX tiling only**. ``Tap``/``Sink`` boundaries
  are required by contract to call ``io_callback`` and can never be exported.
- Bake HuggingFace weights in as **closure constants**, so the artifact is
  self-contained.

What this spike does NOT do: execute a wasm32 vmfb. That needs an emsdk-built IREE
runtime (IREE's browser runtime lives in ``experimental/web`` with no published npm
package). Numerics are verified on the **native** llvm-cpu target instead; the wasm32
target is verified only as far as "compiles and produces a non-empty artifact".
"""

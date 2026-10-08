# Standalone ONNX export

`convert_to_onnx` converts one traceable callable. `export_pipeline` is the `BatchPlan` path: it composes the plan, then calls `convert_to_onnx` for an `onnx`-backend target, so the guards below apply there too. A callable that is not a plan uses `convert_to_onnx` directly.

Needs the onnx extra (`pip install xtrax[onnx]`). A missing toolchain raises `CompileError` naming that extra.

```python
import jax.numpy as jnp
from xtrax.export import ONNX, ONNX_OPSET, convert_to_onnx

def step(x):
    return x * jnp.float32(2)

compiled, census = convert_to_onnx(
    step,
    [jnp.ones((4,), dtype=jnp.float32)],
    ONNX,
)
```

`ONNX` is the registered target (`name="onnx"`, `backend="onnx"`, `verification_level=EXECUTED`). `ONNX_OPSET` is `23`. The return is `(CompileResult, OnnxDtypeCensus)`. `compiled.path` is the `.onnx` file (`out_path` overrides the temp file).

Verify: `src/xtrax/export/onnx.py:67`, `:346-426`, `src/xtrax/export/targets.py:296-302`.

## What this adds over `jax2onnx.to_onnx`

| Guard | Behavior |
|---|---|
| x64 refusal | `jax.config.jax_enable_x64` raises `CompileError` before conversion. The graph would otherwise carry int64/f64. |
| jnp-namespace restore | The `to_onnx` call runs inside `_restoring_jax_namespaces`, which puts back public attributes of `jax.numpy`, `jax.lax`, `jax.nn`, and `jax.random` that the converter replaced or deleted (jax2onnx leaves `jnp.cumsum` patched). |
| ONNX RNG-op refusal | After conversion, `find_onnx_rng_ops` scans the graph, subgraphs, and functions. Any op in `ONNX_RNG_OP_TYPES` (`Bernoulli`, `Multinomial`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`) raises `CompileError`. Pass draws in as inputs. |
| Opset pin | `jax2onnx.to_onnx(..., opset=ONNX_OPSET)` with `ONNX_OPSET = 23`. |

Also: pytree inputs are flattened to `ShapeDtypeStruct` leaves; a model of 2 GiB or more is written with external tensor data; the dtype census counts int64 producers (`TopK` / `ArgMax` index outputs stay int64 inside the graph).

Verify: `src/xtrax/export/onnx.py:113-144`, `:251-260`, `:346-405`.

## Rings and divergence

`xtrax.export.rings` is the comparison ladder. `run_ladder(fn, plan, abstract_inputs, concrete_inputs, *, probe_deps, eager_fn, ...)` validates probe deps, then per input class runs R0 (`r0_replay_gate`), R2a (`r2a_fusion`, which produces the budget), R1 (`r1_target_isa`), R2b (`r2b_lowering`), and R3 (`r3_probe`). A failed R0 skips the later rungs for that class and still returns R0's `RingResult`.

Input-class generators on the same module: `nominal`, `symmetric_geometry`, `magnitude_extremes`, `sub_k_neighbours`. Their lengths come from `BUCKET_LADDER`.

`xtrax.export.divergence` classifies leaves. `compare_pytree(expected, actual)` returns `tuple[LeafDivergence, ...]`. `classify_probes(probes, probe_deps, budgets)` returns `tuple[ProbeReport, ...]` with `DivergenceClass` in the order `UNCOMPARABLE`, `DISCRETE_FLIP`, `INJECTED`, `AMPLIFIED`, `ATTENUATED`, `CLEAN`. `RingResult` is defined there and constructed by the rings.

```python
import numpy as np
from xtrax.export.divergence import compare_pytree

leaves = compare_pytree(
    np.array([1.0, 2.0], dtype=np.float32),
    np.array([1.0, 2.0], dtype=np.float32),
)
```

Verify: `src/xtrax/export/rings.py:98`, `:1794-1827`, `src/xtrax/export/divergence.py:90-102`, `:473`, `:671-676`.

## Parity

`find_onnx_rng_ops(model) -> list[str]` returns `"<op_type>:<node name>"` for each RNG node in an `onnx.ModelProto`. `convert_to_onnx` already refuses a non-empty result.

`verify_onnx_parity(expected, onnx_path, concrete_inputs, *, atol=1e-5, rtol=1e-5) -> LeafParityResult` runs the file on ORT's CPU execution provider. Integer and bool leaves are exact.

`verify_native_parity(expected, vmfb_path, concrete_inputs, *, atol=1e-5, rtol=1e-5, function="main") -> LeafParityResult` runs a native vmfb. `expected` is an independent reference: a value produced by the callable under test compares the program to itself.

Verify: `src/xtrax/export/onnx.py:251-260`, `:477-484`, `src/xtrax/export/parity.py:354-389`.

---
title: Probe evidence for sprint 260922 conformance residuals (P0-P12)
description: Measured JAX 0.11.1 / numpy 2.5.1 CPU behaviours behind spec 260922_conformance-residuals §1.1 — donation carriers in jaxprs, synthesizer row-identity unsoundness, byte bitcasts, and which host-transfer oracles work on CPU
task_id: 260922_conformance-residuals
status: final
---

# Probe evidence: sprint 260922 conformance residuals

Evidence for `.praxia/docs/specs/260922_conformance-residuals.md` §1.1. Every number below
comes from one run of the tracked script, so it can be reproduced and re-checked after a JAX
upgrade:

```bash
JAX_PLATFORMS=cpu uv run --extra dev python \
    scripts/measure_conformance_residuals_260922.py --json out.json
```

Environment: jax 0.11.1, numpy 2.5.1, backend cpu, x64 disabled (the JAX default). Run on
2026-09-22.

## What each finding decides

| Probe | Finding | Decides |
|---|---|---|
| P0 | Donation appears only as `donated_invars` on a `jit` eqn: `(True, False)` for `donate_argnums=0`, `(False, True)` for `donate_argnames='y'`, and on the INNER eqn for a nested donating jit. A jitted callable exposes no donate attribute. | T1 detects donation by walking the jaxpr; nothing else is available. |
| P1 | CPU honours donation: `x.is_deleted()` is True after the call. | T1's `is_deleted` assertions are real discriminators on CI, not vacuous. |
| P1b | An eager function calling a donating inner jit deletes the CALLER's buffer. | Nested donation is a live hazard, so T1 rejects every donation site. |
| P2 | `np.unique(axis=0)` merges `-0.0` with `+0.0` (1 unique row). Two identical NaN rows stay 2 uniques. | T4: the shipped synthesizer merges distinct rows. |
| P3 | `jnp.concatenate` of int32 with float32 promotes to float32; `2**24` and `2**24+1` then compare equal. | T4: shipped `_stack_batch_leaves` merges distinct integer rows. |
| P4 | `jnp.moveaxis` on a numpy int64 leaf gives int32 with x64 off: `2**31+5` becomes `-2147483643`. | T4: numpy leaves must be converted to bytes on the host, never via `jnp`. |
| P5 | `device_put(x, donate=True)` records `copy_semantics=(DONATE_INPUT,)`, not `donated_invars`. Eager use on CPU did not delete the buffer. | T1 screens a second carrier; latent on CPU. |
| P6 | `bitcast_convert_type(., uint8)`: complex64 raises TypeError; bf16/f16 `(3,2)->(3,2,2)`; float8/int8 `(3,2)->(3,2)`; bool via `astype`. | T4 byte recipe: split complex into real and imaginary parts. |
| P7 | `np.asarray` on a typed PRNG key array raises TypeError. | T4/T2 refuse key leaves with a typed error. |
| P8 | A spy on `ArrayImpl._value` sees 0 hits for `np.asarray`, `np.array`, `np.ascontiguousarray` and `.item()`; 1 for `device_get`, `.tolist()`, `float()` and `__array__()`. | A `_value` spy cannot be the transfer oracle. |
| P9 | Signed zero stays distinct after a byte bitcast for f16, bf16, f32, and for complex64's imaginary part via `lax.imag`. | The real/imag split is exact. |
| P10 | `jax.transfer_guard_device_to_host('disallow')` never fires on CPU. | The transfer oracle must be structural (AST) plus a call-level `_to_host` spy. |
| P11 | int4 `(4,3)` bitcast raises ValueError; `(4,2)` packs to `(4,)`. `astype(int8)` is exact (`[-8, 7]`). `itemsize` is 1 for int4, uint4 and float4_e2m1fn while `itemsize_bits` is 4. | Widen sub-byte integers first; detect with `jax.dtypes.itemsize_bits`. |
| P12 | Zero-width rows: `np.unique` gives 1 row, inverse all 0; reshape, bitcast `(5,0,4)->(5,0)` and concatenation all work. | Zero-width leaves need no special case. |

## Raw output

```text
jax 0.11.1, numpy 2.5.1, backend cpu
P0 {"plain": [["mul", null], ["add", null]], "jit_no_donation": [["jit", [false, false]]], "jit_donate_argnums_0": [["jit", [true, false]]], "jit_donate_argnames_y": [["jit", [false, true]]], "nested_inner_donation": [["jit", [false, true]], ["add", null]], "jitted_has_donate_attr": false}
P1 {"p1_top_level_deleted": true, "p1b_caller_buffer_deleted": true}
P2 {"signed_zero_unique_rows": 1, "identical_nan_unique_rows": 2}
P3 {"concat_dtype": "float32", "distinct_rows_compare_equal": true}
P4 {"x64_enabled": false, "moveaxis_dtype": "int32", "values": [-2147483643, -2147483642]}
P5 {"eqns": [{"prim": "device_put", "copy_semantics": "(ArrayCopySemantics.DONATE_INPUT,)"}, {"prim": "mul", "copy_semantics": "None"}], "eager_deletes_caller_buffer": false}
P6 {"complex64": "TypeError", "bfloat16": "(3, 2)->(3, 2, 2)", "bool_": "(3, 2)->(3, 2)", "float16": "(3, 2)->(3, 2, 2)", "float8_e4m3fn": "(3, 2)->(3, 2)", "float8_e5m2": "(3, 2)->(3, 2)", "int8": "(3, 2)->(3, 2)"}
P7 {"np_asarray_typed_keys": "TypeError"}
P8 {"np.asarray": 0, "np.array": 0, "np.ascontiguousarray": 0, "jax.device_get": 1, ".tolist()": 1, "float(a[0])": 1, ".__array__()": 1, "a[0].item()": 0}
P9 {"float16_signed_zero_bytes_distinct": true, "bfloat16_signed_zero_bytes_distinct": true, "float32_signed_zero_bytes_distinct": true, "complex64_imag_signed_zero_distinct": true}
P10 {"np.asarray": "no error (guard did not fire)", "np.array": "no error (guard did not fire)", "jax.device_get": "no error (guard did not fire)"}
P11 {"int4_4x3_bitcast": "ValueError", "int4_4x2_bitcast": "(4, 2)->(4,)", "int4_astype_int8_values": [-8, 7], "itemsize_vs_itemsize_bits": {"int4": [1, 4], "uint4": [1, 4], "int8": [1, 8], "float4_e2m1fn": [1, 4]}}
P12 {"np_unique_zero_width_shape": [1, 0], "np_unique_zero_width_inverse": [0, 0, 0, 0, 0], "jnp_reshape_zero_width": [5, 0], "bitcast_zero_width": "(5, 0, 4)->(5, 0)", "concat_zero_plus_normal": [5, 2]}
```

## Read from code, not measured

- **Axis bug.** `synthesize_dedup_spec` (`dedup_synthesis.py:156-157`) receives an already
  batch-first `(N, F)` array from `_stack_batch_leaves`, then reads `N = stacked.shape[axis]`.
  For `axis != 0` that is the feature width.
- **`_screen_jaxpr` gaps** (`memo.py:190-215`). Sub-jaxprs are found with
  `getattr(param, "eqns", None)`, which never matches `lax.cond`'s tuple-valued `branches`, and
  `if depth > 8: return` silently admits deeper nesting. Filed as backlog #5216.

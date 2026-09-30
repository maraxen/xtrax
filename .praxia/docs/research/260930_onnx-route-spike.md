---
title: 'ONNX route spike: jax2onnx to ORT CPU on xtrax''s generic op classes'
description: 'Three pre-registered bathos runs: every tied sort/argsort/top_k/argmax/scatter case is exact on ORT CPU; jax2onnx 0.16.1 breaks nested jit on jax 0.11; threefry is replaced by ONNX RandomUniform and does not run'
status: draft
task_id: 260930_xtrax-onnx-export-eval
date: '260930'
confidence: 'high for the ORT-CPU measurements (three tracked runs, live controls, records verified); none of it is browser evidence'
sources: 'bathos runs e8f39e99, d64357c2, 367c5de9 (project xtrax); scripts/experiments/onnx_route_spike.py + .bth.toml (prereg 526e53e, 8eae357); aminx origin/main browser_validation specs; jax2onnx plugins/jax/core/jit.py:54-63'
---
# ONNX route spike: jax2onnx to ORT CPU on xtrax's generic op classes

## Question

Should `xtrax.export` gain an `onnx` target next to its IREE targets? The
[260914 route note](260914_browser-inference-routes-jaxjs-jax2onnx.md) named the
risks: tied sorts producing int64 `TopK` indices, and multi-key sort stability.
aminx then shipped a `jax2onnx` → ORT-Web browser sampler that passed exact-token
gates. aminx did not measure the op classes an xtrax target would have to vouch
for *generically*, and it supplies RNG from the host. This spike measures those
op classes.

## What aminx already established (read from aminx `origin/main`, not re-run)

- Export is `jax2onnx.to_onnx(fn, [ShapeDtypeStruct…])` at fixed buckets
  (`scripts/browser_validation/p07_split_export.py:241`). Weights are embedded
  because ORT-Web rejects external data (`p07_knobs_gate.py:1071-1095`).
- Index tensors at graph I/O are int32. The JS side is `Int32Array` throughout.
- RNG is host-supplied (SplitMix64 + Gumbel in JS). `random_split` is unsupported by
  `jax2onnx` (`jax2onnx_spike.py:56-61`).
- Gate results cited in `260929_p07-split-export.md:427-442`:
  - The knobs gate matched tokens 288/288 bitwise against both ORT-CPU and ORT-Web wasm.
  - WebGPU was never exercised: WSL2 exposes no adapter (run `ca0ea202`).
- The export harness gates on `jax_enable_x64` and on "jax2onnx already imported".
  `jax2onnx` patches `jnp` globally at import.

## Design (pre-registered, `scripts/experiments/onnx_route_spike.bth.toml`)

Ten cases × `export_mode ∈ {standard, web}` × ORT CPU EP at
`ORT_DISABLE_ALL` / `ORT_ENABLE_ALL`. Integer and bit outputs are compared
exactly against the JAX oracle. The MLP must match within 1e-5 max-abs. A dtype
census of each graph (including subgraphs) is recorded.

Tie-rich fixtures: 4 distinct int keys and 8 distinct float values over 256 slots.
With 16 top-k slots all tied, tie order alone decides the answer.

Controls, all live in every run:

| Control | Result |
|---|---|
| Descending-index tiebreak changes the argsort answer | true |
| Descending-index tiebreak changes the top_k answer | true |
| Comparator fires on a 2-slot swap | true |
| Comparator stays silent on identity | true |
| Comparator fires on a 1-ulp bit change | true |
| 1e-3 weight-perturbed MLP, converted by the same path, misses the bar | max-abs 0.0101 ≫ 1e-5 |

## Results

| Run | Toolchain | Outcome | Converted | Value-divergent | I/O dtype changed |
|---|---|---|---|---|---|
| `e8f39e99` | jax 0.11.1 + jax2onnx 0.16.1 | partial_conversion | 12/20 | 0 | 0 |
| `d64357c2` | jax 0.10.2 + jax2onnx 0.16.1 | partial_conversion | 18/20 | 0 | 0 |
| `367c5de9` | jax 0.11.1 + jax2onnx 0.17.0 | partial_conversion | 18/20 | 0 | 0 |

All three: `completed`, exit 0, pinned to the pre-registration commit.
ORT 1.30.0, onnx 1.23.0, opset 23.

### 1. Every index-producing case that converts is exact

The following cases are exact under both ORT optimisation levels, in every run
where they convert:

- 2-key tiebreak sort
- stable `argsort` (int and descending float)
- `lax.top_k`
- `top_k` via 2-key sort
- `argmax` on ties
- int scatter-add + gather

The composer-built TinyMLP is within the float bar. The 260914 note hypothesised
(§5) that multi-key sort might re-expose stability dependence through the radix
lowering. That did not happen on ORT CPU, including for *stable* `argsort`, the
op IREE breaks.

### 2. int64 is internal only; graph I/O keeps JAX's dtypes

int64 appears inside every index graph:

- in `TopK` and `ArgMax` outputs (the ONNX spec mandates int64 there)
- in shape plumbing: 22 int64 tensors in the scatter case, from
  `Shape`/`Concat`/`NonZero`/`Gather`

`jax2onnx` casts back, so graph I/O dtypes match JAX's (int32 stays int32) in all
18 converted cases. What this means for ORT-Web's WebGPU EP, which
lacks int64, is **still untested**. The census shows those int64 nodes exist; it
does not show where the WebGPU EP places them.

### 3. jax2onnx 0.16.1 is incompatible with jax 0.11 for any nested `jit`

`jax2onnx/plugins/jax/core/jit.py:59-63` constructs
`jcore.Var(aval, initial_qdd, final_qdd)`. jax 0.11.1's `Var.__init__` takes only
`aval`. Every function that traces an inner `jit` therefore fails to convert with
`TypeError: Var.__init__() takes 2 positional arguments but 4 were given`. That
includes `jnp.argsort(stable=True)` and every `jax.random` draw.

The failure is loud, not silent. `jax2onnx` 0.17.0 fixes it on jax 0.11.1. xtrax
pins `jax>=0.10.2,<0.12`, so an `onnx` extra needs `jax2onnx>=0.17.0`. Whether
0.17.0 still works on jax 0.10.2 was **not tested**.

### 4. Threefry is not preserved: jax2onnx substitutes ONNX's own RNG

`jax.random.uniform(key, …)` converts, but the graph contains `RandomUniform` and
`RandomUniformLike` rather than threefry arithmetic. ONNX's RNG ops are seeded by
attribute, not by the key tensor, so they cannot reproduce JAX's key-determined
bits. The exact seeding semantics are inferred from the op names and have not been
measured. ORT then rejects the graph: `No Op registered for BitCast with
domain_version of 23`.

Today this fails loudly only because of the `BitCast`. A future ORT that registers
`BitCast` would run the graph and return non-JAX random numbers with no error. This
is the ONNX analogue of the IREE split-key miscompile.

Nested `jax.random.split` does not convert at all:
`No plugins registered for primitive 'random_split'`.

### 5. `export_mode="web"` was inert here

Every converted case produced a byte-identical graph under `standard` and `web`.
The mode axis therefore contributes no independent confirmation for these op
classes.

## Implications for an xtrax `onnx` target

1. **Viable.** A generic ONNX target can be `EXECUTED`-verified on ORT CPU. That
   is a stronger level than `wasm32` or the SPIR-V targets can reach today. The
   tie-order failure class that forced IREE's `sort-stability` rule did not
   reproduce.
2. **Refuse in-graph RNG.** The target needs a safety rule that fails on
   `RandomUniform*`/`RandomNormal*`/`Multinomial`/`Bernoulli` in the converted
   graph and on RNG primitives in the jaxpr. aminx has both audits, generic and
   portable: `onnx_audit.find_onnx_rng_ops`, which walks If/Loop/Scan subgraphs,
   and `rng_audit.find_rng_primitives`. Keys should become host inputs, as aminx
   does.
3. **Pin `jax2onnx>=0.17.0`** in the extra, and add the x64 and import-order guards
   to the export entry.
4. **Record the int64 census** in `ExportResult`. aminx has no census; this
   spike's `dtype_census` is the only one. A browser target will need it.
5. **Browser claims stay gated** on real ORT-Web execution. The xtrax
   webgpu-route constraint applies verbatim.

## Not claimed

- Nothing here is ORT-Web or browser evidence.
- `jax2onnx` 0.17.0 on jax 0.10.2 was not measured.
- Only CPUExecutionProvider was used.
- The fixtures are small (N=256) generic ops, not aminx's model. aminx's
  model-level parity is aminx's own record.
- Only one seed was run. Ties are structural in these fixtures, so a second seed
  would test the same property.

# Export

Compile a planned xtrax pipeline to a standalone artifact, via StableHLO and IREE.

Install the toolchain with the `export` extra:

```bash
pip install xtrax[export]
```

`xtrax.export` itself imports on a base install; IREE is loaded lazily, so a
missing extra surfaces as a `CompileError` naming it at compile time.

## Targets and what each one proves

A `Target` pairs an IREE backend with the dtype vocabulary and flags it needs.
`VerificationLevel` records how far the artifact's correctness was established,
which is deliberately not the same for every target:

| Target | Level | What was established |
|---|---|---|
| `NATIVE` | `EXECUTED` | Compiled and run; numerics matched an independent oracle |
| `WASM32` | `CODEGEN_ONLY` | Compiled. Nothing more |
| `VULKAN_SPIRV` | `CODEGEN_ONLY` | Compiled; SPIR-V extracted |
| `METAL_SPIRV` | `CODEGEN_ONLY` | Compiled. Nothing more |

`WASM32` is not executed because doing so needs an emsdk-built IREE runtime,
which has no published package. The SPIR-V targets are not executed because
doing so needs a device this package does not require. `ExportResult.verified`
is unconditionally `False` for a `CODEGEN_ONLY` target; read
`verification_level` to distinguish that from a genuine failure.

No target is registered at `VALIDATED`, and `export_pipeline` raises
`NotImplementedError` for one, rather than reporting a `verified` it has nothing
to compute.

`VULKAN_SPIRV` is the only target that populates `ExportResult.spirv_bytes`.
`METAL_SPIRV` is named for its input dialect, not its output — it emits Metal
Shading Language, so there is no SPIR-V to extract and `spirv_bytes` stays
`None`.

### Dtypes

Each target declares the dtypes it carries. The envelope splits by verification
level rather than by backend, because every backend compiles the same set and
only the runtime differs:

| dtype | `EXECUTED` | `CODEGEN_ONLY` |
|---|---|---|
| `f32`, `f16`, `i32`, `i64`, `i8`, `u32`, `bool` | yes | yes |
| `bf16` | no | yes |
| `f64` | no | no |

`bf16` compiles everywhere and its signature is untouched, but IREE's Python
runtime cannot map bf16 buffers back to numpy, so an executed target cannot run
it and therefore cannot verify it.

`f64` is rejected everywhere. IREE does not refuse it — it demotes it to `f32`
and rewrites the entry point's public signature to match, with a warning rather
than an error. On an executed target that surfaces later as a buffer-level type
mismatch; on a codegen-only target it never surfaces at all, and you get an
artifact that quietly takes and returns `f32`. Cast to `f32` yourself and the
precision loss is a decision rather than a discovery.

### Loading checkpoint weights

`load_hf_weights` reads a safetensors checkpoint and casts anything the target
will not carry, reporting every leaf it touched:

```python
from xtrax.export import NATIVE, load_hf_weights

loaded = load_hf_weights("org/model", target=NATIVE)
loaded.report.dtypes_cast   # ("layer.0.weight: bf16 -> f32", ...)
```

Casting is what makes a bf16 checkpoint verifiable, since `NATIVE` cannot run
bf16. `f64` tensors raise instead of being cast, for the reason above.

## Exporting

```python
from xtrax.export import NATIVE, WASM32, export_pipeline

results = export_pipeline(
    model,
    plan,
    abstract_inputs=[jax.ShapeDtypeStruct(xs.shape, xs.dtype)],
    concrete_inputs=[xs],
    targets=(NATIVE, WASM32),
    reference_fn=lambda inputs: jnp.stack([model(x) for x in inputs[0]]),
)
results["native"].verified   # True when parity passed
results["native"].path       # the compiled artifact, for run_native_vmfb
```

`export_pipeline` is all-or-nothing across `targets`: the first failure aborts
the whole call, and no partial dict is returned.

### `reference_fn` must be independent

An `EXECUTED` target requires `reference_fn`, and it must compute the expected
value from the model directly. Passing
`jax.jit(build_traceable_callable(...))` type-checks and verifies nothing:
comparing the composed callable against itself under two backends detects
lowering divergence only. A composition error — wrong nesting, a dropped
boundary, a mis-shaped carry — changes both sides identically.

## What can cross the boundary

`validate_export_safe` runs before any tracing.

Supported strategies are `Vmap`, `SafeMap`, `Scan`, and `DedupGather`. `Bucket`
is host-tier: pad with `bucketize()` before the boundary. `WhileCarry` has an
unbounded trip count: convert it to a `Scan` with a static length.

Boundary ops are judged by kind:

- **`fuse`** always crosses. It is an in-trace reduction.
- **`tap`** never crosses. A Tap is `T -> T` and feeds downstream, so it cannot
  be dropped on any target.
- **`sink`** crosses only when declared, see below.

### Materializing sinks

A sink runs host code the exported program cannot call, so by default a plan
carrying one is rejected. But a sink that only *records* the per-step values
does not need to run at all: the executor already returns exactly what the sink
receives. `execute_scan_axis` fires `boundary.sink(y)` per step and returns the
stack of those same `y`.

Declaring `materialize=True` says so, and export strips the sink call and
exposes those values as the artifact's output instead:

```python
boundary = AxisBoundary(sink=recording_sink, materialize=True)
result = export_pipeline(model, plan, abstract, [xs],
                         axis_boundaries={"batch": boundary},
                         targets=(NATIVE,), reference_fn=reference)
values = run_native_vmfb(result["native"].path, xs)   # what the sink would have seen
```

Three things to know:

- **It is a precondition, not a proof.** The sink slot accepts any callable, and
  the `T -> None` contract is convention. Stripping guarantees only that the
  call is absent from the exported trace; it says nothing about a side effect a
  non-conforming sink would have performed.
- **It costs memory.** Materializing allocates the full stacked array where an
  `io_callback` sink would have streamed and discarded each step. Hence opt-in.
- **It has no effect outside `xtrax.export`.** An eager run fires the sink
  exactly as before.

`materialize=True` cannot be combined with a `fuse` on the same axis — fuse
collapses the very array materialize needs to expose — and only one axis per
plan may materialize. Both raise named `PlanTopologyError` subclasses.

## Multi-axis plans

One two-axis shape composes: an outer `Vmap` axis wrapping an inner `Scan` axis.
It is the shape `tests/stages/test_nested_ordering.py` certifies, and the
composer follows that recipe rather than generalising past it. Deeper nestings,
and other two-axis pairings, are refused with `MultiAxisCompositionError`.

The initial carry's shape selects how lanes are iterated:

- **Carry batched to the outer axis** — every `scan_init` leaf has the outer
  cardinality as its leading dimension. The outer axis is then a dimension
  rather than a loop, so the composer emits a single `jax.lax.scan` and no
  `jax.vmap` runs at all. This is the recommended form: write the transition's
  per-step logic as ordinary broadcasting array ops.
- **Carry not batched** — lanes can only be iterated by an actual `jax.vmap`.
  That composes when the inner axis's sunk value does not depend on the lane. When
  it does, JAX refuses to vmap an ordered IO callback, and the composer re-raises
  the executor's own guidance as `MultiAxisCompositionError`.

```python
plan = ...                       # outer "lane" Vmap axis, inner "step" Scan axis
init = jnp.arange(batch) * base  # leading dim == the lane axis's cardinality
composed = build_traceable_callable(transition, plan, boundaries, scan_init=init)
```

Boundaries attach to the inner `Scan` axis. On the batched form the outer axis
has no per-lane call site for a `fuse`/`tap`/`sink` to fire at, so one declared
there is refused rather than silently dropped.

`sink` receives the exact value returned as the step's `y`, because both come
from the single `y` the transition returned. Materializing sinks (above)
depends on that: export strips the sink and reads the returned stack instead, so
the two must be the same value.

Ordering is certified in two independent places, because they fail differently.
A composer-level test asserts host-call order on the un-stripped callable,
catching wrong axis nesting or a dropped boundary. A separate test exports the
stripped callable, runs the artifact, and decodes `(lane, step)` back out of its
output — catching a lowering bug that a pre-export test double cannot see. A
third leg runs the stripped callable in pure JAX in between, so a failure points
at either the composition or the lowering rather than at both.

## A footgun worth naming

For a `Scan` axis, the `fn` you pass to `export_pipeline` is always the
transition that gets exported. `Scan.transition` is read only by the eager
`xtrax.tiling.dispatch` path and is never consulted here. Setting both means the
exported artifact can differ from what an eager run of the same plan does.

## WebGPU

Not currently reachable through IREE. IREE 3.11 registers no webgpu backend, and
its Vulkan HAL passes dispatch parameters through push constants, which are not
part of the WebGPU feature set — a shader validator configured the way a browser
is configured rejects the result. Measured in
`.praxia/docs/research/260901_webgpu-export-measurement-pass.md`.

```{automodule} xtrax.export
:members:
:undoc-members:
:show-inheritance:
```

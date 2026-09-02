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

`WASM32` is not executed because doing so needs an emsdk-built IREE runtime,
which has no published package. `ExportResult.verified` is unconditionally
`False` for a `CODEGEN_ONLY` target; read `verification_level` to distinguish
that from a genuine failure.

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

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
| `NATIVE_PORTABLE` | `EXECUTED` | Compiled and run on a fixed, portable CPU baseline (see below) |
| `WASM32` | `CODEGEN_ONLY` | Compiled. Nothing more |
| `VULKAN_SPIRV` | `CODEGEN_ONLY` | Compiled; SPIR-V extracted |
| `METAL_SPIRV` | `CODEGEN_ONLY` | Compiled. Nothing more |

`WASM32` is not executed because doing so needs an emsdk-built IREE runtime,
which has no published package. The SPIR-V targets are not executed because
doing so needs a device this package does not require. `ExportResult.verified`
is unconditionally `False` for a `CODEGEN_ONLY` target; read
`verification_level` to distinguish that from a genuine failure.

### `NATIVE` vs. `NATIVE_PORTABLE`

Both compile via IREE's `llvm-cpu` backend and are `EXECUTED`, but they answer
different questions. `NATIVE` passes `--iree-llvmcpu-target-cpu=host`: it is
tuned to the machine doing the compiling, which is correct for its job as a
parity oracle and wrong for anything handed to someone else — an artifact
built with, say, AVX-512 enabled can fault with an illegal instruction on a
recipient's CPU that lacks it.

`NATIVE_PORTABLE` passes `--iree-llvmcpu-target-cpu=x86-64-v2` instead: a
fixed ISA baseline rather than "whatever this machine has". The portability
claim is narrower than "distributable":

- the artifact is an `embedded-elf-x86_64` module with
  `cpu_features = "+cmov,+mmx,+popcnt,+sse,+sse2,+sse4.2,+cx16,+sahf,+cx8,+crc32,+x87,+fxsr"`,
  using IREE's own ELF loader with no libc or dylib dependency;
- it still declares `Module Dependencies: hal, version >= 6, required` — it
  is **not** a standalone binary, and the recipient needs an IREE runtime
  (the `xtrax[export-runtime]` extra) to load it, not just a compatible CPU;
- x86-64-v2 is **not** "any CPU since 2009". SSE4.2 — the feature that
  defines the v2 baseline — arrived with Intel Nehalem in late 2008 but AMD
  only added it at Bulldozer in 2011, and Intel's own Atom line lacked it
  through Silvermont in 2013. The honest claim is "any x86-64 CPU from
  roughly 2013 onward".

No target triple is passed alongside the CPU flag. Adding
`--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu` produces a
byte-identical artifact (same md5): IREE rewrites the embedded triple to
`x86_64-unknown-unknown-eabi-elf` for its embedded loader regardless of what
triple is requested, so naming an OS there would be inert and would also
misdescribe an artifact that commits to no OS.

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

## Divergence mapping

`xtrax.export.parity.compare` reduces a comparison to one scalar,
`max_abs_diff`, over one array. That answers *is it correct*. Divergence
mapping answers a different question: *where did it stop being correct*.
Both ship, and neither replaces the other — for a divergence carried
entirely by integer indices with every float leaf bit-identical, `compare`
reports `max_abs_diff = 0.0` and passes. It has not lost resolution; it read
zero, because the bug lives on a dtype it does not measure.

### The comparison ladder

A **ring** is a controlled comparison with exactly one independent variable
and a shared reference input: both sides are the same computation, so a
disagreement is attributable to the varied axis and nothing else.

| | Varies | Edits program outputs | Isolates |
|---|---|---|---|
| **R0** *(gate, not a ring)* | — replay, same artifact | no | runtime nondeterminism |
| **R1 Target** | target CPU: `NATIVE` vs `NATIVE_PORTABLE` | no | ISA-dependent codegen |
| **R2a Fusion** | eager vs `jit` — same compiler, different graph | no | the model's own fusion sensitivity |
| **R2b Lowering** | `jit` XLA vs IREE — same graph, different backend | no | export/lowering fidelity |
| **R3 Probe** | cut depth — named intermediates as extra outputs | **yes** | spatial onset inside the model |

**R0 is not a ring.** Replaying one artifact on one input varies nothing.
It is the ladder's validity gate: if the system is nondeterministic, no
ring's disagreement localizes to anything, so R0 runs first and gates every
other rung.

**R1 has two executable legs, not three.** `WASM32` is `CODEGEN_ONLY` (see
above) — a ring needs two runnable sides, so wasm32 cannot be a leg.

Input class (`nominal`, `symmetric_geometry`, `magnitude_extremes`,
`sub_k_neighbours`, …) is not a peer rung. It is a stratification variable
applied to R1/R2/R3 — each generator is labelled in the report with whether
it is in contract, so no verdict rests on an out-of-contract class alone.

### Escalation order

Run R0, then the non-editing rings (R2a, R1, R2b), then R3 last.

The ordering is about interpretability, not cost. R3's result is
uninterpretable until the section-5.4 fidelity precondition passes, because
instrumenting a program changes its own fusion decisions — adding outputs
changes DCE and fusion in both XLA and IREE, and a probe that forces
materialization can suppress the very fusion that caused the divergence
being investigated. A rung that edits what the program returns must
therefore come after every rung that does not. `run_ladder` enforces this
order and refuses to run R3 at all when R0 has failed or the fidelity check
inside R3 trips.

A probe is not a `Tap` or a materializing `sink` — the boundary machinery
described above does not apply here. A probe is simply a named entry in the
step function's own return pytree; nothing about the export boundary needs
to change to add one.

### The tolerance budget is a heuristic, not an error bound

```
budget_leaf = max(ULP_FLOOR, SLACK * m_leaf)   # ULP_FLOOR = 4 ULP, SLACK = 4.0
```

`m_leaf` is R2a's measured eager-vs-`jit` divergence for a float leaf. The
floor exists so that a model whose fusion is a no-op — `eager == jit`
bit-identical, the ordinary case — does not get budget 0 and fail on a
single last-bit difference; the calibration then adapts the budget upward
for models with genuine fusion sensitivity.

A bound on fusion sensitivity is not a bound on lowering fidelity — they are
different failure surfaces measured in the same units. R2a and R2b are
therefore always reported separately, and the budget is advisory on R2b
rather than authoritative.

**Calibration applies to float leaves only.** Integer and bool leaves are
judged by exact match, and no measurement loosens that. The budget is
deliberately inert for the bug class this tooling exists to catch: an index
divergence is either an exact match or a `DISCRETE_FLIP`, never a matter of
degree.

### Reading a `RingResult`

`passed` records that a rung's comparison completed over comparable leaves —
for R1, also that its CPU-feature precondition held — never that the two sides
agree. Divergence is reported per probe in `probes`, classified against R2a's
budgets. R1, R2b and R3 populate `probes` when given `probe_deps` (`run_ladder`
always passes it); R0 and R2a never do. So a flipped index can show up as a
`DISCRETE_FLIP` probe on a rung whose `passed` is `True` — check `probes`, not
just `passed`:

```python
any(p.divergence_class != DivergenceClass.CLEAN for r in results for p in r.probes)
```

When R2a's budgets are incomplete, R1 and R2b omit only the affected probe (and
anything that depends on it) from `probes` and say so in `notes` — every other
declared probe is still classified, and `passed` is unaffected either way.

### Declared dependencies

`probe_deps` names each probe's immediate predecessors, and classification
(`CLEAN`, `AMPLIFIED`, `ATTENUATED`, `INJECTED`, `DISCRETE_FLIP`,
`UNCOMPARABLE`) is always against the worst predecessor. `UNCOMPARABLE` is a
probe with a leaf whose shape or dtype differs between the two sides: no
element-wise comparison ran, and probes downstream of it are never labelled
`INJECTED` or `DISCRETE_FLIP`, since that would assert a comparison that never
happened. It is supplied by the caller, not inferred: dataflow edges cannot be
recovered from a flat output tuple, so an undeclared dependency is an error
rather than a default. It is also a
property of the exported *configuration*, not of the model — a plan with an
optional input branch can drop an edge entirely, so the same model can
legitimately need two different `probe_deps` mappings across two exports.

`validate_probe_deps` checks every declared edge before `run_ladder` runs
anything: for `p -> q`, the ops needed for `p` (its backward slice) must be a
subset of the ops needed for `q`. This is a cheap guard against the likeliest
authoring error — writing a DAG by hand and drawing an edge that is not
actually there — not a proof that the declared graph is complete. It has
three known blind spots:

- a **transitive** edge `p -> r` passes when the true dependency is
  `p -> q -> r`;
- a **passthrough** probe with an empty slice is a subset of everything, so
  an edge into it always passes regardless of whether it is real;
- a **missing** edge is invisible to a check that only validates edges that
  were declared.

`probe_resolution` reuses the same backward slices to report `slice_delta`
per declared edge, as a heuristic proxy for localization resolution rather
than a bound on it: sibling slices overlap and do not partition the module,
a loop body executing many times counts its ops once rather than once per
iteration, and the count is taken from the instrumented module's StableHLO
before IREE's own fusion and DCE — the stage where the divergence actually
lives.

## WebGPU

Not currently reachable through IREE. IREE's `webgpu-spirv` backend exists
upstream but is a build-time plugin that no published wheel enables, so the
installed compiler registers no `webgpu` backend; and IREE's Vulkan HAL passes
dispatch parameters through push constants, which a shader validator configured
the way a browser is configured rejects. Measured in
`.praxia/docs/research/260901_webgpu-export-measurement-pass.md`.

WebGPU has since standardised push constants as Immediates, shipping in Chrome,
so that second point is no longer a permanent property of the web platform. It
does not make the route reachable: the backend is still absent from every
published wheel, and `iree#24650` — `webgpu-spirv` failing on any dispatch that
carries push constants — is still open. See
`.praxia/docs/specs/260910_webgpu-export-route.md` for the full gate.

```{automodule} xtrax.export
:members:
:undoc-members:
:show-inheritance:
```

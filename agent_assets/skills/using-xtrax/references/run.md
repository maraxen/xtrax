> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Run Layer (20% of depth — RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec/make_sink, ZarrStagingSink, zarr_integrity, AxisBoundary, Fuse/Tap/Sink, topology validation, boundary executor)

#### RunSpec: Experiment Configuration

Base class for experiment specifications (custom subclasses define your domain):

```python
from xtrax.run.spec import RunSpec

class MyRunSpec(RunSpec):
    """Custom experiment configuration."""
    model_dim: int
    learning_rate: float
    # Add domain-specific fields alongside the inherited fields:
    # seed: int, axes: list[AxisSpec], carry_specs: list[CarrySpec], boundaries: list[AxisBoundary] | None

run = MyRunSpec(
    seed=42,
    axes=[
        AxisSpec(name="batch", cardinality=1000, default_batch_size=32),
        AxisSpec(name="seqlen", cardinality=512, default_batch_size=128),
    ],
    model_dim=768,
    learning_rate=1e-4,
)
```

Verify: `src/xtrax/run/spec.py`

**Identity factory**: `RunSpec.from_spec(spec)` returns `spec` unchanged (no-op classmethod).  
Verify: `src/xtrax/run/spec.py` (search `from_spec`)

⚠ GAP: `RunSpec`, `CarrySpec`, `DedupSpec`, and stage boundaries are **not exported from top-level `xtrax`**.  
Import via subpackages (all but `DedupSpec` are re-exported at package level):
```python
from xtrax.run import RunSpec              # verify: src/xtrax/run/__init__.py
from xtrax.tiling import CarrySpec         # re-exported from xtrax.tiling.__init__
from xtrax.tiling.dedup import DedupSpec   # not re-exported at xtrax.tiling level
from xtrax.stages import AxisBoundary, Fuse, Tap, Sink  # verify: src/xtrax/stages/__init__.py
```

#### InputResolver: Data Iteration Protocol

Implement the `singledispatch` protocol to resolve data sources:

```python
import functools
from xtrax.run.resolver import InputResolver, RuntimeBundle, FeatureBatch
from xtrax.run.spec import RunSpec

# InputResolver is a Protocol — implement it as a callable class:
class MyResolver:
    """Custom data loader for your domain."""
    
    def __call__(self, spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch:
        """Return a single FeatureBatch for a given RunSpec + RuntimeBundle."""
        # Use functools.singledispatch at module level for spec-type dispatch
        return resolve_batch(spec, bundle)

@functools.singledispatch
def resolve_batch(spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch:
    raise NotImplementedError(f"No resolver for spec type {type(spec)}")

@resolve_batch.register(MyRunSpec)
def _(spec: MyRunSpec, bundle: RuntimeBundle) -> FeatureBatch:
    batch = next(iter(bundle.iterator))  # Pull one batch from materialized iterator
    return FeatureBatch({"inputs": batch})
```

Verify: `src/xtrax/run/resolver.py`

#### RuntimeBundle: Iterator + Model

Pair an iterator with a model:

```python
from xtrax.run.resolver import RuntimeBundle

runtime = RuntimeBundle(
    iterator=my_axis_iterator,  # MapIterator or ScanIterator
    model=my_model,             # eqx.Module
)
```

Verify: `src/xtrax/run/resolver.py`

#### FeatureBatch: Type Alias

`FeatureBatch` is `NewType("FeatureBatch", dict[str, Any])` — a dict-like structure with at minimum `{"inputs": ..., "targets": ...}`:

```python
from xtrax.run.resolver import FeatureBatch

batch: FeatureBatch = {
    "inputs": jax.numpy.ones((32, 10)),
    "targets": jax.numpy.ones((32,)),
}
```

Verify: `src/xtrax/run/resolver.py`

#### SinkSpec + make_sink: Output Routing

Declare how to save results:

```python
from xtrax.run import SinkSpec, make_sink

spec = SinkSpec(
    run_id="2026-08-24T14-00-demo",      # Required -- join key for sink provenance
    output_dir=Path("/path/to/outputs"),  # Directory for output files (None = no output)
    format="zarr",                         # "jsonl" | "h5" | "zarr" | "none" (default: "jsonl")
    flush_every=10,                        # Flush buffer every N stage calls (default: 1)
)

sink = make_sink(spec)  # ZarrStagingSink for "zarr", None for "none"
```

**Driver shortcut — `derive_sink_spec`**: when your driver has a `RunSpec`,
derive the sink spec instead of hand-building one. Run id precedence:
explicit override > `run_spec.run_id` > auto-generated (`run-` + 12 hex
chars, via `new_run_id()`):

```python
from pathlib import Path

from xtrax.run import RunSpec, derive_sink_spec, make_sink

run_spec = RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None,
                   run_id=None)  # optional static field; None = auto-generate
sink_spec = derive_sink_spec(run_spec, output_dir=Path("/path/to/outputs"))
sink = make_sink(sink_spec)  # format defaults to "zarr" here (not "jsonl")
```

Zarr sinks auto-capture static run provenance: git SHA/branch/dirty +
`run_id` + UTC `created_at` on the store's root group, and a minimal
`run_id`/`git_sha` pointer per drained key's group. Git capture never raises
(outside a repo it records `git_sha="unknown"` and emits a `UserWarning`).
Call `sink.finalize()` once at run end to consolidate store metadata; no
`stage()`/`drain()` is legitimate afterwards. A second sink opened against
the same `output_dir` must carry the same `run_id`, or construction raises.
Optional `SinkSpec.extension_schema` (JSON-Schema-style dict) validates
caller attrs at `stage()` time; core field names (`git_sha`, `git_branch`,
`git_dirty`, `run_id`, `created_at`) are reserved.

Verify: `src/xtrax/run/sink.py:14-39`

🚫 HALTS: `make_sink` raises `NotImplementedError` for `"jsonl"` and `"h5"` — only `"zarr"` and `"none"` are backed by a real implementation today; `jsonl`/`h5` remain routing-only stub values pending their own writers.  
Enforcement: `src/xtrax/run/sink.py:32-39`

#### ZarrStagingSink: Keyed Staging for io_callback Streaming (0.4.0a3+)

A keyed staging buffer for JAX `io_callback`-driven streaming output, draining into nested Zarr groups. Generalizes the keyed-staging-then-drain pattern for per-chunk tensor payloads (sequences, logits, encoder intermediates). Domain-specific `io_callback` dispatch stays with the caller — this class owns staging and Zarr storage only.

```python
from xtrax.run import SinkSpec, ZarrStagingSink

sink = ZarrStagingSink(SinkSpec(output_dir=out_dir, format="zarr", flush_every=8))

# Buffer named arrays (and optional JSON-safe metadata) under an opaque key tuple.
# Key components, stringified and joined by "/", become the Zarr group path.
sink.stage((batch_idx, chunk_start), attrs={"model": "v3"}, logits=logits, seqs=seqs)

# Repeated stage() for the same key merges: same-name arrays overwrite, new names accumulate.
# Auto-drains to disk every spec.flush_every stage calls, or explicitly:
sink.drain()   # write all pending payloads + attrs into the Zarr store, clear buffer

payload = sink.take(key)  # pop a still-buffered payload WITHOUT persisting
len(sink)                  # number of keys currently buffered (not yet drained)
```

Verify: `src/xtrax/run/zarr_sink.py:24-125`

⚠ WARN: `zarr` is an optional extra (`pip install xtrax[io]`), imported lazily inside `ZarrStagingSink.__init__` — `xtrax.run` stays importable without it; constructing a sink without zarr raises `ImportError` naming the install command.  
🚫 HALTS: `ZarrStagingSink` requires `spec.format == "zarr"` and a non-None `output_dir` (`ValueError` otherwise). Verify: `src/xtrax/run/zarr_sink.py:36-41`  
⚠ WARN: `take()` discards any pending `attrs` for the key — it returns the in-memory payload without persisting; use `drain()` to persist. `take()` on an unknown key raises `KeyError`.

#### zarr_integrity: Content Digests + Durability (0.4.0a5+)

Content-digest and durability primitives for Zarr directory stores (hoisted from aminx — fully generic). Use for done-markers recording "this output is exactly what I wrote":

```python
from xtrax.run import zarr_content_digest, fsync_tree

fsync_tree(store_path)                    # durabilize directory-of-many-files bottom-up FIRST
digest = zarr_content_digest(store_path)  # deterministic sha256 over full logical content
```

- `zarr_content_digest(path)` — sha256 over the store's paths, attrs, and array data; unaffected by filesystem metadata or which process wrote it.
- `fsync_tree(path)` — fsync every file and directory bottom-up; call before trusting the digest.
- Lower-level building blocks also exported from `xtrax.run`: `canonical_json_bytes`, `normalize_json_value`, `update_array_digest`, `update_zarr_node_digest`, `fsync_file`, `fsync_directory`.

Verify: `src/xtrax/run/zarr_integrity.py`

⚠ WARN: only `zarr_content_digest`/`update_zarr_node_digest` need the `zarr` extra, imported lazily at call time; the JSON/fsync helpers have no zarr dependency. Domain orchestration (locking, atomic promotion, done-marker schemas) is the caller's responsibility.

#### AxisBoundary: Per-Axis Operations

Bundle optional post-processing, monitoring, and output operations for one axis:

```python
from xtrax.stages.boundaries import AxisBoundary, Fuse, Tap, Sink

class MyFuse:
    """Average across axis."""
    def __call__(self, stacked):
        return jax.numpy.mean(stacked, axis=0)

class MyTap:
    """Log values each step."""
    ordered = True
    def __call__(self, x):
        print(f"Step output: {x.shape}")
        return x

class MySink:
    """Write to file."""
    ordered = True
    def __call__(self, x):
        # Import from the vendored shim, never jax.experimental directly
        from xtrax.stages._callback import io_callback
        io_callback(self._write, None, x, ordered=self.ordered)
    def _write(self, x):
        # Host-side: write x to disk (e.g. ZarrStagingSink.stage)
        pass

boundary = AxisBoundary(
    fuse=MyFuse(),   # Pure JAX reducer (inside jit)
    tap=MyTap(),     # Identity + side effect (outside jit, via io_callback)
    sink=MySink(),   # Terminal side effect (outside jit, via io_callback)
)
```

Verify: `src/xtrax/stages/boundaries.py:84-98`

**Invariant**: `AxisBoundary` fields are **all static** (`eqx.field(static=True)`). It has no dynamic leaves.

Topology rules (ordered tap/sink + Vmap conflict; Scan on heterogeneous axis) are enforced at **plan-construction time** by `validate_plan_topology` (0.3.1+, see next section) — the pre-0.3.1 `make_inference_plan` gap is closed. The executor re-checks the Vmap+ordered case at execution time as defense-in-depth (`ExecutorError`).

#### Plan Topology Validation: validate_plan_topology + PlanTopologyError (0.3.1+)

> `validate_plan_topology`/`PlanTopologyError`: 0.3.1+. `axis_boundaries_by_name`: unreleased `main` only (T1-02) — not in the 0.4.0a5 wheel.

Catches structurally-impossible plan/boundary pairings before any JAX trace:

```python
from xtrax.stages import PlanTopologyError, axis_boundaries_by_name, validate_plan_topology

# Adapt RunSpec's positional axes/boundaries lists into a name-keyed Mapping (T1-02).
# RunSpec.boundaries stays a plain list[AxisBoundary] | None, one entry per axis in
# RunSpec.axes order; axis identity gets attached here, at the executor entry.
boundaries_by_name = axis_boundaries_by_name(run_spec.axes, run_spec.boundaries)

validate_plan_topology(plan.decisions, boundaries_by_name)  # None, or raises PlanTopologyError
```

Verify: `src/xtrax/stages/topology.py`

Rules enforced (first violation raises `PlanTopologyError`):
1. 🚫 HALTS: `Scan` strategy on a heterogeneous axis — `jax.lax.scan` requires static carry shape.
2. 🚫 HALTS: `ordered=True` Tap or Sink on a `Vmap` axis — vmap does not preserve step order.

`axis_boundaries_by_name` itself raises `PlanTopologyError` on a length mismatch between `axes` and `boundaries`, and on duplicate axis names (a plain dict would silently keep the last entry, masking a keying bug).

⚠ NOTE: the validator is **structural/duck-typed** — it matches by `type(strategy).__name__`, not `isinstance` against xtrax's own classes, so it works on any library's plan objects with matching field names (e.g. a parallel `aminx.tiling` BatchPlanner whose strategy instances are distinct classes).

#### Boundary Executor: execute_map_axis / execute_scan_axis (Unreleased, T1-04)

The first code that actually runs `AxisBoundary` ops. Two-tier contract: ordered `Tap`/`Sink` fire **inside** the per-axis iterator body (per step), because a run-layer wrapper only ever sees the fully stacked output and physically cannot deliver per-step order; `Fuse` fires once, **after** the axis's iteration completes, over the assembled stacked output — never per-step, and never over a `Scan`'s carry.

```python
from xtrax.stages import ExecutorError, execute_map_axis, execute_scan_axis
from xtrax.tiling import SafeMap, Vmap

# Vmap/SafeMap axis: tap/sink per step, fuse once over stacked ys
ys = execute_map_axis(fn, xs, strategy=SafeMap(batch_size=32), boundary=boundary)

# Scan axis: fn is (carry, x) -> (carry, y); fuse never receives final_carry
final_carry, ys = execute_scan_axis(transition, init, xs, boundary=boundary)
```

Verify: `src/xtrax/stages/executor.py`

🚫 HALTS: `execute_map_axis` with `Vmap` + an ordered tap/sink raises `ExecutorError` — JAX cannot lower this at all (`ValueError: Cannot vmap ordered IO callback`). Defense-in-depth behind `validate_plan_topology`.

⚠ WARN: **`SafeMap` + `ordered=True` silently ignores `batch_size` and runs one element at a time — unconditionally.** `jax.lax.map(..., batch_size=B)` batches via `jax.vmap` internally for ANY B >= 1 (verified empirically — even `batch_size=1` raises the vmap-ordered error); only the no-`batch_size` pure-scan path tolerates `ordered=True`. There is no partially-batched middle ground: an ordered `SafeMap` axis is architecturally identical in cost to a sequential `Scan`. If you need both ordering AND real batching throughput, there isn't one today — consider `Scan`, which makes the sequential cost explicit.

⚠ WARN: `ordered=True` is a real, structural cost, not a default knob: it threads an XLA token as a genuine data dependency between consecutive ordered calls (JEP-10657), so XLA cannot reorder, overlap, or pipeline them with other work. Only set it when correctness genuinely depends on host-observed order; keep `ordered=False` on every other axis's boundary ops.

⚠ NOTE: nested composition (e.g. vmap-of-scan) composes naturally by passing one executor call as the `fn` of an enclosing axis, but ordering preservation under nesting is **not certified** — that is T1-05's stress-harness job, not this module's.

#### Fuse, Tap, Sink Protocols

**Fuse[S, O]** — Pure axis reducer (JAX-traced, inside jit).  
Signature: `Stacked[S] -> Out[O]`  
Example: Average stacked embeddings into a single representation.

**Tap[T]** — Identity + side effect (outside jit, host-side).  
Signature: `T -> T` (passthrough)  
Fields: `ordered: bool` (require step order?)  
Example: Log intermediate tensors to disk.

**Sink[T]** — Terminal side effect (outside jit, host-side).  
Signature: `T -> None` (consumes value, leaves pipeline)  
Fields: `ordered: bool` (require step order?)  
Example: Write final results to H5.

Verify: `src/xtrax/stages/boundaries.py:32-82`

⚠ NOTE: `Tap.ordered` / `Sink.ordered` have a real performance cost, not just a correctness constraint — see the Boundary Executor section above (XLA token dependency, no vmap compatibility, `SafeMap` batch_size silently ignored when ordered).

#### StageBundle: Typed Bag of Optional Callable Stage Slots

`StageBundle` is a pure container (`eqx.Module`, no `__call__`) — **subclass it** and declare fields; every field must be `Optional[Callable]`-shaped:

```python
from collections.abc import Callable
from xtrax.stages.bundle import StageBundle

class MyStages(StageBundle):
    preprocess: Callable | None = None
    encode: Callable | None = None
    postprocess: Callable | None = None

stages = MyStages(preprocess=my_pre, encode=my_encode)

stages.active_stages()      # ["preprocess", "encode"] — field names with non-None callables
stages.has_stage("encode")  # True
```

Verify: `src/xtrax/stages/bundle.py`

🚫 HALTS: `__init_subclass__` validates at class-definition time — every annotated field must be `Optional[Callable]` (bare `Callable`, `Callable[...]`, or a `typing.Protocol` whose only member is `__call__`); unions may have any arity but exactly one `None` member (e.g. `Callable | SomeProtocol | None` is fine). Untyped class attributes are rejected. Violations raise `TypeError`.

⚠ NOTE (0.4.0a2+): the validator resolves PEP 563 postponed annotations via `typing.get_type_hints` — modules with `from __future__ import annotations` work correctly. A field typed against a name not resolvable at module scope raises a deliberate, loud `TypeError` (naming the unresolved annotation), never a silent misclassification.

⚠ WARN: `active_stages()`/`has_stage()` are **Python-side only** — do NOT call them inside JAX traces. Topology is determined by non-None fields at Python dispatch level.

---
name: using-xtrax
description: "Use when writing a padding or bucketing loop for variable-length inputs, a chunked vmap or lax.map over a large batch, a memory-budgeted batch plan, an ONNX or StableHLO export, a resumable training loop, or zarr output sinks. Also when a TOML file should drive train or resume, when batch axes are inferred from a typed function, or when a hand-written Pallas kernel should sit beside a plain-JAX reference. The library is xtrax."
when_to_use: 'Triggers: padding or bucketing loop for variable-length sequences before a JIT step, chunked vmap or lax.map over a large batch, memory-budgeted batching / batch plan that must fit a device budget, ONNX or StableHLO export of a JAX pipeline, resumable training loop / resume training from a checkpoint, zarr output sink / staged zarr writes, TOML file that should launch or resume a training run, infer batch axes from a typed function signature, compose batching primitives into your own CLI, hand-written Pallas kernel beside a plain-JAX reference / replace a use_pallas flag or interpret= switch with xtrax.op.'
---
# using-xtrax

## TIER-1: Read First (Self-Contained)

### Pre-Flight: Compatibility Assertion

Before writing any xtrax code, verify your installation:

```python
import warnings
from pathlib import Path

import xtrax

# This skill was written against the xtrax alpha named in the plugin manifest
# (.praxia/manifest.toml [plugin] version). Warn when the installed package
# differs, then read src/xtrax/__init__.py.
_WRITTEN_AGAINST = "0.4.1b5"
if xtrax.__version__ != _WRITTEN_AGAINST:
    warnings.warn(
        f"using-xtrax was written against xtrax {_WRITTEN_AGAINST}; "
        f"installed xtrax is {xtrax.__version__}. "
        "Read src/xtrax/__init__.py and the sections you are about to copy.",
        stacklevel=2,
    )
```

If you see a version mismatch, verify current behavior directly in the source tree before proceeding with any code example in this skill.

Also verify extras are installed for the optional layers you plan to use:

```bash
# For plan visualization (explain_plan output + render)
pip install xtrax[eda]

# For Zarr-backed output sinks + content digests (ZarrStagingSink, zarr_content_digest)
pip install xtrax[io]

# For the `xtrax` CLI verbs (tyro); for ahead-of-time export (IREE)
pip install xtrax[cli]
pip install xtrax[export]
```

Dependency floor: read the `jax`/`jaxlib` specifiers in `pyproject.toml`'s `dependencies` rather than trusting a number quoted here -- this line said `<0.11` for the whole period the pin was already `<0.12`. The io_callback shim (`xtrax.stages._callback`, shipped 0.4.0a6) pins this same range (`PINNED_JAX_RANGE`, `src/xtrax/stages/_callback.py:33`) and fails loud at import time if the resolved jax drifts outside it.

xtrax does not depend on Equinox: no module under `src/xtrax` imports it, and it is a dev-only requirement of xtrax's own tests. Equinox models still work as plain pytrees (`Trainer`, `Engine`, tiling and export accept them and return the same `eqx.Module` type); install `equinox` yourself if you use it.

---

### JAX Discipline for Domain Library Authors

When building domain libraries on xtrax (custom `RunSpec`, `InputResolver`, `StageBundle`, `AxisBoundary`), three cross-cutting invariants must be preserved:

#### 1. Static vs. Dynamic Fields in `xtrax.Struct`

xtrax's own pytree classes subclass `xtrax.Struct`, a frozen-dataclass pytree base (verify: `src/xtrax/_tree.py:70`; subclasses need no decorator). `AxisBoundary` (and any custom `Struct` you create) separates fields into:
- **Static fields** (`xtrax.field(static=True)`): callables, Python values, never traced
- **Dynamic fields**: JAX arrays, traced at jit time

Equinox is not required for any of this. Equinox models are plain pytrees to xtrax, and a downstream subclass may declare its static fields with `eqx.field(static=True)` or `xtrax.field(static=True)` (verify: `src/xtrax/_tree.py:31`).

Example (verify: `src/xtrax/stages/boundaries.py:84-99` — this skill is a map, not the territory):
```python
class AxisBoundary(Struct):
    fuse: Fuse | BoundaryCallable | None = field(static=True, default=None)  # verify: src/xtrax/stages/boundaries.py:96
    tap: Tap | BoundaryCallable | None = field(static=True, default=None)    # verify: src/xtrax/stages/boundaries.py:97
    sink: Sink | BoundaryCallable | None = field(static=True, default=None)  # verify: src/xtrax/stages/boundaries.py:98
    materialize: bool = field(static=True, default=False)  # verify: boundaries.py:99 -- export-only: declared-materializing sink is stripped
    sink_receives_index: bool = field(static=True, default=False)  # verify: boundaries.py:131 -- opt-in (y, index)
    # No dynamic leaves; tree_flatten returns empty leaves
```

Check your custom modules with:
```python
import jax.tree_util  # verify: src/xtrax/stages/boundaries.py:84-99 (AxisBoundary implementation)
leaves = jax.tree_util.tree_leaves(my_boundary)
assert len(leaves) == 0, "AxisBoundary must have no dynamic leaves"
```

#### 2. PyTree Invariant for AxisBoundary

`AxisBoundary` must flatten to **zero JAX leaves** — it is a static-only structure:

```python
boundary = AxisBoundary(fuse=my_fuse_fn, tap=None, sink=None)
leaves = jax.tree_util.tree_flatten(boundary)[0]
assert leaves == [], "Expected no dynamic leaves in AxisBoundary"
```

This invariant ensures JIT does not retrace when `AxisBoundary` instances change — the static values live in the treedef, which is the compile cache key.

#### 3. JIT Boundary Rules

Three distinct regions exist:

- **Outside jit**: host-side bucketing (`select_bucket`, `bucketize`) runs here, before the
  jitted step. (verify: `src/xtrax/tiling/bucket.py`)

- **Inside jit**: `Fuse` functions (pure JAX axis reducers) run inside the trace.
  ```python
  # Fuse is a pure JAX function: Stacked[S] -> Out[O]
  # Example: fuse stacked embeddings into a single representation
  ```

- **Boundary crossings**: `Tap` and `Sink` use `io_callback` (host-side Python):
  ```python
  # Tap and Sink implementations own their io_callback call.
  # Import it from the vendored shim, never from jax.experimental directly
  # (it IS jax.experimental.io_callback, re-exported after the drift checks):
  from xtrax.stages._callback import io_callback  # verify: src/xtrax/stages/_callback.py
  # The shim pins jax's still-experimental io_callback: version-range and
  # signature checks run at MODULE IMPORT time and raise IoCallbackSignatureError
  # on drift, so an upstream jax move is a loud one-file fix, not a runtime traceback.
  ```
  Cost discipline for these crossings (ordered vs unordered, per-step round-trip
  tax, measured numbers): see the `xtrax-optimizing` skill,
  `references/tier1-host-boundary.md`.

These boundary protocols are the protocol-slot pattern applied to stage boundaries; the general model-structure patterns (variant classes, layer tuples, pure functions, protocol/ABC slots) live in jaxlint's `developing-composable-jax` skill.

Choose JIT decorator based on your model:
- **`xtrax.model_jit`** (preferred): array leaves are traced, every other leaf (callables, Python values, static fields) is a static compile key. Ideal for models with callable static fields (like `AxisBoundary`). It wraps a module-level function and takes the model as an argument (verify: `src/xtrax/_tree.py:237`).
- **`jax.jit`**: All arrays traced, everything else is attempted to be traced (may fail if callables or static values change).

```python
# Preferred: model_jit with static-only callables
@model_jit
def inference_step(model, x):
    # model may have static fields (callables); model_jit keeps them out of the trace
    return model(x)

# Safe for Trainer: Trainer.step runs through xtrax.model_jit (verify: src/xtrax/training/trainer.py:84)
```

---

### Which Primitive for Which Problem

**First answer: `xtrax.intent`.** Write a loop or a map inside a function you jit or differentiate as `xtrax.intent.loop` / `xtrax.intent.map`, and run the function through `xtrax.intent.compile(fn, Context(...))`. The resolver picks a strategy per site from the caller's `Context` (verify: `src/xtrax/intent/_compile.py:1055`, `src/xtrax/intent/_loop.py:367`, `src/xtrax/intent/_map.py:219`):

```
Is the site a loop?
├─ fixed length (xs= or length=) → StaticWhile primal (one while_loop); Scan under AD
│                                   (Scan primal when target="export" and export_format != "stablehlo")
└─ cond= (data-dependent stop)    → WhileCarry (lax.while_loop); forward-mode only; export refuses it
                                    (verify: src/xtrax/intent/_resolve.py:369-386)
Is the site a map over a leading axis?
├─ no memory_budget, or leading size <= 1 → Vmap on both paths
└─ memory_budget set → the Vmap or ChunkedMap that BatchPlanner returns for that axis (tree below)
                        (verify: src/xtrax/intent/_resolve.py:463-481)
   On a CUDA/ROCm backend a ChunkedMap that would enclose a while is moved back to Vmap, or
   refused with UnsatisfiableContextError("scan_around_while") when Vmap does not fit the budget
   (verify: src/xtrax/intent/_resolve.py:921-939)
```

Swap one site's strategy with `compile(..., overrides={label: (primal, diff)})` (`SiteStrategyOverride`, verify: `src/xtrax/intent/_resolve.py:29`). `compile(...).plan_for(*args).table()` shows each site's choice and reason. Full reference: `references/intent.md`.

**A hand-written kernel (Pallas) beside a reference: `xtrax.op`.** `xtrax.op(reference, kernels={"gpu": ..., "cpu": interpret_kernel})` runs the kernel only under `Context(kernels=True)` on a backend it has an entry for, the reference otherwise and under `Context(target="export")` (xtrax's export helpers do not set that target for you); derivatives come from the reference unless a `vjp_kernels` pair is given (verify: `src/xtrax/kernel_seam.py:312-338`). Full reference: `references/op.md`.

What intent buys is one call site that is a `while_loop` when undifferentiated and a `lax.scan` under grad, with no hand-set inference-only flag, exact against both hand-written forms. It does not buy compile time on current JAX: two pre-registered runs on Blackwell (SM120) found the scan and while forms, and `lax.map` around a `while_loop` versus `vmap` around it, compile alike (aminx bathos runs `3b93937a` and `c3c745a5`, both `no_contrast`; criterion 8 ruled met on equivalence, 261025).

**Lower layer: `BatchPlanner`.** Use the tree below directly when you build a `BatchPlan` yourself: bucketed variable-length axes, `DedupGather`, multi-axis plans, and the CLI and export pipelines, none of which intent resolves. `intent.map` plans with `BatchPlanner` when `Context.memory_budget` is set.

**BatchPlanner decision tree** (verify each branch against `src/xtrax/tiling/plan.py:415-529`, `BatchPlanner._decide_strategy`):

```
Is the axis variable-length (e.g., sequences of different sizes)?
├─ YES: bucket_boundaries specified on AxisSpec?
│   ├─ YES → Bucket strategy (length-padding via select_bucket/bucketize)
│   └─ NO → Heterogeneous handling (Tap/Sink + padding outside jit)
│
└─ NO (cardinality fixed): DedupSpec for this axis passed to BatchPlanner(dedup_specs=[...])?
    ├─ YES (repeated elements) → DedupGather strategy (Phase 0b, plan.py:188-)
    │                              (identify K unique items, vmap over K, gather back to N)
    │                              `dedup_eligible=True` ALONE does NOT select it -- it falls
    │                              through to the cardinality rules below (plan.py:433-438)
    │
    └─ NO: Check cardinality vs. default_batch_size:
        ├─ cardinality <= batch_size → Vmap (ChunkedMap if a memory_estimator says it won't fit)
        │
        └─ cardinality > batch_size → ChunkedMap (Vmap if a memory_estimator says the
                                         whole axis fits). Divisibility does not matter:
                                         a ragged final chunk runs as one smaller vmap,
                                         with no padding (after 0.4.0a10, #5565)
```

**Key decision rule** (verify: `src/xtrax/tiling/plan.py:415-529`):

1. **Bucket** — if `bucket_boundaries` is set (variable-length handling)
2. **DedupGather** — only via an explicit `DedupSpec` (Phase 0b); `dedup_eligible=True` alone falls through
3. **Vmap** — if `cardinality <= batch_size` (small, fully-parallel)
4. **ChunkedMap** — if `cardinality > batch_size` (large, chunked; memory-safe). A `cardinality` that is not a multiple of `batch_size` is fine: the final chunk is smaller (through 0.4.0a10 this raised `ValueError` at dispatch)

**Joint-budget mode** (0.4.0a1+): when `BatchPlanner(budget=MemoryBudget(...))` is set, rules 3-4 are replaced for non-bucket axes — every eligible axis starts at `Vmap`, then axes are greedily demoted to `ChunkedMap` in spec order until the whole-plan estimate fits the budget. See TIER-2: Tiling Layer → Joint-Budget Planning.


---

### Minimal Working Pattern (Fully Self-Contained)

This pattern works **without any tier-2 imports or symbols**:

```python
import jax
from xtrax.tiling.plan import AxisSpec, BatchPlanner, BatchPlan  # verify: src/xtrax/tiling/plan.py:31-529
from xtrax.tiling.dispatch import make_axis_dispatch  # verify: src/xtrax/tiling/dispatch.py:31-118

# Step 1: Define axis specification
axis_spec = AxisSpec(
    name="batch",
    cardinality=96,            # 96 samples; any count works -- a count that is not a
    default_batch_size=32,     # multiple of the batch size just gets a smaller last chunk
)

# Step 2: Build batching plan
planner = BatchPlanner()
plan: BatchPlan = planner.plan([axis_spec])

# Step 3: Extract decision for the axis
decision = plan.decisions[0]
print(f"Strategy: {type(decision.strategy).__name__}")
print(f"Reasoning: {decision.reasoning}")

# Step 4: Create dispatch iterator
iterator = make_axis_dispatch(
    decision.strategy,
    axis="batch",
    heterogeneous_axes=set(),
)

# Step 5: Apply iterator to a function
def my_fn(x):
    """Process a single sample."""
    return x * 2

samples = jax.numpy.ones((96, 10))  # (batch, features)
results = iterator(my_fn, samples)   # (batch, features) → apply my_fn to each

print(f"Output shape: {results.shape}")  # (96, 10)
```

**What happened:**
- `AxisSpec` declared the axis (name, size, batch threshold)
- `BatchPlanner.plan()` selected the best strategy (Vmap, ChunkedMap, etc.)
- `make_axis_dispatch()` returned a typed iterator matching the strategy
- Iterator applied `my_fn` to the axis, returning results

This is the core loop. Extend it by:
- Adding more axes to `plan([spec1, spec2, ...])` → multi-axis iteration
- Wrapping results in `AxisBoundary` for post-processing (Fuse/Tap/Sink)
- Using `Scan` strategy via `CarrySpec` for stateful iteration

---

### Workflow Index

Choose your task:

1. **Build a custom domain library** (RunSpec, InputResolver, StageBundle)  
   → Read `references/run.md`

2. **Run tiled inference without recompilation**  
   → Read `references/tiling.md` + `references/run.md`

3. **Implement a training loop**  
   → Read `references/training.md`

4. **Analyze a batching plan before committing**  
   → Read `references/eda.md`

5. **Set up multi-device training or checkpointing**  
   → Read `references/distributed-checkpoint.md`

6. **Run training from TOML or inspect tiling via CLI**  
   → Read `references/cli.md`

7. **Infer AxisSpecs/BundleSchema from a typed function signature**  
   → Read `references/inference.md`

8. **Pad variable-length inputs to one compile per bucket rung**  
   → Read `references/length-bucketing.md`

9. **Replace local copies in an existing domain library**  
   → Read `references/adoption.md`

10. **Export one callable to ONNX (no BatchPlan), or read parity and divergence rings**  
   → Read `references/onnx-standalone.md`

11. **Compare a sampler to a reference implementation**  
   → Read `references/parity.md`

12. **Write a loop or map that picks while_loop/scan or vmap/chunked per call site (xtrax.intent)**  
   → Read `references/intent.md`

13. **Put a Pallas (or other backend) kernel beside a plain-JAX reference, selected by Context instead of a `use_pallas` flag or env var (xtrax.op)**  
   → Read `references/op.md`

---

## TIER-2: Deep Reference

TIER-2 content lives in `references/` — one file per layer, loaded on demand via `Read` (not auto-loaded with this skill). Each file is self-contained for its layer; cross-layer notes point back here or to a sibling file by name.

| Layer | File | Depth | Covers |
|---|---|---|---|
| Tiling | `references/tiling.md` | 40% | AxisSpec, BatchPlanner, Strategies, Dispatch, Iterators, Carry, Dedup, Bucket, Multi-Axis Composition |
| Run | `references/run.md` | 20% | RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec/make_sink, ZarrStagingSink, zarr_integrity, AxisBoundary, Fuse/Tap/Sink, topology validation, boundary executor |
| Training | `references/training.md` | 25% | ResumableState, Trainer, checkify safety, Engine, Callbacks, Optax |
| CLI | `references/cli.md` | E2/E3 | Tyro-delegated verbs: plan/explain/export/run/resume/sweep + graph-validate/graph-plan/graph-author + ledger (`src/xtrax/cli/registry.py:42-53`) |
| EDA | `references/eda.md` | 10% | Plan analysis and visualization |
| Distributed/Checkpoint | `references/distributed-checkpoint.md` | 5% | Pointer pattern for multi-device training, checkpointing |
| Signature Inference | `references/inference.md` | — | xtrax.inference: derive AxisSpecs + BundleSchema from a typed function |
| Export (AOT) | `references/export.md` | — | xtrax.export: export_pipeline, Target/VerificationLevel, native + wasm32 + SPIR-V codegen, dtype envelope, load_hf_weights, materialize stripping, multi-axis composition |
| Length bucketing | `references/length-bucketing.md` | — | AxisSpec.bucket_boundaries, Bucket, host select_bucket/bucketize, BUCKET_LADDER, one compile per rung |
| Adoption | `references/adoption.md` | — | local-copy replacement, SafeMap alias removal, limits, telemetry fail-closed, WhileCarry, ChunkedMap/while_loop, duplicated primitives |
| Standalone ONNX | `references/onnx-standalone.md` | — | convert_to_onnx vs jax2onnx, find_onnx_rng_ops, verify_native_parity, rings, divergence |
| Intent | `references/intent.md` | — | xtrax.intent: Context, compile, loop, map, Plan, overrides, hazards, errors |
| Kernel seam | `references/op.md` | — | xtrax.op: reference + per-backend kernels, selection rules, static args by closure, custom_jvp/custom_vjp derivatives, kernel_parity tests, interpret-mode GPU hazards |

Use the Workflow Index above to pick which file(s) a given task needs — most tasks need one, some (e.g. tiled inference) need two. Don't load a reference file speculatively; load it when the task actually reaches that layer.

---

## Summary

This skill provides a complete, self-contained reference for the xtrax alpha it was written against (the plugin version in `.praxia/manifest.toml`), plus the CHANGELOG `[Unreleased]` changes on main.

**Use TIER-1 to**:
- Verify compatibility (pre-flight)
- Learn JAX discipline for domain library authors
- Understand which primitive solves which problem
- Build your first axis-tiling loop
- Find the right TIER-2 section for your task

**Use TIER-2 to**:
- Deep-dive into one component (tiling, training, EDA, etc.)
- Find enforcement-backed callouts (🚫 HALTS / ⚠ WARN)
- Identify human-in-the-loop investigation stops (🔬 HiTL)
- Locate source code verification points (`verify: src/...:line`)

**All code examples cite their source**: The skill is a map, not the territory. Read the live source at the referenced file:line when in doubt.

---

## Technical Gaps (Known Limitations)

| Gap | Location | Status |
|-----|----------|--------|
| DedupGather large-k regime (k > 256) uses suboptimal power-of-2 bucketing | `src/xtrax/tiling/dedup.py:29` | TODO: implement geometric or mixed bucketing for k > 256 |
| Top-level exports missing (RunSpec, CarrySpec, DedupSpec, AxisBoundary) | `src/xtrax/__init__.py` | By design; use subpackage imports: `from xtrax.run import RunSpec`, `from xtrax.stages import AxisBoundary`, etc. |
| `make_sink` has no writer for `"jsonl"`/`"h5"` | `src/xtrax/run/sink.py:41-58` | Routing-only stub values; `NotImplementedError` until their writers land. Use `"zarr"` (or `"none"`). |
| Ordered `ChunkedMap` axis ignores `batch_size` (runs element-at-a-time) | `src/xtrax/stages/executor.py` | Structural JAX constraint, not fixable locally — see Boundary Executor section; use `Scan` if ordering + explicit sequential cost is acceptable |
The `make_inference_plan` gap noted as of v0.3.0 is closed: plan-time checks now exist via `validate_plan_topology` (`xtrax.stages`, 0.3.1+).

Nested executor composition (vmap-of-scan) ordering is also no longer a gap. The T1-05 stress harness landed and certifies `(lane, step)` call order at `N_TRIALS=20` in `tests/stages/test_nested_ordering.py`, and `xtrax.export`'s composer builds multi-axis plans on that certified recipe (`tests/export/test_multi_axis.py`). The composer still refuses `Bucket` (host-tier) and `WhileCarry` (unbounded trip count) — those are genuine remaining limits, not this one.

---

## For More Information

- **JAX discipline**: See TIER-1 section "JAX Discipline for Domain Library Authors"
- **Decision tree**: See TIER-1 section "Which Primitive for Which Problem"
- **Minimal working example**: See TIER-1 section "Minimal Working Pattern"
- **Live source code**: All code examples cite `verify: src/...:<line>`; read the source at those locations for current behavior

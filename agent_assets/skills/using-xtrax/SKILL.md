---
name: using-xtrax
description: Use when writing JAX pipelines with xtrax, building domain libraries on top of xtrax, running `xtrax run` from TOML (`TrainConfig`), loading your own TOML config via the domain-agnostic `xtrax.config` primitives, composing xtrax's own CLI verbs (`REGISTRY`) into your own CLI, or analyzing batching plans via CLI/EDA (`xtrax plan`/`explain`). Covers: AxisSpec/BatchPlanner/BatchPlan incl. joint-budget planning (MemoryBudget), composition (Fuse/Tap/Sink/AxisBoundary), plan topology validation + the two-tier boundary executor (xtrax.stages), the run layer (RunSpec/InputResolver/StageBundle/SinkSpec/ZarrStagingSink/zarr_integrity), training (Trainer/Engine/ResumableState/init_state), CLI verbs (plan/explain/export/run/resume/sweep + unreleased graph-validate/graph-plan/graph-author), the xtrax.config TOML primitives, EDA, sparsification, the signature-inference layer (xtrax.inference), and ahead-of-time export via the xtrax.export subpackage (export_pipeline/Target/VerificationLevel/materialize/load_hf_weights, native + wasm32 + SPIR-V codegen).
xtrax_version: 0.4.0a8
triggers:
  - writing JAX pipeline with xtrax
  - building domain library on xtrax
  - AxisSpec / BatchPlanner / BatchPlan / AxisBoundary
  - Fuse / Tap / Sink / RunSpec / CarrySpec
  - MemoryBudget / BudgetInfeasibleError / device_memory_budget / lowered_memory_estimate
  - validate_plan_topology / PlanTopologyError / execute_map_axis / execute_scan_axis
  - SinkSpec / make_sink / ZarrStagingSink / zarr_content_digest / fsync_tree
  - explain_plan / render / EDA
  - sparsify_model / SparsePolicy
  - infer_bundle / BundleSchema / AxisOverride / axis_config
  - signature inference / xtrax.inference / AxisRole / AmbiguousAxisError
  - xtrax run / xtrax plan / xtrax explain / xtrax export / xtrax resume / xtrax sweep
  - xtrax.export / export_pipeline / Target / VerificationLevel / CODEGEN_ONLY
  - IREE / vmfb / wasm32 / SPIR-V / ahead-of-time export / load_hf_weights
  - TrainConfig / load_config / ConfigError / init_state
  - load_fn / CLIError / CLIImportError / REGISTRY (stable public xtrax.cli primitives)
  - xtrax graph-validate / xtrax graph-plan / xtrax graph-author (unreleased, main-only)
  - GraphValidateArgs / GraphPlanArgs / GraphAuthorArgs / validate_graph / TemplateGenerator
  - xtrax.config / load_toml_document / require_sections / require_field
  - check_schema_version / classify_schema_version / SchemaVersionStatus
  - REGISTRY composition / building your own CLI on xtrax's verbs
---

# using-xtrax

## TIER-1: Read First (Self-Contained)

### Pre-Flight: Compatibility Assertion

Before writing any xtrax code, verify your installation:

```python
import xtrax

# Version check — the alpha this skill was written against is the `xtrax_version` in
# its own frontmatter (gated against __version__ by audit-project-hygiene, so it does
# not drift). The 0.4.0 alpha line moves fast; any 0.4.0aN is close enough, but
# re-verify sections touched by later alphas.
assert xtrax.__version__.startswith("0.4.0"), f"Expected xtrax 0.4.0aN, got {xtrax.__version__}"

# Verify in live source: read src/xtrax/__init__.py:1 to confirm __version__ definition
# This assertion is blind to forks maintaining the same version string without a code bump
```

If you see a version mismatch, verify current behavior directly in the source tree before proceeding with any code example in this skill.

Also verify extras are installed for the optional layers you plan to use:

```bash
# For plan visualization (explain_plan output + render)
pip install xtrax[eda]

# For Zarr-backed output sinks + content digests (ZarrStagingSink, zarr_content_digest)
pip install xtrax[io]
```

Dependency floor: read the `jax`/`jaxlib` specifiers in `pyproject.toml`'s `dependencies` rather than trusting a number quoted here -- this line said `<0.11` for the whole period the pin was already `<0.12`. The io_callback shim (`xtrax.stages._callback` — unreleased `main` only, T1-03; not in the 0.4.0a5 wheel) pins this same range and fails loud at import time if the resolved jax drifts outside it.

---

### JAX Discipline for Domain Library Authors

When building domain libraries on xtrax (custom `RunSpec`, `InputResolver`, `StageBundle`, `AxisBoundary`), three cross-cutting invariants must be preserved:

#### 1. Static vs. Dynamic Fields in `eqx.Module`

`AxisBoundary` (and any custom `eqx.Module` you create) separates fields into:
- **Static fields** (`eqx.field(static=True)`): callables, Python values, never traced
- **Dynamic fields**: JAX arrays, traced at jit time

Example (verify: `src/xtrax/stages/boundaries.py:84-98` — this skill is a map, not the territory):
```python
class AxisBoundary(eqx.Module):
    fuse: Fuse | BoundaryCallable | None = eqx.field(static=True, default=None)  # verify: src/xtrax/stages/boundaries.py:96
    tap: Tap | BoundaryCallable | None = eqx.field(static=True, default=None)    # verify: src/xtrax/stages/boundaries.py:97
    sink: Sink | BoundaryCallable | None = eqx.field(static=True, default=None)  # verify: src/xtrax/stages/boundaries.py:98
    # No dynamic leaves; tree_flatten returns empty leaves
```

Check your custom modules with:
```python
import jax.tree_util  # verify: src/xtrax/stages/boundaries.py:84-98 (AxisBoundary implementation)
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

This invariant ensures JIT does not retrace when `AxisBoundary` instances change — the structure is cached by Equinox.

#### 3. JIT Boundary Rules

Three distinct regions exist:

- **Outside jit**: `sparsify_model(model, policy)` MUST run here. (verify: `src/xtrax/sparse/inference.py:44`)
  ```python
  🚫 HALTS RuntimeError if sparsify_model is called inside jax.jit
  # Enforcement at src/xtrax/sparse/inference.py:44-55 (assert_not_tracing)
  ```

- **Inside jit**: `Fuse` functions (pure JAX axis reducers) run inside the trace.
  ```python
  # Fuse is a pure JAX function: Stacked[S] -> Out[O]
  # Example: fuse stacked embeddings into a single representation
  ```

- **Boundary crossings**: `Tap` and `Sink` use `io_callback` (host-side Python):
  ```python
  # Tap and Sink implementations own their io_callback call.
  # Import it from the vendored shim, never from jax.experimental directly
  # (shim is unreleased main only, T1-03 — on the 0.4.0a5 wheel fall back to
  # jax.experimental.io_callback):
  from xtrax.stages._callback import io_callback  # verify: src/xtrax/stages/_callback.py
  # The shim pins jax's still-experimental io_callback: version-range and
  # signature checks run at MODULE IMPORT time and raise IoCallbackSignatureError
  # on drift, so an upstream jax move is a loud one-file fix, not a runtime traceback.
  ```
  Cost discipline for these crossings (ordered vs unordered, per-step round-trip
  tax, measured numbers): see the `xtrax-optimizing` skill,
  `references/tier1-host-boundary.md`.

Choose JIT decorator based on your model:
- **`eqx.filter_jit`** (preferred): JAX arrays are traced, static fields are held constant. Ideal for models with callable static fields (like `AxisBoundary`).
- **`jax.jit`**: All arrays traced, everything else is attempted to be traced (may fail if callables or static values change).

```python
# Preferred: filter_jit with static-only callables
@eqx.filter_jit
def inference_step(model: eqx.Module, x):
    # model may have static fields (callables); filter_jit handles correctly
    return model(x)

# Safe for Trainer: Trainer.step is @eqx.filter_jit (verify: src/xtrax/training/trainer.py:31)
```

---

### Which Primitive for Which Problem

**Decision tree** (verify each branch against `src/xtrax/tiling/plan.py:123-150` BatchPlanner rules):

```
Is the axis variable-length (e.g., sequences of different sizes)?
├─ YES: bucket_boundaries specified on AxisSpec?
│   ├─ YES → Bucket strategy (length-padding via select_bucket/bucketize)
│   └─ NO → Heterogeneous handling (Tap/Sink + padding outside jit)
│
└─ NO (cardinality fixed): dedup_eligible=True?
    ├─ YES (repeated elements) → DedupGather strategy
    │                              (Phase 0: identify unique items, Phase 1: vmap over K unique,
    │                               Phase 2: scatter results back to original N positions)
    │
    └─ NO (all distinct): Check cardinality vs. default_batch_size:
        ├─ cardinality <= batch_size → Vmap (fully parallel vectorization)
        │
        ├─ cardinality > batch_size AND divisible → SafeMap (chunked vmap, memory-bounded)
        │
        └─ cardinality > batch_size AND NOT divisible → SafeMap + deferred warning
                                                        (last chunk is smaller; OK if handled)
```

**Key decision rule** (verify: `src/xtrax/tiling/plan.py:121-141`):

1. **Bucket** — if `bucket_boundaries` is set (variable-length handling)
2. **DedupGather** — if `dedup_eligible=True` (repeated elements)
3. **Vmap** — if `cardinality <= batch_size` (small, fully-parallel)
4. **SafeMap** — if `cardinality > batch_size` (large, chunked; memory-safe)

**Joint-budget mode** (0.4.0a1+): when `BatchPlanner(budget=MemoryBudget(...))` is set, rules 3-4 are replaced for non-bucket axes — every eligible axis starts at `Vmap`, then axes are greedily demoted to `SafeMap` in spec order until the whole-plan estimate fits the budget. See TIER-2: Tiling Layer → Joint-Budget Planning.

---

### Minimal Working Pattern (Fully Self-Contained)

This pattern works **without any tier-2 imports or symbols**:

```python
import jax
from xtrax.tiling.plan import AxisSpec, BatchPlanner, BatchPlan  # verify: src/xtrax/tiling/plan.py:26-120
from xtrax.tiling.dispatch import make_axis_dispatch  # verify: src/xtrax/tiling/dispatch.py:31-102

# Step 1: Define axis specification
axis_spec = AxisSpec(
    name="batch",
    cardinality=100,           # 100 samples
    default_batch_size=32,     # chunk size for SafeMap
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

samples = jax.numpy.ones((100, 10))  # (batch, features)
results = iterator(my_fn, samples)   # (batch, features) → apply my_fn to each

print(f"Output shape: {results.shape}")  # (100, 10)
```

**What happened:**
- `AxisSpec` declared the axis (name, size, batch threshold)
- `BatchPlanner.plan()` selected the best strategy (Vmap, SafeMap, etc.)
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

5. **Apply sparsification (structured pruning at inference)**  
   → Read `references/sparse-distributed.md`

6. **Run training from TOML or inspect tiling via CLI**  
   → Read `references/cli.md`

7. **Infer AxisSpecs/BundleSchema from a typed function signature**  
   → Read `references/inference.md`

---

## TIER-2: Deep Reference

TIER-2 content lives in `references/` — one file per layer, loaded on demand via `Read` (not auto-loaded with this skill). Each file is self-contained for its layer; cross-layer notes point back here or to a sibling file by name.

| Layer | File | Depth | Covers |
|---|---|---|---|
| Tiling | `references/tiling.md` | 40% | AxisSpec, BatchPlanner, Strategies, Dispatch, Iterators, Carry, Dedup, Bucket, Multi-Axis Composition |
| Run | `references/run.md` | 20% | RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec/make_sink, ZarrStagingSink, zarr_integrity, AxisBoundary, Fuse/Tap/Sink, topology validation, boundary executor |
| Training | `references/training.md` | 25% | ResumableState, Trainer, SafetyTrainStep, Engine, Callbacks, Optax |
| CLI | `references/cli.md` | E2/E3 | Tyro-delegated verbs: plan/explain/export/run/resume/sweep + graph-validate/graph-plan/graph-author |
| EDA | `references/eda.md` | 10% | Plan analysis and visualization |
| Sparse/Distributed/Checkpoint | `references/sparse-distributed.md` | 5% | Pointer pattern for structured pruning, multi-device training, checkpointing |
| Signature Inference | `references/inference.md` | — | xtrax.inference: derive AxisSpecs + BundleSchema from a typed function |
| Export (AOT) | `references/export.md` | — | xtrax.export: export_pipeline, Target/VerificationLevel, native + wasm32 + SPIR-V codegen, dtype envelope, load_hf_weights, materialize stripping, multi-axis composition |

Use the Workflow Index above to pick which file(s) a given task needs — most tasks need one, some (e.g. tiled inference) need two. Don't load a reference file speculatively; load it when the task actually reaches that layer.

---

## Summary

This skill provides a complete, self-contained reference for the xtrax alpha named in its frontmatter `xtrax_version`, plus unreleased `main`.

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
| `make_sink` has no writer for `"jsonl"`/`"h5"` | `src/xtrax/run/sink.py:32-39` | Routing-only stub values; `NotImplementedError` until their writers land. Use `"zarr"` (or `"none"`). |
| Ordered `SafeMap` axis ignores `batch_size` (runs element-at-a-time) | `src/xtrax/stages/executor.py` | Structural JAX constraint, not fixable locally — see Boundary Executor section; use `Scan` if ordering + explicit sequential cost is acceptable |
The `make_inference_plan` gap noted as of v0.3.0 is closed: plan-time checks now exist via `validate_plan_topology` (`xtrax.stages`, 0.3.1+).

Nested executor composition (vmap-of-scan) ordering is also no longer a gap. The T1-05 stress harness landed and certifies `(lane, step)` call order at `N_TRIALS=20` in `tests/stages/test_nested_ordering.py`, and `xtrax.export`'s composer builds multi-axis plans on that certified recipe (`tests/export/test_multi_axis.py`). The composer still refuses `Bucket` (host-tier) and `WhileCarry` (unbounded trip count) — those are genuine remaining limits, not this one.

---

## For More Information

- **JAX discipline**: See TIER-1 section "JAX Discipline for Domain Library Authors"
- **Decision tree**: See TIER-1 section "Which Primitive for Which Problem"
- **Minimal working example**: See TIER-1 section "Minimal Working Pattern"
- **Live source code**: All code examples cite `verify: src/...:<line>`; read the source at those locations for current behavior

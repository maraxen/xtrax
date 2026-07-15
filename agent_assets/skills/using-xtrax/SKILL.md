---
name: using-xtrax
description: Use when writing JAX pipelines with xtrax, building domain libraries on top of xtrax, running `xtrax run` from TOML (`TrainConfig`), loading your own TOML config via the domain-agnostic `xtrax.config` primitives, composing xtrax's own CLI verbs (`REGISTRY`) into your own CLI, or analyzing batching plans via CLI/EDA (`xtrax plan`/`explain`). Covers: AxisSpec/BatchPlanner/BatchPlan incl. joint-budget planning (MemoryBudget), composition (Fuse/Tap/Sink/AxisBoundary), plan topology validation + the two-tier boundary executor (xtrax.stages), the run layer (RunSpec/InputResolver/StageBundle/SinkSpec/ZarrStagingSink/zarr_integrity), training (Trainer/Engine/ResumableState/init_state), CLI verbs (plan/explain/export/run/resume/sweep + unreleased graph-validate/graph-plan/graph-author), the xtrax.config TOML primitives, EDA, sparsification, and the signature-inference layer (xtrax.inference). xtrax v0.4.0a5 + main.
xtrax_version: 0.4.0a5
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

# Version check — this skill is written for v0.4.0a5 (0.4.0 alpha line moves fast;
# any 0.4.0aN is close enough, but re-verify sections touched by later alphas)
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

Dependency floor: `jax>=0.10.2,<0.11` / `jaxlib>=0.10.2,<0.11` (verify: `pyproject.toml:7`). The io_callback shim (`xtrax.stages._callback` — unreleased `main` only, T1-03; not in the 0.4.0a5 wheel) pins this same range and fails loud at import time if the resolved jax drifts outside it.

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
   → Read TIER-2: Run Layer (20%)

2. **Run tiled inference without recompilation**  
   → Read TIER-2: Tiling Layer (40%) + Run Layer (20%)

3. **Implement a training loop**  
   → Read TIER-2: Training Layer (25%)

4. **Analyze a batching plan before committing**  
   → Read TIER-2: EDA (10%)

5. **Apply sparsification (structured pruning at inference)**  
   → Read TIER-2: Sparse/Distributed/Checkpoint (5%)

6. **Run training from TOML or inspect tiling via CLI**  
   → Read TIER-2: CLI Layer (E2/E3)

---

## TIER-2: Deep Reference

### Tiling Layer (40% of depth — AxisSpec, BatchPlanner, Strategies, Dispatch, Iterators, Carry, Dedup, Bucket)

#### AxisSpec: Axis Specification

Declare a single axis to be tiled:

```python
from xtrax.tiling.plan import AxisSpec  # verify: src/xtrax/tiling/plan.py:26-93

spec = AxisSpec(
    name="batch",                      # verify: src/xtrax/tiling/plan.py:43 (Human-readable axis name)
    cardinality=1000,                  # Total elements on this axis
    default_batch_size=32,             # Chunk size for SafeMap
    tile_granularity=1,                # Alignment (default 1 = no constraint)
    heterogeneous=False,               # Elements have varying shapes?
    dedup_eligible=False,              # Repeated elements?
    bucket_boundaries=None,            # Variable-length bucketing? (optional)
)
```

Verify all fields: `src/xtrax/tiling/plan.py:26-49`

**Deprecation notice — verify scoping**:

⚠ WARN: `AxisSpec.batch_size` is **deprecated** (since v0.3.0).  
Use `AxisSpec.default_batch_size` instead.  
Enforcement: `DeprecationWarning` from `src/xtrax/tiling/plan.py:76-91`

**IMPORTANT SCOPE**: The `.batch_size` deprecation applies **only to `AxisSpec`**.  
The following remain live, correct fields and require NO changes:
- `AxisDecision.batch_size` — the chosen batch size for this axis
- `SafeMap.batch_size` — the tile size in the SafeMap strategy
- `AxisStatsEntry["batch_size"]` — EDA output field
- `safe_map(batch_size=...)` — parameter to safe_map function

⚠ WARN: `AxisSpec.granularity` is **deprecated** (since v0.3.0).  
Use `AxisSpec.tile_granularity` instead.  
Enforcement: `DeprecationWarning` from `src/xtrax/tiling/plan.py:85-91`

#### AxisSpec Validation

🚫 HALTS: Empty `bucket_boundaries` raises `ValueError`.  
Enforcement: `src/xtrax/tiling/plan.py:59-61`
```python
# This raises ValueError
AxisSpec(name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=())
```

🚫 HALTS: Non-ascending `bucket_boundaries` raises `ValueError`.  
Enforcement: `src/xtrax/tiling/plan.py:68-74`
```python
# This raises ValueError: boundaries must be strictly ascending
AxisSpec(name="seq", cardinality=10, default_batch_size=4, bucket_boundaries=(16, 8))
```

🚫 HALTS: Non-positive values in `bucket_boundaries` raise `ValueError`.  
Enforcement: `src/xtrax/tiling/plan.py:63-67`

#### BatchPlanner: Strategy Selection

`BatchPlanner.plan()` analyzes each `AxisSpec` and assigns a strategy:

```python
from xtrax.tiling.plan import BatchPlanner  # verify: src/xtrax/tiling/plan.py:121-150

planner = BatchPlanner(
    memory_estimator=None,      # Optional per-axis estimator; mutually exclusive with budget
    carry_specs=None,           # Optional: list[CarrySpec] for Scan axes
    dedup_specs=None,           # Optional: list[DedupSpec] for dedup configuration
    heterogeneous_axes=None,    # Optional: set[str] of axes with variable shapes
    budget=None,                # Optional: MemoryBudget for joint-budget mode (0.4.0a1+)
)

plan = planner.plan([spec1, spec2, ...])
```

Verify: `src/xtrax/tiling/plan.py:143-150`

**Output**: `BatchPlan` with `decisions: tuple[AxisDecision, ...]`  
Each `AxisDecision` contains:
- `spec: AxisSpec` — the input specification
- `batch_size: int` — final chosen batch size
- `reasoning: str` — human-readable explanation
- `strategy: AxisStrategy` — Vmap | SafeMap | Scan | DedupGather | Bucket

#### Joint-Budget Planning (0.4.0a1+): MemoryBudget + Estimators

`BatchPlanner(budget=MemoryBudget(bytes=..., estimate=...))` replaces the independent per-axis rules 3-5 with whole-plan greedy demotion: every eligible axis starts at `Vmap`, then axes with `cardinality > default_batch_size` are demoted to `SafeMap` **in the order specs were given** until the joint estimate fits the budget. Callers express demotion priority by spec order (axes they are most willing to sequentialize first). Carry/dedup/bucket decisions stay fixed but participate in the estimate; budget-mode reasoning strings carry the byte numbers for `xtrax explain`.

```python
from xtrax.tiling import (  # all exported at xtrax.tiling level, same tier as CarrySpec
    BatchPlanner, MemoryBudget, BudgetInfeasibleError,
    device_memory_budget, lowered_memory_estimate,
)

budget = MemoryBudget(
    bytes=device_memory_budget(fraction=0.9),   # bytes from XLA allocator's bytes_limit
    estimate=my_estimate_fn,                    # Sequence[AxisDecision] -> estimated peak bytes
)
planner = BatchPlanner(budget=budget)
plan = planner.plan(specs)  # may raise BudgetInfeasibleError
```

Verify: `src/xtrax/tiling/budget.py:23-56`, `src/xtrax/tiling/estimators.py:27-97`

**Strict by design** (unlike per-axis `memory_estimator`, which swallows estimator errors):
- 🚫 HALTS: `budget` and `memory_estimator` are mutually exclusive — passing both raises.
- 🚫 HALTS: estimator exceptions propagate unchanged; there is no silent fallback in budget mode.
- 🚫 HALTS: `BudgetInfeasibleError` when every demotion candidate is already `SafeMap` and the joint estimate still exceeds `budget.bytes` — the message names budget, final estimate, and per-axis strategy state.
- 🚫 HALTS: `MemoryBudget.__post_init__` rejects non-int/non-positive `bytes` and non-callable `estimate`. Verify: `src/xtrax/tiling/budget.py:50-56`

**Native estimator building blocks** (`xtrax.tiling.estimators`):
- `device_memory_budget(fraction=0.9, device=None) -> int` — budget bytes from the XLA allocator's `Device.memory_stats()["bytes_limit"]`; fails loud when the backend reports no stats (e.g. some CPU builds).
- `lowered_memory_estimate(fn, *abstract_args) -> int` — AOT-compiles from `ShapeDtypeStruct`s and returns XLA's own buffer-assignment bytes (argument + output + temp) via `Compiled.memory_analysis()`.

Spec: `.praxia/docs/specs/260706_joint-budget-batch-planner.md`

#### Strategies: The Five Axis Patterns

**1. Vmap** — Fully parallel, stateless.  
Selected when: `cardinality <= batch_size`  
Behavior: `jax.vmap(fn)` over the axis. All elements processed in parallel.  
Verify: `src/xtrax/tiling/strategy.py:42-46`

```python
from xtrax.tiling.strategy import Vmap  # verify: src/xtrax/tiling/strategy.py:42-46

strategy = Vmap()
# Applied via: results = jax.vmap(fn)(inputs)  # verify: src/xtrax/tiling/dispatch.py:95-96
```

**2. SafeMap** — Chunked vmap, memory-safe.  
Selected when: `cardinality > batch_size`  
Behavior: Chunks inputs into `batch_size` chunks, applies vmap to each chunk, concatenates.  
Verify: `src/xtrax/tiling/strategy.py:49-53`

```python
from xtrax.tiling.strategy import SafeMap  # verify: src/xtrax/tiling/strategy.py:49-53

strategy = SafeMap(batch_size=32)
# Applied via: results = safe_map(fn, inputs, batch_size=32)  # verify: src/xtrax/transforms/map.py
```

Memory estimation (optional): Provide a `memory_estimator` to `BatchPlanner` to prevent Vmap if estimated memory > device limit.

**3. Scan** — Carry-bearing sequential iteration.  
Selected when: `CarrySpec` declares this axis as stateful (e.g., accumulating loss, sampling state).  
Behavior: `jax.lax.scan(transition, init, xs)` threads carry through iterations.  
Verify: `src/xtrax/tiling/strategy.py:56-62`

```python
from xtrax.tiling.strategy import Scan

def transition(carry, x):
    """(carry_in, x) -> (carry_out, y)"""
    new_carry = carry + x
    return new_carry, x * 2

strategy = Scan(transition=transition, init=0.0)
# Applied via: final_carry, results = jax.lax.scan(transition, init, xs)
```

Declare a Scan axis via `CarrySpec` (see below).

**4. DedupGather** — Deduplication for repeated elements.  
Selected when: `dedup_eligible=True` and `DedupSpec` identifies repeated elements.  
Behavior: (Phase 0) Extract unique indices; (Phase 1) vmap over K unique; (Phase 2) gather back to N.  
Verify: `src/xtrax/tiling/strategy.py:65-86`

```python
from xtrax.tiling.strategy import DedupGather

strategy = DedupGather(
    unique_indices=np.array([0, 1, 0, 2]),  # 4 elements → 3 unique
    index_map=np.array([0, 1, 0, 2]),       # inverse: position i uses unique slot index_map[i]
    k=3,                                    # raw unique count
    k_bucket=4,                             # padded bucket (power of 2, >= k)
    dedup_fn=_default_dedup_fn,            # select unique by index
    gather_fn=_default_gather_fn,          # scatter results back
)
```

🚫 HALTS: `DedupGather` cannot be passed to `make_axis_dispatch`.  
It is handled by internal library dispatch, not for direct user iteration.  
Enforcement: `src/xtrax/tiling/dispatch.py:78-82` raises `DispatchRejected`

**5. Bucket** — Variable-length bucketing (host-side padding).  
Selected when: `bucket_boundaries` is set on `AxisSpec`.  
Behavior: Pads variable-length inputs to the nearest boundary, creating a fixed set of XLA programs.  
Verify: `src/xtrax/tiling/strategy.py:88-107`

```python
from xtrax.tiling.strategy import Bucket
from xtrax.tiling.bucket import select_bucket, bucketize

boundaries = (32, 64, 128)  # Pad up to nearest boundary
strategy = Bucket(boundaries=boundaries)

# Host-side operation: select bucket, pad, send to jit
bucket_idx = select_bucket(sequence_length=50, boundaries=boundaries)  # → 1 (64)
padded_seq = bucketize(sequence, boundaries=boundaries)                # → (64,)
```

🚫 HALTS: `Bucket` cannot be passed to `make_axis_dispatch`.  
It is a host-side strategy; padding and bucketing happen before JAX.  
Enforcement: falls through to exhaustiveness `TypeError` at `src/xtrax/tiling/dispatch.py:101-102` (no dedicated branch — compare `DedupGather`, which raises `DispatchRejected` at lines 78-82).

#### Dispatch: Converting Strategy to Iterator

`make_axis_dispatch()` converts a strategy to a callable iterator:

```python
from xtrax.tiling.dispatch import make_axis_dispatch

iterator = make_axis_dispatch(
    strategy=decision.strategy,    # Vmap | SafeMap | Scan (NOT DedupGather or Bucket)
    axis="batch",                   # Name of the axis (for error messages)
    heterogeneous_axes={"state"},  # Set of axes with variable-shape elements
)

# Iterator is one of: VmapIterator, SafeMapIterator, JaxScanIterator
results = iterator(fn, inputs, in_axes=0)
```

Verify: `src/xtrax/tiling/dispatch.py:31-102`

**Rejection rules:**

🚫 HALTS: `DedupGather` is rejected.  
Enforcement: `src/xtrax/tiling/dispatch.py:78-82`  
Reason: DedupGather is handled by internal library dispatch (`axis_dispatch`), not exposed to user `make_axis_dispatch`.

🚫 HALTS: `Scan` on a heterogeneous axis is rejected.  
Enforcement: `src/xtrax/tiling/dispatch.py:84-92`  
Reason: `jax.lax.scan` requires static carry shape; variable-geometry state is incompatible.

#### Iterators: Three Patterns

**MapIterator** (stateless): Vmap and SafeMap

```python
from xtrax.tiling.iterator import VmapIterator, SafeMapIterator

# VmapIterator: jax.vmap
vmap_iter = VmapIterator()
results = vmap_iter(fn, inputs, in_axes=0)  # fn applied in parallel

# SafeMapIterator: chunked vmap
safemap_iter = SafeMapIterator(tile=32)
results = safemap_iter(fn, inputs, in_axes=0)  # fn applied in chunks of 32
```

Verify: `src/xtrax/tiling/iterator.py:28-59`

**ScanIterator** (carry-bearing): Scan

```python
from xtrax.tiling.iterator import JaxScanIterator

def transition(carry, x):
    new_carry = carry + x.sum()
    return new_carry, x * 2

scan_iter = JaxScanIterator()
final_carry, results = scan_iter(transition, init=0.0, xs=inputs)
```

Verify: `src/xtrax/tiling/iterator.py:62-81`

#### CarrySpec: Declare Scan Axes

Declare which axis uses `Scan` strategy (carry-bearing iteration):

```python
from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.strategy import ScanTransition

def transition(carry: dict, x) -> tuple:
    """(carry_in, x) -> (carry_out, y)"""
    carry_in['loss'] += x['loss']
    return carry_in, x

carry_spec = CarrySpec(
    axis_name="n_samples",
    init={"loss": 0.0},                # Initial carry (must be static shape at trace time)
    transition=transition,             # (carry, x) -> (carry, y)
    ordered_sinks=True,                # Guarantee step order for io_callback?
)

# Pass to planner:
planner = BatchPlanner(carry_specs=[carry_spec])
plan = planner.plan([axis_spec_for_n_samples])
```

Verify: `src/xtrax/tiling/carry.py:21-45`

🔬 HiTL: **CarrySpec init static shape**  
**Trigger**: When `CarrySpec.init` contains shapes that are not provably static at Python time.  
**Question**: "Verify that `init` shape is static at JAX trace time before `BatchPlanner.plan()`. Dynamic shapes fail at `jax.lax.scan` compilation. Is this shape static: {shape}?"  
**Consequence**: Proceeding without confirmation may cause cryptic trace-time shape mismatch errors.  
Block until confirmed.

#### DedupSpec and get_k_bucket

Configure deduplication for repeated elements:

```python
from xtrax.tiling.dedup import DedupSpec, get_k_bucket

# Identify unique elements in a batch
batch = jax.numpy.array([0, 1, 0, 2, 1, 1])
unique_vals, unique_indices = jax.numpy.unique(batch, return_index=True)
k = len(unique_indices)  # 3 unique elements

# Configure dedup strategy (k_bucket is computed internally — do NOT pass it)
spec = DedupSpec(
    axis_name="batch",              # Must match AxisSpec.name of a dedup_eligible axis
    unique_indices=unique_indices,  # (k,) indices of unique elements in original
    index_map=...,                  # (n,) inverse: position i uses result from slot index_map[i]
    k=k,                            # Number of distinct elements (== len(unique_indices))
)

# Pass to planner:
planner = BatchPlanner(dedup_specs=[spec])
plan = planner.plan([axis_spec_with_dedup_eligible_true])
```

Verify: `src/xtrax/tiling/dedup.py`

🔬 HiTL: **DedupSpec k > 256**  
**Trigger**: When `k > 256` (more than 256 unique elements).  
**Question**: "k={k} exceeds 256 — power-of-2 bucketing wastes up to 2× compute here (see `src/xtrax/tiling/dedup.py:29` TODO). Proceed with powers-of-2 or define custom bucket boundaries?"  
**Consequence**: Large k values use suboptimal bucketing. Custom buckets (e.g., geometric progression 1.5×) may reduce waste.  
Block until confirmed.

⚠ GAP: DedupGather large-k regime (k > 256)  
Current implementation uses powers-of-2 bucketing (`get_k_bucket(k)` rounds up to next power).  
For k > 256, this wastes up to 2× compute per element (worst case: k=257 → bucket=512).  
**TODO** at `src/xtrax/tiling/dedup.py:29`: Implement geometric or mixed bucketing.  
**Status**: Not fixed as of v0.4.0a5 — use with caution for large k.

#### Bucket: Variable-Length Axis Handling

For axes with variable-length elements (e.g., sequences), use bucketing to limit recompilation:

```python
from xtrax.tiling.bucket import select_bucket, bucketize

# Declare buckets on AxisSpec
spec = AxisSpec(
    name="seq",
    cardinality=1000,
    default_batch_size=32,
    bucket_boundaries=(32, 64, 128, 256),  # Pad to nearest boundary
)

# At runtime: select bucket and pad
seq_length = 50
bucket_idx = select_bucket(seq_length, boundaries=spec.bucket_boundaries)  # → 1 (64)
padded_seq = bucketize(sequence, boundaries=spec.bucket_boundaries)        # → (64,)
```

Verify: `src/xtrax/tiling/bucket.py`

🔬 HiTL: **bucket_boundaries tradeoff**  
**Trigger**: Setting `bucket_boundaries` on an AxisSpec.  
**Question**: "Boundaries {boundaries} → {n_buckets} compiled XLA programs. Each adds compilation latency; each gap adds padding waste. Review tradeoff and confirm boundaries?"  
**Consequence**: Too many buckets → compile overhead. Too few → large padding waste.  
Block until confirmed.

---

### Run Layer (20% of depth — RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec/make_sink, ZarrStagingSink, zarr_integrity, AxisBoundary, Fuse/Tap/Sink, topology validation, boundary executor)

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
    output_dir=Path("/path/to/outputs"),  # Directory for output files (None = no output)
    format="zarr",                         # "jsonl" | "h5" | "zarr" | "none" (default: "jsonl")
    flush_every=10,                        # Flush buffer every N stage calls (default: 1)
)

sink = make_sink(spec)  # ZarrStagingSink for "zarr", None for "none"
```

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

---

### Training Layer (25% of depth — ResumableState, Trainer, SafetyTrainStep, Engine, Callbacks, Optax)

#### ResumableState: Training State

Immutable training state that can be saved and restored:

```python
from xtrax.training.types import ResumableState  # verify: src/xtrax/training/types.py

state = ResumableState(
    step=jnp.int32(0),         # jnp.int32 scalar — dynamic leaf; plain 0 (Python int) also accepted
    key=jax.random.key(0),     # PRNG key
    model=my_model,            # eqx.Module (trainable parameters)
    opt_state=None,            # Optimizer state (optax format)
)
```

Verify: `src/xtrax/training/types.py`

**Mutation pattern** (using `eqx.tree_at`):

```python
import equinox as eqx

# Update model and step in state
new_state = eqx.tree_at(
    lambda s: (s.model, s.step),
    state,
    (new_model, state.step + 1),
)
```

Verify: `src/xtrax/training/trainer.py:67-70`

#### Trainer: Single-Model Training Step

Execute one supervised training step:

```python
from xtrax.training.types import LossFunction  # verify: src/xtrax/training/types.py
from xtrax.training.trainer import Trainer  # verify: src/xtrax/training/trainer.py:12-74
import optax

loss_fn: LossFunction = lambda pred, target: jnp.mean((pred - target) ** 2)
optimizer = optax.adam(learning_rate=1e-4)

trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)

# Execute step
new_state, metrics = trainer.step(state, batch)  # verify: src/xtrax/training/trainer.py:31-74
# metrics = {"loss": scalar}
# new_state.step incremented by 1
```

Verify: `src/xtrax/training/trainer.py:12-74`

**Invariant**: `Trainer.step` is `@eqx.filter_jit` decorated (trace only JAX arrays).

#### SafetyTrainStep: Gradient Safety

Numerical safety wrappers for gradient computation:

```python
from xtrax.training.types import SafetyTrainStep

# Gradient clipping, NaN detection, etc.
safety = SafetyTrainStep(
    grad_clip_norm=1.0,      # Clip gradients by norm
    check_nans=True,          # Detect NaN losses
)

# Applied inside Trainer.step or Engine
```

Verify: `src/xtrax/training.types`

#### Engine: Async Training Loop

High-level training orchestration with callbacks:

```python
from xtrax.engine.engine import Engine
import asyncio

engine = Engine(
    trainer=trainer,
    data_loader=resolver,
    callbacks=[callback1, callback2],
)

# Async iteration
async def train():
    async for new_state, metrics in engine.fit(state, num_epochs=10):
        print(f"Step {new_state.step}, Loss: {metrics['loss']}")

# Blocking alternative: fit_sync
for new_state, metrics in engine.fit_sync(state, num_epochs=10):
    print(f"Step {new_state.step}, Loss: {metrics['loss']}")
```

Verify: `src/xtrax/engine/engine.py`

⚠ NOTE: `Engine.fit` is **async**. Use `fit_sync()` for blocking usage.

#### Callback Protocol

Extend training with custom hooks:

```python
from xtrax.training.types import Callback

class LoggingCallback(Callback):
    """Log metrics every N steps."""
    
    def on_step_end(self, state, metrics):
        if state.step % 100 == 0:
            print(f"Step {state.step}: {metrics}")

    def on_epoch_end(self, state, epoch: int):
        print(f"Epoch {epoch} end")

trainer = Trainer(...)
engine = Engine(trainer=trainer, callbacks=[LoggingCallback()])
```

Verify: `src/xtrax/training/types.py`

**Callback hooks** (7 total, verify: `src/xtrax/training/types.py:32-40`):
- `on_train_start(state)` — Before training begins
- `on_train_end(state)` — After all training
- `on_resume(state)` — When resuming from a checkpoint
- `on_epoch_start(state, epoch: int)` — Before epoch (note `epoch` arg)
- `on_epoch_end(state, epoch: int)` — After epoch (note `epoch` arg)
- `on_step_start(state)` — Before step
- `on_step_end(state, metrics)` — After step, receives metrics dict

⚠ NOTE: Callback hooks run **Python-side, outside JAX traces**. Mutating state in callbacks has no effect on training.

#### Optax Integration

Create learning rate schedules and optimizer chains:

```python
from xtrax.training import make_optimizer, adamw_with_schedule
import optax

# Simple Adam
opt = optax.adam(learning_rate=1e-4)

# Adam with learning rate schedule
schedule = optax.exponential_decay(
    init_value=1e-4,
    transition_steps=1000,
    decay_rate=0.96,
)
opt_with_schedule = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.adam(learning_rate=schedule),
)

# Utility functions
opt = make_optimizer(learning_rate=1e-4)
opt = adamw_with_schedule(init_lr=1e-4, warmup_steps=1000, total_steps=10000)
```

Verify: `src/xtrax/training/optim.py` (definitions); re-exported at `src/xtrax/training/__init__.py:4`

---

### CLI Layer (E2/E3) — Tyro-delegated verbs: plan/explain/export/run/resume/sweep + graph-validate/graph-plan/graph-author

> **Availability**: six verbs shipped in the 0.3.0 release (E2: `plan`/`explain`/`export`; E3: `run`/`resume`/`sweep`). Three more — `graph-validate`/`graph-plan`/`graph-author` (T1-10/T1-11/T1-12) — are **unreleased, main-only** (no CHANGELOG entry yet, not in the 0.4.0a5 wheel), same convention as this doc's other main-only flags (e.g. the `io_callback` shim, `axis_boundaries_by_name`). Verify against `REGISTRY` directly (`src/xtrax/cli/registry.py`) before relying on a verb count — this doc is a map, not the territory.

#### Verb Registry (Tyro-delegated)

All CLI verbs are registered in `REGISTRY` — a single dict mapping verb name → `(ArgsClass, run_fn)`:

```python
from xtrax.cli.registry import REGISTRY  # verify: src/xtrax/cli/registry.py:41-51

# REGISTRY keys (E2/E3 — 0.3.0 release):
#   "plan"    → (PlanArgs, run_plan)       — infer_bundle + BatchPlanner, print summary
#   "explain" → (ExplainArgs, run_explain) — infer_bundle + plan + explain_plan + emit
#   "export"  → (ExportArgs, run_export)   — export plan artifacts
#   "run"     → (RunArgs, run_run)         — load_config → run_from_config → Engine.fit_sync
#   "resume"  → (ResumeArgs, run_resume)   — read manifest → reconstruct state from latest ckpt → train N more epochs
#   "sweep"   → (SweepArgs, run_sweep)     — sequential in-process grid search over a sweep TOML

# REGISTRY keys (T1-10/11/12 — unreleased, main-only):
#   "graph-validate" → (GraphValidateArgs, run_graph_validate) — load <ir.json>, run validate_graph, write audit_verdict back
#   "graph-plan"     → (GraphPlanArgs, run_graph_plan)         — load <ir.json>, resolve a node's callable_ref, plan it via plan_from_fn
#   "graph-author"   → (GraphAuthorArgs, run_graph_author)     — free-generate a candidate IR via TemplateGenerator, validate in-process, write it
```

`entrypoint.main()` builds a tyro subcommand dict from `REGISTRY` and dispatches the parsed `ArgsClass` instance to its `run_fn`. Verify: `src/xtrax/cli/entrypoint.py:19-48`

#### `xtrax run config.toml` Flow

End-to-end training from a TOML file:

```
config.toml
  → load_config(path)          # tomllib parse + validation  — verify: src/xtrax/cli/config.py:38-56
  → TrainConfig                # cli-private dataclass       — verify: src/xtrax/cli/config.py:15-29
  → run_from_config(cfg)       # cli-private glue            — verify: src/xtrax/cli/run.py:38-80
      → resolve model/optimizer/loss/data via load_fn (import-path strings)
      → init_state(model, optimizer, seed)   # public API    — verify: src/xtrax/training/state.py:9-20
      → config_hash(cfg_dict)  # run_id derivation           — verify: src/xtrax/cli/hash.py:7-20
      → write_manifest(...)    # always before fit_sync      — verify: src/xtrax/cli/manifest.py:56-77
      → Engine(Trainer(...)).fit_sync(state, data, ...)
```

CLI entry: `run_run(RunArgs(config="config.toml"))` catches `ConfigError` and exits with a clean message. Verify: `src/xtrax/cli/run_verb.py:14-20`

#### `xtrax resume <run-id> --epochs N` Flow

Resume a prior run from its latest orbax checkpoint, training N **additional** epochs into a new sibling run dir:

```
xtrax resume <run-id> --epochs N [--manifest-path PATH]
  → read_manifest(run_id)                 # locate the run's manifest.json
  → resolve_components(...)               # re-resolve model/optimizer/loss/data from import paths
  → load_checkpoint(...)                  # reconstruct ResumableState from latest orbax ckpt
  → write_manifest_dict(...)              # new sibling run dir under .xtrax/runs/
  → Engine(Trainer(...)).fit_sync(...)
```

`ResumeArgs`: positional `run_id`, required `epochs: int`, optional `manifest_path` (if the run dir was moved). Raises `ResumeError` (subclasses `CLIError`) on a missing/invalid manifest or checkpoint. Verify: `src/xtrax/cli/resume_verb.py:18-30`

#### `xtrax sweep sweep_config.toml` Flow

Sequential in-process grid search. The sweep config is a normal training TOML plus a `[sweep.axes]` section whose leaves are **lists** — the grid is the cartesian product, and each combination overrides the base config for one run:

```toml
# ... normal [model]/[optimizer]/[loss]/[data] sections ...

[sweep.axes]
seed = [42, 43]
optimizer.kwargs.peak_lr = [1e-3, 3e-4]   # nested keys via dotted tables or nesting
```

Properties (verify: `src/xtrax/cli/sweep_verb.py`):
- Sweep manifest written incrementally and atomically per combination (`tempfile.mkstemp` + `os.replace` before each run executes); per-run fault tolerance (one failed combination doesn't kill the sweep).
- JAX compilation cache reused across combinations (single process, sequential — isolates compilation and execution memory via `gc` between runs).
- 🚫 HALTS: `ConfigError` if `[sweep]` is not a table or any `sweep.axes` leaf is not a list.

`SweepArgs`: positional `config_path` only. Verify: `src/xtrax/cli/sweep_verb.py:34-37`

#### `xtrax graph-validate <ir.json>` Flow (unreleased, main-only, T1-10)

Validates a D4 IR document in place and writes the audit verdict back into it:

```
xtrax graph-validate <ir.json>
  → load_graph(path)               # parse D4 IR document          — verify: src/xtrax/composition/serialize.py
  → validate_graph(graph, root=cwd)  # deterministic validation gate — verify: src/xtrax/composition/validate.py
  → dump_graph(result.graph, path)   # write audit_verdict back, same path
  → print JSON envelope {schema_version, failure_count, failures[]}
```

`GraphValidateArgs`: positional `ir_path` only. Malformed input (missing/unknown `schema_version`, unresolvable `callable_ref`) raises `SystemExit` with a clean message. Any node not `verdict=PASS` exits 1 after printing the envelope. Registered as the flat verb `graph-validate` — `entrypoint.py`'s tyro dispatch is a flat `dict[str, (ArgsClass, run_fn)]` with no nested-subcommand support, so this is not the DAG doc's informal two-word `graph validate`. Verify: `src/xtrax/cli/graph_verb.py`

#### `xtrax graph-plan <ir.json> <node-id> [--shapes ...]` Flow (unreleased, main-only, T1-11)

Resolves a named node's `callable_ref` from a D4 IR document and plans it — the CLI-consumed half of AC1's graph→plan parity proof:

```
xtrax graph-plan <ir.json> <node_id> [--shapes "x=(4,)f32"]
  → load_graph(path)                       # same loader graph-validate uses
  → nodes_by_id[node_id].callable_ref       # CLIError if node_id not found (lists available ids)
  → plan_from_fn(callable_ref, shapes)      # same plan_from_fn helper run_plan's bare --fn/--shapes path uses
  → print_plan_summary(plan)
```

`GraphPlanArgs`: positional `ir_path`, required `node_id`, optional `shapes` (default `""`; see `xtrax.cli.shapes.parse_shapes` grammar). A node's `callable_ref` post-`load_graph` resolves to the identical live function object `load_fn` would resolve from a bare `module.path:symbol` string — both use the same convention, so this path is provably convergent with `run`'s `--fn` resolution, not just coincidentally similar. Verify: `src/xtrax/cli/graph_plan_verb.py`

#### `xtrax graph-author <out.json> [--seed N] [--num-nodes N]` Flow (unreleased, main-only, T1-12)

The default generate-then-validate authoring front-end — free-generates a candidate graph and validates it in-process before writing:

```
xtrax graph-author <out_path> [--seed 0] [--num-nodes 3]
  → TemplateGenerator().generate(seed, num_nodes)   # deterministic free-generation
  → validate_graph(graph, root=cwd)                 # same gate graph-validate uses, run in-process
  → dump_graph(result.graph, out_path)
  → print JSON envelope {schema_version, failure_count, failures[]}
```

`GraphAuthorArgs`: positional `out_path`, `seed: int = 0`, `num_nodes: int = 3`. "Authors ≥1 graph passing graph-validate" is enforced directly here as an in-process assertion (`SystemExit(1)` on any non-`PASS` verdict), not left for a caller to separately verify by running `graph-validate` afterward. An invalid `num_nodes` or unwritable `out_path` raises `SystemExit` with a clean message. Verify: `src/xtrax/cli/graph_author_verb.py`

#### Key Types

| Symbol | Module | Role |
|--------|--------|------|
| `TrainConfig` | `xtrax.cli.config` | Parsed training config (`schema_version`, `model`, `optimizer`, `loss`, `data`, `seed`, `num_epochs`) |
| `ConfigError` | `xtrax.cli.config` | Invalid/incomplete TOML; subclasses `CLIError` |
| `load_config` | `xtrax.cli.config` | Parse + validate TOML path → `TrainConfig`; composed from `xtrax.config`'s primitives (below) |
| `init_state` | `xtrax.training` | **Public API** — build `ResumableState` from model + optimizer + seed |
| `config_hash` | `xtrax.cli.hash` | cli-private — stable 12-char hex hash for run-id derivation |
| `write_manifest` | `xtrax.cli.manifest` | cli-private — always-write `manifest.json` under `.xtrax/runs/<run_id>/` |
| `load_fn` | `xtrax.cli` (also `xtrax.cli.loader`) | **Stable public API** — domain-agnostic `module.path:symbol` → callable resolver; safe for downstream packages to import directly |
| `CLIError` / `CLIImportError` | `xtrax.cli` (also `xtrax.cli.errors`) | **Stable public API** — downstream CLIs may subclass `CLIError` for their own fail-loud error types (mirrors `ConfigError`/`ResumeError` in-repo) |
| `REGISTRY` | `xtrax.cli` (also `xtrax.cli.registry`) | **Stable public API** (keys/shape only) — `verb_name -> (ArgsClass, run_fn)`; see the REGISTRY-composition pattern below |
| `load_toml_document` | `xtrax.config` | **Stable public API** — domain-agnostic TOML parse, wraps IO/decode errors into a caller-supplied `error_cls` |
| `require_sections` | `xtrax.config` | **Stable public API** — presence check naming *every* missing section, not just the first |
| `require_field` | `xtrax.config` | **Stable public API** — extract + validate a field against an arbitrary predicate |
| `check_schema_version` / `classify_schema_version` | `xtrax.config` | **Stable public API** — schema-version validation; `classify_schema_version` is the public extension seam for future status kinds |

`init_state` is re-exported from `xtrax.training` (`__all__` at `src/xtrax/training/__init__.py:14`). `TrainConfig`/`load_config`/`ConfigError` stay in `xtrax.cli.config` — cli-private, not top-level `xtrax` exports, and training-shaped (not suitable for a non-training consumer to import directly). By contrast, `load_fn`/`CLIError`/`CLIImportError`/`REGISTRY` ARE declared public at `xtrax.cli.__all__`, and `xtrax.config`'s four primitives are a fully domain-agnostic top-level module — a downstream consumer that needs training-config-shaped TOML loading should compose `xtrax.config`'s primitives directly for its own dataclass shape (see the Minimal `xtrax.config` Usage example below), not mirror `TrainConfig`'s pattern by hand and not import `TrainConfig` itself.

#### Minimal `xtrax.config` Usage (domain-agnostic, not training-shaped)

```python
from dataclasses import dataclass
from xtrax.config import load_toml_document, check_schema_version, require_sections, require_field

class InferConfigError(Exception):
    """A downstream package's own error type -- xtrax.config never hardcodes one."""

@dataclass
class InferConfig:
    schema_version: int
    model: dict
    checkpoint: dict

def load_infer_config(path: str) -> InferConfig:
    raw = load_toml_document(path, InferConfigError)
    check_schema_version(raw, current=1, error_cls=InferConfigError)
    require_sections(raw, ("model", "checkpoint"), InferConfigError)
    return InferConfig(schema_version=raw["schema_version"], model=raw["model"], checkpoint=raw["checkpoint"])
```

Verify: `src/xtrax/config.py`; `xtrax.cli.config.load_config` is the dog-fooded reference usage (`src/xtrax/cli/config.py`). Spec: `.praxia/docs/specs/260715_generic-fail-loud-toml-to-dataclass-conf.md`.

#### Minimal `config.toml` Skeleton

Each section uses import-path `path`/`factory` keys plus optional `kwargs`. Verify against `tests/cli/test_config.py:16-37`:

```toml
schema_version = 1
seed = 42
num_epochs = 3

[model]
path = "mylib.models:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = { learning_rate = 1e-3, total_steps = 300 }

[loss]
path = "mylib.losses:mse_loss"
kwargs = {}

[data]
factory = "mylib.data:make_dataset"
kwargs = {}
batch_size = 4
```

🚫 HALTS: Missing `schema_version` or any of `[model]`, `[optimizer]`, `[loss]`, `[data]` raises `ConfigError`, naming **every** missing section (not just the first).  
🚫 HALTS: `num_epochs` must be a positive int; `seed` must be an int.  
Enforcement: `src/xtrax/cli/config.py` (composed from `xtrax.config`'s `check_schema_version`/`require_sections`/`require_field`).

#### Tyro-Free Import Rule

`import xtrax.cli` must **not** pull `tyro` at module level (AC2 import isolation):

- `xtrax.cli.__init__` exports `CLIError`, `CLIImportError`, `ShapeParseError`, `load_fn`, `REGISTRY`, and a lazy `main()` that imports `entrypoint` on demand. `REGISTRY` is exposed via a PEP 562 module-level `__getattr__` — accessing it lazily imports `xtrax.cli.registry` (and thus all 9 built-in verb modules) on demand, so a bare `import xtrax.cli` stays as lightweight as before this export was added. Verify: `src/xtrax/cli/__init__.py`
- `entrypoint.main()` imports `tyro` **inside** the function body. Verify: `src/xtrax/cli/entrypoint.py:30-31`

Test pattern (mirrors E2 isolation tests): `assert "tyro" not in sys.modules` immediately after `import xtrax.cli`.

#### REGISTRY Composition: Reusing xtrax's Verbs in Your Own CLI

A downstream package building its own tyro-dispatched CLI (rather than getting a verb hosted through `xtrax`'s own binary — that entry-points-plugin approach was considered and explicitly **deferred**, see `.praxia/docs/specs/260715_entry-points-based-xtrax-cli-verb-regist.md`) can reuse xtrax's own verbs today, with zero xtrax code changes:

```python
import tyro
from xtrax.cli import REGISTRY

my_verbs = {"infer": (InferArgs, run_infer)}  # your own package's verbs
merged = {**REGISTRY, **my_verbs}

subcommands = {name: args_cls for name, (args_cls, _fn) in merged.items()}
selected = tyro.extras.subcommand_cli_from_dict(subcommands)
for name, (args_cls, run_fn) in merged.items():
    if args_cls is type(selected):
        run_fn(selected)
```

**Stability boundary:** `REGISTRY`'s **keys and dict shape** (`verb_name -> (ArgsClass, run_fn)`) are a stable, documented contract. The `ArgsClass`/`run_fn` internal typing is **provisional**, not independently versioned — don't rely on a specific verb's `ArgsClass` field set staying fixed across xtrax releases beyond what its own docs promise.

🚫 HALTS: a verb-name or `ArgsClass` field-name collision between `REGISTRY` and your own verbs is your responsibility to avoid — `tyro.extras.subcommand_cli_from_dict` does not detect or warn on one. Verify: `tests/cli/test_registry_composition.py`.

---

### EDA: Plan Analysis and Visualization (10% of depth)

#### extract_plan_stats: Structured Analysis

Extract statistics from a `BatchPlan`:

```python
from xtrax.eda.stats import extract_plan_stats  # verify: src/xtrax/eda/stats.py

plan = planner.plan([spec1, spec2, ...])
stats = extract_plan_stats(plan)

# stats is a dict[str, Any] with:
# {
#   "axes": [
#       {"name": "batch", "cardinality": 100, "batch_size": 32, "reasoning": "...", ...},  # verify: src/xtrax/eda/types.py
#       ...
#   ],
#   ...
# }
```

Verify: `src/xtrax/eda/stats.py`

#### explain_plan: Guaranteed Non-Empty Reasoning

Wrapper around `extract_plan_stats` ensuring all reasoning fields are non-empty:

```python
from xtrax.eda.explain import explain_plan

stats = explain_plan(plan)
# All stats["axes"][i]["reasoning"] are guaranteed non-empty strings
```

Verify: `src/xtrax/eda/explain.py:14-41`

#### EDA-as-Planning-Audit Workflow

Before committing to a batching strategy, audit the plan:

```python
# Step 1: Build plan
plan = planner.plan([spec1, spec2, ...])

# Step 2: Inspect reasoning
stats = explain_plan(plan)
for axis in stats["axes"]:
    print(f"{axis['name']}: {axis['strategy']} ({axis['reasoning']})")

# Step 3: Visualize (if xtrax[eda] installed)
from xtrax.eda import render  # implemented in viz.py, re-exported via eda/__init__.py

html = render(plan)
with open("plan.html", "w") as f:
    f.write(html)
```

**Benefit**: Catch suboptimal strategy choices (e.g., SafeMap when Vmap would fit) before first JIT compilation.

#### analyze_dedup, analyze_bucket

Per-axis statistics:

```python
from xtrax.eda.stats import analyze_dedup, analyze_bucket

# Dedup analysis
dedup_stats = analyze_dedup(decision)  # For DedupGather decisions

# Bucket analysis
bucket_stats = analyze_bucket(decision)  # For Bucket decisions
```

Verify: `src/xtrax/eda/stats.py`

#### render: HTML Visualization

Generate interactive HTML plan visualization:

```python
from xtrax.eda import render  # implemented in viz.py, re-exported via eda/__init__.py

html = render(plan)
# html is a string of HTML
```

⚠ WARN: `render()` requires `pip install xtrax[eda]` (extras).  
Import is lazy — no error at module load time, but `render()` call will fail if extras not installed.

#### plan_to_dataframe: Pandas Export

Export plan stats to a pandas DataFrame:

```python
from xtrax.eda import plan_to_dataframe  # lazy re-export from eda/export.py, same pattern as render()

df = plan_to_dataframe(plan)
# DataFrame with columns: name, cardinality, strategy, batch_size, reasoning, ...
```

⚠ WARN: `plan_to_dataframe()` requires `pip install xtrax[eda]` (extras, same as `render()`).

Verify: `src/xtrax/eda/export.py:23` (implementation); `src/xtrax/eda/__init__.py:37-44` (lazy re-export wrapper). `from xtrax.eda.stats import plan_to_dataframe` does NOT work — `stats.py` does not define or import this symbol.

---

### Sparse / Distributed / Checkpoint (5% of depth — Pointer Pattern)

#### Sparsification: Structured Pruning

Convert a dense model to sparse (BCOO) format at inference time:

```python
from xtrax.sparse import sparsify_model, make_sparse_forward_fn  # verify: src/xtrax/sparse/inference.py
from xtrax.sparse.policy import SparsePolicy
import equinox as eqx

policy = SparsePolicy(target_sparsity=0.9)

# BEFORE jit: sparsify the model  # verify: src/xtrax/sparse/inference.py:44-55
sparse_model = sparsify_model(model, policy)

# RECOMMENDED: Use closure pattern
forward_fn = make_sparse_forward_fn(sparse_model)
result = jax.jit(forward_fn)(x)

# ALTERNATIVE: Pass to eqx.filter_jit (holds BCOO as static)
@eqx.filter_jit
def inference(x):
    return sparse_model(x)

result = inference(x)
```

Verify: `src/xtrax/sparse/inference.py`

🚫 HALTS: `sparsify_model` **cannot** be called inside `jax.jit`.  
Enforcement: `RuntimeError` from `assert_not_tracing` at `src/xtrax/sparse/inference.py:44-55`  
Reason: BCOO structure is non-static, must be created on host.

#### Distributed: Multi-Device Training

Initialize distributed context:

```python
from xtrax import init_dist, is_distributed, LogicalMesh, with_manual_axes

init_dist(backend="xmap")  # or "pjit"

if is_distributed():
    mesh = LogicalMesh(shape=(2, 4))  # 2×4 device mesh
    with with_manual_axes(mesh):
        # Distributed training code
        pass
```

Verify: `src/xtrax/distributed/` (full reference deferred to source)

#### Checkpoint: Save/Load Training State

Persist training state for resumption:

```python
from xtrax import save_checkpoint, load_checkpoint

# Save
save_checkpoint(state, directory="/path/to/ckpt")

# Load
state = load_checkpoint(directory="/path/to/ckpt")
```

Verify: `src/xtrax/checkpoint/` (see orbax docs for full checkpoint manager API)

---

### Signature Inference (xtrax.inference) — derive AxisSpecs + BundleSchema from a typed function

> **Availability**: shipped in the 0.3.0 release (Tier-1 MVP, E1).  
> Import paths: `from xtrax.inference import ...` (all 8 public symbols re-exported from `__init__`).  
> `AxisRole` lives canonically in `xtrax.tiling.roles` (zero xtrax deps) and is re-exported by `xtrax.inference` for convenience. Verify: `src/xtrax/inference/errors.py:12`, `src/xtrax/tiling/roles.py:14`.

#### `infer_bundle`: The Entrypoint

```python
from xtrax.inference import infer_bundle, BundleSchema, AxisOverride, axis_config
from xtrax.tiling.plan import AxisSpec

schema, axes = infer_bundle(
    fn,                         # pure, traceable JAX function  — verify: src/xtrax/inference/api.py:15
    abstract_inputs,            # Sequence[ShapeDtypeStruct | (shape, dtype)]
    verify_against=None,        # optional Sequence of concrete inputs
)
# returns: tuple[BundleSchema, list[AxisSpec]]  — verify: src/xtrax/inference/api.py:20
```

Exact signature (verify: `src/xtrax/inference/api.py:15-20`):
```python
def infer_bundle(
    fn: Any,
    abstract_inputs: Sequence[Any],
    *,
    verify_against: Sequence[Any] | None = None,
) -> tuple[BundleSchema, list[AxisSpec]]:
```

Internally: calls `jax.eval_shape` (zero FLOPs) to extract the output schema, reads any `@axis_config` sidecar on `fn`, synthesizes one `AxisSpec` per qualifying input leaf (ndim >= 1), and optionally calls `verify_structure`.  
Verify: `src/xtrax/inference/api.py:58-78`

> **CLI cross-link**: `xtrax plan` and `xtrax explain` both call `infer_bundle` internally (load `--fn` import path + parse `--shapes`, then plan). See TIER-2: CLI Layer (E2/E3). `explain` adds `explain_plan` + format emission (`json`/`text`/`html`/`png`). Verify: `src/xtrax/cli/plan.py:31-39`, `src/xtrax/cli/explain.py:52-60`

#### Fail-Loud Model: `AxisRole.KNOWN` vs `AxisRole.UNKNOWN`

Every synthesized `AxisSpec` carries a `role` field (verify: `src/xtrax/tiling/plan.py:50`):

- **`AxisRole.KNOWN`** (default for hand-written specs) — planner proceeds normally.
- **`AxisRole.UNKNOWN`** — sentinel set on every axis that `infer_bundle` cannot resolve (i.e., bare functions with no `@axis_config`).

```python
# AxisRole is an enum with two MVP members:
class AxisRole(enum.Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
# verify: src/xtrax/tiling/roles.py:14-26
```

🚫 HALTS: `BatchPlanner.plan()` raises `AmbiguousAxisError` for any `AxisSpec` with `role == AxisRole.UNKNOWN`.  
Enforcement: `src/xtrax/tiling/plan.py:259-266`  
Message: `"axis '<name>' has an unresolved role; declare it with @axis_config or provide an override before planning."`

This is intentional: `infer_bundle` on a zero-config function produces UNKNOWN axes, and the planner never silently proceeds with ambiguous axes. You must resolve every axis before planning.

#### `@axis_config`: Tier-1 Override (Resolve UNKNOWN → KNOWN)

```python
from xtrax.inference import axis_config, AxisOverride

@axis_config(
    AxisOverride(name="batch", default_batch_size=32),   # positional: axis 0
    AxisOverride(name="seq",   default_batch_size=128),  # positional: axis 1
)
def my_fn(x, y):
    ...
# verify: src/xtrax/inference/config.py:44-74
```

`AxisOverride` fields (verify: `src/xtrax/inference/config.py:9-38`):

```python
@dataclass(frozen=True)
class AxisOverride:
    name: str                              # Required. Human-readable axis name.
    default_batch_size: int                # Required. Batch size (NOT inferrable from shape).
    cardinality: int | None = None         # Override leading-dim cardinality (None = infer).
    tile_granularity: int = 1              # Alignment granularity (default 1).
    heterogeneous: bool = False            # Variable-shape elements?
    dedup_eligible: bool = False           # Eligible for deduplication?
    bucket_boundaries: tuple[int,...] | None = None  # Variable-length bucketing?
```

`@axis_config` stores overrides as `__xtrax_axis_config__` on the decorated function (zero call-path overhead) and returns the function unchanged. Resolving an axis via override sets its `AxisSpec.role` to `KNOWN`. `default_batch_size` is REQUIRED because batch size is never inferrable from shape alone (Assumption A3 of the inference design). Verify: `src/xtrax/inference/config.py:18-19`

#### `BundleSchema`: Output Schema

```python
from xtrax.inference import BundleSchema
# verify: src/xtrax/inference/schema.py:12-27

@dataclass
class BundleSchema:
    fields: dict[str, ShapeDtypeStruct]  # leaf name -> ShapeDtypeStruct (from eval_shape output)
    carry_specs: list[Any] | None = None  # passthrough seam; always None in MVP
```

Field names are recovered from the eval_shape output pytree's key_path:
- `GetAttrKey.name` — dataclass / eqx.Module field names
- `DictKey.key` — dict keys
- Positional fallback `out_{i}` for SequenceKey or bare leaves

Verify: `src/xtrax/inference/schema.py:30-93`

#### `verify_structure` / `verify_against=`: Purity Guard

`verify_structure` runs `fn` concretely on `concrete_inputs`, compares the pytree structure, leaf shapes, and dtypes against the abstract eval_shape output, and raises `StructureMismatchError` on any divergence.

```python
from xtrax.inference.verify import verify_structure  # verify: src/xtrax/inference/verify.py:19-96
from xtrax.inference import StructureMismatchError

# Via infer_bundle:
schema, axes = infer_bundle(fn, abstract_inputs, verify_against=concrete_inputs)

# Direct call:
verify_structure(fn, abstract_inputs, concrete_inputs)  # returns None or raises
```

`StructureMismatchError` is raised when `jax.eval_shape`'s abstract output structure diverges from actual execution — e.g., due to data-dependent control flow.  
Verify: `src/xtrax/inference/errors.py:28-41`

#### jaxtyping Note

jaxtyping is **optional**: the inference layer never hard-imports it. The Tier-2 jaxtyping dim-name role adapter (which would map annotated dim names to concrete `AxisRole` values) is deferred and not part of the MVP.

#### Minimal Working Example

```python
import jax
from jax import ShapeDtypeStruct
import numpy as np
from xtrax.inference import infer_bundle, axis_config, AxisOverride, AxisRole
from xtrax.tiling.plan import BatchPlanner

def encode(x, y):
    """Two-input function: (batch, feat), (batch, feat) -> (batch, feat)."""
    return x + y

# --- Zero-config: UNKNOWN axes, planner will fail loud ---
abstract = [ShapeDtypeStruct((32, 128), np.float32),
            ShapeDtypeStruct((32, 128), np.float32)]
schema, axes = infer_bundle(encode, abstract)
print(schema.fields)        # {"out_0": ShapeDtypeStruct(shape=(32, 128), dtype=float32)}
print(axes[0].role)         # AxisRole.UNKNOWN — not annotated, cannot plan yet

planner = BatchPlanner()
# planner.plan(axes)        # 🚫 HALTS: AmbiguousAxisError — axis 'axis_0' has unresolved role

# --- With @axis_config: KNOWN axes, planner proceeds ---
@axis_config(
    AxisOverride(name="batch", default_batch_size=32),
    AxisOverride(name="batch", default_batch_size=32),  # one override per input
)
def encode_annotated(x, y):
    return x + y

schema2, axes2 = infer_bundle(encode_annotated, abstract)
print(axes2[0].role)        # AxisRole.KNOWN
print(axes2[0].name)        # "batch"

plan = planner.plan(axes2)  # succeeds — all axes KNOWN
print(plan.decisions[0].reasoning)  # "cardinality <= batch_size → Vmap"
```

#### Deferred (Not in MVP)

- **Tier-2 jaxtyping dim-name adapter**: maps annotated dimension names to concrete `AxisRole` values; would eliminate `@axis_config` for jaxtyping-annotated functions.
- **libcst Bundle codegen**: generate `BundleSchema` as typed Python source from inferred schema.
- **CarrySpec auto-derivation**: automatically derive `CarrySpec` from function return type annotations.

---

## Summary

This skill provides a complete, self-contained reference for xtrax v0.4.0a5 (+ Unreleased `main`).

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

## Technical Gaps (Known Limitations in v0.4.0a5)

| Gap | Location | Status |
|-----|----------|--------|
| DedupGather large-k regime (k > 256) uses suboptimal power-of-2 bucketing | `src/xtrax/tiling/dedup.py:29` | TODO: implement geometric or mixed bucketing for k > 256 |
| Top-level exports missing (RunSpec, CarrySpec, DedupSpec, AxisBoundary) | `src/xtrax/__init__.py` | By design; use subpackage imports: `from xtrax.run import RunSpec`, `from xtrax.stages import AxisBoundary`, etc. |
| `make_sink` has no writer for `"jsonl"`/`"h5"` | `src/xtrax/run/sink.py:32-39` | Routing-only stub values; `NotImplementedError` until their writers land. Use `"zarr"` (or `"none"`). |
| Ordered `SafeMap` axis ignores `batch_size` (runs element-at-a-time) | `src/xtrax/stages/executor.py` | Structural JAX constraint, not fixable locally — see Boundary Executor section; use `Scan` if ordering + explicit sequential cost is acceptable |
| Nested executor composition (vmap-of-scan) ordering not certified | `src/xtrax/stages/executor.py` | T1-05 stress harness pending |

The `make_inference_plan` gap noted as of v0.3.0 is closed: plan-time checks now exist via `validate_plan_topology` (`xtrax.stages`, 0.3.1+).

---

## For More Information

- **JAX discipline**: See TIER-1 section "JAX Discipline for Domain Library Authors"
- **Decision tree**: See TIER-1 section "Which Primitive for Which Problem"
- **Minimal working example**: See TIER-1 section "Minimal Working Pattern"
- **Live source code**: All code examples cite `verify: src/...:<line>`; read the source at those locations for current behavior

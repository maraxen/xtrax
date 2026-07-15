> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Tiling Layer (40% of depth — AxisSpec, BatchPlanner, Strategies, Dispatch, Iterators, Carry, Dedup, Bucket)

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

#### Multi-Axis Composition

`BatchPlanner` composes axes by **simple independence**: build one `AxisSpec` per axis (plus a matching `DedupSpec`/`CarrySpec` where applicable), pass the whole list to a single `planner.plan([...])` call, and each axis resolves to its own strategy — no rule anywhere reads two axes' specs together. The only cross-axis interactions in the entire pipeline are joint-budget demotion order (Joint-Budget Planning, above) and `validate_plan_topology` (Run Layer, below), which checks each axis's strategy only against that same axis's own boundary/heterogeneity.

Worked example — one dedup-eligible axis (`tokens`) + one variable-length axis (`seqlen`) in the same plan:

```python
import numpy as np
from xtrax.tiling.plan import AxisSpec, BatchPlanner
from xtrax.tiling.dedup import DedupSpec   # not re-exported at xtrax.tiling level

# Axis A: 8 elements, only 3 distinct → dedup
tokens_spec = AxisSpec(name="tokens", cardinality=8, default_batch_size=4,
                       dedup_eligible=True)
# Axis B: variable-length sequences → bucket (Rule 1: bucket_boundaries set → Bucket)
seqlen_spec = AxisSpec(name="seqlen", cardinality=1000, default_batch_size=32,
                       bucket_boundaries=(32, 64, 128))

# DedupSpec is DATA-DEPENDENT: caller must have already inspected the batch
# (e.g. np.unique on the host) to compute these — unlike Bucket/Vmap/SafeMap,
# which are static config.
dedup = DedupSpec(
    axis_name="tokens",                                       # matches AxisSpec.name
    unique_indices=np.array([0, 1, 3], dtype=np.int32),       # (k,) first-occurrence slots
    index_map=np.array([0, 1, 0, 2, 1, 1, 0, 2], dtype=np.int32),  # (n,) position i → slot
    k=3,
)

planner = BatchPlanner(dedup_specs=[dedup])
plan = planner.plan([tokens_spec, seqlen_spec])

# plan.decisions, in spec order (verified live):
#   [0] tokens -> DedupGather | "dedup-gather (DedupSpec for 'tokens', k=3, k_bucket=4)"   # Phase 0b
#   [1] seqlen -> Bucket      | "bucket_boundaries=(32, 64, 128) → Bucket (host-side padding)"  # Rule 1
```

Verify: `src/xtrax/tiling/plan.py:184-283` (plan(), independent per-axis loop), `src/xtrax/tiling/plan.py:242-257` (Phase 0b DedupSpec pre-demotion), `src/xtrax/tiling/plan.py:397-408` (Bucket Rule 1), `src/xtrax/tiling/dedup.py:46-90` (DedupSpec fields + `__post_init__` validation: `len(unique_indices) == k`, `index_map` covers exactly `[0, k)`)

**Executing the mixed plan.** There is no single call that dispatches a whole multi-strategy `BatchPlan` — the caller iterates `plan.decisions` and routes each axis by strategy type. Bucket axes are handled on the host **before** the jit boundary; DedupGather axes go through the eager `axis_dispatch()` shim; Vmap/SafeMap/Scan axes use `make_axis_dispatch()` iterators (Dispatch subsection, above). xtrax provides the per-axis primitives, not a mixed-strategy executor.

```python
from xtrax.tiling.bucket import select_bucket, bucketize
from xtrax.tiling.dispatch import axis_dispatch

dedup_decision, bucket_decision = plan.decisions

# 1. Bucket axis: pad on the host, in plain Python, BEFORE jit — no dispatch call.
boundaries = bucket_decision.strategy.boundaries
bucket_idx = select_bucket(seq_length, boundaries=boundaries)
padded_seq = bucketize(sequence, boundaries=boundaries)

# 2. DedupGather axis: eager three-phase shim (dedup → safe_map → gather).
ys = axis_dispatch(dedup_decision.strategy, fn, xs)
# internally: dedup_fn(xs, unique_indices) → safe_map(fn, ...) → gather_fn(ys, index_map)
```

Verify: `src/xtrax/tiling/dispatch.py:105-174` (axis_dispatch eager shim — handles Vmap, SafeMap, Scan, DedupGather), `src/xtrax/tiling/dispatch.py:152-161` (DedupGather three-phase execution). Confirmed live this session: `axis_dispatch(dedup_decision.strategy, lambda x: x * 2, jnp.array([10,20,10,30,20,20,10,30]))` → `[20 40 20 60 40 40 20 60]` — round-trips correctly through dedup→map→gather.

🚫 HALTS: Neither mixed-plan strategy goes through `make_axis_dispatch()`.  
`make_axis_dispatch(DedupGather(...))` raises `DispatchRejected` ("handled elsewhere ... Use DedupGather via BatchPlanner + _dispatch_axis"); `make_axis_dispatch(Bucket(...))` falls through to the exhaustiveness `TypeError: Unknown strategy type` (no dedicated branch).  
Enforcement: `src/xtrax/tiling/dispatch.py:78-82` (DedupGather), `src/xtrax/tiling/dispatch.py:101-102` (Bucket)

🚫 HALTS: `axis_dispatch(Bucket(...), fn, xs)` also raises `TypeError`, by design: "Bucket is a host-side strategy and is not executed by axis_dispatch. Pad to a bucket shape on the host with select_bucket()/bucketize() before your jitted step, then dispatch the per-bucket compute with a device-tier strategy (e.g. Vmap/SafeMap)."  
Enforcement: `src/xtrax/tiling/dispatch.py:163-171`

⚠ WARN: `DedupSpec` inputs are data-dependent. `unique_indices`/`index_map`/`k` must be computed from the **actual batch** on the host before `planner.plan()` is called; if the batch changes, rebuild the `DedupSpec` and re-plan. The other axes in a composed plan (Bucket boundaries, Vmap/SafeMap cardinality) are static config and survive batch changes unchanged.

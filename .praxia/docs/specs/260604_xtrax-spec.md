---
title: xtrax — High-Performance Composable JAX Library
task_id: 260604_xtrax-shape
date: 260604
status: draft-r3
oracle_round: 2
---

# Specification: xtrax

## 1. Purpose and Scope

xtrax is a domain-agnostic, high-performance composable JAX library for external consumers. It provides the primitives that domain-specific model libraries (e.g., prxteinmpnn) build on top of: batched execution strategies (tiling), composable training infrastructure, safe JAX operator wrappers, distributed sharding utilities, checkpointing, optimizer composition helpers, and data pipeline abstractions. It sits above JAX and Equinox and delegates optimizer logic to Optax. Target: Python 3.13+.

### What xtrax is NOT
- Not a model library (no layers, no architectures).
- Not a replacement for Equinox primitives (`eqx.Module`, `filter_jit`, `filter_grad`, `filter_vmap`, `eqx.nn.*`, `partition`/`combine`/`tree_at`).
- Not a replacement for Optax optimizers, schedules, gradient clipping, or EMA.
- Not a replacement for Orbax — wraps it.
- Not a replacement for Grain — wraps it.
- No `Sequential`, `pipe()`, or `LayerStack` combinators.
- No Python-loop hot paths anywhere.

### Primary consumers
External developers building domain libraries on JAX + Equinox.

---

## 2. Package Layout

```
src/xtrax/
  __init__.py
  transforms/   map.py, scan.py
  tiling/       strategy.py, plan.py, iterator.py, dedup.py, dispatch.py
  stages/       protocols.py, bundle.py
  training/     state.py, losses.py, callbacks.py, trainer.py, step.py,
                combinators.py, optim.py, accumulate.py
  engine/       engine.py
  data/         module.py, streaming.py
  distributed/  sharding.py, dist.py
  safety/       numerics.py, checkify.py, preemption.py
  checkpoint/   orbax.py
  io/           callbacks.py
```

---

## 3. Module Contracts

### 3.1 `xtrax.transforms.map` — `safe_map`

```python
def safe_map(
    fn: Callable[[PyTree], PyTree],
    xs: PyTree,
    batch_size: int | None = None,
) -> PyTree: ...
```

`batch_size` is a **vmap/lax.map threshold and chunk size**, not a padding target.

**Invariants:**
- `batch_size is None` or `n <= batch_size` → `jax.vmap(fn)(xs)`.
- `n > batch_size` AND `n % batch_size == 0` → `jax.lax.map(fn, xs, batch_size=batch_size)` (chunked vmap inside lax loop).
- `n > batch_size` AND `n % batch_size != 0` → raises `ValueError("safe_map: xs leading axis {n} not divisible by batch_size {batch_size}")` before any JAX tracing.
- Result shape for valid inputs is identical to `jax.vmap(fn)(xs)`.
- `fn` must be JAX-traceable.

**JAX version note:** `jax.lax.map(batch_size=...)` requires `jax>=0.4.36`. This drives the dependency floor (§4).

**Does NOT:** Handle heterogeneous batch sizes across axes.

**Gate:** `uv run pytest tests/transforms/test_map.py -v` covering:
- `batch_size=None` → vmap
- `batch_size=n` → vmap
- `batch_size=50`, n=100 (divisible) → lax.map, result matches vmap
- `batch_size=30`, n=100 (non-divisible) → `ValueError`
- pytree xs; shape invariant

---

### 3.2 `xtrax.transforms.scan` — `safe_scan`

```python
def safe_scan(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
) -> tuple[Carry, Y]: ...
```

**Invariants:**
- Raises Python `ValueError` before tracing if `length == 0` or inferred xs leading-axis is 0.
- `init=None` is **not rejected** by `safe_scan`. `None` is a legal empty-carry pytree in `jax.lax.scan`. The init-required check belongs exclusively to the `Scan` dispatch strategy (§3.7), not this standalone transform. Do not add an `init is None` guard here.
- Delegates entirely to `jax.lax.scan` after validation.
- Signature mirrors `jax.lax.scan` exactly.

**Gate:** `uv run pytest tests/transforms/test_scan.py -v` — normal scan, empty-xs `ValueError`, `reverse=True`, `unroll>1`.

---

### 3.3 `xtrax.tiling.strategy` — `AxisStrategy`

```python
@dataclass(frozen=True)
class Vmap: pass

@dataclass(frozen=True)
class SafeMap:
    batch_size: int  # threshold and chunk size passed to safe_map

@dataclass(frozen=True)
class Scan:
    transition: ScanTransition
    # No batch_size — safe_scan has no chunking; Scan is sequential carry-bearing

@dataclass(frozen=True)
class DedupGather:
    dedup_fn: DedupFn
    gather_fn: GatherFn
    k_bucket: int  # metadata: max unique elements bucket size (power-of-2 padded); not used by dispatch

AxisStrategy = Vmap | SafeMap | Scan | DedupGather

@runtime_checkable
class ScanTransition(Protocol):
    def __call__(self, carry: PyTree, x: PyTree) -> tuple[PyTree, PyTree]: ...

@runtime_checkable
class DedupFn(Protocol):
    def __call__(self, xs: PyTree) -> tuple[PyTree, Array]:
        # Returns (deduped_xs, gather_indices) — both required by dispatch
        ...

@runtime_checkable
class GatherFn(Protocol):
    def __call__(self, deduped_ys: PyTree, indices: Array) -> PyTree: ...
```

**`Scan` strategy note:** `Scan` is **caller-constructed only**. `BatchPlanner` never selects it — it has no selection rule mapping from `AxisSpec` fields to `Scan`. Callers that need carry-bearing iteration construct `Scan(transition=my_fn)` directly.

**Invariants:**
- Exactly four variants; `assert_never` enforces exhaustiveness in `make_axis_dispatch`.
- All frozen dataclasses; protocols `runtime_checkable`.

**Implementation freedom:** Four separate dataclasses vs. single discriminated dataclass — provided the union alias stays `AxisStrategy = Vmap | SafeMap | Scan | DedupGather`.

---

### 3.4 `xtrax.tiling.plan` — `AxisSpec`, `AxisDecision`, `BatchPlan`, `BatchPlanner`

```python
@dataclass(frozen=True)
class AxisSpec:
    name: str
    cardinality: int
    batch_size: int
    granularity: int = 1
    heterogeneous: bool = False
    dedup_eligible: bool = False

@dataclass(frozen=True)
class AxisDecision:
    spec: AxisSpec
    batch_size: int
    reasoning: str
    strategy: AxisStrategy  # never Scan — planner never selects Scan

@dataclass(frozen=True)
class BatchPlan:
    decisions: tuple[AxisDecision, ...]

class BatchPlanner:
    def __init__(
        self,
        memory_estimator: Callable[[AxisSpec], int] | None = None,
    ) -> None: ...
    def plan(self, specs: Sequence[AxisSpec]) -> BatchPlan: ...
```

**Default selection rules (when `memory_estimator` is None):**
1. `dedup_eligible is True` → `DedupGather`
2. `cardinality <= batch_size` → `Vmap`
3. `cardinality > batch_size` AND `cardinality % batch_size == 0` → `SafeMap(batch_size)`
4. `cardinality > batch_size` AND `cardinality % batch_size != 0` → `SafeMap(batch_size)` with a warning logged via Python `warnings.warn`; fixer must not raise here. **Deferred-failure contract:** this plan is intentionally non-executable as returned — `make_axis_dispatch` will raise `ValueError` at dispatch time (§3.1 non-divisibility rule). The warning is a hard pre-execution error in disguise; callers must correct divisibility before dispatching.

**`memory_estimator` contract:** When provided, the estimator returns an estimated memory cost (in bytes) for the given `AxisSpec` if processed via `Vmap`. If the estimate exceeds a planner-internal threshold (default: `jax.devices()[0].memory_stats().get("bytes_limit", 4 * 2**30)`), `SafeMap` is preferred over `Vmap` regardless of cardinality. If the memory stats query fails, fall back to default selection rules silently.

**Invariants:**
- `plan` is pure Python — no JAX tracing.
- `Scan` is **never returned** by `plan`. Callers that want `Scan` construct `AxisDecision` manually.
- `BatchPlan.decisions` length equals `len(specs)`.

**Does NOT:** Accept device arrays. Does not call JAX (except optionally for memory stats). Does not execute the batch.

---

### 3.5 `xtrax.tiling.iterator` — Iterators

```python
@runtime_checkable
class MapIterator(Protocol):
    def __call__(self, fn: Callable, xs: PyTree) -> PyTree: ...

class VmapIterator(eqx.Module):
    def __call__(self, fn: Callable, xs: PyTree) -> PyTree: ...  # jax.vmap(fn)(xs)

class SafeMapIterator(eqx.Module):
    batch_size: int
    def __call__(self, fn: Callable, xs: PyTree) -> PyTree: ...  # safe_map(fn, xs, batch_size)

@runtime_checkable
class ScanIterator(Protocol):
    def __call__(self, fn: Callable, init: PyTree, xs: PyTree) -> tuple[PyTree, PyTree]: ...

class JaxScanIterator(eqx.Module):
    def __call__(self, fn: Callable, init: PyTree, xs: PyTree) -> tuple[PyTree, PyTree]: ...
    # Delegates to safe_scan(fn, init, xs)
```

**Invariants:** Concrete classes are `eqx.Module`; protocols are `runtime_checkable`. `VmapIterator` and `SafeMapIterator` produce identical outputs for same inputs when `batch_size >= n` (both use vmap). `SafeMapIterator` propagates `ValueError` from `safe_map` for non-divisible n.

---

### 3.6 `xtrax.tiling.dedup` — `get_k_bucket`, `DedupSpec`

```python
def get_k_bucket(n: int) -> int:
    # Mandated integer implementation: 1 << (n - 1).bit_length()
    # Returns the smallest power of 2 >= n.
    # Raises ValueError for n <= 0.
    # Examples: get_k_bucket(1)=1, get_k_bucket(7)=8, get_k_bucket(8)=8, get_k_bucket(9)=16

@dataclass(frozen=True)
class DedupSpec:
    k_bucket: int   # must be power of 2; __post_init__ validates with (k_bucket & (k_bucket-1)) == 0
    max_unique: int # maximum number of unique elements (k_bucket >= max_unique enforced)
```

**`DedupSpec.k_bucket` semantics:** Metadata field indicating the power-of-2 padded allocation size for unique-element arrays. Callers use it to pre-allocate `dedup_fn` output buffers. `dispatch` does not use `k_bucket` — it is purely for callers. Similarly `DedupGather.k_bucket` in §3.3 is metadata for callers sizing their allocation; dispatch ignores it.

**`DedupSpec.__post_init__` mandated checks:**
- `k_bucket <= 0` → `ValueError`
- `k_bucket & (k_bucket - 1) != 0` → `ValueError("k_bucket must be a power of 2")`
- `k_bucket < max_unique` → `ValueError("k_bucket must be >= max_unique")`

---

### 3.7 `xtrax.tiling.dispatch` — `make_axis_dispatch`, `DispatchRejected`

```python
class DispatchRejected(Exception): pass

def make_axis_dispatch(
    strategy: AxisStrategy,
    fn: Callable,
    xs: PyTree,
    init: PyTree | None = None,
) -> PyTree: ...
```

**Mandated dispatch behavior (exhaustive; `assert_never` required):**

- `Vmap` → `jax.vmap(fn)(xs)`

- `SafeMap` → `safe_map(fn, xs, strategy.batch_size)`

- `Scan` → if `init is None`: raise `ValueError("Scan strategy requires init")`; else `safe_scan(strategy.transition, init, xs)` — `strategy.transition` is passed as `fn` arg to `safe_scan`, NOT `fn` from the dispatch call signature. Specifically: `safe_scan(fn=lambda carry, x: strategy.transition(carry, x), init=init, xs=xs)`, then the outer `fn` from the dispatch signature is ignored. **Note:** this means for `Scan`, `fn` in the dispatch call signature is unused; callers pass `fn=None` or any placeholder; the actual computation is fully determined by `strategy.transition`.

- `DedupGather` → three-phase:
  1. `deduped_xs, gather_indices = strategy.dedup_fn(xs)` — unpack exactly two values
  2. `deduped_ys = safe_map(fn, deduped_xs, batch_size=None)` — vmap over deduped (no chunking)
  3. `result = strategy.gather_fn(deduped_ys, gather_indices)` — scatter back

`DispatchRejected` is a guard for unknown future variants only; the four current variants never raise it.

---

### 3.8 `xtrax.stages.protocols` — `TransformFn`, `RollingFn`, `FuseFn`

```python
@runtime_checkable
class TransformFn(Protocol[In, Out]):
    def __call__(self, x: In) -> Out: ...

@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
    def __call__(self, carry: Carry, x: In) -> tuple[Carry, Out]: ...

@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
    def __call__(self, items: PerItem) -> Combined: ...
```

No concrete implementations in xtrax. No domain-specific slot names. These are the generic tier-1 protocols from which domain libraries (e.g., aminx StageSet slots) specialize.

---

### 3.9 `xtrax.stages.bundle` — `StageBundle`

```python
class StageBundle(eqx.Module):
    """Typed bag of optional callable stage slots. Subclass and declare fields.
    Topology determined by non-None fields at Python dispatch level.
    No runtime branching inside JAX traces — this is a CALLER PRECONDITION, not enforced.
    """
    def active_stages(self) -> list[str]: ...   # Python-side only
    def has_stage(self, name: str) -> bool: ... # Python-side only
```

**Invariants:**
- `eqx.Module` subclass. All user-declared fields must be `Optional[Callable]`.
- `active_stages()` and `has_stage()` are **caller-preconditioned** to only be called Python-side, never inside a JAX-traced context. xtrax does not enforce this at runtime — it is documented as a hard precondition. Violations produce undefined JAX tracing behavior, not a clean error.
- No `__call__` on `StageBundle` itself — pure container.
- No branching on stage presence inside JAX traces; callers select code paths before calling JAX.

**Phase 7 addition:** `__init_subclass__` validates all annotated fields are `Optional[Callable]`; raises `TypeError` at class definition time on violation.

---

### 3.10 `xtrax.training.state` — `ResumableState`

```python
class ResumableState(eqx.Module):
    step: Array          # jnp.int32 scalar — dynamic leaf, NOT static (avoids per-step recompilation)
    key: Array
    model: eqx.Module
    opt_state: PyTree
    extras: dict[str, PyTree] = eqx.field(default_factory=dict)
```

No mutation — updates via `eqx.tree_at`. `extras` is Orbax-serializable. `step` as dynamic `Array` avoids recompilation on every increment.

---

### 3.11 `xtrax.training.losses` — `LossFunction`

```python
@runtime_checkable
class LossFunction(Protocol):
    def __call__(self, predictions: PyTree, targets: PyTree) -> Array: ...
    # Return must be scalar Array (shape ()). Must be JAX-traceable.
```

---

### 3.12 `xtrax.training.callbacks` — `Callback`

```python
@runtime_checkable
class Callback(Protocol):
    def on_train_start(self, state: ResumableState) -> None: ...
    def on_train_end(self, state: ResumableState) -> None: ...
    def on_resume(self, state: ResumableState) -> None: ...
    def on_epoch_start(self, state: ResumableState, epoch: int) -> None: ...
    def on_epoch_end(self, state: ResumableState, epoch: int) -> None: ...
    def on_step_start(self, state: ResumableState) -> None: ...
    def on_step_end(self, state: ResumableState, metrics: dict[str, Array]) -> None: ...
```

All hooks run Python-side (outside JAX traces). Mutating `state` in a callback has no effect on training — `state` is immutable and the training loop uses the return value of `trainer.step`, not the callback's view.

---

### 3.13 `xtrax.training.trainer` — `Trainer`

```python
class Trainer(eqx.Module):
    loss_fn: LossFunction
    optimizer: optax.GradientTransformation

    @eqx.filter_jit
    def step(
        self,
        state: ResumableState,
        batch: PyTree,
    ) -> tuple[ResumableState, dict[str, Array]]: ...
```

**Invariants:**
- `step` is `eqx.filter_jit`-decorated; pure function of `(state, batch)`.
- Gradient via `eqx.filter_value_and_grad(loss_fn_wrapped, has_aux=False)(state.model)`.
- Update call: `updates, new_opt_state = self.optimizer.update(grads, state.opt_state, eqx.filter(state.model, eqx.is_array))` — model params **always** passed as the third argument (required for weight-decay compatibility).
- Model updated via `eqx.apply_updates(state.model, updates)`.
- Returns new `ResumableState` with `step = state.step + 1` and updated `opt_state` and `model`.
- Metrics dict: at minimum `{"loss": scalar_loss}`.

**Does NOT:** Manage checkpointing, callbacks, sharding.

---

### 3.14 `xtrax.training.step` — `SafetyTrainStep`, `create_train_step`

```python
class SafetyTrainStep(eqx.Module):
    trainer: Trainer
    safety_manager: SafetyManager

    @eqx.filter_jit
    def step(
        self,
        state: ResumableState,
        batch: PyTree,
    ) -> tuple[ResumableState, dict[str, Array]]: ...

def create_train_step(
    loss_fn: LossFunction,
    optimizer: optax.GradientTransformation,
    safety: bool = False,
    safety_manager: SafetyManager | None = None,
) -> Trainer | SafetyTrainStep: ...
```

`safety=False` → `Trainer`. `safety=True` → `SafetyTrainStep`. Public `.step()` signature identical for both types.

---

### 3.15 `xtrax.training.combinators` — `WeightedLoss`, `MultiTaskLoss`

```python
class WeightedLoss(eqx.Module):
    loss_fn: LossFunction
    weight: float = eqx.field(static=True)

    def __call__(self, predictions: PyTree, targets: PyTree) -> Array: ...
    # Mandated: weight * self.loss_fn(predictions, targets)

class MultiTaskLoss(eqx.Module):
    losses: tuple[WeightedLoss, ...]

    def __call__(
        self,
        predictions: tuple[PyTree, ...],
        targets: tuple[PyTree, ...],
    ) -> Array: ...
    # Mandated: jnp.sum(jnp.stack([l(p, t) for l, p, t in zip(self.losses, predictions, targets)]))
    # Static-length tuple comprehension is correct here — it unrolls at trace time (not a data-axis
    # hot path). The §1 "no Python loops" rule applies to data-axis iteration, NOT to static structural
    # unrolling over fixed-length tuples. Do NOT attempt jax.tree.map here; the three tuples are
    # zipped (losses, predictions, targets) and not co-mappable as a pytree.
```

**Invariants:**
- Both satisfy `isinstance(x, LossFunction)`.
- Python `assert len(predictions) == len(targets) == len(self.losses)` before any JAX tracing.
- Inner computation: each `WeightedLoss.__call__` is JAX-traceable; stacking via `jnp.stack` on the tuple of scalars is valid (tuple is a static-length pytree).

---

### 3.16 `xtrax.training.optim` — Optimizer utilities

```python
def no_bias_wd_mask(params: PyTree) -> PyTree:
    # Mandated exactly: jax.tree.map(lambda x: x.ndim != 1, params)
    # Returns bool pytree: True = apply weight decay, False = skip
    # Note: 1-D embedding weights will be excluded — document this trade-off

def make_optimizer(
    base: optax.GradientTransformation,
    clip_norm: float | None = 1.0,
) -> optax.GradientTransformation:
    # clip_norm=None → return base unchanged
    # Otherwise mandated: optax.chain(optax.clip_by_global_norm(clip_norm), base)
    # clip MUST come before base in the chain

def adamw_with_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,   # total steps INCLUDING warmup — NOT total - warmup (common footgun)
    weight_decay: float = 1e-2,
    clip_norm: float | None = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    wd_mask: Callable = no_bias_wd_mask,
) -> optax.GradientTransformation:
    # Mandated implementation:
    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0, peak_value=peak_lr,
    #     warmup_steps=warmup_steps, decay_steps=total_steps, end_value=0.0)
    # base = optax.adamw(
    #     learning_rate=schedule, weight_decay=weight_decay,
    #     b1=b1, b2=b2, eps=eps,
    #     mask=wd_mask)   # <-- mask IS passed; mandatory
    # return make_optimizer(base, clip_norm=clip_norm)

def partition_labels(
    model: eqx.Module,
    frozen_filter: Callable[[Any], bool],
    frozen_label: str = "frozen",
    train_label: str = "train",
) -> PyTree:
    # Returns label pytree where every leaf is either frozen_label or train_label.
    # Mandated: use eqx.partition(model, frozen_filter) to split, then
    # map frozen leaves → frozen_label, non-frozen leaves → train_label.
    # The returned structure must be identical to the model pytree structure
    # with string leaves — compatible with optax.partition(transforms, label_tree).
```

No Python loops inside any returned `GradientTransformation`. Does not re-implement any optimizer logic.

---

### 3.17 `xtrax.training.accumulate` — `accumulate_grads`

```python
def accumulate_grads(
    loss_fn: Callable[[PyTree, PyTree], Array],
    params: PyTree,
    microbatches: PyTree,   # leading axis shape: (num_microbatches, microbatch_size, ...)
    filter_spec: Callable | None = None,  # defaults to eqx.is_array
) -> tuple[PyTree, Array]:
    # Returns (mean_grads, mean_loss).
    # Mandated: jax.lax.scan over microbatch leading axis. No Python for-loop.
```

**Preconditions (Python-side, raise before tracing):**
- All leaves of `microbatches` must share the same leading axis size `num_microbatches`.
- All microbatches must have equal second-axis size `microbatch_size`. Reason: `mean_grads == full_batch_grads` equivalence holds only when all microbatches are equal-sized and `loss_fn` computes a mean over the batch dimension.

**Equivalence guarantee:** Given equal-sized microbatches of size `M` and `loss_fn` computing `jnp.mean` over the batch dimension, `accumulate_grads` result matches full-batch gradient within `atol=1e-5, rtol=1e-5` for float32.

**Gate:** `jax.make_jaxpr` of the inner body contains a `scan` primitive. Grad matches full-batch grad within `atol=1e-5` (tested with 4 equal microbatches of size 25, total batch 100).

---

### 3.18 `xtrax.engine.engine` — `Engine`

```python
class Engine(eqx.Module):
    trainer: Trainer | SafetyTrainStep
    callbacks: tuple[Callback, ...]
    validation_callbacks: tuple[Callback, ...] = ()

    async def fit(
        self,
        state: ResumableState,
        data: DataModule,
        num_epochs: int,
        checkpoint_dir: str | Path | None = None,
    ) -> ResumableState: ...

    async def eval(
        self,
        state: ResumableState,
        data: DataModule,
        loss_fn: LossFunction | None = None,
    ) -> dict[str, Array]: ...

    def fit_sync(self, *args, **kwargs) -> ResumableState:
        return asyncio.run(self.fit(*args, **kwargs))
```

**Epoch semantics:** `Engine.fit(num_epochs=N)` controls the outer training loop — exactly N epochs. `DataModule.num_epochs` controls how the dataset iterator cycles internally (None = cycle indefinitely until training ends). When both are set, `Engine.fit(num_epochs)` is the authoritative stop condition. Mid-epoch dataset exhaustion raises `StopIteration` which `Engine` catches and treats as epoch end.

**Invariants:**
- `fit` increments `state.step` by 1 per batch (each call to `trainer.step`).
- `checkpoint_dir` set → after each epoch: `save_checkpoint(manager, state)` then `manager.wait_until_finished()` (synchronous flush — note this serializes async checkpoint benefit; acceptable for correctness).
- `eval`: wraps `state.model` in `eqx.nn.inference_mode(state.model)` before iterating (disables dropout/stochastic ops in eqx modules; no-op for models without such layers). Does not call `trainer.step`. Metrics per batch collected as pytree of scalars; averaged via `jax.tree.map(jnp.mean, jnp.stack(...))` across batches.
- `eval` fires `validation_callbacks` hooks; `fit` fires `callbacks`. No cross-firing.
- `BoundedCallbackHandler` used internally to dispatch callbacks asynchronously without blocking the JAX execution path.

**`asyncio` contract:** `Engine.fit` and `Engine.eval` are coroutines; callers must run them in an asyncio event loop. `fit_sync` provides the convenience `asyncio.run()` wrapper for single-threaded use. xtrax does not manage the event loop beyond `asyncio.run`. Callers with an existing event loop must call `await engine.fit(...)` directly.

**JAX/asyncio interaction:** `trainer.step` (JIT-compiled) may block the GIL during XLA dispatch. Callbacks and checkpoint saves run in the same event loop. For throughput-sensitive workloads, callbacks should be non-blocking (use `io_callback` internally or offload to threads).

**Implementation freedom:** Internal async scheduling (asyncio.Queue, asyncio.gather, sequential awaits) is implementer's choice provided the invariants above hold.

---

### 3.19 `xtrax.data.module` — `DataModule`

```python
class DataModule(eqx.Module):
    dataset: Any
    batch_size: int = eqx.field(static=True)
    num_epochs: int | None = eqx.field(static=True)  # None = cycle indefinitely
    seed: int = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    collate_fn: Callable | None = eqx.field(static=True, default=None)
    # Default collate_fn when None: jax.tree.map(jnp.stack, items)

    def train_iter(self) -> Iterator[PyTree]: ...
    def eval_iter(self) -> Iterator[PyTree]: ...
```

`distributed=True` without prior `init_dist()` call raises `RuntimeError` on first `train_iter()` or `eval_iter()` call.

---

### 3.20 `xtrax.data.streaming` — `BucketIterator`, `create_distributed_pipeline`

```python
class BucketIterator:
    def __init__(
        self,
        dataset: Any,
        bucket_boundaries: list[int],   # N sorted boundary values
        batch_sizes: list[int],         # N+1 batch sizes (one per bucket)
        length_fn: Callable[[Any], int],
    ) -> None: ...
    # Raises ValueError at construction if len(batch_sizes) != len(bucket_boundaries) + 1

    def __iter__(self) -> Iterator[PyTree]: ...

def create_distributed_pipeline(
    dataset: Any,
    global_batch_size: int,
    num_devices: int,
    seed: int,
) -> Iterator[PyTree]: ...
# Per-device batch size = global_batch_size // num_devices
# Uses Grain's grain.load with grain.ShardOptions(shard_index=jax.process_index(), shard_count=num_devices)
# Raises ValueError if global_batch_size % num_devices != 0
```

---

### 3.21 `xtrax.distributed.sharding` — `ShardingPolicy`, mesh helpers

```python
class ShardingPolicy(eqx.Module):
    rules: tuple[tuple[str, PartitionSpec], ...] = eqx.field(static=True)
    # Ordered; first match via re.search(pattern, path) wins

    def get_partition_spec(self, path: str) -> PartitionSpec: ...
    # Default (no match): PartitionSpec() — fully replicated

    def apply_to_pytree(self, pytree: PyTree) -> PyTree: ...
    # Returns same-structure pytree with leaves replaced by PartitionSpec

def get_device_mesh(shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh: ...
# Raises ValueError if math.prod(shape) != len(jax.devices())

def get_hardware_mesh_profile() -> dict[str, Any]:
    # Required keys: 'device_type' (str), 'num_devices' (int),
    #                'recommended_shape' (tuple[int,...]), 'recommended_axis_names' (tuple[str,...])
    # Never raises — CPU single-device fallback: shape=(1,), axis_names=('batch',)
```

**Does NOT:** Apply sharding or call `jax.device_put`. Pure advisory.

---

### 3.22 `xtrax.distributed.dist` — `init_dist`

```python
def init_dist(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
) -> None: ...
```

Idempotent — same args on repeat call: no-op. Different args on second call: `RuntimeError`. Auto-discovery order: explicit args → SLURM env vars (`SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_JOB_NODELIST`) → single-process localhost fallback (coordinator=`localhost:1234`, num_processes=1, process_id=0). Localhost fallback calls `jax.distributed.initialize` with single-process config.

---

### 3.23 `xtrax.safety.numerics` — `safe_norm`, `safe_reciprocal`

```python
def safe_norm(x: Array, axis=None, keepdims: bool = False, eps: float = 1e-8) -> Array:
    # Mandated: jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + eps**2)

def safe_reciprocal(x: Array, eps: float = 1e-8) -> Array:
    # Mandated: 1.0 / (x + eps)
```

Both JAX-traceable; no Python conditionals on array values. Gradient of `safe_norm` at `x=0` is `0 / eps` which is finite (no NaN).

---

### 3.24 `xtrax.safety.checkify` — `SafetyManager`, `with_safety`

```python
class SafetyManager(eqx.Module):
    enabled: bool = eqx.field(static=True)
    check_nans: bool = eqx.field(static=True)
    check_infs: bool = eqx.field(static=True)

    def wrap(self, fn: Callable) -> Callable: ...
    # enabled=False → returns fn unchanged (strict identity, no wrapper overhead)
    # enabled=True → returns checkify-wrapped fn that raises Python exception on host

def with_safety(fn: Callable, manager: SafetyManager) -> Callable:
    # Convenience alias for manager.wrap(fn)
```

`enabled=False` → strict identity (zero overhead; must not introduce trace-time overhead via any wrapper). `enabled=True` → `jax.experimental.checkify.checkify` with error kinds derived from `check_nans`/`check_infs`. Errors raised as Python exceptions on host after computation completes.

---

### 3.25 `xtrax.safety.preemption` — `PreemptionHandler`

```python
class PreemptionHandler:
    def __init__(self, save_fn: Callable[[], None], rank: int = 0) -> None: ...
    def register(self) -> None: ...  # installs SIGUSR1 + SIGTERM handlers; idempotent
    @property
    def preempted(self) -> bool: ...
```

`save_fn` called at most once. Only `rank == 0` calls `save_fn`. `preempted` backed by `threading.Event` (thread-safe). `register()` is idempotent (second call is a no-op).

---

### 3.26 `xtrax.checkpoint.orbax` — Checkpoint wrappers

```python
def get_checkpoint_manager(
    directory: str | Path,
    max_to_keep: int = 5,
    keep_period: int | None = None,
) -> orbax.checkpoint.CheckpointManager: ...

def save_checkpoint(
    manager: orbax.checkpoint.CheckpointManager,
    state: ResumableState,
    step: int | None = None,   # defaults to int(state.step); must be called Python-side (not inside JIT)
) -> None: ...
# Precondition: must not be called inside a JAX-traced context; int(state.step) would fail otherwise.

def load_checkpoint(
    manager: orbax.checkpoint.CheckpointManager,
    state_template: ResumableState,
    step: int | None = None,   # None → latest
) -> ResumableState: ...
```

No custom serialization. `load_checkpoint` on empty dir or unknown step raises `FileNotFoundError`.

---

### 3.27 `xtrax.io.callbacks` — `async_indexed_stream`, `BoundedCallbackHandler`

```python
async def async_indexed_stream(
    iterable: Iterable[T],
    buffer_size: int = 2,
) -> AsyncIterator[tuple[int, T]]: ...
# Semantics: prefetches up to buffer_size items ahead in a background thread.
# Indices are monotonically increasing from 0.
# Exceptions from the underlying iterable are caught and re-raised on the next yield to the consumer.
# Thread-safe: uses asyncio.to_thread for blocking iteration.

class BoundedCallbackHandler:
    def __init__(self, max_concurrent: int = 4) -> None: ...
    # Internally: asyncio.Semaphore(max_concurrent) — mandated

    async def submit(self, coro: Coroutine) -> None: ...
    # Acquires semaphore before starting coro; releases on completion or exception.
    # Exceptions in coro are logged (not propagated) to avoid cancelling the training loop.

    async def wait_all(self) -> None: ...
    # Blocks until all submitted coroutines complete.
    # Must be called at end of training to flush pending callbacks.
```

---

## 4. Dependency Declaration

```toml
[project]
name = "xtrax"
version = "0.1.0"
description = "High-performance composable JAX library"
readme = "README.md"
requires-python = ">=3.13"

dependencies = [
    "jax>=0.4.36",
    "jaxlib>=0.4.36",
    "equinox>=0.11.0",
    "optax>=0.2.3",
    "orbax-checkpoint>=0.6.0",
    "grain>=0.2.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4.0",
    "pyright>=1.1.360",
    "jaxtyping>=0.2.28",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py313"
select = ["E", "F", "I", "UP"]
```

**Version rationale:**
- `jax>=0.4.36` — conservatively pins above version where `jax.lax.map(batch_size=...)` is confirmed stable. Verify against `jax.version.__version__` in CI; fail fast if below floor.
- `equinox>=0.11.0` — stable `filter_jit`, `inference_mode`
- `optax>=0.2.3` — `warmup_cosine_decay_schedule`, `incremental_update`
- `orbax-checkpoint>=0.6.0` — async checkpoint manager
- `grain>=0.2.0` — distributed shard-by-worker API

**Note on existing pyproject.toml:** The current `pyproject.toml` in the repo is a blank scaffold (`dependencies = []`). Task 0.1 replaces it entirely with the above.

---

## 5. Fixer Task Decomposition

### Phase 0: Scaffold
**Task 0.1** — Replace `pyproject.toml` with §4 content; create `src/xtrax/` tree stubs; create `ruff.toml`; create `tests/` mirror stubs.
Gate: `uv run ruff check .` exits 0; `uv run pytest --collect-only` exits 0 with 0 tests.

### Phase 1: Transforms
**Task 1.1** — `safe_map` + tests. Gate: `uv run pytest tests/transforms/test_map.py -v`
  - batch_size=None → vmap path
  - batch_size=200, n=100 → vmap path
  - batch_size=50, n=100 (divisible) → lax.map path, result matches vmap
  - batch_size=30, n=100 (non-divisible) → ValueError
  - pytree xs; shape invariant

**Task 1.2** — `safe_scan` + tests. Gate: `uv run pytest tests/transforms/test_scan.py -v`

Phase gate: `uv run pytest tests/transforms/ -v` all pass; ruff clean.

### Phase 2: Tiling
**Task 2.1** — Strategy types. Gate: 4 variants instantiate; isinstance checks pass; Scan has no batch_size field.
**Task 2.2** — BatchPlan/BatchPlanner (deps: 2.1). Gate: Vmap/SafeMap/DedupGather selection per rules; memory_estimator override tested; Scan never returned by plan.
**Task 2.3** — Iterators (deps: Phase 1, 2.1). Gate: VmapIterator == SafeMapIterator when batch_size>=n; SafeMapIterator propagates ValueError for non-divisible.
**Task 2.4** — Dedup. Gate: get_k_bucket integer bit-length correct; k_bucket=3 raises ValueError; k_bucket < max_unique raises ValueError.
**Task 2.5** — Dispatch (deps: 2.1–2.4, Phase 1). Gate: each strategy dispatches per §3.7; Scan with init=None raises ValueError; DedupGather unpacking correct.
Phase gate: `uv run pytest tests/tiling/ -v` all pass; ruff clean.

### Phase 3: Safety and Stages (parallel)
**Task 3.1** — safe_norm, safe_reciprocal. Gate: grad at zero is finite; reciprocal(0.0) = 1/eps.
**Task 3.2** — SafetyManager, with_safety. Gate: enabled=False is strict identity (checked via `fn is wrapped_fn`); NaN detected when enabled.
**Task 3.3** — PreemptionHandler. Gate: SIGUSR1 triggers save_fn once; rank≠0 does not call save_fn; register() idempotent.
**Task 3.4** — Stage protocols. Gate: lambdas pass isinstance for all three protocols.
**Task 3.5** — StageBundle (deps: 3.4). Gate: active_stages/has_stage correct; valid eqx.Module pytree; no __call__.
Phase gate: `uv run pytest tests/safety/ tests/stages/ -v` all pass.

### Phase 4: Training Core and Data
**Task 4.1** — ResumableState, LossFunction, Callback + tests. Gate: step is jnp.int32 Array leaf; isinstance checks pass.
**Task 4.2** — Trainer (deps: 4.1). Gate: step+1 per call; loss decreases 10 steps trivial regression; params passed as 3rd arg to optimizer.update (verified via optax mock).
**Task 4.3** — SafetyTrainStep, create_train_step (deps: 4.2, 3.2). Gate: factory returns correct type; both expose identical .step() signature; safety=True detects NaN.
**Task 4.4** — Loss combinators (deps: 4.1). Gate: WeightedLoss=weight×loss; MultiTaskLoss=sum of weighted; both pass isinstance(LossFunction); tuple-length mismatch raises Python assert.
**Task 4.5** — Optimizer utilities. Gate: no_bias_wd_mask returns False for 1-D; make_optimizer clip-then-base order; adamw_with_schedule passes mask=wd_mask (verified via optax mock); partition_labels produces string-leaf pytree matching model structure.
**Task 4.6** — accumulate_grads (deps: 4.1, 4.2). Gate: lax.scan in jaxpr; atol=1e-5 vs full-batch (4 equal microbatches of 25); unequal-size raises ValueError.
**Task 4.7** — DataModule + streaming. Gate: train_iter yields correct shapes; BucketIterator mismatched sizes raises ValueError at construction; distributed=True without init_dist raises RuntimeError; create_distributed_pipeline raises on non-divisible global_batch_size.
Phase gate: `uv run pytest tests/training/ tests/data/ -v` all pass.

### Phase 5: Distributed and Checkpoint
**Task 5.1** — ShardingPolicy, mesh helpers. Gate: first-match-wins with overlapping patterns; get_hardware_mesh_profile returns required keys on CPU; get_device_mesh raises ValueError for shape mismatch.
**Task 5.2** — init_dist. Gate: idempotent same args; different args raises RuntimeError; SLURM env mocked; localhost fallback works.
**Task 5.3** — Checkpoint wrappers (deps: 4.1). Gate: save/load round-trip preserves all ResumableState fields; empty dir raises FileNotFoundError; save_checkpoint called outside JIT.
Phase gate: `uv run pytest tests/distributed/ tests/checkpoint/ -v` all pass.

### Phase 6: Engine and IO
**Task 6.1** — async_indexed_stream, BoundedCallbackHandler. Gate: correct (index, item) pairs; concurrency bounded by semaphore; wait_all flushes; exception propagation from iterable reaches consumer.
**Task 6.2** — Engine (deps: 4.1–4.3, 4.7, 5.3, 6.1). Gate:
  - fit over 2 epochs → state.step == num_batches * 2
  - All 7 Callback hooks fired in correct order
  - eval returns metrics dict; state.step unchanged
  - eval fires validation_callbacks only
  - Checkpoint written + waited after each epoch when checkpoint_dir set
  - eval wraps model in inference_mode
  - fit_sync works via asyncio.run
Phase gate: `uv run pytest tests/ -v` full suite; ruff clean.

### Phase 7: Novel Additions
**Task 7.1** — StageBundle __init_subclass__ validation. Gate: invalid field type raises TypeError at class definition.
**Task 7.2** — MultiTaskLoss dynamic weight schedule (weight_schedule: Callable[[int], Array] | None). Semver → 0.2.0. Gate: existing tests pass; schedule applied per step.
Phase gate: `uv run pytest tests/ -v` all pass; ruff clean.

---

## 6. Acceptance Criteria

- `safe_map(fn, xs, batch_size=200)` with n=100 (vmap path) matches `jax.vmap(fn)(xs)`.
- `safe_map(fn, xs, batch_size=50)` with n=100 (lax.map path, divisible) matches `jax.vmap(fn)(xs)`.
- `safe_map(fn, xs, batch_size=30)` with n=100 (non-divisible) raises `ValueError`.
- `safe_scan` with length=0 raises `ValueError` before JAX tracing.
- `AxisSpec(cardinality=100, batch_size=200)` → default planner selects `Vmap`.
- `AxisSpec(cardinality=300, batch_size=200, dedup_eligible=False)` → default planner selects `SafeMap(batch_size=200)`.
- `AxisSpec(cardinality=100, dedup_eligible=True)` → default planner selects `DedupGather`.
- `get_k_bucket(7)` returns 8; `get_k_bucket(8)` returns 8; `get_k_bucket(0)` raises `ValueError`.
- `DedupSpec(k_bucket=3, max_unique=3)` raises `ValueError` (not power of 2).
- `DedupSpec(k_bucket=4, max_unique=8)` raises `ValueError` (k_bucket < max_unique).
- `SafetyManager(enabled=False)` + `with_safety(fn, mgr)` → returned callable is `fn` (identity).
- `SafetyManager(enabled=True)` wrapping NaN-producing jitted fn → Python exception on host.
- `ResumableState` saved then loaded → all fields bit-identical within JAX dtype precision.
- `accumulate_grads` with 4 microbatches of size 25 matches full-batch grad within `atol=1e-5`.
- `accumulate_grads` with unequal microbatch sizes raises `ValueError`.
- `make_axis_dispatch(Scan(transition=fn), fn_ignored, xs, init=None)` raises `ValueError`.
- `Engine.fit` over 2 epochs with recorder: on_train_start ×1, on_epoch_start ×2, on_step_start ×N, on_step_end ×N, on_epoch_end ×2, on_train_end ×1.
- `init_dist()` twice with identical args → no exception, initialized once.
- `ShardingPolicy(rules=[("weight", PartitionSpec("devices")), (".*", PartitionSpec())])` with path `"encoder.weight"` → `PartitionSpec("devices")`.
- `PreemptionHandler(save_fn, rank=1)` + SIGUSR1 → `save_fn` not called.
- `DataModule(distributed=True)` without `init_dist()` → `train_iter()` raises `RuntimeError`.
- `BucketIterator(boundaries=[10,20], batch_sizes=[8,16])` raises `ValueError` (batch_sizes len mismatch).
- `create_distributed_pipeline(ds, global_batch_size=100, num_devices=3)` raises `ValueError` (non-divisible).
- `no_bias_wd_mask({"w": jnp.ones((10,10)), "b": jnp.zeros(10)})` → `{"w": True, "b": False}`.

---

## 7. Risk Table

| Risk | Severity | Mitigation |
|------|----------|------------|
| `jax.lax.map(batch_size=...)` not at pinned floor | Critical | Floor bumped to jax>=0.4.36; verify in CI with `assert jax.__version__ >= "0.4.36"` |
| Grain API instability across 0.2.x | Medium | Pin `grain>=0.2.0,<1.0`; isolate all Grain calls to `data/`; test `create_distributed_pipeline` with mocked dataset |
| Orbax async checkpoint race | Medium | `manager.wait_until_finished()` after every `save_checkpoint` in Engine; trade-off: serializes async benefit (acceptable for correctness) |
| `checkify` import path changes | Low | Isolated to `safety/checkify.py`; only import changes on migration |
| `accumulate_grads` float32 drift | Medium | Gate uses atol=1e-5; drift documented; equal-microbatch precondition enforced |
| SLURM auto-discovery on non-SLURM clusters | Medium | Localhost single-process fallback; tested with mocked empty SLURM env |
| `Engine.fit` asyncio + JAX GIL blocking | Medium | Callbacks must be non-blocking; document clearly; `BoundedCallbackHandler` semaphore prevents callback pile-up |
| `StageBundle` topology branching in JAX trace | High (unmitigated) | Caller precondition only — no runtime enforcement; document with capital-WARNING; Phase 7 `__init_subclass__` only validates field types, not call-site usage |
| `MultiTaskLoss` shape mismatch silent at trace | Medium | Python `assert` before JAX operations |
| `no_bias_wd_mask` excludes 1-D embedding weights | Low | Document ndim!=1 rule; consumers override with custom mask |
| `partition_labels` path format depends on eqx version | Low | Pin equinox minor; test against eqx's own pytree path representation |
| `save_checkpoint` called inside JIT | Medium | Document "must be called Python-side" precondition; int(state.step) will fail inside trace, providing a natural error |

---

## 8. Open Questions (resolved)

1. `Trainer.optimizer` field — resolved: dynamic pytree field (eqx.Module leaf).
2. `Engine` from-config factory — deferred to future spec revision.
3. `DataModule.collate_fn` — resolved: optional; default `jax.tree.map(jnp.stack, items)`.
4. `Engine.eval` metrics aggregation — resolved: `jnp.mean` per metric over all batches; scalar per metric.
5. `accumulate_grads` filter_spec — resolved: defaults to `eqx.is_array`.
6. `Scan` planner selection — resolved: Scan is caller-constructed only; BatchPlanner never selects it.
7. `memory_estimator` semantics — resolved: specified in §3.4; queries device memory stats when provided.
8. `Engine.fit` vs `DataModule.num_epochs` precedence — resolved: Engine.fit(num_epochs) is authoritative; DataModule.num_epochs controls dataset cycling.
9. `safe_map` non-divisible remainder — resolved: raises ValueError; divisibility is a precondition.
10. `get_k_bucket` FP fragility — resolved: mandate integer bit_length implementation.
11. `safe_scan` init=None — resolved: `None` is a legal lax.scan carry; safe_scan intentionally does NOT reject it. The init-required check lives in §3.7 Scan dispatch only. See §3.2 invariants.
12. `BatchPlanner` rule 4 deferred failure — resolved: plan warns but does not raise; `make_axis_dispatch` will raise ValueError at dispatch time. Documented as deferred-failure contract in §3.4.

---

## 9. References

- Equinox: https://docs.kidger.site/equinox/
- Optax: https://optax.readthedocs.io/
- Orbax: https://orbax.readthedocs.io/
- Grain: https://github.com/google/grain
- JAX distributed: https://jax.readthedocs.io/en/latest/multi_process.html
- JAX checkify: https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html
- Source extraction: `prxteinmpnn/src/aminx/` (tiling), `jaxbeans/src/jaxbeans/` (training stack)
- task_id: `260604_xtrax-shape`

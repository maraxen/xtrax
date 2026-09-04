# Architecture

xtrax is organized into eleven composable subpackages, each with a single responsibility.

## Package Organization

```
xtrax/
  transforms/    — Low-level JAX wrappers: safe_map, safe_scan
  tiling/        — Axis strategies and batching: AxisSpec, BatchPlanner, dispatch
  stages/        — Generic pipeline protocols and bundles
  training/      — Training loop: Trainer, ResumableState, losses, optimizers, callbacks
  engine/        — High-level orchestrator: Engine, async iteration, checkpoint coordination
  data/          — Data pipelines: DataModule, Grain integration, distributed sharding
  distributed/   — Multi-process coordination: init_dist, mesh helpers, sharding policies
  checkpoint/    — Checkpoint management: Orbax wrappers, save/load
  safety/        — Numerics and debugging: safe_norm, NaN/Inf detection, preemption handling
  sparse/        — Sparse matrix management: SparsePolicy, SparseMaskManager
  io/            — I/O utilities: async callbacks, bounded concurrency
```

## Design Philosophy

### 1. Composition Over Inheritance

xtrax uses **protocols** (structural subtyping) instead of base classes. This lets domain libraries (e.g., your model library) define their own types without depending on xtrax's class hierarchy.

For example, `LossFunction` is a protocol:

```python
class LossFunction(Protocol):
    def __call__(self, predictions: PyTree, targets: PyTree) -> Array:
        ...
```

Any callable that returns a scalar is a valid loss — no inheritance needed.
The same structural-subtyping principle applies to xtrax's boundary
types (pytree-of-arrays, pytree-of-`ShapeDtypeStruct`); see
{doc}`dependency-boundaries` for how that interlingua, plus xtrax's
substrate/adapter and JAX-namespace-stability policies, are scoped.

### 2. Static Shape for Recompile Safety

JAX's JIT compiler traces and compiles based on shape. xtrax is aggressive about static shapes:

- `AxisSpec.default_batch_size` is static (passed to `@jax.vmap` or `@jax.lax.map`).
- `SparsePolicy.nse_budget` is static (determines BCOO format).
- `DataModule.batch_size` is static.

This prevents "shape explosion" — silent recompilations every time batch size changes or sparsity patterns shift.

### 3. Immutable State, Pure Functions

Training state is never mutated in place. Every `.step()` call is a pure function:

```python
new_state, metrics = trainer.step(state, batch)  # state is unchanged
```

This enables safe checkpointing, reproducibility, and distributed training without synchronization headaches.

### 4. Delegation, Not Reimplementation

xtrax doesn't reimplement solved problems:

- **Equinox** handles JAX modules and AD filters.
- **Optax** handles optimizers, schedules, and gradient transforms.
- **Orbax** handles checkpointing and async saves.
- **Grain** handles data loading and sharding.
- **JAX** handles sparse arrays and distributed primitives.

xtrax provides a **composable glue layer** that makes these libraries work together seamlessly.

## Layer 1: Transforms

**Module**: `xtrax.transforms`

Provides safe wrappers around JAX's map and scan operations:

- `safe_map(fn, xs, batch_size)`: Vmap if xs is small; chunk and loop otherwise. Raises `ValueError` if xs leading dimension isn't divisible by batch_size.
- `safe_scan(fn, init, xs)`: Identical to `jax.lax.scan`; wrapper validates pre-trace.

These are low-level building blocks used by the tiling layer.

## Layer 2: Tiling and Axis Strategies

**Module**: `xtrax.tiling`

Defines how to schedule computation across axes:

- `AxisSpec`: Declares an axis with cardinality, batch size, and eligibility for deduplication.
- `BatchPlanner`: Selects a strategy (Vmap, SafeMap, DedupGather) for each axis automatically.
- `make_axis_dispatch()`: Executes the selected strategy on data.

Example:

```python
specs = [AxisSpec(name="batch", cardinality=1000, default_batch_size=256)]
plan = BatchPlanner().plan(specs)
result = make_axis_dispatch(plan.decisions[0].strategy, my_fn, my_data)
```

### Strategy Selection Rules

The `BatchPlanner` uses these rules (in order):

1. If `dedup_eligible=True`, use `DedupGather`.
2. If `cardinality <= default_batch_size`, use `Vmap`.
3. If `cardinality > default_batch_size` and divisible, use `SafeMap`.
4. If `cardinality > default_batch_size` and NOT divisible, use `SafeMap` with a warning (will error at dispatch time).

For custom strategies (e.g., Scan for RNNs), construct `AxisDecision` directly.

## Layer 3: Stages and Pipelines

**Module**: `xtrax.stages`

Defines generic pipeline protocols for composable transformations:

- `TransformFn`: A stateless transformation (`x → y`).
- `RollingFn`: Stateful computation with carry (`carry, x → carry, y`).
- `FuseFn`: Combine multiple results (`[y1, y2, ...] → z`).

`StageBundle` is a typed container for optional pipeline stages. Domain libraries (e.g., your model library) subclass it:

```python
class MyPipeline(StageBundle):
    encode: Optional[TransformFn]
    process: Optional[RollingFn]
    decode: Optional[TransformFn]
```

At Python dispatch time, check which stages are active and build the computation graph accordingly. Never branch on stage presence inside JAX traces — that violates JIT's purity model.

## Layer 4: Training Core

**Module**: `xtrax.training`

The heart of xtrax. Provides:

- `ResumableState`: Immutable bundle of model, optimizer state, step counter, and metadata.
- `Trainer`: Wraps loss + optimizer. The `.step()` method is JIT-compiled.
- `SafetyTrainStep`: Wraps `Trainer` with NaN/Inf detection (optional).
- Loss combinators: `WeightedLoss`, `MultiTaskLoss`.
- Optimizer helpers: `adamw_with_schedule`, `no_bias_wd_mask`, `partition_labels`.
- Gradient accumulation: `accumulate_grads` for microbatch training.

All loss functions conform to the `LossFunction` protocol. All training loops are pure functions of `(state, batch) → (state, metrics)`.

## Layer 5: Engine and Orchestration

**Module**: `xtrax.engine`

High-level training harness. `Engine` orchestrates:

- **Epoch loop**: Multiple passes over data.
- **Callbacks**: Lifecycle hooks (`on_train_start`, `on_epoch_end`, etc.).
- **Checkpointing**: Automatic save after each epoch.
- **Validation**: Separate eval loop with `validation_callbacks`.
- **Async coordination**: Prevents callback pile-up with bounded concurrency.

Example:

```python
engine = Engine(trainer=trainer, callbacks=[LogMetrics(), EarlyStopping()])
final_state = await engine.fit(state, data, num_epochs=10, checkpoint_dir="./ckpts")
```

`Engine.fit()` is async; use `fit_sync()` for blocking code.

## Layer 6: Data Pipelines

**Module**: `xtrax.data`

Couples datasets with batch configuration:

- `DataModule`: Wraps a dataset with batch_size, num_epochs, seed, and collate function.
- `BucketIterator`: Sorts data by length and batches into variable-sized groups (e.g., short sequences in larger batches, long sequences in smaller batches for fair memory usage).
- `create_distributed_pipeline`: Uses Grain to shard data across processes.

All iterators yield PyTrees of batches, compatible with `Trainer.step()`.

## Layer 7: Distributed Training

**Module**: `xtrax.distributed`

Multi-process training via JAX's SPMD APIs:

- `init_dist()`: Initialize distributed communication (idempotent).
- `LogicalMesh`: Describes device topology.
- `ShardingPolicy`: Pattern-based rules for partitioning parameters across the mesh.
- `get_device_mesh()`: Helpers for constructing mesh shapes.

Delegation: xtrax just wraps JAX's distributed primitives. The actual sharding is done by Equinox and JAX's `device_put_sharded()`.

## Layer 8: Checkpointing

**Module**: `xtrax.checkpoint`

Orbax integration for checkpointing:

- `save_checkpoint()`: Serialize state to disk (async in Orbax, synchronous in this interface).
- `load_checkpoint()`: Deserialize state from disk.

Orbax handles all the heavy lifting (async I/O, cloud storage, versioning). xtrax just provides convenient wrappers.

## Layer 9: Safety and Debugging

**Module**: `xtrax.safety`

Optional guards and numeric utilities:

- `safe_norm(x)`: Compute norm with epsilon to prevent NaN at zero.
- `safe_reciprocal(x)`: Compute `1/x` safely.
- `SafetyManager` + `with_safety()`: Wrap JIT functions with NaN/Inf detection via `jax.experimental.checkify`.
- `PreemptionHandler`: Catch SIGUSR1/SIGTERM and trigger checkpointing.

All optional, zero overhead when disabled.

## Layer 10: Sparse Matrix Management

**Module**: `xtrax.sparse`

Structured sparsification for inference and constrained training:

- `SparseConfig`: Specifies `nse_budget`, update schedule, and fallback mode.
- `SparsePolicy`: Computes and applies sparse masks.
- `SparseMaskManager`: Python-side coordinator that recomputes masks on schedule.

Masks are step-scheduled and step-cached to prevent XLA retrace. When true nonzeros exceed the budget, either fall back to dense or error — your choice.

## Layer 11: I/O and Callbacks

**Module**: `xtrax.io`

Utilities for non-blocking I/O:

- `async_indexed_stream()`: Async generator that prefetches from a blocking iterable.
- `BoundedCallbackHandler`: Bounded concurrency coroutine executor (semaphore pattern).

Used internally by `Engine` to dispatch callbacks asynchronously without blocking the training loop.

## Data Flow: A Training Step

Here's how a batch moves through xtrax:

```
1. DataModule yields batch (PyTree)
2. Engine.fit() calls trainer.step(state, batch)
3. Trainer._compute_loss(state.model, batch) inside filter_jit
4. JAX traces: forward pass, backward, optimizer update
5. ResumableState returned with incremented step
6. Callbacks fire (Python-side, outside jit)
7. Engine.fit() checks checkpoint schedule and saves if needed
8. Loop back to step 1
```

All computation inside `filter_jit` is pure JAX. All I/O, callbacks, and state coordination happen Python-side.

## Concurrency and Async

`Engine.fit()` is async (coroutine) but not concurrent within a single process:

- Trainer steps run sequentially (one per batch).
- Callbacks run sequentially in their own async tasks, bounded by a semaphore.
- Checkpointing is async in Orbax but synchronously awaited in xtrax (trade-off: simpler semantics).

For true parallelism, use multiple processes (via `init_dist()`) or distribute batches to multiple threads (outside xtrax's scope).

## Type Safety

xtrax uses type hints extensively:

- **Protocols** (`LossFunction`, `Callback`, `TransformFn`) define contracts without base classes.
- **Generics** (`PyTree`, `Array`, `ResumableState`) clarify pytree shapes.
- **Literal types** (`fallback_mode: Literal["dense_mask", "error"]`) catch mistakes at static-analysis time.

All public APIs are fully typed. Use `ty` or `mypy` to catch bugs before runtime.

## Performance Characteristics

- **Trainer.step()**: One JIT-compiled loop per model per batch-shape. Negligible Python overhead.
- **Engine.fit()**: Callback overhead is bounded by the semaphore; typical overhead <1%.
- **Distributed data**: Grain handles sharding efficiently; xtrax adds <1% overhead.
- **Sparse masks**: Zero overhead when sparsity is constant (cached masks). Retrace only on schedule.

xtrax is a thin, composable layer — performance is dominated by JAX and your model, not by xtrax.

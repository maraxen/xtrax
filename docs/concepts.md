# Core Concepts

This page introduces the key ideas that make xtrax powerful.

## Composable Training Infrastructure

xtrax builds a modular training stack on top of JAX and Equinox. Rather than a rigid training loop, we provide composable primitives:

- **Trainer**: A lightweight wrapper around your model and loss. Computes gradients, applies updates, and returns the new state. 100% pure JAX — JIT-compiled every time.
- **ResumableState**: Bundles model, optimizer state, training step counter, and arbitrary metadata. Designed for checkpointing and resumption.
- **Engine**: A higher-level orchestrator that handles multiple epochs, callbacks, validation, and checkpointing automatically.

This separation lets you choose your level of abstraction: start with `Trainer.step()` for fine-grained control, or use `Engine.fit()` for the full training harness.

## Axis Strategies: Composable Tiling

Real models and datasets have multiple axes that must be scheduled for compute. xtrax provides four **axis strategies** to handle them:

1. **Vmap**: Vectorize the entire axis using `jax.vmap`. Fastest when memory permits.
2. **SafeMap**: Chunk the axis into batches and loop (`jax.lax.map` with batch-wise vmap). When an axis is too large to vectorize, SafeMap chunks it automatically.
3. **Scan**: Carry-bearing sequential iteration. For axes that must maintain state across steps (e.g., RNN hidden states).
4. **DedupGather**: When an axis has repeated elements, deduplicate them, compute on unique values only, then scatter results. Transparent speedup for skewed distributions.

You declare an axis with `AxisSpec` — specify its cardinality, preferred batch size, and whether deduplication is eligible. The `BatchPlanner` automatically selects the best strategy:

```python
from xtrax import AxisSpec, BatchPlanner

specs = [
    AxisSpec(name="batch", cardinality=1024, default_batch_size=256),
    AxisSpec(name="time", cardinality=100, default_batch_size=100),
]

planner = BatchPlanner()
plan = planner.plan(specs)

for decision in plan.decisions:
    print(f"{decision.spec.name}: {decision.strategy}")
```

The result is a `BatchPlan` — an executable schedule that can be dispatched via `make_axis_dispatch()`.

## Sparse Inference with Fixed Sparsity

xtrax provides **structured sparse matrix multiplication** for inference, inference-only domains, and memory-constrained training. The sparse module applies a learned sparsity pattern with a fixed "number of stored elements" (nse) budget — guaranteeing constant XLA compile shapes even as sparsity patterns change.

Key concept: **masks are step-scheduled**. Sparse masks can be recomputed on a fixed schedule (e.g., every N steps), and xtrax handles the transition safely:

- `SparsePolicy` defines which elements to keep (top-k by magnitude).
- `SparseMaskManager` applies masks on a schedule and caches them Python-side.
- When true nonzeros exceed the budget, fall back to dense or error out — your choice.

This is transparent to the model: just wrap your weight matrix with a mask, and xtrax handles layout, padding, and recompilation safety.

## Distributed Training with Mesh Sharding

xtrax integrates JAX's multi-process SPMD sharding. After calling `init_dist()`, models are automatically partitioned across devices:

```python
from xtrax import init_dist, LogicalMesh

init_dist()  # Initialize multi-process communication

# Define a device mesh (e.g., 2×4 devices)
mesh_shape = (2, 4)
mesh = LogicalMesh(shape=mesh_shape, axis_names=("batch", "model"))

# Apply sharding policies to your model
from xtrax import ShardingPolicy
policy = ShardingPolicy(
    rules=[
        ("encoder.weight", PartitionSpec("model", None)),
        ("decoder.weight", PartitionSpec(None, "model")),
    ]
)
```

Sharding policies are pattern-based: the first rule matching a parameter path wins. Full replication is the default for unmatched parameters.

## Loss Composition and Multi-Task Learning

xtrax provides utilities for composing losses:

- **WeightedLoss**: Scale any loss by a fixed weight.
- **MultiTaskLoss**: Sum multiple weighted losses. Useful for multi-task learning.

Both conform to the `LossFunction` protocol, so they work anywhere a loss is expected:

```python
from xtrax import LossFunction, MultiTaskLoss, WeightedLoss

task1_loss = WeightedLoss(my_loss_fn_1, weight=0.6)
task2_loss = WeightedLoss(my_loss_fn_2, weight=0.4)
combined = MultiTaskLoss(losses=(task1_loss, task2_loss))

trainer = Trainer(loss_fn=combined, optimizer=optimizer)
```

## Safety Checks: NaN/Inf Detection

Training is fragile. Gradients can explode, weights can underflow, and bugs hide in the details. xtrax provides optional safety checks:

```python
from xtrax import create_train_step

trainer = create_train_step(
    loss_fn=loss,
    optimizer=optimizer,
    safety=True,  # Enable NaN/Inf detection
)
```

With `safety=True`, every backward pass is guarded by `jax.experimental.checkify`, which flags NaN or Inf values in gradients and raises a Python exception on the host. Zero performance cost when disabled.

## Data Pipelines with Grain

xtrax wraps Google's Grain library for efficient data loading with built-in distributed support. A `DataModule` couples your dataset with batch configuration:

```python
from xtrax import DataModule

data = DataModule(
    dataset=my_dataset,
    batch_size=32,
    num_epochs=10,
    seed=42,
    distributed=True,  # Auto-shard across processes
)

for batch in data.train_iter():
    # Each process sees a different subset of the data
    state, metrics = trainer.step(state, batch)
```

## Checkpointing and Resumption

Training interruptions are inevitable. xtrax delegates checkpointing to Orbax (Google's checkpoint library) and provides a simple interface:

```python
from xtrax import save_checkpoint, load_checkpoint

# Save after each epoch
checkpoint_manager = get_checkpoint_manager(directory="./checkpoints")
save_checkpoint(checkpoint_manager, state, step=epoch)

# Resume from a checkpoint
state = load_checkpoint(checkpoint_manager, template_state, step=latest)
```

`ResumableState` includes a `step` counter and `extras` dict for custom metadata, making it easy to resume training exactly where you left off.

## Callbacks and Monitoring

The `Engine` fires lifecycle callbacks:

```python
from xtrax import Callback

class LogMetricsCallback(Callback):
    def on_step_end(self, state: ResumableState, metrics: dict) -> None:
        if state.step % 100 == 0:
            print(f"Step {state.step}: {metrics['loss']:.4f}")

engine = Engine(trainer=trainer, callbacks=[LogMetricsCallback()])
```

All 7 callback hooks (`on_train_start`, `on_epoch_start`, `on_step_start`, etc.) fire in order, and they run Python-side — perfect for logging, early stopping, or metric aggregation.

## Pure Functions and Immutable State

Every xtrax training step is a pure function: same inputs → same outputs, no side effects. This is JAX's model, and xtrax follows it strictly. `ResumableState` is immutable; you never mutate a state in place. This design enables:

- **Reproducibility**: Train the same model twice with the same seed and get identical results.
- **Checkpointing**: Save and resume is trivial when state is immutable.
- **Distributed training**: No synchronization headaches from mutable state.
- **Debugging**: Replay any step by keeping the inputs.

Every training loop is a fold: `state_0 → step 1 → state_1 → step 2 → state_2 → ...`.

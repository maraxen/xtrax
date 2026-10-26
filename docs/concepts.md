# Core Concepts

This page introduces xtrax's main abstractions and how they fit together. For why the
tiling layer exists at all, see [Why xtrax exists](why-xtrax.md).

## Composable Training Infrastructure

xtrax builds a modular training stack on top of JAX and Equinox. Rather than a rigid training loop, we provide composable primitives:

- **Trainer**: A lightweight wrapper around your model and loss. Computes gradients, applies updates, and returns the new state. 100% pure JAX — JIT-compiled every time.
- **ResumableState**: Bundles model, optimizer state, training step counter, and arbitrary metadata. Designed for checkpointing and resumption.
- **Engine**: A higher-level orchestrator that handles multiple epochs, callbacks, validation, and checkpointing automatically.

This separation lets you choose your level of abstraction: start with `Trainer.step()` for fine-grained control, or use `Engine.fit()` for the full training harness.

## Axis Strategies: Composable Tiling

Real models and datasets have multiple axes that must be scheduled for compute. xtrax provides four **axis strategies** to handle them:

1. **Vmap**: Vectorize the entire axis using `jax.vmap`. Fastest when memory permits.
2. **ChunkedMap**: Chunk the axis into batches and loop (`jax.lax.map` with batch-wise vmap). When an axis is too large to vectorize, ChunkedMap chunks it automatically.
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

## Distributed Training with Mesh Sharding

xtrax builds on JAX's multi-process SPMD sharding. Start the processes with `jax.distributed.initialize()` and build the device mesh with `jax.make_mesh`:

```python
import jax

jax.distributed.initialize()  # Initialize multi-process communication

# Define a device mesh (e.g., 2×4 devices)
mesh = jax.make_mesh((2, 4), ("batch", "model"))
```

:::{note}
`init_dist`, `is_distributed`, `get_device_mesh`, `get_hardware_mesh_profile`, and `ShardingPolicy` are deprecated in the probe release (xtrax #6107): each warns on access, names its replacement, and otherwise behaves as before. `codemods/thin-wrappers-to-jax/` rewrites the call sites that have a one-line replacement and reports the rest. `DataModule(distributed=True)` now checks `jax.process_count() > 1`.
:::

## Loss Composition and Multi-Task Learning

`ComposedLoss` sums named loss terms with traced weights, so changing a weight does not recompile a jitted step. It returns `(loss, aux)`, with each term's unweighted value in `aux`; pair it with `Trainer(has_aux=True)`:

```python
from xtrax import ComposedLoss

combined = ComposedLoss(
    terms=[("task1", my_loss_fn_1), ("task2", my_loss_fn_2)],
    weights=[0.6, 0.4],
)

trainer = Trainer(loss_fn=combined, optimizer=optimizer, has_aux=True)
```

:::{note}
`WeightedLoss` and `MultiTaskLoss` are deprecated in the probe release (xtrax #6107): each warns on access and otherwise behaves as before.
:::

## Safety Checks: NaN/Inf Detection

Training is fragile. Gradients can explode, weights can underflow, and bugs hide in the details. `jax.experimental.checkify` flags NaN and division-by-zero in a jitted step and raises on the host:

```python
import jax
from jax.experimental import checkify

trainer = create_train_step(loss_fn=loss, optimizer=optimizer)
checked = jax.jit(checkify.checkify(trainer.step, errors=checkify.float_checks))

err, (state, metrics) = checked(state, batch)
err.throw()
```

`checkify.float_checks` covers NaN and division by zero, not an `Inf` that comes from overflow.

:::{note}
`create_train_step(..., safety=True)`, `SafetyTrainStep`, `SafetyManager`, and `with_safety` are deprecated in the probe release (xtrax #6107): each warns and otherwise behaves as before. `safety=False` is not deprecated.
:::

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

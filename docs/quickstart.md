# Quick Start

Get up and running with xtrax in 30 seconds.

## Installation

```bash
pip install xtrax
```

Requires Python 3.13+ and JAX.

## Your First Training Loop

Here's a complete example: define a model, create a trainer, and run a training step.

```python
from xtrax import Trainer, Engine, ResumableState, adamw_with_schedule
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

# Define a simple model (two-layer MLP)
key = jax.random.key(0)
key, subkey = jax.random.split(key)

model = eqx.nn.MLP(
    input_size=64,
    output_size=1,
    width_size=128,
    depth=2,
    key=subkey,
)

# Define a loss function
def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

# Create an optimizer with learning rate schedule
optimizer = adamw_with_schedule(
    peak_lr=1e-3,
    warmup_steps=100,
    total_steps=1000,
)

# Create the trainer
trainer = Trainer(loss_fn=mse_loss, optimizer=optimizer)

# Initialize training state
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
state = ResumableState(
    step=jnp.int32(0),
    key=key,
    model=model,
    opt_state=opt_state,
)

# Create a batch
batch = {
    "inputs": jnp.ones((32, 64)),
    "targets": jnp.ones((32, 1)),
}

# Run a training step
new_state, metrics = trainer.step(state, batch)
print(f"Loss: {metrics['loss']:.4f}")
```

## What's Happening

1. **Model definition**: We use Equinox to define a JAX model.
2. **Loss function**: Any callable that returns a scalar loss.
3. **Optimizer**: xtrax provides helpers like `adamw_with_schedule` that wrap Optax.
4. **Trainer**: Wraps the model, loss, and optimizer. The `.step()` method is JIT-compiled.
5. **State**: `ResumableState` bundles model parameters, optimizer state, and training metadata.
6. **Training step**: Call `.step(state, batch)` to compute gradients, update parameters, and return the new state.

## Next Steps

- **Distributed training**: Use `Engine` for multi-process training with automatic checkpointing.
- **Composable batching**: Use `AxisSpec` and `BatchPlanner` to apply vmap, batch-wise loops, or structured sparsity.
- **Data pipelines**: Load data with `DataModule` and `create_distributed_pipeline`.
- **Safety checks**: Enable NaN/Inf detection with `create_train_step(safety=True)`.

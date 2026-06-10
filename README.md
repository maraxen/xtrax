# xtrax

High-performance composable JAX library for training and inference.

[![PyPI version](https://img.shields.io/pypi/v/xtrax.svg)](https://pypi.org/project/xtrax/)
[![CI Status](https://github.com/maraxen/xtrax/actions/workflows/ci.yml/badge.svg)](https://github.com/maraxen/xtrax/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/xtrax/badge/?version=latest)](https://xtrax.readthedocs.io/en/latest/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Why xtrax?

- **Composable training steps** — Use `Trainer` or `SafetyTrainStep` with your own loss functions and optimizers
- **Safety-checked arithmetic** — Avoid NaN/Inf propagation with safe operations (`safe_norm`, `safe_reciprocal`)
- **Flexible tiling strategies** — Partition computations with `AxisSpec`, `BatchPlan`, and `Vmap`/`SafeMap` for data and model parallelism
- **Inference sparsification** — Apply structured sparsity masks with `SparseConfig` and `sparsify_model`
- **Distributed helpers** — Initialize and coordinate multi-GPU/TPU training with `init_dist`, `LogicalMesh`, and sharding utilities

## Installation

```bash
pip install xtrax
```

Requires Python 3.13 or later.

## Quick Start

```python
import jax
import optax
from xtrax import Trainer, ResumableState, Engine, save_checkpoint, load_checkpoint

# 1. Create a simple loss function
def loss_fn(model, batch):
    predictions = model(batch["inputs"])
    return jnp.mean((predictions - batch["targets"]) ** 2)

# 2. Set up trainer with optimizer
optimizer = optax.adam(1e-3)
trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)

# 3. Initialize training state
model = ...  # Your equinox model
opt_state = optimizer.init(...)
state = ResumableState(model=model, opt_state=opt_state, step=0)

# 4. Run a training step
new_state, metrics = trainer.step(state, batch={"inputs": x, "targets": y})
print(f"Loss: {metrics['loss']}")
```

For a complete training loop with callbacks and checkpointing, use the `Engine`:

```python
from xtrax import Engine, DataModule

# Create or load a DataModule (must implement train_iter())
data = DataModule(...)

# Create an engine with trainer and optional callbacks
engine = Engine(trainer=trainer)

# Run multi-epoch training with checkpoint saving
final_state = engine.fit_sync(
    state=state,
    data=data,
    num_epochs=10,
    checkpoint_dir="./checkpoints"
)
```

## Getting Results Out

### Streaming Callbacks

Log metrics asynchronously to files or external services:

```python
from xtrax.io import BoundedCallbackHandler, async_indexed_stream

# Create a custom async callback
class LogCallback:
    async def on_step_end(self, state, metrics):
        print(f"Step {state.step}: {metrics}")

# Use in your Engine
engine = Engine(
    trainer=trainer,
    callbacks=[LogCallback()]
)
```

### Checkpoint Save and Load

Save model state and restore for inference or resumption:

```python
from xtrax import save_checkpoint, load_checkpoint

# After training
save_checkpoint(checkpoint_dir="./checkpoints/final", state=final_state)

# Load for inference
restored_state = load_checkpoint(checkpoint_dir="./checkpoints/final")
model = restored_state.model

# Run inference
predictions = model(test_inputs)
```

## Documentation

Full API docs, architecture guides, and advanced examples at [xtrax.readthedocs.io](https://xtrax.readthedocs.io).

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

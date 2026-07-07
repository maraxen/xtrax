# xtrax

[![PyPI - Version](https://img.shields.io/pypi/v/xtrax.svg)](https://pypi.org/project/xtrax/)
[![Tests](https://github.com/maraxen/xtrax/actions/workflows/ci.yml/badge.svg)](https://github.com/maraxen/xtrax/actions)
[![Docs](https://img.shields.io/readthedocs/xtrax.svg)](https://xtrax.readthedocs.io)
[![Coverage](https://img.shields.io/badge/coverage-tier1%2095%25-brightgreen)](https://github.com/maraxen/xtrax)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/maraxen/xtrax/blob/main/LICENSE)

A set of composable building blocks for JAX/Equinox training loops — engine + trainer orchestration, safety-checked steps, axis tiling strategies, inference-time sparsification, distributed/sharding helpers, streaming output callbacks, and orbax checkpointing — extracted from the author's research code.

## Status

**xtrax is alpha, experimental software** built primarily for the author's personal research use. APIs may change without notice between releases; no backward-compatibility guarantees pre-1.0. Issues and pull requests are welcome, but support is best-effort — the project exists first to serve the author's own JAX training workflows.

## Why xtrax?

`jax.jit` specializes and freezes: it traces your function into a program where every shape is a compile-time constant, and XLA plans all memory for that program up front. The compiler makes the program you gave it fast, but it never restructures it — it cannot narrow a `vmap` that doesn't fit in memory, cannot batch ragged inputs without a recompile per shape, and cannot notice that most of a batch is duplicates. Those decisions happen in Python, before tracing — and the code that makes them is exactly what gets copy-pasted between research projects.

xtrax packages that pre-trace layer, plus the conveniences that surround it:

- **Axis tiling** — declare axes with `AxisSpec`; `BatchPlanner` selects `Vmap`, `SafeMap` (chunked via `jax.lax.map`), `Scan`, bucketing, or dedup-gather per axis, and `xtrax explain` reports why
- **Composable training steps** — `Trainer` or `SafetyTrainStep` with your own loss functions and optimizers
- **Safety-checked arithmetic** — opt-in checkify NaN/Inf detection and safe ops (`safe_norm`, `safe_reciprocal`)
- **Inference sparsification** — structured sparsity masks with `SparseConfig` and `sparsify_model`, fixed compile shapes
- **Distributed helpers** — `init_dist`, `LogicalMesh`, and sharding utilities over JAX's native machinery

For the full rationale — why this layer has to live above the JIT boundary — see [Why xtrax exists](https://xtrax.readthedocs.io/en/latest/why-xtrax.html).

## Installation

```bash
pip install xtrax
```

Requires Python 3.13 or later.

## Quick Start

```python
import jax.numpy as jnp
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

## Project links

- [GitHub repository](https://github.com/maraxen/xtrax)
- [Issue tracker](https://github.com/maraxen/xtrax/issues)
- [Changelog](CHANGELOG.md)
- [Contributing guide](CONTRIBUTING.md)
- [Citation metadata](CITATION.cff)

## License

Licensed under the [Apache License 2.0](https://github.com/maraxen/xtrax/blob/main/LICENSE).

## Citation

If you use xtrax in research, please cite it:

```bibtex
@software{xtrax,
  title = {xtrax: High-Performance Composable JAX Training},
  author = {Russo, Marielle},
  version = {0.4.0a2},
  year = {2026},
  url = {https://github.com/maraxen/xtrax}
}
```

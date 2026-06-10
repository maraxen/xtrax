# xtrax

[![PyPI - Version](https://img.shields.io/pypi/v/xtrax.svg)](https://pypi.org/project/xtrax/)
[![Tests](https://github.com/maraxen/xtrax/workflows/Test/badge.svg)](https://github.com/maraxen/xtrax/actions)
[![Docs](https://img.shields.io/readthedocs/xtrax.svg)](https://xtrax.readthedocs.io)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](https://github.com/maraxen/xtrax)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/maraxen/xtrax/blob/main/LICENSE)

High-performance composable JAX training library.

## Why xtrax?

- **Composable architecture**: Modular training pipeline designed for flexibility and extensibility
- **Distributed training**: Built-in support for multi-device and distributed JAX training
- **Sparse operations**: Efficient sparse model support with automatic sparsification
- **Tiling and batching**: Advanced tiling strategies for complex distributed scenarios
- **Production-ready**: Designed for real-world training workflows with checkpointing and safety features

## Install

```bash
pip install xtrax
```

Requires Python 3.13+.

## Quick Start

```python
from xtrax import Trainer, Engine

# Create a simple training configuration
trainer = Trainer()

# Create an engine for distributed execution
engine = Engine()

# Define your loss and training logic
trainer.train(engine=engine)
```

For more details, see the [documentation](https://xtrax.readthedocs.io).

## Features

### Streaming outputs

Use `BoundedCallbackHandler` to capture and stream training outputs during execution. Integrate custom callbacks to monitor metrics and log results in real time.

### Checkpointing

Save and restore training state with orbax checkpoints via `save_checkpoint()` and `load_checkpoint()`. Ensures resumable training across interruptions.

## Documentation

Full API documentation and tutorials are available at [https://xtrax.readthedocs.io](https://xtrax.readthedocs.io).

## License

Licensed under the [Apache License 2.0](https://github.com/maraxen/xtrax/blob/main/LICENSE).

## Citation

If you use xtrax in research, please cite it:

```bibtex
@software{xtrax,
  title = {xtrax: High-Performance Composable JAX Training},
  author = {Russo, Marielle},
  year = {2026},
  url = {https://github.com/maraxen/xtrax}
}
```

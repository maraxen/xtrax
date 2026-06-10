# xtrax

High-performance composable JAX library for advanced training workflows.

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
concepts
architecture
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/overview
api/engine
api/training
api/data
api/tiling
api/sparse
api/distributed
api/transforms
api/safety
api/stages
api/output-sinks
```

```{toctree}
:maxdepth: 1
:caption: Advanced

advanced/debugging
```

## Features

- **Composable**: Build training pipelines from reusable components
- **High-performance**: Native JAX implementation with JIT compilation
- **Distributed**: Built-in support for multi-GPU and multi-node training
- **Type-safe**: Full type hints for IDE support and documentation
- **Production-ready**: Checkpoint management and safety utilities

## Quick Links

- [GitHub Repository](https://github.com/maraxen/xtrax)
- [Issue Tracker](https://github.com/maraxen/xtrax/issues)
- [Changelog](https://github.com/maraxen/xtrax/blob/main/CHANGELOG.md)

## Installation

```bash
pip install xtrax
```

For development:

```bash
uv sync --all-groups
```

## Contributing

See the [Contributing Guide](https://github.com/maraxen/xtrax/blob/main/CONTRIBUTING.md) for how to get involved.

## License

Apache License 2.0 - see [LICENSE](https://github.com/maraxen/xtrax/blob/main/LICENSE) for details.

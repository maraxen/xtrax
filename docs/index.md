# xtrax

Composable building blocks for JAX/Equinox training and batched inference — axis
tiling strategies, trainer/engine orchestration, safety-checked steps, structured
sparsification, sharding helpers, and orbax checkpointing.

```{note}
**xtrax is alpha, experimental software.** Built primarily for the author's personal research use, APIs may change without notice between releases. No backward-compatibility guarantees pre-1.0. Issues and PRs welcome; support is best-effort.
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
why-xtrax
concepts
architecture
dependency-boundaries
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/overview
api/engine
api/training
api/data
api/tiling
api/inference
api/sparse
api/distributed
api/transforms
api/safety
api/stages
api/export
api/output-sinks
api/eda
```

```{toctree}
:maxdepth: 1
:caption: Advanced

advanced/debugging
advanced/eda-guide
```

## What's here

- **Axis tiling** — declare axes with `AxisSpec`; `BatchPlanner` picks vmap, chunked
  map, scan, bucketing, or dedup-gather per axis, and `xtrax explain` reports why.
  For the rationale, see [Why xtrax exists](why-xtrax.md).
- **Training conveniences** — `Trainer` / `Engine` / `ResumableState` for Equinox
  models, with lifecycle callbacks and orbax checkpointing.
- **Safety checks** — opt-in checkify NaN/Inf detection, plus numerically safe ops
  (`safe_norm`, `safe_reciprocal`).
- **Structured sparsity** — fixed-nse sparse inference with stable compile shapes.
- **Sharding and data helpers** — thin wrappers over `jit` auto-sharding /
  `shard_map` and grain.

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

Build documentation locally (warnings are errors):

```bash
just audit-docs-build
```

Or directly:

```bash
uv sync --group docs --extra eda
uv run sphinx-build -W -n -b html docs docs/_build
```

## Contributing

See the [Contributing Guide](https://github.com/maraxen/xtrax/blob/main/CONTRIBUTING.md) for how to get involved.

## License

Apache License 2.0 - see [LICENSE](https://github.com/maraxen/xtrax/blob/main/LICENSE) for details.

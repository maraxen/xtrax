# Contributing to xtrax

Thank you for contributing to xtrax! This document outlines our development workflow, testing standards, and release process.

## Development Setup

Use `uv` for dependency and environment management:

```bash
uv sync --extra dev
```

This installs all runtime and dev dependencies, including testing, linting, type-checking, and documentation tools.

## Running Tests

Run the full test suite with pytest:

```bash
uv run pytest
```

**Coverage requirement:** 90% minimum. Tests are gated in CI.

Current state: 414 tests passing, 96.5% coverage.

## Code Quality

### Linting and Formatting

Check code style with ruff (lint and format):

```bash
uv run ruff check .
uv run ruff format .
```

All code must pass ruff checks before commit.

### Type Checking

Run pyright for type safety:

```bash
uv run pyright
```

All code must pass type checking.

## Building Documentation

Build docs locally with Sphinx:

```bash
uv sync --only-group docs
uv run sphinx-build -W -n -b html docs docs/_build
```

- `-W`: Treat warnings as errors
- `-n`: Show warnings about missing references
- `-b html`: Build HTML output

Documentation is hosted on ReadTheDocs and published on every commit to main.

## Release Process

Releases are automated:

1. **Tag a release**: Create a git tag matching `v*` (e.g., `v0.2.0`)
2. **Automated workflow**: `.github/workflows/publish.yml` triggers on tag push
3. **TestPyPI**: Package is built and tested on TestPyPI first
4. **PyPI**: If TestPyPI succeeds, published to PyPI via OIDC trusted publishing

### Release Checklist

Before tagging:

- [ ] Bump version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Ensure all tests pass: `uv run pytest`
- [ ] Ensure coverage meets gate: `uv run pytest --cov`
- [ ] Build docs locally: `uv run sphinx-build -W -n -b html docs docs/_build`
- [ ] Commit changes
- [ ] Create tag: `git tag -a v0.X.Y -m "Release 0.X.Y"`
- [ ] Push tag: `git push origin v0.X.Y`

## Code Style Guidelines

- Follow existing code patterns in the codebase
- Use type hints consistently
- Document public APIs with docstrings
- Write clear, focused commits with descriptive messages
- Test new functionality thoroughly

## Questions?

Open an issue on [GitHub](https://github.com/maraxen/xtrax/issues) or check the [docs](https://xtrax.readthedocs.io).

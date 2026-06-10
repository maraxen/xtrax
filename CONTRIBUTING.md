# Contributing to xtrax

Thank you for your interest in contributing! We welcome pull requests and issue reports.

## Development Setup

Clone the repository and set up the development environment:

```bash
uv sync --extra dev
```

## Running Tests

Run the full test suite:

```bash
uv run pytest
```

Tests include unit tests, integration tests, and benchmarks. All tests must pass before submitting a PR.

## Code Quality

The project enforces strict code quality standards via:

### Linting and Formatting

```bash
uv run ruff check .
uv run ruff format .
```

### Type Checking

```bash
uv run pyright
```

### Coverage

The project enforces a minimum of **90% code coverage**. Coverage reports are generated automatically with pytest:

```bash
uv run pytest
```

The HTML coverage report is available at `.coverage_html/index.html`.

## Documentation

Build documentation locally:

```bash
uv run sphinx-build -W -n -b html docs docs/_build
```

The `-W` flag treats warnings as errors, and `-n` enables nitpicky mode. Documentation is auto-published to [https://xtrax.readthedocs.io](https://xtrax.readthedocs.io) on each release.

## Releasing

Releases are automated via git tags:

1. Ensure all tests pass and coverage meets the 90% gate
2. Tag the commit: `git tag v0.2.x` (or the appropriate version)
3. Push the tag: `git push origin v0.2.x`
4. GitHub Actions will automatically publish to PyPI via OIDC

## Code Style

- Follow PEP 8 (enforced by ruff)
- Type annotations required for all public functions
- Docstrings for all public modules, classes, and functions
- Import sorting via ruff's isort integration

## Reporting Issues

Please include:

- A clear description of the issue
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
- Python version and relevant dependency versions

Thank you for contributing!

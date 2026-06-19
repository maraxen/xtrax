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
uv run ty check src/
```

### Coverage

Tiered coverage gates enforce scoped product and optional-extra surfaces:

```bash
just audit-coverage-tier1   # shipped xtrax (excl. eda/devtools)
just audit-coverage-tier2   # xtrax.eda optional extra
```

The HTML coverage report from pytest is available at `.coverage_html/index.html`.
Coverage artifacts (`.coverage`, `coverage.xml`, `.coverage_html/`) are local-only
and must not be committed.

### Deterministic audit track

```bash
just audit-deterministic
```

## Documentation

Build documentation locally:

```bash
uv sync --group docs --extra eda
uv run sphinx-build -W -n -b html docs docs/_build
```

The `-W` flag treats warnings as errors, and `-n` enables nitpicky mode. Documentation is auto-published to [https://xtrax.readthedocs.io](https://xtrax.readthedocs.io) on each release.

## Releasing

Releases use OIDC Trusted Publishing (no stored PyPI tokens). **Do not push a
release tag until the full distribution audit passes** (`just audit-deterministic`,
`just audit-coverage-tier1`, and `just audit-publish-oidc`).

### Human prerequisites (before first publish)

Configure Trusted Publishers on both indexes (backlog **#1454**):

1. [TestPyPI](https://test.pypi.org/manage/account/publishing/) — project `xtrax`,
   workflow `.github/workflows/publish.yml`, environment `testpypi`
2. [PyPI](https://pypi.org/manage/account/publishing/) — same workflow, environment `pypi`

### Release checklist

1. Ensure `just audit-deterministic`, `just audit-coverage-tier1`, and
   `just audit-publish-oidc` pass locally
2. Confirm TestPyPI and PyPI Trusted Publisher settings are configured (#1454)
3. Tag the commit: `git tag v0.3.x` (match `src/xtrax/__init__.py`)
4. Push the tag: `git push origin v0.3.x`
5. GitHub Actions publishes to **TestPyPI** first, then **PyPI** via OIDC

`workflow_dispatch` on the publish workflow runs build + wheel smoke only; upload
jobs require a `v*` tag push.

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

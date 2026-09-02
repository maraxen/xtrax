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
uv run ty check src/
```

All code must pass type checking.

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

- `-W`: Treat warnings as errors
- `-n`: Show warnings about missing references
- `-b html`: Build HTML output

Documentation is hosted on ReadTheDocs and published on every commit to main.

Releases use OIDC Trusted Publishing (no stored PyPI tokens). **Do not push a
release tag until the full distribution audit passes** (`just audit-deterministic`,
`just audit-coverage-tier1`, and `just audit-publish-oidc`).

### Human prerequisites (before first publish)

Configure the Trusted Publisher (backlog **#1454**):

1. [PyPI](https://pypi.org/manage/account/publishing/) — project `xtrax`,
   workflow `.github/workflows/publish.yml`, environment `pypi`

Publishing goes straight to PyPI. A TestPyPI staging stage was removed on
2026-07-02, so a tag push is the first and only publish — there is no dry run
ahead of it.

### Release checklist

1. Bump the version in **both** `src/xtrax/__init__.py` and `CITATION.cff`, and
   close the `[Unreleased]` changelog section. `audit-project-hygiene` fails if
   the two version sites disagree
2. Ensure `just audit-deterministic`, `just audit-coverage-tier1`, and
   `just audit-publish-oidc` pass locally
3. Run `just audit-release-readiness`. For a version that is not yet published
   it reports `BLOCKED_AUTOMATED` with **exactly one** blocker — the n9 probe
   invalidating on `git tag 'vX' not found locally`, or on PyPI having no such
   release. That is the gate working as designed: it checks that the version,
   the tag and the PyPI release agree, which cannot be true before you publish.
   What matters is that nothing *else* is listed
4. Confirm the PyPI Trusted Publisher is configured (#1454)
5. Tag the commit: `git tag v0.4.0aN` (match `src/xtrax/__init__.py`)
6. Push the tag: `git push origin v0.4.0aN`. This publishes to PyPI
   immediately — there is no staging index and no dry run
7. Once the publish workflow succeeds, re-run `just audit-release-readiness`.
   With the tag pushed and the release live, the probe stops invalidating and
   the verdict becomes `READY`. A tag pushed while other blockers are open
   still publishes, because nothing enforces the gate for you
5. GitHub Actions publishes to **TestPyPI** first, then **PyPI** via OIDC

`workflow_dispatch` on the publish workflow runs build + wheel smoke only; upload
jobs require a `v*` tag push.

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

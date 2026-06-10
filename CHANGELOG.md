# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-10

### Added
- **Flat lazy API**: Direct imports from `xtrax` (e.g., `from xtrax import Trainer, Engine`) via lazy loading `__getattr__` pattern
- **Sphinx + RTD documentation**: Full API autodoc, furo theme, and ReadTheDocs configuration
- **CI/CD workflow**: GitHub Actions for lint (ruff), format checks, type checking (pyright), pytest with 90% coverage gate, and OIDC-based automated PyPI publishing
- **Apache-2.0 license and PyPI metadata**: Full package metadata, author attribution, and py.typed marker for type checking support

### Changed
- **Version reconciliation**: Bumped from 0.1.0 to 0.2.0 with single-source-of-truth in `src/xtrax/__init__.py`

### Details
- Public API now includes all core training, engine/IO, data, tiling, sparse, distributed, transforms, safety, and stages modules
- Output-sink surface: `BoundedCallbackHandler` for streaming outputs and orbax checkpoint support (`save_checkpoint`, `load_checkpoint`)
- Documentation published to https://xtrax.readthedocs.io
- CI enforces 90% code coverage and passes type checking before merge
- PyPI releases automated via git tags (e.g., `git tag v0.2.0`)


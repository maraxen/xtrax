# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-10

### Added

- **Distribution readiness**: Apache-2.0 license, PyPI metadata, py.typed marker for type checking support
- **Lazy public API**: 43 curated top-level imports (Trainer, Engine, AxisSpec, BatchPlan, etc.) via PEP 562 lazy loading; bare `import xtrax` overhead <1ms
- **Documentation**: Sphinx-powered docs hosted on RTD with furo theme, quickstart guide, concepts, and architecture diagrams
- **CI/CD**: GitHub Actions workflow for lint (ruff), type-check (pyright), test (pytest, 414 tests at 96.5% coverage), with 90% coverage gate
- **Publish pipeline**: Trusted publishing via OIDC to TestPyPI and PyPI; automated on git tags matching v*

### Fixed

- **Version reconciliation**: Single-sourced version 0.2.0 via hatchling; removed version duplication across files

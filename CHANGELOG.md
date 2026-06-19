# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`xtrax.eda` — EDA visualization subpackage** (optional extras: `pip install xtrax[eda]`):
  A two-layer exploratory data analysis interface for inspecting `BatchPlan` outputs from
  the tiling subsystem.

  _Stats layer_ (stdlib + numpy only, no extras required):
  - `extract_plan_stats(plan: BatchPlan) -> PlanStatsDict` — extracts strategy distribution,
    axis metadata, dedup/bucket statistics, and memory warnings.
  - `explain_plan(plan: BatchPlan) -> PlanStatsDict` — like `extract_plan_stats` with
    guaranteed non-empty `reasoning` strings per axis.
  - `analyze_dedup(decision: AxisDecision) -> DedupStatsEntry` — dedup ratio, padding waste,
    unique vs padded counts.
  - `analyze_bucket(decision: AxisDecision) -> BucketStatsEntry` — bucket boundaries and count.

  _Viz layer_ (requires `pip install xtrax[eda]`):
  - `render(plan, fmt, path, stats_transform, metadata, logger, panels) -> bytes | str | None`
    — single entry point for PNG (bytes), SVG (bytes), HTML (str) output. Seaborn/matplotlib
    backend; headless via `Agg`. Supports post-stats transform hook, JSON metadata sidecar,
    panel filtering, and a structural `PlanLogger` protocol for wandb/tensorboard adapters.
  - `plan_to_dataframe(stats: PlanStatsDict) -> pd.DataFrame` — one row per axis.

  _Types_ (no extras):
  - `PlanStatsDict` — fully-typed `TypedDict` for the stats surface.
  - `PlanLogger` — structural `Protocol`; xtrax never imports wandb or tensorboard.
  - `PanelName` — `Literal["strategy","cardinality","dedup","bucket","memory","reasoning"]`.

  (`src/xtrax/eda/`, `tests/eda/`, `docs/advanced/eda-guide.md`, `docs/api/eda.md`)

### Changed

- Distribution readiness: tiered coverage gates (`tier1_core` 90/80, `tier2_eda` 90/75),
  docs plumbing gate (`just audit-docs-build`), LibCST added-types diff gate, and
  deterministic audit track expansion (`just audit-deterministic`).
- `pyproject.toml`: Added `eda` to both `[project.optional-dependencies]` and
  `[dependency-groups]` (`pandas>=2.0`, `matplotlib>=3.8`, `seaborn>=0.13`).

## [0.2.1] - 2026-06-14

### Added

- **`CarrySpec`, `CarryShape`, `DedupSpec`** ported from `aminx.tiling` (T2.2-2.3): Three
  new types for declaring pre-committed axis strategies before the `BatchPlanner` budget
  loop.
  - `CarrySpec(axis_name, init, transition, ordered_sinks)` — declares an axis as a
    `jax.lax.scan` carry; `__post_init__` guards against heterogeneous axis names.
  - `CarryShape(name, shape, dtype)` — typed carry-buffer descriptor; `materialize()`
    returns a zero-initialized buffer.
  - `DedupSpec` + `get_k_bucket` — declares an axis for dedup-gather; `get_k_bucket`
    rounds cardinality up to the next power of 2.
  ([`src/xtrax/tiling/carry.py`](src/xtrax/tiling/carry.py),
   [`src/xtrax/tiling/carry_shape.py`](src/xtrax/tiling/carry_shape.py),
   [`src/xtrax/tiling/dedup.py`](src/xtrax/tiling/dedup.py))

- **`BatchPlanner` Phase 0 and Phase 0b** (T2.2-2.3): `BatchPlanner.plan()` now accepts
  `carry_specs: list[CarrySpec] | None` and `dedup_specs: list[DedupSpec] | None`.
  Phase 0 pre-commits declared carry axes to `Scan` before the cardinality/budget loop;
  Phase 0b pre-commits dedup-eligible axes to `DedupGather`. Remaining axes proceed
  through the existing budget rules unchanged.
  ([`src/xtrax/tiling/plan.py`](src/xtrax/tiling/plan.py))

- **Factory `make_axis_dispatch` + iterator types** (T2.4): `make_axis_dispatch(strategy)`
  is now a pure factory — it takes a strategy and returns an iterator object, not a result.
  Three iterator types ported from `aminx.tiling`:
  - `VmapIterator` — wraps `jax.vmap`
  - `SafeMapIterator` — chunk-order-stable chunked map
  - `JaxScanIterator` — `jax.lax.scan`; returns `(final_carry, stacked_outputs)`
  - `MapIterator` — eager Python-level map
  `DispatchRejected` raised for `DedupGather` (handled upstream by `BatchPlanner`).
  Backward-compat shim `axis_dispatch(strategy, fn, xs, init=None)` preserves the prior
  eager 4-arg call.
  ([`src/xtrax/tiling/dispatch.py`](src/xtrax/tiling/dispatch.py),
   [`src/xtrax/tiling/iterator.py`](src/xtrax/tiling/iterator.py))

- **`Scan.init` field** (T2.1): `Scan` strategy now carries an optional `init: Any | None`
  for configurable carry initialization. Default `None` (zero-init, backward-compatible).
  ([`src/xtrax/tiling/strategy.py`](src/xtrax/tiling/strategy.py))

### Exports

All new types and exceptions exported from `xtrax.tiling`:
`CarrySpec`, `CarryShape`, `DedupSpec`, `get_k_bucket`,
`VmapIterator`, `SafeMapIterator`, `JaxScanIterator`, `MapIterator`,
`DispatchRejected`, `axis_dispatch`.

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


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Sink provenance tracking** (`xtrax.run`): `ZarrStagingSink` now auto-captures
  static run provenance for downstream consumers. `SinkSpec` gains a required
  `run_id` and an optional JSON-Schema-style `extension_schema`. The store's
  root group receives the full record (`git_sha`, `git_branch`, `git_dirty`,
  `run_id`, `created_at` as ISO-8601 UTC); each drained key's group gets a
  minimal `run_id`/`git_sha` pointer. Git capture never raises (falls back to
  `git_sha="unknown"` with a `UserWarning`). Core field names are reserved
  against caller attrs; schema validation happens at `stage()` time. New
  `finalize()` method consolidates store metadata exactly once and locks the
  sink; opening a second sink on the same `output_dir` with a different
  `run_id` now raises. (spec: #96, task 260824_default-sink-provenance-tracking)

### Changed

- **Breaking**: `SinkSpec.run_id` is now a required constructor argument.

## [0.4.0a6] - 2026-08-20

### Added

- **`WhileCarry`** (`xtrax.tiling`): `lax.while_loop`-backed `AxisStrategy`
  for carry-only loops (no input sequence, no per-step `ys`).
  `CarrySpec(collect_outputs=False)` pre-demotes to `WhileCarry`;
  `make_axis_dispatch` returns `WhileLoopIterator`. Not reverse-mode AD
  safe. (#81 spec, #83 implementation)

## [0.4.0a5] - 2026-07-08

### Added

- **`xtrax.run.zarr_integrity`** (`xtrax.run`): content-digest and durability
  primitives for Zarr directory stores, hoisted out of aminx's
  `host/campaign.py` -- they were fully generic (no domain logic) and
  belong in the shared layer. `zarr_content_digest(path)` computes a
  deterministic sha256 over a store's full logical content (paths, attrs,
  array data), unaffected by filesystem metadata or which process wrote
  it; `fsync_tree(path)` durabilizes a directory-of-many-files store
  bottom-up before that digest is trusted. Also exports the lower-level
  building blocks (`canonical_json_bytes`, `normalize_json_value`,
  `update_array_digest`, `update_zarr_node_digest`, `fsync_file`,
  `fsync_directory`) for callers assembling their own verification/durability
  logic on top. `zarr_content_digest`/`update_zarr_node_digest` require the
  optional `zarr` dependency at call time only (`pip install xtrax[io]`);
  importing `xtrax.run` never requires zarr installed.
  (`src/xtrax/run/zarr_integrity.py`, `tests/run/test_zarr_integrity.py`)

## [0.4.0a4] - 2026-07-07

### Added

- **`ZarrStagingSink.stage()` gains `attrs`** (`xtrax.run`): optional
  `dict[str, Any]` of JSON-safe scalar metadata written to the Zarr group's
  `.attrs` on drain, merging the same way staged arrays merge across
  repeated `stage()` calls for the same key. `take()` discards pending
  attrs (it returns the in-memory payload without persisting). Fills the
  gap for consumers whose payloads mix array data with small provenance
  metadata that has no array-shaped equivalent.
  (`src/xtrax/run/zarr_sink.py`, `tests/run/test_zarr_sink.py`)

## [0.4.0a3] - 2026-07-07

### Added

- **`xtrax.run.ZarrStagingSink`** (`xtrax.run`): a keyed staging buffer for
  JAX `io_callback`-driven streaming output, draining into nested Zarr
  groups. `SinkSpec.format` gains `"zarr"`; a new `make_sink(spec)` factory
  dispatches to the real implementation (`zarr`/`none` today; `jsonl`/`h5`
  remain routing-only stubs pending their own writers). Generalizes the
  keyed-staging-then-drain pattern used by consumers with per-chunk tensor
  payloads (sequences, logits, encoder intermediates, accumulated tensors) —
  domain-specific `io_callback` dispatch stays with the caller; this module
  owns staging and Zarr storage only.

  Zarr is a new optional extra (`pip install xtrax[io]`), imported lazily
  inside `ZarrStagingSink.__init__` — `xtrax.run` remains fully importable
  without it; only constructing a Zarr sink requires the dependency, with a
  clear `ImportError` pointing at the install command otherwise.
  (`src/xtrax/run/sink.py`, `src/xtrax/run/zarr_sink.py`,
  `tests/run/test_sink.py`, `tests/run/test_zarr_sink.py`)

## [0.4.0a2] - 2026-07-07

### Fixed

- **`StageBundle.__init_subclass__` validator** (`xtrax.stages`): three limitations
  that blocked domain code from adopting `StageBundle` were fixed.
  - **PEP 563 blindness**: modules using `from __future__ import annotations`
    left field annotations as unevaluated strings; the validator now resolves
    them via `typing.get_type_hints(cls, include_extras=True)` and raises a
    clear `TypeError` (naming the unresolved annotation) instead of silently
    misclassifying fields.
  - **Structural-callable `Protocol`s rejected**: fields typed as a
    `typing.Protocol` whose only member is `__call__` are now accepted as
    callable-shaped, alongside plain `Callable`.
  - **Union check hardcoded to exactly two args**: `X | None` unions now
    validate for any arity — N-args-with-exactly-one-`None`, not just the
    2-arg case — so e.g. `Callable | SomeProtocol | None` fields validate
    correctly instead of raising.
  (`src/xtrax/stages/bundle.py`, `tests/stages/test_bundle.py`)

## [0.4.0a1] - 2026-07-06

### Added

- **Joint-budget planning mode for `BatchPlanner`** (`xtrax.tiling`):
  `BatchPlanner(budget=MemoryBudget(bytes=..., estimate=...))` replaces the
  independent per-axis rules with whole-plan greedy demotion — every eligible
  axis starts at `Vmap`, then axes with `cardinality > default_batch_size` are
  demoted to `SafeMap` in the order specs were given until the joint estimate
  fits the budget. Callers express demotion priority by spec order. Strict by
  design: mutually exclusive with the per-axis `memory_estimator`, estimator
  exceptions propagate, and an unfittable plan raises `BudgetInfeasibleError`.
  Carry/dedup/bucket decisions stay fixed but participate in the estimate;
  budget-mode reasoning strings carry the byte numbers for `xtrax explain`.

  Native-tooling estimator building blocks (`xtrax.tiling.estimators`):
  - `device_memory_budget(fraction=0.9, device=None)` — budget bytes from the
    XLA allocator's `Device.memory_stats()["bytes_limit"]`; fails loud when
    the backend reports no stats.
  - `lowered_memory_estimate(fn, *abstract_args)` — AOT-compiles from
    `ShapeDtypeStruct`s and returns XLA's own buffer-assignment bytes
    (argument + output + temp) via `Compiled.memory_analysis()`.

  Exports are tiling-level (`xtrax.tiling`), same tier as `CarrySpec` — no
  root public-API change. Spec:
  `.praxia/docs/specs/260706_joint-budget-batch-planner.md`.
  (`src/xtrax/tiling/budget.py`, `src/xtrax/tiling/estimators.py`,
  `tests/tiling/test_budget_plan.py`, `tests/tiling/test_estimators.py`)

### Fixed

- **CI recovery**: install `just` via `uv tool` (runner image stopped shipping
  it); publish-OIDC gate aligned with the no-TestPyPI decision; coverage DAG
  gate now prints the pytest output tail on failure (failures were previously
  undiagnosable from CI logs); praxia-CLI emit smoke test skips when the
  binary is absent; CITATION/README version metadata synced to `__version__`.

### Changed

- **Docs positioning**: new `docs/why-xtrax.md` (why the tiling layer lives
  above the JIT boundary); README "Why xtrax?" and docs index rewritten to
  match; `.claude/workflows/port-validation.js` now tracked in-repo.

## [0.3.1] - 2026-07-02

### Added

- **Plan topology validator** (`xtrax.stages`): `PlanTopologyError` raised on
  invalid stage-plan topologies.

### Changed

- **Publish workflow**: straight to PyPI via OIDC Trusted Publishing;
  TestPyPI staging dropped by decision.

## [0.3.0] - 2026-07-02

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

- **`xtrax.inference` — Signature inference subpackage** (Tier-1 MVP, E1):
  Zero-config axis detection and fail-loud semantics for batched JAX computations. Enables
  automatic extraction of output schemas and input axis specifications, with explicit role
  assignment (KNOWN via `@axis_config` or UNKNOWN with fail-loud guards).

  _Public API_:
  - `infer_bundle(fn, abstract_inputs, *, verify_against=None) -> (BundleSchema, list[AxisSpec])`
    — main entrypoint; infers output schema and axis specs from abstract inputs.
  - `@axis_config(*AxisOverride(...))` — decorator for Tier-1 axis resolution; attaches
    overrides positionally to leading axes; each override specifies `name` and required
    `default_batch_size` (Assumption A3: not inferable from shape alone).
  - `AxisOverride` — dataclass for single-axis configuration; fields include `name`,
    `default_batch_size`, `cardinality`, `tile_granularity`, `heterogeneous`, `dedup_eligible`,
    `bucket_boundaries`.
  - `BundleSchema` — output structure mapping field names to `ShapeDtypeStruct`; carries
    optional `carry_specs` list (deferred for T2+, always None in MVP).
  - `AxisRole` — Enum with KNOWN (axis resolved) and UNKNOWN (axis ambiguous, fail-loud);
    future tiers extend with concrete roles (BATCH, SEQUENCE, etc.).
  - `AmbiguousAxisError` — raised by `BatchPlanner.plan()` when axis role is UNKNOWN.
  - `StructureMismatchError` — raised when `verify_against` outputs diverge from abstract-traced.
  - `synthesize_axes(abstract_inputs, overrides=None) -> list[AxisSpec]` — lower-level factory
    that synthesizes AxisSpec with explicit role assignment.

  _Deferred (T2+)_:
  - Concrete axis roles (BATCH, SEQUENCE, FEATURE) with domain-specific planner behavior.
  - jaxtyping dimension-name adapter for role inference without decorators.
  - CarrySpec auto-derivation for RNN-like stateful axes.
  - LibCST Bundle codegen for boilerplate `@axis_config` and dataclass generation.

  (`src/xtrax/inference/`, `tests/inference/`, `docs/api/inference.md`)

- **`xtrax run` / `xtrax resume` / `xtrax sweep` — CLI training verbs** (E3):
  TOML-driven training, checkpoint resume, and local grid-search sweep.
  - `xtrax run config.toml` — resolve model/optimizer/loss/data from import paths, write
    manifest, train with orbax checkpointing.
  - `xtrax resume <run-id> --epochs N` — read manifest, reconstruct state from latest
    checkpoint, train for N additional epochs into a new sibling run dir.
  - `xtrax sweep sweep_config.toml` — sequential in-process grid search with atomic
    sweep manifest, JAX compilation cache reuse, and per-run fault tolerance.

  (`src/xtrax/cli/`, `tests/cli/`)

### Changed

- Distribution readiness: tiered coverage gates (`tier1_core` 90/80, `tier2_eda` 90/75),
  docs plumbing gate (`just audit-docs-build`), narrative docs gate (`just audit-narrative-docs`),
  output-sink docs gate (`just audit-output-sink-docs`), publish OIDC gate
  (`just audit-publish-oidc`), release readiness convergence audit
  (`just audit-release-readiness`), LibCST added-types diff gate, and deterministic audit
  track expansion (`just audit-deterministic`).
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

- **Distribution readiness**: Apache-2.0 license, PyPI metadata, py.typed marker for type checking support
- **Lazy public API**: 43 curated top-level imports (Trainer, Engine, AxisSpec, BatchPlan, etc.) via PEP 562 lazy loading; bare `import xtrax` overhead <1ms
- **Documentation**: Sphinx-powered docs hosted on RTD with furo theme, quickstart guide, concepts, and architecture diagrams
- **CI/CD**: GitHub Actions workflow for lint (ruff), type-check (pyright), test (pytest, 414 tests at 96.5% coverage), with 90% coverage gate
- **Publish pipeline**: Trusted publishing via OIDC to TestPyPI and PyPI; automated on git tags matching v*

### Fixed

- **Version reconciliation**: Single-sourced version 0.2.0 via hatchling; removed version duplication across files

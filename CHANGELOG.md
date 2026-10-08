# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Per-axis input invariance** (#2598, #2599): `AxisSpec.varying_inputs`
  names the inputs that vary along a mapped axis; every other named input is
  invariant along it. `declare_varying_inputs` rejects an unknown axis or
  input name. The list round-trips on `BatchPlan` and is printed by
  `xtrax plan` and `xtrax explain`. `verify_axis_invariance` traces with
  `jax.make_jaxpr` and reports outputs and intermediates that do not read a
  varying input. `claimed_invariant_outputs` (axis to output indices) checks
  a hoisting claim: a claimed output that reads a varying input raises
  `InputInvarianceError`, naming the axis, the output, and the input.
  Invariant inputs may still combine with varying ones (weights times data).
  The check walks callees inside `jax.checkpoint`,
  `custom_jvp`, and `jit`/`pjit`.
- **Module provenance for ported code** (#2590, #2591, #2592):
  `xtrax.provenance.Provenance` records an upstream repository, a revision
  (commit sha or tag), an SPDX licence, and a relationship (`vendored`,
  `ported`, `derived`, `inspired`). A missing revision or licence requires
  `waiver_reason`. Ported modules set `__provenance__`.
  `scripts/audit_port_provenance.py` fails when a `port/manifests` kernel, or
  a module docstring that says the code was ported, vendored, adapted, or
  upstreamed from somewhere, has no declaration. The check is
  `just audit-port-provenance`, and `just audit-port` runs it (CI job
  `audit-port`). `scripts/port_init.py` prints a declaration from the
  arguments the porter supplies.
- **Sink session and opt-in element index** (#2587, #2588): `xtrax.stages.sink_session`
  opens a host sink, yields an ordered `io_callback` pinned to
  `xtrax.stages._callback`, and closes the session on the way out, including
  when the traced region raises. `AxisBoundary(sink_receives_index=True)` passes
  the mapped-axis index to the sink as `(y, index)` for `Vmap`, `ChunkedMap`
  (including a remainder chunk), and `Scan`. The default call stays `sink(y)`.
- **Skill delivery** (#2107, #2596, #2597): joint-budget `memory_estimator` and
  `MemoryBudget` estimates in the using-xtrax tiling skill come from
  `lowered_memory_estimate` on a representative tile, scaled by live tile
  counts. `.praxia/manifest.toml` lists every `agent_assets/skills` skill and
  tracks `xtrax.__version__`. `scripts/install_skills.py --check` reports
  installed copies that are missing or whose `xtrax_version` differs from the
  repo (`--target` or `XTRAX_SKILLS_TARGET`).
- **`WhileLoopWithYsIterator`** (#2589): a `lax.while_loop` iterator that writes
  each step's `y` into a preallocated buffer of caller-supplied `max_steps`
  and returns `(final_carry, ys_buffer, length)`. Buffer fill is `0`. When
  `length == max_steps` and `cond(final_carry)` is still true, the cap stopped
  the loop and every index holds a body output.

- **Grain input pipelines** (#2085): `xtrax.data.build_input_pipeline` builds a
  domain-free Grain pipeline (process shard, shuffle, repeat, optional
  caller-supplied fixed-length pad, `numpy.stack` batch, threaded prefetch,
  optional `mp_prefetch`, optional `device_put`). `create_distributed_pipeline`
  shards that pipeline with `ShardByJaxProcess` (`jax.process_index` /
  `jax.process_count`) instead of returning the dataset unchanged, and batches
  at `global_batch_size // num_devices`. `DataModule(use_grain_pipeline=True)`
  yields from the same builder (train shuffles, eval does not). Read threads
  default to 1 and multiprocessing workers default to 0 — on cheap reads, 8
  threads were slower than 1 (22.6k vs 2.1k examples/s). `mp_prefetch` marks
  absl flags parsed so the first batch does not raise `UnparsedFlagAccessError`
  outside an absl app. Grain stays an optional import (`xtrax[data]`);
  importing `xtrax.data` does not import it.

- **Input-pipeline stall profiler** (#2086): `xtrax.data.profile_input_pipeline`
  reports `examples_per_s` and `wait_fraction` (time a simulated train step
  spent waiting on data, over the step's wall time). It runs a slow-source
  negative control (1 read thread, prefetch 1; `wait_fraction` must exceed
  0.5) and an instant-source positive control (`wait_fraction` under 0.05),
  records whether those bounds held, and can raise `InputPipelineControlError`
  when they did not. Each measurement is emitted as an
  `xtrax.profiling.ProbeRecord`.
- **`xtrax.testing` reference-parity harness** (#2324): framework-neutral primitives
  for comparing a candidate sampler to an independent oracle. `order_from_randn`
  and `InjectedSource` feed one host order/noise/uniform draw to both sides.
  `teacher_forced_lane` compares conditional logits in both decoding directions,
  with an alphabet-index map. `collapsed_sampler_lane` compares temperature-0
  argmax and reports near-ties. `distributional_lane` reports per-step chi-square
  and total variation plus a minimum detectable TV, and reports PASS only after
  `require_negative_control` has rejected a perturbed knob. `knob_coverage` lists
  declared knobs that were not varied. `assert_distinct_callables` refuses a
  run-twice self-comparison.

- **using-xtrax parity reference** (#2325): `agent_assets/skills/using-xtrax/references/parity.md`
  on shared host randomness, the four comparison strategies, and the requirement
  that a sampler comparison call the public entry point with a must-fail control.
- **Chunk- and resume-invariant PRNG key stream** (#2544): `xtrax.random.element_keys`
  folds a base key, or an integer seed via `jax.random.key`, with the global element
  index as `int32`. The derivation matches aminx `compute_sample_keys` bit-for-bit, so
  keys depend only on `(base key, global index)`. `make_chunk_plan` and `iter_chunk_keys`
  yield the same keys for any chunk size or resume point.
- **`xtrax.profiling.iter_jaxpr_eqns` / `sub_jaxprs`**: one public jaxpr walker.
  It yields every equation, recursing through `pjit`/`jit`, `scan`, `while`
  cond and body, `cond` branches, `custom_jvp`/`custom_vjp`, and `remat`.
  `xtrax.export.safety` and `xtrax.profiling.loop_scaling` both use it (debts
  #2526 and #2505).

- **`convert_to_onnx(..., model_name=, embed_external_data=)`** (debt #2526):
  `model_name` (default `"xtrax_export"`) is written on the graph.
  `embed_external_data=True` stores every tensor in the protobuf, including
  Loop-subgraph constants that an external-data sidecar leaves unloadable in
  ORT-Web. `export_pipeline` forwards both arguments. `find_onnx_rng_ops` now
  descends into subgraphs nested in `FunctionProto` bodies, and
  `onnx_unknown_domain_census` counts op domains outside `""` and `"ai.onnx"`.

- **`xtrax.export.rings.make_input_class(generator, ..., label=)`** (debt #2526):
  input classes are built from a caller-supplied generator callable.

### Fixed

- **BatchPlanner per-axis `memory_estimator`** (#2593, #2594): an estimator
  that raises fails `plan()` with `RuntimeError` naming the axis, instead of
  falling back to the cardinality rules and selecting Vmap. A missing device
  `bytes_limit` is still logged once and compared against the documented 4 GiB
  default. `memory_estimator=None` keeps those cardinality rules; when they pick
  Vmap without an estimate, the planner logs a warning once per planner and
  axis. When `AxisSpec.element_input_bytes` is
  set, an estimate below that per-element input size raises `ValueError`
  naming the axis, the estimate, and the bound. Specs that omit it have no
  element shape or dtype, so that check is skipped.

### Changed

- **`extent_scaling_report`** flags a loop only when both per-iteration work and
  trip count grow with the extent (debt #2505). `jnp.searchsorted` lowers to a
  scan of `ceil(log2(n))` steps; a vector of queries makes each step's work
  grow, but the trip count does not, so that loop is no longer flagged. A
  `while` with no static length is run once per extent to count its iterations.

### Deprecated

- **`xtrax.export.rings.symmetric_geometry` and `sub_k_neighbours`** (debt #2526):
  protein/MPNN input generators. They emit `DeprecationWarning` and will be
  removed in the next release. Callers pass their own generator to
  `make_input_class`. `BUCKET_LADDER` is unchanged.
- **`xtrax.run` output contract** (debt #2523): `make_sink` accepts `format="memory"`
  and returns a `MemorySink` with the same stage/drain/finalize protocol as
  `ZarrStagingSink`, including read-back. `ZarrStagingSink` appends along the
  leading axis across drains when `SinkSpec.append=True` (the default remains
  overwrite). `finalize()` consolidates, fsyncs, and digests the store, returning
  a `SinkReceipt` (`path`, `digest`, `digest_algo_version`, `run_id`, `seed`).
  `derive_sink_spec` copies `RunSpec.seed` and, when `output_dir` is omitted,
  `RunSpec.output_root`. `RunSpec` gains optional static fields `output_root`,
  `device_count`, `precision`, and `shard_lineage`. `SinkSpec.provenance` injects
  precomputed git state or a path to capture from; the default no longer shells
  out from `Path.cwd()`. Exclusive store roots record `producer` and `xtrax_version`;
  durable roots keep those fields inside the `xtrax.store` record so the root
  attr set stays exactly that record.
  `level_schemas` supplies a JSON schema per key depth. `DIGEST_ALGO_VERSION`
  labels the zarr content digest. `canonical_hash`
  (`CANONICAL_HASH_ALGO_VERSION`) is the sha256-of-canonical-JSON helper used by
  run-layer document digests. `atomic_write_bytes` / `atomic_write_text` write
  via temp file, fsync, replace, and directory fsync. `xtrax run` writes
  `manifest.json` with `atomic_write_text`.

### Changed

- **Reserved sink attr names** (debt #2523): `producer` and `xtrax_version` are
  now reserved, alongside the existing provenance names. Staging an attr with
  either name raises, and both are excluded from the default zarr content
  digest. This is a minor compatibility break for callers who stored their own
  attrs under those names.
- **Memory digest version** (debt #2523): memory-sink receipts record
  `MEMORY_DIGEST_ALGO_VERSION`, not `DIGEST_ALGO_VERSION`. The memory digest
  covers array names and array bytes only (caller attrs do not change it). It
  is a different algorithm from the zarr content digest and the two are not
  comparable.
- **CLI provenance** (debt #2523): `xtrax run` passes the process working
  directory as sink provenance, so the metrics store records that checkout's
  git HEAD. Outside a git repository the sink warns and records
  `git_sha="unknown"`. Omitting `SinkSpec.provenance` in the library still
  does not shell out.
- **Trainer key threading, auxiliary metrics, and engine hooks** (#2525).
  `Trainer` accepts `takes_key` and `has_aux` (both default off, so
  `loss_fn(predictions, targets) -> scalar` is unchanged). With `takes_key`,
  `step` splits `state.key` once and calls `loss_fn(model, batch, key)`. With
  `has_aux`, that callable returns `(loss, aux)` and the aux dict is merged
  into the metrics. `accumulate_grads(..., has_aux=True)` also returns the
  mean aux pytree: each leaf is `jnp.mean` over the microbatch axis, the same
  reduction as the mean loss. `Engine.fit` / `fit_sync` gain a validation hook
  (`validate_fn`, `validate_every_steps`, `validate_every_epochs`), early
  stopping (`EarlyStopping`: metric, mode, patience, min_delta), and
  step-cadence checkpoints (`checkpoint_every_steps`). `Engine.restore` loads
  a checkpoint's `state.key` and `state.extras`. Fail-closed RunLedger
  behaviour on `fit` is unchanged.

- **Traced-weight loss composition** (#2088). `ComposedLoss` sums terms of
  the form `fn(predictions, targets, **per_batch_flags) -> scalar`. Term
  weights, and optional per-output placement weights, are traced arrays, so
  changing a weight or a per-batch flag value does not recompile. The return
  value is `(weighted_loss, aux)` where `aux` holds the unweighted per-term
  scalars consumed by `Trainer(has_aux=True)`. `WeightedLoss` remains the
  static-float combinator.
- **`xtrax.profiling.count_backend_compiles` and `assert_no_recompile_after`** (#2083):
  context manager counting JAX `/jax/core/compile/backend_compile_duration` events
  (count and seconds), plus a helper that asserts a stepped function does not
  backend-compile after warmup. The private `unregister_event_duration_listener`
  hook falls back to the public `jax.monitoring` alias and raises if JAX moves both.

- **`xtrax.profiling.load_trace_events` and `hlo_text_for`** (#2084): load Perfetto
  `traceEvents` from a `jax.profiler.trace` directory, and extract compiled HLO text
  from both `jax.jit` and `eqx.filter_jit`. `hlo_text_for` raises `TypeError` when
  the compiled object has no HLO text, instead of returning `None`.

### Fixed

- **CPU fusion scope attribution** (#2084): `scope_map_from_hlo_text` maps each
  fusion instruction (the `hlo_op` a CPU trace executes, e.g. `add_add_fusion.3`)
  to the named_scope label of its fused computation's instructions, or to the
  fusion instruction's own `op_name` when the body has none. `parse_scopes` no
  longer returns an empty attribution for fused CPU steps.
- **Planner helpers for joint-budget consumers** (#2521): `MemoryBudget` mode
  fixes a heterogeneous axis to `ChunkedMap` and never assigns it `Vmap`.
  `BatchPlan.decision_for(axis_name)` returns that axis's decision
  (`KeyError` names unknown axes). `plan_axis` is a single-axis wrapper over
  `BatchPlanner` (an int bytes-per-element estimate, or a callable measured
  with `lowered_memory_estimate`). `estimate_memory_theoretical` is a
  domain-free product-of-extents estimator in `xtrax.tiling`. Body-mode
  `Scan` in `axis_dispatch`, and whether `CarrySpec.transition` executes,
  are unchanged and remain deferred (#2543).
- **Host-side bucketing primitives on `xtrax.tiling`** (#2522): `BUCKET_LADDER`
  (64..2048) now lives in `xtrax.tiling` so runtime code does not import the
  export stack. `xtrax.export.rings.BUCKET_LADDER` is the same object. New
  helpers: `valid_span` (valid span from the leading edge through the last
  valid position), `select_rung` (smallest ladder rung at least the span,
  capped at the caller's current length), `trim_axis`, and `pad_axis`.

- **Layered config resolution** (#2524): `xtrax.config.resolve_layered` resolves
  one key through an explicit argument, an environment variable (empty or
  `none` disables that layer), `[tool.<app>]` in the nearest `pyproject.toml`,
  the per-machine `${XDG_CONFIG_HOME:-~/.config}/<app>/config.toml`, then a
  default, and reports which layer decided. A malformed TOML file raises
  `ValueError`. `resolve_memory_budget` builds on it: configured values are
  absolute byte counts; otherwise it uses `device_memory_budget` (source
  `device`) and, when the device reports no `bytes_limit`, a logged 4 GiB
  default scaled by `headroom`.

### Changed

- **`BatchPlanner` memory-limit fallback is no longer silent** (#2524): the
  per-axis `memory_estimator` path reads the device limit via
  `device_memory_budget(fraction=1.0)`, so a reported `bytes_limit` is still
  compared in full. When the device does not report `bytes_limit`, the planner
  logs once and uses the documented 4 GiB default
  (`xtrax.tiling.estimators.DEFAULT_DEVICE_MEMORY_BYTES`) instead of
  substituting that figure with no record. Planning decisions on devices that
  do not report a limit are unchanged. `device_memory_budget` itself still
  raises when the runtime cannot answer.
- **using-xtrax skill**: end-to-end length bucketing (`AxisSpec.bucket_boundaries`,
  host `select_bucket`/`bucketize`, `BUCKET_LADDER`, one compile per rung; #2497),
  a local-copy replacement table and the 0.4.0a12 `SafeMap` alias removal pointing
  at `codemods/safemap-to-chunkedmap/` (#2498), standalone `convert_to_onnx` versus
  raw jax2onnx plus rings/divergence (#2499), limits and telemetry fail-closed
  (`LedgerUnavailableError`; #2500), duplicated primitives including
  `synthesize_dedup_spec` (#2501), `WhileCarry` for inference-only loops (#2103),
  and the `ChunkedMap`/`lax.map` scan-of-while compile hazard (#2105).
### Fixed

- **Skill examples match installed call signatures** (`agent_assets/skills`, #2496).
  Copy-paste blocks for `select_bucket` / `bucketize`, `SafetyTrainStep`, `Engine.fit`,
  `make_optimizer` / `adamw_with_schedule`, distributed init, and checkpoints now follow
  current source. `tests/skills/test_skill_code_blocks.py` parses every fenced Python
  block, resolves `xtrax` names, and binds literal keyword arguments. A block whose
  nearest non-blank line above the fence is `<!-- skill-check: skip -->` is skipped.
  The using-xtrax preflight compares frontmatter `xtrax_version` with `xtrax.__version__`
  and warns on mismatch.
- **xtrax skill descriptions load for the task** (`agent_assets/skills`, #2502).
  Frontmatter descriptions and triggers are phrased around padding and bucketing,
  chunked maps, memory-budgeted batching, ONNX or StableHLO export, resumable
  training, zarr sinks, citable measurements, slow scans, numerical divergence,
  and shared-filesystem reads.
- **`chunked_map` never vmaps an axis of length 1** (#2520). `jax.lax.map(..., batch_size=k)`
  vmaps each chunk, so `batch_size=1`, a remainder of 1, and a leading axis of length 1
  emitted a vmap-of-1 (the miscompile aminx #2391 hit on TITAN RTX). Those cases now run
  unbatched: a sequential `lax.map`, a direct call on the peeled last element, or a direct
  call when the whole axis has length 1. The same guard covers every xtrax-dispatched vmap:
  `ChunkedMapIterator`, `VmapIterator` (including tree-structured `in_axes` and `None`
  prefixes), unordered `execute_map_axis(Vmap)`, `axis_dispatch(Vmap)`, and the per-row
  and dedup-gather maps in `verify_dedup_outputs`. Ordered paths are unchanged. Values
  and order are unchanged.

## [0.4.0a12] - 2026-10-01

### Added

- **`xtrax.run.digest` module**: Digest functions for reproducibility and durability
  infrastructure (#181, U4, spec demistify 261001_preemption-safe-cimist-fitting):
  `canonical_digest`, `array_digest`, `source_fingerprint`, `numerics_env`. Used by
  pipeline resumption and done-marker verification to detect content changes
  independent of timing, process, or operational metadata.

- **Reserved `xtrax.` attribute namespace** (#181, U0, spec demistify
  261001_preemption-safe-cimist-fitting): Zarr attrs starting with the prefix
  `"xtrax."` are excluded from `zarr_content_digest` by default (pass
  `include_provenance=True` to include them). The sink's new `stamp_reserved()`
  method is the sanctioned writer of such attrs, ensuring internal bookkeeping
  (e.g. run reports, status flags) does not affect content-based reproducibility.
  `ZarrStagingSink.stage()` now rejects caller attrs in the reserved namespace.

- **`xtrax.run.zarr_commit` module** (#181, U1-U2, spec demistify
  261001_preemption-safe-cimist-fitting): Durable atomic-commit primitives for
  zarr v3 directory stores. Provides crash-safe staging + rename patterns:
  `commit_key` (atomically commit a staged group), `lookup` (classify key state:
  Missing, Reuse, Stale, Corrupt), `create_store` / `open_store` (initialize and
  open durable stores with identity payloads and prefix trees), `committed_keys`
  (enumerate committed keys), `gc_staging` (garbage-collect staging dirs),
  `write_staged_group` (stage arrays + attrs). Frozen dataclasses `CommitRecord`,
  `Committed`, `Duplicate` for immutable record-keeping; exception hierarchy
  `DurableStoreError` + specific subclasses for diagnostics. Includes fault
  injection support (XTRAX_FAULT_INJECT env var) for crash atomicity testing.

- **`ZarrStagingSink` durable create-or-join mode** (#181, U3, spec demistify
  261001_preemption-safe-cimist-fitting): `SinkSpec` gains defaulted fields
  `open_mode` (`"exclusive"` | `"create_or_join"`), `store_identity` and
  `prefixes` (also accepted by `derive_sink_spec`). With
  `open_mode="create_or_join"` the sink atomically creates the store root (or
  joins an existing one after an identity + prefix check), never opens the
  target with `mode="a"` and never rewrites root attrs, and commits a writer
  record at `("_xtrax_writers", run_id)` before the constructor returns. In
  this mode `stage()` takes `input_digest` (required), `input_payload`,
  `commit_meta` and `env_extra`; `drain()` commits each key atomically via
  staging + rename and returns `{key: Committed | Duplicate}`
  (`CommitConflictError` on a differing digest). New `lookup`,
  `committed_keys`, `gc_staging`, `close()` and context-manager support;
  `finalize()` raises (durable stores are never consolidated) and
  `stamp_reserved` only targets already-committed keys. `drain()` returns `{}`
  in the unchanged `exclusive` mode. The sink's shared constants moved to the
  new import-free `xtrax.run._sink_names` to break an import cycle (still
  re-exported from `zarr_sink`).

  Deliberate deviations from the exclusive-mode contract, and hardening rules of the
  durable mode:
  - `stamp_reserved(())` on the root **raises** in durable mode (a durable store never
    rewrites root attrs); stamp a committed key instead. Concurrent stamps of the same
    name on the same key by different writers are last-writer-wins.
  - Durable `drain()` **reports auto-flushed outcomes**: it returns the
    `Committed | Duplicate` outcome of every key committed since the previous explicit
    `drain()`, including keys an auto-flush (`flush_every`) committed inside `stage()`.
    A `CommitConflictError` can therefore also surface from `stage()`.
  - **Exclusive mode refuses durable stores**: opening a directory whose root carries
    `xtrax.store` with the default `open_mode="exclusive"` raises `ValueError` instead of
    rewriting the durable store's root.
  - A key can never be nested inside a committed key: `commit_key` raises
    `UnknownPrefixError` if the parent, or any ancestor below the store root, is a
    committed key; `stage()` requires the key's parent to equal or sit under a declared
    `SinkSpec.prefixes` entry (the reserved writers prefix excluded).
  - An existing **empty** directory is treated as absent by `create_or_join`; a non-empty
    non-store directory still raises `NotADurableStoreError`. The output directory is
    resolved through symlinks so staging lands beside the real target, and a staging /
    store device mismatch raises `ValueError`. `run_id` is validated as a key part before
    any filesystem work.
  - `close()` now ends the sink (all modes): `stage`/`drain`/`stamp_reserved` afterwards
    raise `RuntimeError`, and undrained keys are discarded with a `UserWarning`.
  - `lookup(verify=True)` propagates `OSError`/`MemoryError` from the content-digest
    computation instead of reporting `Corrupt`; zarr decode errors stay `Corrupt` (the
    reason now names the exception type). `gc_staging` renames each candidate to
    `<name>.gc-<uuid8>` before deleting it.
- **`xtrax-io-on-shared-filesystems` skill** (`agent_assets/skills/`). Reference for
  choosing a read pattern on NFS/Lustre/GPFS: cost is requests x latency, not bytes;
  what `md.iterload(stride=k)` and an offset-table seek reader actually read; the
  CPU-bound check first, then `nfsiostat` per-op RTT/kB-op (not `/proc/<pid>/io`
  call size, which stdio buffering makes small even for good sequential reads) to
  separate latency-bound from bandwidth-bound; stage-by-large-copy, reduce-once/read-many and
  filesystem-keyed reader selection; and the same-filesystem A/B gate before an
  I/O-path change ships. Baseline agents without it misstated what a strided
  `iterload` reads and proposed adding parallel readers to a bandwidth-bound pool.

### Removed

- **The deprecated `SafeMap` / `SafeMapIterator` / `safe_map` aliases** (#5680). They
  shipped for one release (0.4.0a11) as `DeprecationWarning` aliases of `ChunkedMap` /
  `ChunkedMapIterator` / `chunked_map` (#3644) and now raise `AttributeError` /
  `ImportError`. **Breaking.** Migrate with the ast-grep rules in
  `codemods/safemap-to-chunkedmap/`. Name-based strategy matching (`stages.topology`,
  `export.composer`) still accepts the class name `"SafeMap"`, for consumers' own
  duck-typed classes of that name (aminx's, until its deprecation). The port wave now
  targets `xtrax.transforms.map.chunked_map` and is re-sealed (manifest hash recomputed
  with `scripts/audit_port_oracle_seal.py`; the reference oracle is unchanged).

### Fixed

- **`memoize_jaxpr` admits typed PRNG keys** (`xtrax.inference`, #5679). A
  `jax.random.key(...)` argument raised `TypeError: Cannot interpret 'key<fry>' as a data
  type` during leaf classification, and a function closing over one raised from the program
  digest. A typed key now digests its impl name and `key_data` bits, so equal bits under two
  impls are different entries; key plumbing (`split`, `fold_in`) on a typed key is memoized
  and a draw is refused by the purity screen, as for a raw `PRNGKey`. Any other extended
  dtype raises `MemoKeyUnsupportedLeafError`.
- **Native/IREE parity compares integer and bool outputs exactly, per output leaf**
  (`xtrax.export`, #5688). `compare` / `verify_native_parity` used `np.allclose(rtol=1e-5)`
  for every dtype, so an index output of `1_000_009` passed for `1_000_000` and
  `ExportResult.verified` was True for a wrong native artifact, while the same program
  failed on the onnx target. Both backends now share `compare_leaves`: one result per
  output leaf, integers and bools exact, any dtype change a failure.
  `verify_native_parity` returns a `LeafParityResult` (a `ParityResult` subclass), and a
  multi-output native entry point is compared leaf by leaf instead of being stacked into
  one array. A nested list of scalars for a single output is read as one array on every
  backend (onnx used to split it into scalar leaves); a tuple of arrays is never stacked,
  so an export that collapsed two outputs into one cannot verify. Under
  `jax_enable_x64`, a NumPy oracle left at NumPy's 64-bit default (float64/int64/uint64)
  is not a dtype mismatch against its 32-bit counterpart; values are still compared,
  exactly for integers, and any other narrowing is a mismatch. An integer reference
  against a float output, or a bool <-> int swap, fails with `max_abs_diff` inf. Both
  backends narrow concrete inputs the same way (`parity.narrow_inputs`), so a float64
  NumPy input to an f32 export now runs on native too. The comparison label reads
  `(exact comparison)`. **Behaviour change:** a native export that only passed through
  float tolerance on an integer output now fails. `LeafParityResult` and the new `compare_leaves` live in
  `xtrax.export.parity` (`LeafParityResult` is still importable from
  `xtrax.export.onnx`).

- **The export gate judges the program actually exported** (`xtrax.export`, #5690).
  The op rules traced the per-element `fn` against the BATCHED `abstract_inputs` and
  swallowed the trace failure, so for an ordinary per-element function the
  "unsuppressible" `onnx-in-graph-rng` rule (and the IREE op rules) never ran; an RNG
  draw was refused only after conversion, by the graph backstop. `export_pipeline` now
  checks topology, composes the callable once, traces it once
  (`safety.trace_for_export_safety`), and every target's gate judges that trace
  (new `traced_jaxpr=` on `check_export_safety` / `validate_export_safe`). The
  `convert_to_onnx` namespace guard also restores a public JAX attribute that a
  conversion *deleted*, not only one it replaced.

- **ONNX failures surface as the documented error types** (`xtrax.export`, #5689).
  The dtype gate now judges every leaf of a pytree input (`abstract_inputs[0]['a']`) and
  the program's OUTPUTS, for every target (a bf16-returning program passed `native`);
  for `ONNX` it also judges every op by its operands' and results' dtypes, except the
  data-movement primitives ORT runs at any dtype (casts, reshape, transpose, slices,
  concatenate, gather, rev, reduce_sum). So `x.astype(jnp.bfloat16) + 1` is a
  `DtypeNotSupportedError` at plan time instead of a raw `onnxruntime`
  `NOT_IMPLEMENTED` at session creation, while the precision-emulation idiom
  `x.astype(jnp.bfloat16).astype(jnp.float32)` passes. Both the allowlist and the
  envelope (f16 and the integer types run as intermediates) are pinned by ORT tests.
  One cause is one blocker: an input dtype is not reported again by the ops that use
  it. `run_onnx` narrows inputs (no device copy) and raises `CompileError`, naming the
  declared and given inputs, when ORT cannot load or run the graph. A model of 2 GiB
  or more is written with every large tensor -- Constant-node attributes included --
  in `<model>.onnx.data`, where serialization used to raise a raw `ValueError`; a
  stale data file is removed first, and an unrelated file of that name in the CWD no
  longer fails the export.

### Changed

- **`export_pipeline` traces and composes once for all targets** (#5691), where it
  traced `fn` once per target in the gate loop and rebuilt the composed callable once
  per target in the compile loop.

## [0.4.0a11] - 2026-10-01

### Added

- **`xtrax.export.divergence` / `xtrax.export.rings`: divergence mapping** (#5095, #156).
  Locates *where* a compiled artifact departs from production JAX, instead of
  `parity.compare`'s single scalar. `compare_pytree` reports per-leaf `LeafDivergence`
  (ULP distance for floats; exact match for integers and bools, with no budget, ever),
  and `classify_probes` labels each named probe with a `DivergenceClass` in precedence
  order (`UNCOMPARABLE`, `DISCRETE_FLIP`, `INJECTED`, `AMPLIFIED`, `ATTENUATED`, `CLEAN`)
  against fusion-noise budgets measured on the host (`budget_leaf`, `budget_key`). The
  ring runners `r0_replay_gate`, `r1_target_isa`, `r2a_fusion`, `r2b_lowering`, `r3_probe`
  and `run_ladder` separate replay, target-ISA, fusion and lowering effects. Ring
  execution needs the `export` extra; `divergence` itself imports no IREE.

- **`xtrax.export.ONNX`: an ONNX export target, verified on ONNX Runtime** (new `onnx`
  extra: `jax2onnx>=0.17.0,<0.18`, `onnx`, `onnxruntime`; no IREE). `export_pipeline(...,
  targets=(ONNX,))` converts the composed callable with jax2onnx at opset 23 and runs the
  `.onnx` file on ORT's CPU execution provider against `reference_fn`. The target is
  `EXECUTED`. Parity is per output leaf (`LeafParityResult`): integer and bool leaves must
  match exactly and every leaf must keep its dtype. `ExportResult.onnx_census` records where
  int64 occurs inside the graph (from `TopK`/`ArgMax`); graph I/O keeps JAX's dtypes.
  `Target.backend` (`Backend.IREE` / `Backend.ONNX`) now selects the export-safety op
  rules. The four IREE rules apply to IREE targets only, because stable sorts and
  `lax.top_k` were measured exact on ORT. `ONNX` adds `"onnx-in-graph-rng"`
  (unsuppressible), because jax2onnx lowers `jax.random` draws to key-ignoring ONNX
  `RandomUniform`. Conversion refuses `jax_enable_x64`, and it undoes the `jnp.cumsum`
  patch that a process's first jax2onnx call leaves behind. `IREE_TARGETS` is the five
  IREE targets; `ALL_TARGETS` adds `ONNX`; `compile_for_target` refuses a non-IREE target.
  `ExportResult.artifact_bytes` is a backend-neutral alias of `vmfb_bytes`. Measured in
  `.praxia/docs/research/260930_onnx-route-spike.md`. The verification covers ORT CPU
  only, not ORT Web.

- **`xtrax.tiling.dedup_synthesis.verify_dedup_outputs(spec, fn, xs, *, rtol, atol)`**
  (#5217): checks claim (ii), that dispatching `fn` through the dedup path
  (vmap over canonical rows, then gather) reproduces per-row `jax.vmap(fn)(xs)`. The
  comparison is numeric (`allclose`, NaN equal to NaN; integer and bool outputs exact),
  never bitwise: XLA fuses the two programs differently, and float32 outputs legitimately
  differ (3.3e-6 measured at N=2000, K=7). It raises `DedupOutputMismatchError` with
  the worst error and first bad row. Both paths are jitted by default (`jit=True`), as
  production dispatch is, and equal infinities compare equal. It compares on device and
  moves one mask through the module's single `_to_host` route. It complements `verify_dedup_spec`,
  which checks input-row identity.

- **`MemoPolicy(execute_screened=True)`** (`xtrax.inference`, #5233): on a cache miss,
  run the jitted **screened program** of that call signature instead of the function
  eagerly, so a path the trace never took (`isinstance(x, jax.core.Tracer)`, identity
  checks, a caught `ConcretizationTypeError`) cannot be what gets cached. The runner
  is built from the same closed jaxpr the screen inspected; if it has to be rebuilt,
  a re-trace whose program differs from the screened one raises `MemoStalenessError`.
  Spot checks under this policy re-run the screened program, not the function
  eagerly. Opt-in; the default is unchanged. Costs one compile per signature, and a
  miss returns fresh output buffers, never an argument's.

- **`xtrax.profiling.loop_scaling`: flag loop bodies whose per-iteration cost grows with
  the loop's own extent** (debt #1983). `loop_bodies(fn, *args)` traces `fn` (nothing
  is compiled or run) and reports every `scan`/`while` body at any depth with ONE
  iteration's work and the trip count -- the multiplication XLA's `cost_analysis`
  hides by counting a `while` body once. `extent_scaling_report(fn, make_args, n)`
  traces at `n` and `2n` and flags bodies whose per-iteration work grows (ratio near
  2.0 for full-extent recompute, near 1.0 for incremental), the O(L^2)
  autoregressive-sampler shape that correctness tests and small-L wall clocks miss.
  In-place `dynamic_update_slice`/`scatter` are costed by their update, so a correct
  incremental body writing its row into an `(L, ...)` buffer is not a false positive.
  Work is a ratio-oriented proxy (dot_general FLOPs + output elements), not a
  FLOP-exact model. The debt's third check (consumption ratio) is not implemented.

- **`xtrax.tiling.dedup_synthesis.verify_dedup_spec`**: checks a `DedupSpec`'s
  row-identity claim (spec §4.3/§10.2-10.3, backlog #5172). Every row is
  compared **bitwise, per leaf, in native byte layout** against its claimed
  canonical row — the same byte definition `synthesize_dedup_spec` uses, so
  the two functions cannot drift. A mismatch in *any* leaf fails the row
  (union semantics); `DedupSpecVerificationError.first_bad_row`/`n_bad`/
  `leaf_index` describe the first failure. Structural checks (array type,
  integer dtype, `k`/length/bounds) run first, entirely on host, before any
  device→host transfer; the row check then moves exactly one `(N, L)` boolean
  mismatch mask (`N * L` bytes). Checks row-equality only (claim i); the
  numeric compute-equivalence claim (ii, "does re-running `fn` on the deduped
  rows reproduce `fn`'s full output") needs `fn` and is follow-up #5217.
- **`xtrax.inference` donation rejection, both directions**: `memoize_jaxpr`
  now rejects, at admission, any wrapped function whose traced jaxpr carries a
  donation marker on any equation at any nesting depth — `jit`/`pjit`
  `donate_argnums`/`donate_argnames`, and `device_put(..., donate=True)` via
  its `copy_semantics` operand. Previously only the output side was half-built
  and the input side was entirely unchecked (`test_donation_rejected_at_wrap`
  asserted an unrelated policy field and never exercised donation). New
  `MemoDonationError(MemoImpurityError)`, exported from
  `xtrax.inference.errors.__all__` and `xtrax.inference`, carries a structured
  `.sites` tuple naming each offending equation, its carrier
  (`"donated_invars"`/`"copy_semantics"`), and — for top-level equations —
  which flattened input leaf it aliases. Two known fail-open admission paths
  are not fixed (the screen runs once, keyed off the first call's shape, and
  never traces kwargs) and are pinned with strict `xfail` markers citing
  #5214/#5215.

### Fixed

- **`ZarrStagingSink.drain` errors name the payload that failed** (`xtrax.run`, #5552).
  A zarr write error now carries the staged key, group path, array name, shape and
  dtype in its message, on the same exception object (type unchanged), and says the
  pending buffer was not cleared. The context is in the message itself because JAX's
  `io_callback` re-renders only the original exception's message line, not notes or
  chained causes; previously a drain from the activation-capture path surfaced as an
  opaque `CpuCallback error` naming only zarr internals.

- **CLI shape specs: the documented form works and errors show a real example**
  (`xtrax.cli`, #5174). `<dtype>` in the grammar was a placeholder but read as literal
  syntax, so `x=(4,3)<float32>` was rejected. The parser now accepts the long aliases
  `float32`/`float64`/`int32` and an optional `<...>` around the dtype, reports a
  missing dtype specifically, and every error message shows `e.g. x=(4,3)f32`.

- **`BatchPlanner.plan()`: duplicate `DedupSpec`s raise; dedup never makes a budget
  plan infeasible** (`xtrax.tiling`, #5175). Two `DedupSpec`s for one axis now raise
  `DedupSpecCollisionError` (plan() routes through `merge_dedup_specs`) instead of
  silently keeping the last. **Behaviour change** for callers passing duplicates. In
  joint-budget mode a DedupGather axis was a fixed decision, so adding a `DedupSpec`
  could turn a plan that fit into `BudgetInfeasibleError`. If every ordinary demotion
  still leaves the plan over budget, dedup axes are now planned without dedup as a
  last resort, one at a time in spec order until the plan fits, each with a
  `RuntimeWarning` and `"DedupSpec dropped"` in the decision's `reasoning`. The error is raised only if that fails too.

- **`memoize_jaxpr` purity screen audited against the primitives JAX registers**
  (`xtrax.inference`, #5234). Five of the eleven banned names were registered by
  neither JAX 0.10.2 nor 0.11.1 (the "stateful" list banned nothing at all). The screen
  now rejects `debug_print` and `debug_callback` (a hit would skip the side effect),
  `random_gamma`, `rng_uniform`, `threefry2x32`/`threefry4x32` and `philox2x32`/`philox4x32`,
  and **a function that
  closes over a mutable `jax.Ref`**, which previously crashed in the program digest
  with `ValueError: Out of bound indexer`. Key plumbing on a key argument
  (`random_split`, `random_fold_in`, `random_wrap`, `random_unwrap`, `random_clone`)
  is admitted explicitly. Tests fail if any listed name stops being registered, or if
  a registered random/rng/callback/debug-family primitive is left unclassified.
  **Behaviour change:** a memoized function that calls `jax.debug.print` is now rejected.

- **Chunked mapping handles a ragged final chunk** (`xtrax.transforms`,
  `xtrax.tiling`, #5565). An axis whose cardinality is not a multiple of its batch
  size used to plan as SafeMap with a `RuntimeWarning`, then raise `ValueError` at
  dispatch. Any data-driven axis could hit this; an MSA depth of 50,713 = 13·47·83 has
  no usable divisor. `jax.lax.map(batch_size=...)` already runs the remainder as one
  smaller vmapped chunk, with no padding and peak memory still bounded by the batch
  size. xtrax's own divisibility check was the only obstacle, and it is removed (verified
  equal to `vmap` on JAX 0.10.2 and 0.11.1, including n=50,713 with batch 512). The
  planner no longer warns, and the decision's reasoning notes the ragged remainder.
  **Behaviour change:** a non-divisible axis whose `memory_estimator` says it fits now
  plans as `Vmap`, like a divisible one. The old Rule 5 forced chunking there only
  because dispatch was going to fail.

- **`ZarrStagingSink.drain` stores 0-d and zero-length payloads** (`xtrax.run`, #161).
  Chunks are now rank-matched (`tuple(max(d, 1) for d in shape)`); previously a
  0-d per-step scalar, or any array with a zero-length dimension such as
  `(0, 3)`, raised inside zarr -- surfacing from `io_callback` as an opaque
  `JaxRuntimeError: INTERNAL: CpuCallback error calling callback`.
- **`synthesize_dedup_spec` row identity is now exact native bytes per leaf**,
  not numpy-float equality over a promoted, dtype-concatenated view (spec
  §4.3/§5, backlog #5172). The predecessor's `jnp.concatenate` of raw leaves
  silently merged rows that are not the same bits:
  - `+0.0`/`-0.0` compared equal (`np.unique(axis=0)` merges signed zeros) —
    they are now distinct, so `k` may rise, possibly past `max_unique_k`
    (turning a `"synthesized"` result into `"k_over_limit"`).
  - Mixed-dtype leaves (e.g. an int32 leaf beside a float32 leaf) were
    promoted to a common floating dtype before comparison, silently
    collapsing distinct integers (`2**24` and `2**24+1` compared equal).
  - A numpy int64/float64 leaf was truncated to int32/float32 by the implicit
    `jnp.moveaxis` conversion under x64-off, silently merging wide values.
  - `axis != 0` read the wrong dimension for `N` (`stacked.shape[axis]` on an
    already batch-first array), producing a spurious `ValueError` or a
    garbage duplication ratio; `axis != 0` now works correctly.
  - `transfer_bytes_spent`/`k_bucket_bytes` now report true per-leaf byte
    widths instead of the promoted-dtype width (e.g. an int8 leaf beside a
    float32 leaf now counts 5 bytes/row, not the promoted 8).
  - Bitwise-identical NaN rows now deduplicate (`k` may fall), which is also
    now sound rather than an artifact of promotion.
  - Typed PRNG key leaves are now refused with `DedupSynthesisUnsupportedError`
    naming `jax.random.key_data`, where they used to raise a raw `TypeError`
    from deep inside JAX. Sub-byte integer leaves (int2/int4/uint2/uint4) are
    compared exactly after a lossless widening `astype`; sub-byte float leaves
    (float4/float6) are refused rather than silently widened through a lossy
    numeric conversion.
  Every change is in the sound direction (see risk table, spec §12); none is a
  regression. `tests/tiling/test_dedup_synthesis.py` (PR #104) passes
  unmodified against the new implementation.
- **`added-types-diff` gate no longer trips on nested `def`s** (backlog #5205,
  verified live on PR #156's `visit` closure inside `_topological_order`):
  `_PublicFunctionCollector.visit_FunctionDef` now stops descending into
  function bodies after recording a public-named def, so a nested def is
  never mis-reported as "unable to locate callable", and no longer silently
  overwrites a same-named top-level def's entry in the gate's base/head maps
  — previously this could either spuriously flag an unrelated top-level def
  as changed, or, more seriously, hide a real signature change to it (both
  base and head maps held the unchanged nested def under that key). A
  factory pattern (`foo = _make_foo()` with a nested `def foo`) is now
  invisible to the gate rather than accidentally loud; covering assigned
  callables would be a gate feature, not this fix.

### Changed

- **Divergence rungs R1 and R2b report per-probe divergence; `UNCOMPARABLE` class**
  (`xtrax.export`, #5210, #5209, #157). Both rungs used to return `passed=True` with
  `probes=()` for arbitrarily large value divergence, so `all(r.passed for r in
  results)` read an integer index flip as "no divergence". `passed` stays structural
  (the precondition held and every leaf was comparable). `r1_target_isa` and
  `r2b_lowering` now take `probe_deps=` (forwarded by `run_ladder`) and populate
  `probes` with `classify_probes` against R2a's budgets; check `probes`, not `passed`,
  for value divergence. A probe with a leaf that could not be compared (shape/dtype
  mismatch) is now `UNCOMPARABLE` rather than `AMPLIFIED`.

- **`memoize_jaxpr` cache hits are cheaper** (`xtrax.inference`, #5241, #160): arguments
  are flattened once per call (was three times), the purity and donation screens walk
  the jaxpr once, and integer bounds are cached. Keys and screening results are
  unchanged (pinned against the previous implementation).

- **`export_pipeline` runs every target's safety gate first, then evaluates `reference_fn`
  once, then compiles**, rather than gating, compiling and calling `reference_fn` per target. That makes the oracle independent of
  toolchain side effects on `jax.numpy`.
- **`[tool.uv] override-dependencies` lifts jax2onnx's `orbax-checkpoint<0.11.37` cap** in
  xtrax's own lock, so the `onnx` extra adds packages without pulling every environment
  down to orbax 0.11.x (the lock keeps 0.12.1). The ONNX and checkpoint suites pass on orbax
  0.12.x. A `pip install xtrax[onnx]` from PyPI still gets jax2onnx's own cap.

- **`SafeMap` → `ChunkedMap`, `SafeMapIterator` → `ChunkedMapIterator`, `safe_map` →
  `chunked_map`** (#3644). `safe_map` already means something else in JAX
  (`jax._src.util.safe_map`, a length-checked map), and "Safe" suggested the
  `xtrax.safety` subsystem; the strategy is memory-bounded chunking. **The old names
  import for one release as deprecated aliases** (each access raises
  `DeprecationWarning`); they are removed in the release after. Migrate mechanically
  with `codemods/safemap-to-chunkedmap/`: ast-grep rules for Python plus a Markdown
  script, with a fixture pair showing the exact transformation (see its README).
  **Consumers that dispatch on the strategy's class *name*** (aminx's
  `kernel_dispatch.py` compares `type(strategy).__name__ == "SafeMap"`) must accept
  `"ChunkedMap"` before upgrading. The alias constructs the new class, and the codemod
  rewrites that comparison. xtrax's own name-based matching accepts both names for
  this release. Strategy-name strings in EDA output (`strategy_counts` keys, the
  `strategy` column) are now `"ChunkedMap"`, and `RuntimeBundle.iterator`'s annotation
  names `ChunkedMapIterator`. The sealed `port/` apparatus still
  references `xtrax.transforms.map.safe_map` through the alias, so it must be re-sealed
  before the aliases are removed.

- **Dependency floors raised to versions that actually work** (debt #738). With
  jax at its own floor (0.10.2), `equinox>=0.11.0` and `orbax-checkpoint>=0.6.0`
  could not `import xtrax` at all (`jax.core.Primitive`, `jax.sharding.PositionalSharding`
  and `jax.experimental.layout.DeviceLocalLayout` were removed from jax). New floors,
  each the lowest release that installs on Python 3.13 and passes the
  dependency-sensitive suite at jax 0.10.2: `equinox>=0.13.1`,
  `orbax-checkpoint>=0.11.17`, `numpy>=2.1` (already forced by jax), and in the
  extras `zarr>=3.0.8` and `tyro>=0.9.1`. `optax>=0.2.3` is unchanged and verified.
  A new `dependency-floors` CI job resolves core + io + cli with
  `--resolution lowest-direct`, fails if any floor is not exactly what installed
  (`scripts/check_dependency_floors.py`), and runs the tests there.
- **`xtrax.checkpoint` moved off orbax's deprecated `CheckpointManager` kwargs**
  (debt #739). `max_to_keep`/`keep_period` become an `AnyPreservationPolicy` of
  `LatestN` + `EveryNSteps` (same retention: the latest N plus every multiple of
  the period), `item_handlers=` becomes a `handler_registry`, and `items=`
  save/restore becomes `args=`. The public signatures and on-disk layout are
  unchanged, and checkpoints written by earlier releases restore as before
  (pinned by a test). orbax's v1 API is not adopted: as of orbax 0.12.1 it is
  still `orbax.checkpoint.experimental.v1`.
- **`ZarrStagingSink` enforces `extension_schema` `required` fields at `drain()`,
  not per `stage()` call** (`xtrax.run`, debt #1540). Per-call enforcement
  contradicted the documented merge-on-repeat contract: a key whose required
  fields arrived in a later `stage()` call was rejected. Value-type checks stay
  fail-fast at `stage()`. `drain()` checks every key with staged attrs against
  its on-disk attrs merged with pending ones, raises before writing anything,
  and leaves the buffer intact. An auto-flush (`flush_every`) is a drain, so a
  split must complete within `flush_every` calls.

## [0.4.0a10] - 2026-09-13

Release theme: the export boundary now refuses, at plan time, the constructs
measured to break through IREE — rather than letting them compile and diverge
or abort. Four of the five rules below were found by measuring a real model's
exported artifact against eager JAX, and two of them produce **no compile
error and no exception**: the artifact runs and returns wrong integers, which a
float-tolerance parity check reports as `max_abs_diff 0.0` and passes.

### Added

- **Export-safety rule `"unlegalizable-op"`**: refuses `jax.lax.top_k`, which
  IREE's StableHLO importer rejects outright on every target (it lowers to a
  `stablehlo.composite` wrapping `chlo.top_k`, marked explicitly illegal). This
  rule is **not** suppressible — no caller can accept a risk that is a hard
  compile failure. The suggested replacement,
  `jnp.argsort(-x, axis=-1, stable=True)[..., :k]`, itself trips the next rule.
- **Export-safety rule `"sort-stability"`**: flags a stable sort/argsort, whose
  tie order IREE does not preserve relative to XLA. Measured: an integer
  `sort_key_val` with 64 slots and 4 distinct keys differs at **45 of 64**
  positions. Nothing crashes and no float leaf changes, so a parity gate built
  on float magnitude cannot see it at any tolerance. Fold an explicit index
  tiebreak into the sort key rather than relying on backend stability.
- **Export-safety rule `"random-permutation"`**: flags `jax.random.permutation`,
  which IREE compiles into a valid but **wrong** permutation when the key is a
  split half and the sibling half is also consumed — 248 of 256 positions
  differ. Suppressible via `acknowledged`. It is **necessary but not
  sufficient**: the underlying defect is the split-derived key, not the
  permutation, so a clean run of this rule does not certify a model's
  randomness. Reported upstream as
  [iree-org/iree#24927](https://github.com/iree-org/iree/issues/24927).
- **Export-safety rule `"unbatched-threefry-key"`**: flags threefry driven by a
  key that arrives as a **runtime input** with at most one lane. Unlike the
  rules above this is not a wrong answer but a **hard abort at invocation** —
  IREE's HAL rejects the command buffer with a bogus `2**39`-ish length against
  a 192-byte binding. The boundary is the lane count, not the rank of the key
  input and not `vmap`-versus-not: an un-batched draw aborts from `(2,)`,
  `(1,2)` and `(4,2)` inputs alike, one lane aborts, two or more is bit-exact.
  A **constant** key is genuinely exempt (it stays exact under
  `--iree-opt-const-eval=false`), so the rule tests input provenance rather
  than matching the primitive alone. Two measured workarounds, both named in
  the blocker text: give the key two or more lanes, or set
  `jax_threefry_partitionable=False`. Reported upstream as
  [iree-org/iree#24929](https://github.com/iree-org/iree/issues/24929).
- **`xtrax.export.targets.NATIVE_PORTABLE`**: a fifth export target, `"native-portable"`,
  `EXECUTED` like `NATIVE` but compiled with a fixed `--iree-llvmcpu-target-cpu=x86-64-v2`
  baseline instead of `=host`, so the artifact runs on a recipient's CPU rather than only
  the one that compiled it (any x86-64 CPU from roughly 2013 onward — not "any CPU since
  2009"). It still requires an IREE runtime on the recipient's machine; it is not a
  standalone binary. `NATIVE` is unchanged and remains host-tuned, for use as a parity
  oracle only.
- **`export-runtime` extra**: `iree-base-runtime`, `safetensors` and `huggingface_hub` —
  what a consumer needs to *load and run* an exported artifact. The existing `export`
  extra now aliases it and adds `iree-base-compiler`, which is what xtrax needs to
  *build* one. The compiler is 349 MB installed against ~7 MB for the runtime, so a
  downstream package that only executes artifacts no longer pays for a toolchain it
  never invokes.

### Fixed

- **Pytree inputs are flattened at vmfb invocation.** `export_pipeline`'s
  EXECUTED path passed a pytree straight to the runtime, which accepts only
  flat arrays, so any model whose entry point takes a structured input failed
  at invocation rather than at export — late, and with an error naming the
  runtime instead of the call site.
- **A materializing inner sink is refused on the literal-`vmap` route.**
  `compose_vmap_of_scan`'s literal-`vmap` route discarded the materialized
  `ys` and returned only the final carry, so a sink declared `materialize=True`
  silently lost every value but the last. It now refuses rather than
  under-reporting.
- **`zarr_content_digest` now excludes provenance attrs by default**: run-ID,
  git SHA/branch/dirty, and creation timestamp are excluded from the digest
  unless explicitly included via the new `include_provenance=True` parameter
  (same parameter added to `update_zarr_node_digest` and `run_repro_floor`, the
  latter so a caller holding a digest pinned before this change has a way to
  reproduce it). This restores the
  intended contract that digest values are unaffected by which process or
  session wrote the store. **Digest values computed before this change do not
  match values computed after it for any sink-written store** — stored done-marker
  digests will mismatch and must be recomputed.
- **`export-toolchain-tests` now fails on a skipped test.** The job ran a bare
  `pytest tests/export/ -q`, which was exactly as green with every export test skipped
  as with all of them run — so the "tested against real IREE 3.11" claim it exists to
  make had nothing enforcing it. A silently-failed toolchain install now fails the job
  instead of passing it.
- **Corrected the 0.4.0a8 entry below**, which described `WASM32` as `EXECUTED`. It has
  always been registered `CODEGEN_ONLY` (`targets.py`); executing a wasm artifact needs
  an emsdk-built IREE runtime that no published package provides. The code was right and
  the changelog was wrong, in the direction that overstates what ships.

### Changed

- **`grain` and `pytest-asyncio` are no longer runtime dependencies.** Both were
  declared in `[project].dependencies` but imported nowhere under `src/`.
  `grain` moves to a new `data` extra — consumers who need it must now install
  `xtrax[data]`. `pytest-asyncio` is removed entirely from runtime deps; it
  remains in the `dev` extra, where a test-only plugin belongs. **This is
  consumer-visible:** anyone relying on `pip install xtrax` to pull `grain` or
  `pytest-asyncio` must now ask for them explicitly.
- **The `dev` and `eda` dependency-groups are now thin aliases of their
  matching extras** (`dev = ["xtrax[dev]"]`). Previously each group duplicated
  its extra's contents and had drifted apart, so `uv sync --group <x>` kept the
  group and dropped the extra — silently uninstalling `beartype`, `chex`,
  `interrogate`, `jaxlint` and `libcst`. An alias has no content of its own and
  so cannot diverge. A new contract enforces the shape for any name declared in
  both tables.

## [0.4.0a9] - 2026-09-07

### Added

- **`derive_sink_spec` and `new_run_id` on the public API**: both names now
  resolve from the `xtrax` root and are listed in `distribution/public_api.toml`
  as tier-1 exports. Backlog #4457 item (2) deliberately held them back until a
  real driver consumed the seam, so that the surface would be shaped by a caller
  rather than by speculation; the run CLI now builds its sink exclusively
  through `derive_sink_spec`/`make_sink`, which discharges that condition. They
  are promoted through the lazy-export path the contract requires, so
  `xtrax.derive_sink_spec` still imports `xtrax.run` only on first attribute
  access rather than at package import.

### Changed

- **The `controller` extra is capped at `bathos<0.14`** (was an uncapped
  `>=0.13.0a1`). Once CI depended on that extra, an uncapped alpha meant an
  unrelated bathos release could turn the board red without a commit here.


## [0.4.0a8] - 2026-09-02

### Added

- **`xtrax.export`**: compiles a `BatchPlan` to a standalone artifact via
  StableHLO and IREE. `export_pipeline` folds a plan into one traceable
  callable, exports it, compiles it for one or more targets, and verifies
  numerical parity against the original JAX callable where the target can be
  executed. Four targets ship — `NATIVE` (`EXECUTED`), `WASM32`,
  `VULKAN_SPIRV`, and `METAL_SPIRV` (all `CODEGEN_ONLY`) — with `VerificationLevel`
  recording how far each one is actually checked, so a `CODEGEN_ONLY` artifact
  never reads as verified. SPIR-V shaders are extracted from `vulkan-spirv`
  builds, magic-filtered so `metal-spirv`'s MSL dump is rejected rather than
  mistaken for a shader. The IREE toolchain is imported lazily behind the
  `export` extra: a missing toolchain surfaces at compile time with a clear
  error, never as an ImportError at module load.
  (spec: `.praxia/docs/specs/260901_xtrax-export-webgpu.md`)
- **Multi-axis export composition**: `build_traceable_callable` handles an outer
  `Vmap` axis wrapping an inner `Scan` axis, following the composition recipe
  certified by `tests/stages/test_nested_ordering.py` literally rather than
  routing through the executor's scan helper. A plan whose shape forces a
  literal `jax.vmap` around a lane-dependent ordered `Tap`/`Sink` is refused
  with `MultiAxisCompositionError`, carrying the executor's own guidance rather
  than a second paraphrase of it. Previously any plan with more than one axis
  was refused outright.
- **Export-safety gate** (`check_export_safety`, `validate_export_safe`): blocks
  BCOO leaves and out-of-envelope dtypes before any compile runs, reporting
  every blocker at once. Leaves reachable through `fn`'s closure are scanned
  alongside `abstract_inputs`, which is what catches a weight held in an
  Equinox module — the commonest shape, since weights are never arguments.
- **`materialize` boundary kind** (`xtrax.stages`): marks a sink whose values
  the export should carry as real outputs, stripped to a no-op inside the
  exported artifact so the compiled graph stays pure.
- **`load_hf_weights`**: loads safetensors checkpoints, casting any dtype the
  chosen target cannot verify and reporting every cast leaf untruncated. `f64`
  raises rather than casting silently.

### Changed

- The export dtype envelope is measured against IREE 3.11.0 rather than derived
  from a backend's published numeric model. Two results are worth stating
  because both contradict the obvious assumption: `f64` is rejected on **every**
  target including `NATIVE`, because IREE does not refuse it — `ConvertTypesPass`
  demotes it to `f32` and rewrites the entry point's public signature with a
  warning, which on a `CODEGEN_ONLY` target is never surfaced at all. And `bf16`
  splits by verification level rather than by backend: it compiles everywhere
  with its signature untouched, and fails only in IREE's runtime
  buffer-to-numpy mapping, so a `CODEGEN_ONLY` target carries it while an
  `EXECUTED` target cannot verify it.
- `huggingface_hub` pinned to `>=1,<2`. The previous `>=0.24,<2` spanned the
  1.0 break and was resolving across API generations.

### Fixed

- The release-readiness gate no longer requires markers for a staging index that
  was retired. `distribution/release_readiness.toml` asserted that
  `publish.yml` contained `publish-testpypi` and `test.pypi.org/legacy`, but the
  TestPyPI job was deliberately dropped on 2026-07-02 ("publishing to PyPI only
  by decision"). The gate had therefore reported `BLOCKED_AUTOMATED` for two
  months on a condition no longer wanted, with every other check passing.
  `just audit-release-readiness` was the only place that failure could appear,
  and nothing in CI ran it, so it surfaced only when someone went to cut a
  release. Its hermetic half now runs inside `audit-deterministic` as
  `audit-release-readiness-contract`, alongside every other
  `tests/distribution` gate, and the test asserts against the config's own
  marker list rather than restating it — the duplication is what let the two
  drift apart.

### Note

- No WebGPU target ships. IREE's Vulkan HAL passes dispatch parameters through
  push constants, which are not a WebGPU capability, so naga rejects every
  SPIR-V module IREE emits and no flag removes them. The question is tracked as
  research rather than left implied by a `VALIDATED` target that nothing
  registers.

## [0.4.0a7] - 2026-08-27

### Added

- **CSE runtime-optimization layer** (`xtrax.inference`, `xtrax.tiling`): `analyze_cse`
  detects structurally duplicate jaxpr equations via a union-find fixpoint pass
  (opcode + params + operand identity, matching XLA's `hlo_cse.cc` equivalence rule),
  exposed via a new `xtrax explain --report cse` CLI path. Dedup-spec synthesis
  (`xtrax.tiling.dedup_synthesis`) adds two-stage sample-then-exact duplicate
  detection for tiling batches, avoiding an O(N) device-to-host transfer unless
  sampled duplication justifies the exact pass. `xtrax.inference.memo` gains a
  `MemoizedCallable` Protocol and a dual-signature `@overload` for `memoize_jaxpr`.
  Jury-audited over two review rounds. (spec: `.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md`)
- **`xtrax.profiling` core module + `xtrax-probing` skill**: `ProbeRecord` schema
  with a fail-closed claim-validity contract (an unsupported claim raises at
  construction, never silently passes) upstreamed from prolix's measurement
  tooling, plus HLO-as-text + real Perfetto runtime-trace parsing (`trace.py`),
  a pytest-benchmark-to-`ProbeRecord` bridge (`bench.py`), and report generation.
  New `xtrax-probing` skill documents the contract, stage-0/1/2 probe drivers,
  and gate/controller integration.
- **`RunSpec.run_id` + `derive_sink_spec` seam** (`xtrax.run`): `RunSpec` gains
  an optional `run_id` (defaults to `None`) and a `new_run_id()` helper; the new
  `derive_sink_spec` seam threads run identity from `RunSpec` through to sink
  construction. The `xtrax run` CLI verb now persists this provenance
  automatically via the same seam.
- First real GPU stage-2 `ProbeRecord`s captured against real hardware (aminx
  L40S dogfood) — validates the stage-0/1/2 taxonomy end-to-end for the first
  time; coverage is currently single-GPU/single-vendor, broader multi-hardware
  campaign automation is tracked separately.
- **`xtrax-optimizing` skill + Tier-gated probe drivers** (`agent_assets/skills/xtrax-optimizing`,
  `scripts/prof_stage{0,1}_*.py`): three-tier taxonomy separating host-boundary
  mechanics (ordered/unordered Tap/Sink cost), data movement
  (`async_indexed_stream` prefetch overlap), and composition-level changes
  (on-the-fly vs materialized one-hot), each exemplified by a driver that emits
  claim-valid ProbeRecords: `prof_stage0_onehot_cost` (never-execute cost
  analysis), `prof_stage1_onehot_micro` (named-scope attribution +
  parity gate before measurement), `prof_stage1_host_boundary` (correctness
  gate + per-variant dispatch counts), `prof_stage1_feed_overlap` (regime
  guard in-record). Smoke/unit pins in `tests/scripts/test_prof_optimizing_drivers.py`.
  (scope: `.praxia/docs/specs/260825_jax-optimizing-skill-scope.md`)
- **Sink provenance tracking** (`xtrax.run`): `ZarrStagingSink` now auto-captures
  static run provenance for downstream consumers. `SinkSpec` gains a required
  `run_id` and an optional JSON-Schema-style `extension_schema`. The store's
  root group receives the full record (`git_sha`, `git_branch`, `git_dirty`,
  `run_id`, `created_at` as ISO-8601 UTC); each drained key's group gets a
  minimal `run_id`/`git_sha` pointer. Git capture never raises (falls back to
  `git_sha="unknown"` with a `UserWarning`). Core field names are reserved
  against caller attrs; schema validation happens at `stage()` time. New
  `finalize()` method consolidates store metadata exactly once (refusing to
  run while staged payloads are undrained) and locks the sink; opening a
  second sink on the same `output_dir` with a different
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

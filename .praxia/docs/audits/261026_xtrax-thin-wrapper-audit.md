---
title: 'xtrax thin-wrapper audit: what to migrate off Equinox and what to deprecate'
description: 'Every src/xtrax subpackage classified decision / domain-infra / thin-wrapper, with eqx usage, consumer call sites at origin/main, and a keep/deprecate/split call; shrinks the 19-class Equinox migration to 13 and corrects four spec assumptions.'
status: final
task_id: 261023_equinox-decouple-pallas
date: '261026'
backlog_ids: '6107'
base_sha: '90ea208'
verdict: 'Of 19 eqx.Module classes, 13 migrate to Struct and 6 are better deprecated or kept out of the pytree layer; 4 spec assumptions need correction before step 2.'
---
# xtrax thin-wrapper audit: what to migrate off Equinox and what to deprecate

Backlog #6107. Spec under audit: `.praxia/docs/specs/261023_equinox-decoupling.md`. Line numbers are at
xtrax `90ea208`; consumer lines are at each repo's `origin/main` (aminx `0729c45`, plegadx `cb16660`,
prolix `f6200c2`, demistify `edb73b0`, denxity `1fa0a23`, asr `7f7a7d7`). JAX 0.11.1, optax 0.2.8,
equinox 0.13.8, orbax-checkpoint 0.12.1 read from `.venv/lib/python3.13/site-packages`.

## Summary

1. **Few subpackages are thin wrappers.** 18 of 23 subpackages (and the top-level modules) are decisions
   or xtrax-owned infrastructure. Two are thin as a whole: `distributed` and `io`. Three split:
   `sparse` (two renames plus an unconsumed policy), `safety` (`manager.py` thin, `ops.py` not) and
   `training` (the checkify step, two loss sums and two optimizer helpers thin, the rest not).
2. **Deprecate, do not migrate (6 of the 19 classes):** `ShardingPolicy`, `SafetyManager`,
   `SafetyTrainStep`, `WeightedLoss`, `MultiTaskLoss`, and `SparsePolicy` (the last as a plain frozen
   dataclass, not a pytree). None has a consumer at origin/main. Each is a rename or a few lines of a JAX
   call (sections 3 and 4). The other 13 migrate to `Struct`.
3. **Also deprecate (no eqx, no consumer):** `xtrax.distributed` (`jax.distributed.initialize()` and
   `jax.make_mesh` already do it), `xtrax.io` (an alias of `xtrax.engine.io`), `sparse_filter_jit`
   (literally `eqx.filter_jit`), `make_sparse_forward_fn`, `partition_labels`, and optionally
   `accumulate_grads`. Deprecating `SafetyTrainStep` and `sparse_filter_jit` removes two of the three
   planned `model_jit` callers.
4. **Keep although thin:** `chunked_map` is not a rename of `lax.map(batch_size=)`; it carries the size-1
   vmap guard (#2520) and a remainder-of-1 peel that JAX does not. `safe_scan` (demistify tests its
   zero-length error), `safe_norm` (finite gradient at 0, probed), `JaxScanIterator` and
   `WhileLoopIterator` (aminx stores them as pytree fields), `Trainer` (the `xtrax run` and `xtrax resume`
   verbs build it), `checkpoint` (pins the on-disk layout aminx resumes from).
5. **Spec corrections:** (a) `DataModule._epochs` is a static `init=False` field, which makes
   `jax.tree_util.register_dataclass` raise `ValueError` (probed); (b) aminx subclasses `RunSpec`
   (`aminx/run/spec.py:144`) and the spec lists only plegadx; (c) `tiling._BatchPlanWrapper` exists only
   for `eqx.filter_jit` and prolix imports it (`prolix/fitting/config.py:22`); (d) demistify hashes
   `xtrax/tiling/`, `xtrax/transforms/` and the `chunked_map`/`safe_scan` files into its shard fingerprint
   (`demistify/shard_identity.py:166-175`).
6. **The backlog's deprecation pattern citation is wrong.** `tombstone.py` is an audit-finding
   suppression ledger (`tombstone.py:1-84`), not a deprecation mechanism. The real pattern is `_renamed.py`
   plus a PEP 562 `__getattr__` alias plus an ast-grep codemod, kept one release, then removed (section 6).
7. **Decisions for the user:** whether `accumulate_grads` goes (needs a parity probe against
   `optax.microbatching`); whether the whole `xtrax.sparse` package goes (zero consumers); how
   `DataModule(distributed=True)` guards once `init_dist` is deprecated.

## 1. Method and limits

- Classification: **decision** = encodes a policy or contract a consumer would otherwise re-implement;
  **domain-infra** = xtrax-owned infrastructure with no JAX counterpart (CLI, ledger, gates, sinks);
  **thin-wrapper** = the body reduces to a JAX, optax or orbax call. Test applied to each: does deleting it
  make a consumer re-implement a decision, or only change an import?
- Equinox use is an AST sweep of `src/xtrax` (`/tmp/.../eqx_classes.py`, throwaway; matches the spec's 19
  classes exactly). 19 files import equinox (spec §1 says 20; the extra count is not reproducible).
  Mentions in docstrings and comments (`export/pipeline.py:136`, `training/optim.py:109`,
  `profiling/trace.py:575+`, `devtools/gates/dataclass_shape_diff.py:58`) are not uses.
- One-line-wrapper sweep: an AST pass listing public functions whose whole body is `return <jax/optax/eqx
  call>`. It found `sparse_filter_jit`, `safe_norm`, `safe_reciprocal`, `no_bias_wd_mask`,
  `partition_labels`, `CarryShape.materialize`, `JaxScanIterator.__call__` (`tiling/iterator.py:167`),
  `WhileLoopIterator.__call__` (`:189`) and two profiling helpers. Everything else was read by hand.
- Consumers: `git grep` of `*.py` on each repo's `origin/main` (read-only). Single-line `from xtrax... import`
  lines were parsed; multi-line imports were found by a second targeted symbol grep. Not scanned: repos
  outside the six, dynamic imports built from strings, non-Python files. "None" below means none in those
  six repos at those SHAs.
- Not run: GPU anything. The size-1 vmap miscompile (aminx #2391) is taken from aminx's report, not
  reproduced here.

## 2. Classification of every subpackage

Lines = total lines in `src/xtrax/<pkg>`. Eqx = equinox API used in code.

| Subpackage | Lines / files | Class | Eqx use | Consumers (origin/main) | Recommendation |
|---|---|---|---|---|---|
| checkpoint | 131 / 2 | decision (over orbax) | none | aminx `ebm/training.py:127`; plegadx `checkpoint.py:21` | keep. Pins item name `"state"` and the retention union (`orbax.py:21,48-60`) |
| cli | 3298 / 25 | domain-infra | none | plegadx `cli/entrypoint.py:24` (`CLIError`), `cli/registry.py:43` (`REGISTRY`) | keep |
| composition | 951 / 7 | domain-infra | none | none | keep |
| data | 687 / 4 | decision (Grain composition, measured defaults `pipeline.py:17-24`) | `module.py:5,18,39-48`: `Module`, `field` x8 | aminx `tests/ebm/test_training.py:25` only | keep, migrate `DataModule` after fixing the `init=False` field (section 5) |
| devtools | 4817 / 28 | domain-infra | `gates/_performance_probes.py:9,30,38`: `nn.Linear` x2 | none (plegadx `scripts/port_emit.py:149` uses `findings`, not devtools) | keep; replace `nn.Linear` per spec §4 |
| distributed | 341 / 3 | **thin-wrapper** | `sharding.py:7,13,23`: `Module`, `field` | none | **deprecate** (section 4, T2) |
| eda | 973 / 6 | domain-infra | none | aminx `scripts/ebm/bucket_boundary_check.py:228`; plegadx 3 files; demistify `scripts/validation/f458_batch_mi_kernel_parity.py:363` | keep |
| engine | 797 / 3 | decision (fail-closed telemetry, cadence, early stop) | `engine.py:20,275,288-290,472-473`: `Module`, `field` x3, `nn.inference_mode`, `tree_at` | aminx `ebm/training.py:128` | keep; migrate. All fields static, so a plain frozen dataclass is enough |
| export | 7000 / 12 | decision (parity rings, safety census, toolchain pins) | none | aminx scripts (`export.ONNX`, `check_export_safety`, ...), `src/aminx/agent/tools.py:123`, `host/bucketing.py:79` (`BUCKET_LADDER`) | keep |
| inference | 3360 / 13 | decision | none (docstring mentions only, `schema.py:19,42,114`) | aminx `sampling/hoisted.py:18` (`make_hoisted`) | keep |
| intent | 3949 / 11 | decision (strategy resolution under a `Context`) | none | aminx `inference/decode/autoregressive.py:42-43` (guarded import) | keep |
| io | 41 / 2 | **thin-wrapper** (self-alias) | none | none (aminx vendored its own, `utils/_vendored_callbacks.py:13-17`) | **deprecate**; collapse into `engine.io` |
| loop | 4369 / 26 | domain-infra (research-loop gates) | none | none | keep; `checkified_execution.py:48,93` must drop `SafetyManager` |
| profiling | 3326 / 14 | domain-infra (claim-validity contract) | none | aminx scripts; asr scripts (`profiling.claims`, `loop_scaling`, `emitters`) | keep |
| run | 4149 / 16 | decision (sinks, digests, commit protocol) | `spec.py:8,14,28-36`: `Module`, `field` x6; `resolver.py:8,39`: annotation only | aminx `run/spec.py:22,144` (subclass), `host/runner.py:1965`; plegadx `infra/runspec.py:14`, `infra/xtrax_shell.py:16-17,131`; demistify `shard_identity.py:32` (digests) | keep; migrate `RunSpec`; drop the `resolver.py:8` import |
| safety | 162 / 4 | **split** | `manager.py:6,11,20-22`: `Module`, `field` x3 | denxity `distances/metrics.py:45`, `solvers.py:13` (`safe_norm`) | keep `ops.py`, `preemption.py`; deprecate `manager.py` |
| sparse | 311 / 5 | **split** | `policy.py:5,16-17`; `inference.py:33,178`: `filter_jit`, `is_array`, `Module` annotations | none | deprecate the two wrappers; decide on the rest (section 4, T1) |
| stages | 1376 / 9 | decision (topology rules, ordered-callback constraints) | `boundaries.py:84-131`: `Module`, `field` x5; `bundle.py:57`: `Module` | aminx `types/boundaries.py:5`; plegadx `_boundary.py:19`, `infra/runspec.py:15` | keep; migrate |
| telemetry | 2305 / 9 | domain-infra (append-only run ledger) | `ir.py:100,107,118,123`: lazy `filter_make_jaxpr`, `filter_jit` | none (Engine uses it, `engine.py:26-28`) | keep; port the two fallbacks in `ir.py` |
| testing | 1121 / 4 | domain-infra (parity lanes) | none | none | keep |
| tiling | 3637 / 14 | decision (planner, budget, bucket ladder, dedup) | `iterator.py:93,125,160,182,204`: `Module` x5, `field` x2 | all six repos (`AxisSpec` alone is imported in 19 aminx files) | keep; migrate the 5 iterators (section 3) |
| training | 832 / 8 | **split** | `trainer.py`, `step.py`, `types.py`, `loss.py`, `state.py`, `grad.py` (all of the spec's step-3 API) | aminx `training/trainer.py:19-20,124`, `ebm/training.py:129-130`; plegadx `harness.py:70-72`, `equinox_aflf_sampler.py:50` | keep `types`, `state`, `optim` core, `Trainer`, `ComposedLoss`; deprecate the rest (section 4) |
| transforms | 175 / 3 | decision (`chunked_map`) + thin (`safe_scan`, kept) | none | demistify `_tiling.py:62`; denxity `_tiling.py:63` | keep |
| top-level modules | 1584 / 8 | domain-infra, except `random.py` (decision) | none | aminx `run/spec_json.py:21` (`config.classify_schema_version`); plegadx `scripts/port_emit.py:149` (`findings`) | keep |

## 3. The 19 `eqx.Module` classes

Spec step numbers in the last column. "Struct" means the spec's §3 `Struct`.

| Class | Location | Verdict | Why | Spec step |
|---|---|---|---|---|
| `DataModule` | `data/module.py:18` | migrate, **fix first** | static `init=False` field `_epochs` (`:48`) breaks `register_dataclass` (section 5) | 2 |
| `AxisBoundary` | `stages/boundaries.py:84` | migrate | aminx re-exports it (`types/boundaries.py:5`); plegadx walks it | 2 |
| `StageBundle` | `stages/bundle.py:57` | migrate | plegadx subclasses it (`runspec.py:15`) | 2 |
| `VmapIterator`, `ChunkedMapIterator`, `WhileLoopWithYsIterator` | `tiling/iterator.py:93,125,204` | migrate | decisions (#2520 guard, tile, buffered ys); aminx stores `MapIterator`s as non-static pytree fields (`decode/autoregressive.py:349-350`) | 2 |
| `JaxScanIterator`, `WhileLoopIterator` | `tiling/iterator.py:160,182` | migrate as zero-field `Struct`s, **do not deprecate** | bodies are `lax.scan` / `lax.while_loop` renames (`:167`, `:189`), but aminx holds `wave_iterator: ScanIterator` as a pytree field (`autoregressive.py:350`), imports `JaxScanIterator` for it (`decode/factory.py:19`) and asserts the class name (`tests/tiling/test_dispatch_via_xtrax_parity.py:106,153`) | 2 |
| `Engine` | `engine/engine.py:275` | migrate; plain frozen dataclass suffices | every field static and no jit in `engine.py` takes `self` (grep "jit" finds two comments); aminx only constructs it and calls `fit_sync` | 2-3 |
| `Trainer` | `training/trainer.py:13` | migrate | `xtrax run` and `xtrax resume` build it (`cli/run.py:116`, `cli/resume_verb.py:107`) | 3 |
| `ResumableState` | `training/types.py:41` | migrate | aminx and plegadx construct it; plegadx `tree_at`s it | 3 |
| `ComposedLoss` | `training/loss.py:112` | migrate, last | not a rename (traced weights, aux dict, placement weights), but no consumer and the spec's hardest class (custom `__init__`, `__check_init__`) | 3 |
| `RunSpec` | `run/spec.py:14` | migrate | aminx and plegadx subclass it | 4 |
| **`SafetyTrainStep`** | `training/step.py:24` | **deprecate** | `checkify(trainer.step, errors=float_checks)` + `err.throw()` (`:55,88`); no consumer | 3 |
| **`SafetyManager`** | `safety/manager.py:11` | **deprecate** | `jax.jit(checkify.checkify(fn, errors=float_checks))` + `err.throw()` (`:44,48`); `check_nans`/`check_infs` are never read (`:20-22` vs `:38-51`) | 2 |
| **`WeightedLoss`** | `training/loss.py:16` | **deprecate** | `self.weight * self.loss_fn(...)` (`:44`); no consumer | 3 |
| **`MultiTaskLoss`** | `training/loss.py:47` | **deprecate** | `jnp.sum(jnp.stack([...]))` times an optional scalar (`:95-107`); no consumer | 3 |
| **`ShardingPolicy`** | `distributed/sharding.py:13` | **deprecate** with the package | regex to `PartitionSpec` through `tree_map_with_path` (`:35-58`); one static field, so zero pytree leaves; no consumer | 2 |
| **`SparsePolicy`** | `sparse/policy.py:16` | keep as a plain frozen dataclass, not a `Struct` | one static `config` field, zero leaves, only ever an argument to non-jitted `sparsify_model` (`sparse/inference.py:57-116`) | 2 |

Net: 13 migrate (12 as `Struct`, `Engine` as a plain dataclass; `ComposedLoss` last, `DataModule` after
its fix), 5 deprecate, 1 plain dataclass. Besides the class count, the deprecations remove
`SafetyTrainStep._step_jit` and `sparse_filter_jit` from the `model_jit` caller list (spec §3), leaving
`Trainer.step` as its one real caller.

## 4. Thin wrappers

For each: what xtrax does, the verified JAX equivalent, consumer call sites, and the deletion test.

### T1. `sparse_filter_jit`, `make_sparse_forward_fn` (and the BCOO claim)

- xtrax: `sparse_filter_jit(fn, **kwargs)` is `return eqx.filter_jit(fn, **kwargs)`
  (`sparse/inference.py:178`). `make_sparse_forward_fn(fn, model)` is a closure over `model`
  (`:140-143`), i.e. `functools.partial` with the arguments reordered.
- Equivalent: `jax.jit`. A `BCOO` argument is a registered pytree whose `data` and `indices` are ordinary
  array leaves, so plain `jax.jit(f)(bcoo, x)` compiles once for same-`nse` inputs.
- **The docstring is wrong.** `sparse/inference.py:15-17,149-157` says `eqx.filter_jit` holds `BCOO` as
  static. Probe (appendix A1): under `eqx.filter_jit` the `BCOO` leaves are `DynamicJaxprTracer`s, i.e.
  traced; `eqx.is_array(bcoo)` is False but its two leaves are arrays. One trace for two calls with equal
  `nse`, identical to plain `jax.jit` (1 trace). The no-retrace property comes from fixed `nse`, not from
  static handling.
- Consumers: none in the six repos.
- Deletion test: an import change.
- Deprecation: both names get a call-time `DeprecationWarning` shim, then removal (section 6). Replacement
  text in the warning: `jax.jit(fn)` and `functools.partial(fn, model)`.
- Rest of `xtrax.sparse` (`SparsePolicy.make_mask` is a magnitude top-k by `argsort`, `policy.py:25-30`;
  `sparsify_model`; `SparseMaskManager`) is a decision with no consumer. If kept: `eqx.is_array` default and
  `eqx.Module` annotations (`inference.py:57-61,120`) become `_tree.is_array` and `PyTree`. User call:
  deprecate the whole package instead.

### T2. `xtrax.distributed`

- `init_dist` (`distributed/init.py:27-131`): fills missing arguments from `SLURM_NTASKS`, `SLURM_PROCID`,
  `SLURM_JOB_NODELIST`, then `jax.distributed.initialize`. JAX does this itself: with no arguments,
  `initialize` auto-detects Slurm, Open MPI and TPU (`jax/_src/distributed.py:245-257`), and
  `SlurmCluster` parses the node list, including `node[001-0015]`, and derives a per-job port
  (`jax/_src/clusters/slurm_cluster.py:20-58`). xtrax's `_derive_coordinator_from_nodelist` keeps the
  bracket text for `node-[001-003]` (`init.py:17-24`, its own comment says so) and hard-codes port 1234.
  It also reads `SLURM_JOB_NODELIST` where JAX reads `SLURM_STEP_NODELIST` (`slurm_cluster.py:21`).
  xtrax skips the JAX call for one process (`init.py:118-124`).
- `get_device_mesh` (`sharding.py:83-112`): `np.array(jax.devices()).reshape(shape)` plus a device-count
  check. Equivalent `jax.make_mesh(axis_sizes, axis_names)` (`jax/__init__.py:155`,
  `_src/sharding_impls.py:900-975`), which also orders devices for the physical topology.
- `get_hardware_mesh_profile` (`:115-174`): the `==2`, `% 8 == 0` and default branches return the same
  value (`:149-159`). No JAX equivalent needed; it carries no decision.
- Consumers: none. Only xtrax-internal coupling: `DataModule` reads a module flag that `init_dist` sets
  (`data/module.py:9-15,64-67`; `init.py:127-129`).
- **Open point for the user:** `jax.distributed.is_initialized()` exists (`jax/_src/distributed.py:340`) but
  is False after xtrax's single-process `init_dist` path, which marks "initialized" without calling JAX
  (`init.py:118-129`). Re-basing the guard needs a rule for the one-process case.
- Deletion test: an import change plus the DataModule guard.

### T3. `SafetyManager`, `with_safety`, `SafetyTrainStep`, `create_train_step(safety=True)`

- xtrax: `jax.jit(checkify.checkify(fn, errors=checkify.float_checks))`, then `err.throw()` on the host
  (`safety/manager.py:44-49`; `training/step.py:55,85-88`).
- Equivalent: exactly that expression. `checkify.float_checks = nan_checks | div_checks`
  (`jax/_src/checkify.py:1170-1173`); `Error.throw` at `:250`.
- Two defects found on the way: `check_nans` and `check_infs` are declared and never read, and the
  docstring says `float_checks` "covers NaN + Inf + overflow" (`manager.py:43`) while JAX defines it as NaN
  plus division by zero. `loop/checkified_execution.py:13-18` already records the dead fields.
- Consumers: none. Internal: `loop/checkified_execution.py:48,93`, `engine/engine.py:29,288` (annotation).
- Deletion test: an import change; the gate in `loop/` calls `checkify` directly.
- `create_train_step` keeps its `safety=False` path (returns `Trainer`) until `safety=True` is removed.

### T4. `WeightedLoss`, `MultiTaskLoss`

- `weight * loss_fn(p, t)` (`training/loss.py:44`); `sum(loss_i(p_i, t_i))` times an optional schedule
  (`:95-107`). Equivalent: a lambda or `jnp.sum`. No JAX or optax function replaces them, but there is
  nothing to replace: no consumer imports either, and `ComposedLoss` (kept) covers the same use with
  traced weights (it returns `(loss, aux)`, so it pairs with `Trainer(has_aux=True)`).
- Deletion test: an import change at most; nobody imports them.

### T5. `make_optimizer`, `partition_labels`; `accumulate_grads`

- `make_optimizer(base, clip_norm=1.0)` is `optax.chain(optax.clip_by_global_norm(clip_norm), base)`
  (`training/optim.py:43`; optax `transforms/_clipping.py:78`). Consumer: aminx `training/trainer.py:19,124`.
  The default `clip_norm=1.0` is a silent policy, so deleting it makes aminx state the clip explicitly. Keep
  (trivial cost, no eqx), note the default in the docstring.
- `partition_labels` is `jax.tree.map(lambda leaf: frozen if f(leaf) else train, model)`
  (`optim.py:124-127`), meant for `optax.partition` (`optax/transforms/_combining.py:176`; its docstring
  names `optax.partition`, which exists). No consumer. Deprecate.
- `adamw_with_schedule` and `no_bias_wd_mask` are optax compositions (`optim.py:76-95`; `optax/_src/alias.py:598`,
  `schedules/_schedule.py:617`) with defaults optax does not share (`weight_decay=1e-2` against optax's
  `1e-4`). Used by aminx (`ebm/training.py:129`, `training/trainer.py:19`) and plegadx
  (`harness.py:70`). Decision. Keep.
- `accumulate_grads` (`training/grad.py:11-127`): `lax.scan` over pre-stacked microbatches with
  `eqx.filter_value_and_grad`, unweighted mean of grads, loss and aux. Partial equivalent:
  `optax.microbatching.microbatch` / `micro_grad` (`optax/microbatching/_microbatching.py:324,555`), which
  reshapes an un-stacked batch itself. Not probed for numerical equality. No consumer. Deprecate only after
  a parity probe; otherwise it must migrate off `eqx.filter_value_and_grad` and `eqx.is_array`.

### T6. `xtrax.io`

- `io/callbacks.py:31` re-exports `xtrax.engine.io` (the docstring says "thin re-export"). No JAX
  counterpart is involved; it is a second import path. No consumer (aminx vendored its own and removed the
  unused `BoundedCallbackHandler`, `utils/_vendored_callbacks.py:13-17`). Deprecate the package; keep
  `engine.io`, which `Engine` uses (`engine.py:25`).

### T7. Thin, but kept

| Item | Mechanism (verified) | What xtrax adds | Why keep |
|---|---|---|---|
| `safe_scan` (`transforms/scan.py:6-47`) | `jax.lax.scan(...)` (`:47`) | `ValueError` on a zero-length axis (`:43-44`). `lax.scan` accepts length 0: probe A3 | demistify asserts the error (`tests/test_tiling_seam.py:257-265`), calls it at `_tiling.py:406`; denxity `_tiling.py:568`; `stages/executor.py:310` |
| `safe_norm`, `safe_reciprocal` (`safety/ops.py:5,28`) | `jnp.sqrt(sum(x**2) + eps**2)`; `1/(x+eps)` | gradient at 0: `jax.grad(jnp.linalg.norm)(0)` is NaN, `safe_norm` gives 0 (probe A5) | denxity has 8 call sites (`distances/metrics.py:116-216`, `solvers.py:122`) |
| `JaxScanIterator`, `WhileLoopIterator` | `lax.scan` (`iterator.py:167`), `lax.while_loop` (`:189`) | a pytree node that downstream modules store (section 3) | aminx |
| `stages/_callback.io_callback` (`:113`) | `jax.experimental.io_callback` (`experimental/__init__.py:25`) | import-time jax-range and signature check (`:110-111`) | the in-repo precedent for pinning an experimental JAX API; a candidate model for the Pallas seam |
| `checkpoint/orbax.py` | `ocp.CheckpointManager` | item `"state"`, `LatestN` union `EveryNSteps`, absolute path, `wait_until_finished` (`:21,48-60,86`) | aminx and plegadx resume from this layout |
| `profiling/jaxpr.py` | `jax.extend.core.subjaxprs` (`extend/core/__init__.py:51`) | `subjaxprs` is one level and sees only `Jaxpr` and tuples (`jax/_src/core.py:316-330`); xtrax recurses and opens lists and mappings (`jaxpr.py:19-73`) | partial overlap; the profiling package is a leaf that must not import jax at load (`profiling/__init__.py`) |
| `random.element_keys` | `jax.vmap(fold_in(key, i))` over `arange(start, start+count)` | explicit index contract, int32 index dtype (`random.py:5-17`). On this JAX `split(key, n)[i] == fold_in(key, i)` (probe A4), but only while `jax_threefry_partitionable` is True, and split costs O(n) | no consumer; keep |
| `tiling/estimators.lowered_memory_estimate` (`:72-109`) | `jax.jit(fn).lower().compile().memory_analysis()` | sums argument, output, temp; raises on missing analysis | aminx `host/runner.py:760`, `sampling/multistate_poe.py:52` |
| `inference/abstract.build_abstract_inputs` (`:11-105`) | `jax.ShapeDtypeStruct` per leaf | `(shape, dtype)`-tuple leaf convention and validation | `infer_bundle` (`inference/api.py:59`) |

### T8. The known overlap: `ChunkedMap` and `chunked_map`

The mechanism of an unordered `ChunkedMap` axis is `jax.lax.map(f, xs, batch_size=B)`
(`stages/executor.py:274` calls `chunked_map`, which calls it at `transforms/map.py:115,124`). JAX's
`lax.map` with a batch size is a `scan` over `vmap`ped chunks plus a separate `vmap` of the remainder
(`jax/_src/lax/control_flow/loops.py:2775-2792`; `_batch_and_remainder` at `:2702-2722`). So `lax.map`
vmaps a remainder of length 1 and every chunk of `batch_size=1`. `chunked_map` exists to avoid exactly
that: length-1 axis is a direct call (`map.py:99-100`), `batch_size=1` is a plain sequential `lax.map`
(`:107-108`), a remainder of 1 is peeled and run unbatched (`:112-121`). The reason is the GPU miscompile of
a vmap over a length-1 axis (xtrax #2520, aminx #2391; aminx carries a private copy of the same logic,
`aminx/utils/safe_map.py`, whose docstring says "the same guard belongs in xtrax's `chunked_map`", `:39`).
Not reproduced here (CPU only). Verdict: **decision, keep**. The `ChunkedMap` strategy descriptor
(`tiling/strategy.py:66`) is plain data. The ordered branch (`executor.py:272`) is an unbatched `lax.map`,
i.e. a scan, by design.

One stale consumer note: denxity's docstring still says chunking "requires N % batch_size == 0"
(`denxity/_tiling.py:410`); #5565 removed that (`transforms/map.py:77-79`).

## 5. Findings the decoupling spec should absorb

1. **`DataModule._epochs` breaks `Struct`.** `data/module.py:48` is `eqx.field(static=True, init=False,
   default_factory=itertools.count)`. `jax.tree_util.register_dataclass` requires meta plus data fields to
   equal the dataclass's `init=True` fields (`jax/_src/tree_util.py:1134-1149`). Probe A2: registering
   such a class raises `ValueError ... Unexpected fields: {'_epochs'}`; with `drop_fields=['_epochs']` it
   registers, but every `tree_unflatten` builds a fresh counter, so the epoch seed `seed + k`
   (`module.py:51-55`) silently restarts after any `tree_map` or `dataclasses.replace`. Today it survives:
   `eqx.Module` keeps static fields as aux data, and the same flatten/unflatten round trip returns the same
   counter object (probe A2). Spec §1 filed
   `DataModule` under "plain fields". Needs: move the counter out of the pytree (a class-level or
   `object.__setattr__` holder) and add `DataModule` to the §6 semantics tests.
2. **aminx subclasses `RunSpec`.** `class RunSpec(_XtraxRunSpec)` at `aminx/run/spec.py:144`, mixing
   `dataclasses.field(default_factory=...)` (`:15,148-155`), `eqx.field(static=True)` (`:156-160`) and eqx
   sub-modules `IOConfig` etc. as data fields. Spec §2 names plegadx only. aminx also renamed its
   `precision` field to dodge `RunSpec.precision` (`:152`). Add aminx's class to probe 3.
3. **`tiling._BatchPlanWrapper`** (`_plan_wrapper.py:5-28`) hashes a `BatchPlan` by `id()` "for use with
   `eqx.filter_jit`", is in `tiling.__all__`, and prolix imports it (`prolix/fitting/config.py:22`).
   `model_jit`'s `_Static` hashing (spec §3) is the replacement; until then it is a break risk.
4. **demistify fingerprints xtrax source.** `shard_identity.py:166-175` hashes `xtrax/tiling`,
   `xtrax/transforms` and the files defining `chunked_map` and `safe_scan`, eagerly at import (`:281`), so
   every xtrax change under those directories (step 2 touches `tiling/iterator.py`) changes
   `code_fingerprint` and relabels future shards. It also does `inspect.getsourcefile(xtrax.chunked_map)`,
   so moving `chunked_map` or `safe_scan` out of the top-level namespace raises at import.
5. **Two eqx uses spec §1 does not list as code paths:** `telemetry/ir.py:100-124` falls back to
   `eqx.filter_make_jaxpr` and `eqx.filter_jit(fn).lower` when plain tracing fails (it is why a model with
   a callable field still gets IR captured), and `run/resolver.py:8,39` imports equinox for an annotation.
6. **`Trainer` cannot be deprecated** without replacing the `xtrax run`/`resume` path (`cli/run.py:116`,
   `cli/resume_verb.py:107`). Both big consumers copied its update step instead of calling it: aminx
   `EBMTrainStep` (`ebm/training.py:404-418`) and plegadx `TrainStep` ("byte-for-byte identical",
   `harness.py:5-30`; `equinox_aflf_sampler.py:27-31`). Cheaper than either: expose the 4-line update as a
   function (`filter_value_and_grad`, `optimizer.update(..., filtered_params)`, `apply_updates`).
7. Already-broken consumer imports on origin/main, independent of this work: plegadx
   `scripts/run_intellifold_e2e_030_ensemble.py:240-241` imports `SafeMap`/`safe_map`; prolix
   `ensemble_dispatch.py:26` and `tiling/xtrax_adapter.py:19` import `SafeMap` from `xtrax.tiling.strategy`
   under a pin of `xtrax>=0.4.0a5,<0.5` (`pyproject.toml:32`). Both names were removed in d657bf6.

## 6. Deprecation mechanics (what the repo actually does)

`tombstone.py` is not part of it. The shipped pattern, from `c78b296` (add) and `d657bf6` (remove):

1. `src/xtrax/_renamed.py`: a `RENAMED` map and `deprecated_alias(module, name, namespace)` that warns with
   `DeprecationWarning` (using `skip_file_prefixes` so the blame lands on the caller, including when
   jaxtyping's import hook adds frames) and returns the new object.
2. A PEP 562 module `__getattr__` in each module that exported the old name.
3. `codemods/safemap-to-chunkedmap/` (ast-grep `rules.yml`, a Markdown script, a before/after fixture pair)
   so consumers migrate mechanically.
4. CHANGELOG `### Deprecated` with the codemod pointer (`CHANGELOG.md:818-829`); one release later
   `### Removed` (`CHANGELOG.md:521-531`) and the alias tests become removal tests with a control that the
   new names resolve.
5. `_renamed.py` keeps only what name-based matching still needs (`CHUNKED_MAP_NAMES`, `_renamed.py:14`),
   because aminx dispatches on `type(strategy).__name__`.

Adaptations for the items above:

- `deprecated_alias` returns `namespace[new]`, which only works when the replacement is an xtrax object.
  When the replacement is a JAX function (`jax.jit`, `checkify`, `jax.distributed.initialize`), the shim
  must return a thin function or class that warns when called, with the replacement named in the message.
- The names are imported eagerly in `training/__init__.py:3-20`, `tiling/__init__.py` and
  `sparse/__init__.py`, and listed in `xtrax/__init__.py` (`__all__` and `_LAZY`, `:3-146`). Those entries
  must move behind `__getattr__` before the hook can fire.
- Classes that consumers store in pytrees or compare by name (`JaxScanIterator`, `ChunkedMap`) need the
  alias to return the same class so `type(x).__name__` is unchanged. This is why they are not on the list.
- Codemod rules are one-liners here: `sparse_filter_jit($F)` to `jax.jit($F)`; `init_dist()` to
  `jax.distributed.initialize()`; `get_device_mesh($S, $N)` to `jax.make_mesh($S, $N)`. Only
  `ShardingPolicy`, `SafetyManager`/`SafetyTrainStep` and the two losses need prose, since they are classes.
- Because no consumer imports any of the deprecated names, one release of warning is enough. aminx was the
  only consumer the previous deprecation waited for.

## 7. Verification status

| Claim | Status |
|---|---|
| 19 eqx classes, listed in section 3 | verified (AST sweep, matches spec) |
| `lax.map(batch_size=)` vmaps the remainder | verified (`loops.py:2775-2792`) |
| `lax.scan` accepts length 0, so `safe_scan`'s guard is xtrax's | verified (probe A3, jax 0.11.1) |
| `BCOO` leaves are traced under `eqx.filter_jit`; retraces equal plain `jax.jit` | verified (probe A1) |
| `register_dataclass` rejects a static `init=False` field | verified (probe A2, plus source `tree_util.py:1134-1149`) |
| `grad(jnp.linalg.norm)(0) = nan`, `grad(safe_norm)(0) = 0` | verified (probe A5) |
| `split(key, n)[i] == fold_in(key, i)` with `jax_threefry_partitionable=True` | verified (probe A4); config-dependent |
| `jax.distributed.initialize()` auto-detects Slurm | read from source (`distributed.py:245`, `slurm_cluster.py`), not run |
| size-1 vmap GPU miscompile | unverified (aminx report only, CPU here) |
| `optax.microbatching` equals `accumulate_grads` | unverified, not probed |
| Consumer scan complete | verified for `*.py` at the six SHAs; other repos and string-built imports not scanned |

## Appendix A. Probes

Throwaway scripts, run with `uv run --no-sync python -I` on jax 0.11.1, equinox 0.13.8, CPU. Sources are
reproduced so a number here can be re-derived without the session.

**A1 (BCOO under filter_jit).** `w = BCOO.fromdense(jnp.eye(4))`; `@eqx.filter_jit def f(m, x): traces.append(
[type(l).__name__ for l in jax.tree_util.tree_leaves(m)]); return m @ x`; call `f(w, x)` then
`f(BCOO.fromdense(2.0*jnp.eye(4)), x)`. Output: `BCOO leaves under tree_flatten: ['ArrayImpl','ArrayImpl']`,
`is_array(BCOO): False`, `trace count after two calls with same-nse BCOO: 1`, `leaf types inside trace:
['DynamicJaxprTracer','DynamicJaxprTracer']`, `plain jax.jit trace count (BCOO arg, same nse): 1`.

**A2 (init=False static field).**
```python
@dataclasses.dataclass(frozen=True)
class M:
    batch: int = dataclasses.field(metadata={"static": True})
    _epochs: object = dataclasses.field(metadata={"static": True}, init=False,
                                        default_factory=itertools.count)
jax.tree_util.register_dataclass(M)   # ValueError: ... Unexpected fields: {'_epochs'}
# with drop_fields=["_epochs"]: registers; tree_unflatten(*tree_flatten(m)[::-1]) builds a new counter
# ("counter survives flatten/unflatten: False | next on rebuilt: 0")
# the same shape as an eqx.Module with eqx.field(static=True, init=False, ...):
# "eqx: counter survives flatten/unflatten: True"
```

**A3 (zero-length scan).** `jax.lax.scan(lambda c, x: (c + x, x), 0.0, jnp.zeros((0,)))` returns carry `0.0`
and `ys.shape == (0,)`; `jax.lax.map(f, jnp.zeros((0,)), batch_size=3).shape == (0,)`. No error.

**A4 (split vs fold_in).** `jax.config.jax_threefry_partitionable` is `True`;
`split(key,10)[3:6] == split(key,6)[3:6]` and equals `vmap(fold_in(key, i))(arange(3,6,int32))`, compared
on `key_data`.

**A5 (safe_norm).** `jax.grad(jnp.linalg.norm)(zeros(3))` is `[nan nan nan]`; `jax.grad(safe_norm)(zeros(3))`
is `[0. 0. 0.]`; the hand-rolled `sqrt(sum(v**2) + 1e-8**2)` gradient equals `safe_norm`'s.

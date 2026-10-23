---
title: 'Equinox decoupling: native pytree layer with Equinox passthrough'
description: Make xtrax run without equinox while staying compatible with eqx.Module users and downstream subclasses (Part 1 of the equinox/Pallas plan)
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261023'
backlog_ids: ''
adversarial_review: ''
---
# Equinox decoupling: native pytree layer with Equinox passthrough


## Goal

xtrax imports, trains, tiles and exports with `equinox` absent, and nothing changes for anyone who
has it installed: Equinox models still train, `eqx.filter_jit` still sees xtrax objects as modules,
and downstream subclasses of xtrax classes keep working without edits.

Part 2 (Pallas kernels, `xtrax.kernels`) is a separate spec; it depends on §3's `Struct` base for
kernel configs. Spike evidence: `research/261023_pallas-block-sparse-spike.md`.

## 1. Current coupling (origin/main 26958dc)

Inventory by AST sweep over `src/xtrax` (throwaway sweep, see §6 for the shipped version):
20 files import equinox; 19 classes subclass `eqx.Module`; API used: `field` (48), `Module` (46),
`filter_jit` (26), `is_array` (8), `filter_value_and_grad` (6), `tree_at` (5), `filter` (4),
`nn.inference_mode` (3), `nn.Linear` (2, devtools probes only), `partition`, `apply_updates`,
`filter_make_jaxpr` (1 each). No `converter=`, `AbstractVar`, `filter_vmap` or eqx serialisation.
`src/xtrax/checkpoint/` has no equinox reference.

Features a replacement must reproduce, per class:

| Class | Location | Needs |
|---|---|---|
| `ComposedLoss` | `training/loss.py:112` | custom `__init__` assigning fields; `__check_init__`; callable non-static fields (`terms`) |
| `RunSpec` | `run/spec.py:14` | `__check_init__`; `default_factory`; **subclassed downstream** |
| `StageBundle` | `stages/bundle.py:57` | **subclassed downstream** |
| `AxisBoundary` | `stages/boundaries.py:84` | static-only (zero leaves); found downstream by `isinstance(…, eqx.Module)` walk |
| `ResumableState` | `training/types.py:41` | constructed downstream; downstream calls `eqx.tree_at` on it |
| `Trainer` | `training/trainer.py:13` | `@eqx.filter_jit` on method `step` (self traced) |
| `SafetyTrainStep` | `training/step.py:24` | `@eqx.filter_jit` on method `_step_jit`; checkify inside |
| `Engine` | `engine/engine.py:275` | static fields only; `eqx.nn.inference_mode` + `tree_at` at `:428` |
| others (11) | tiling iterators, `DataModule`, policies, `SafetyManager` | plain fields / `__call__` |

## 2. Downstream contract (why the base cannot simply become a dataclass)

- plegadx subclasses `RunSpec` and `StageBundle` and declares `eqx.field(static=True)` on the
  subclasses (`~/projects/plegadx/src/plegadx/infra/runspec.py:43,88,238,314,353,…`).
- plegadx finds `AxisBoundary`s by recursing only through `isinstance(obj, eqx.Module)`
  (`plegadx/_boundary.py:231,243`).
- aminx (`training/trainer.py:175,210,800`, `ebm/training.py:505`) and plegadx
  (`adapters/aflf/equinox_aflf_sampler.py:305,442`) construct `ResumableState`; plegadx `eqx.tree_at`s it.
- The shipped `using-xtrax` skill (`src/xtrax/agent_plugin.json`, skill sources) teaches
  `eqx.Module`/`eqx.field` for custom `RunSpec`/`AxisBoundary`.

A native dataclass base breaks the first two silently (subclass fields lose static-ness or the
walker stops finding boundaries). Hence §3.

## 3. Design: `xtrax._tree`, one seam with two backends

New private module `src/xtrax/_tree/` exposing the only names the rest of xtrax may use:

```
Struct, field, is_array, partition, combine, filter, tree_at, apply_updates,
filter_jit, filter_value_and_grad, filter_make_jaxpr, inference_mode, BACKEND
```

**Backend resolution** (explicit > env > default, reported, fails loud on a bad value):
`XTRAX_TREE_BACKEND` = `equinox` | `native` | unset. Unset → `equinox` if importable, else `native`.
`equinox` requested but not importable → `ImportError` naming the extra. `BACKEND` and
`backend_source()` ("env" | "auto") are inspectable, and `xtrax doctor`/telemetry records them.

- **`equinox` backend:** re-exports `eqx.Module`, `eqx.field`, `eqx.filter_jit`, … unchanged.
  With equinox installed (every current consumer), class identity, `isinstance(…, eqx.Module)`
  and subclassing with `eqx.field` are exactly what they are today. Zero behaviour change.
- **`native` backend:** about 250 lines on `jax.tree_util`:
  - `Struct`: metaclass/`__init_subclass__` that applies `dataclass(frozen=True)` and
    `jax.tree_util.register_dataclass`, splitting fields into data/meta by
    `field(static=True)` metadata. Supports a custom `__init__` (assignment is allowed until
    `__init__` returns, then frozen, as in Equinox), and calls `__check_init__` along the MRO after
    `__init__`. Plain fields that hold non-arrays (callables) are data leaves, as in Equinox.
  - `filter_jit`: partitions args by `is_array`; arrays traced, the non-array remainder (incl.
    callable leaves) hashed into the static key. Works as a method decorator through `__get__`, so
    `self` is partitioned like any other argument (needed by `Trainer.step`, `SafetyTrainStep._step_jit`).
  - `filter_value_and_grad`: differentiates w.r.t. inexact-array leaves of the first argument.
  - `tree_at`, `partition`/`combine`, `apply_updates`: straightforward `tree_util` versions.
  - `inference_mode(model)`: `__xtrax_inference__()` if defined; else, if `"equinox" in sys.modules`
    and the model is an `eqx.Module`, defer to `eqx.nn.inference_mode`; else identity. Never imports equinox.

Verified 261023 on jax 0.11.1 / equinox 0.13.8 (probe: `register_dataclass` +
`field(metadata=dict(static=True))`): an array-only change does not retrace, a static change does;
`eqx.tree_at` works on such a dataclass; an `eqx.Module` holding one passes `eqx.filter_jit`;
the dataclass is **not** `isinstance(…, eqx.Module)` (which is why the default is the equinox backend).

**User models** are typed `Model = PyTree` (alias in `xtrax._tree`), not `eqx.Module`
(`training/types.py:53`, `training/state.py:10`, `sparse/inference.py:58-61,120`).

## 4. Packaging

- `pyproject.toml`: move `equinox>=0.13.1` from `dependencies` to `[project.optional-dependencies].equinox`;
  add it to `dev`. Every equinox-using consumer already declares equinox itself (§7), so none
  needs an edit. **Decision D2 below.**
- `devtools/gates/_performance_probes.py` uses `eqx.nn.Linear` → guard behind the `equinox`
  backend or rewrite as a two-array MLP.

## 5. Migration order (one PR each, each green on both backends)

1. `_tree` module + its own tests (parity suite §6) — no call sites change.
2. Static-only / plain classes: tiling iterators, `stages/*`, policies, `DataModule`, `SafetyManager`,
   `Engine` fields. Mechanical `eqx.` → `_tree.` rewrite.
3. Training path: `Trainer`, `SafetyTrainStep`, `ResumableState`, `ComposedLoss` (custom `__init__`
   + `__check_init__`), `training/grad.py`, `engine.py:428` inference mode.
4. `RunSpec`, `sparse/inference.py` (BCOO path keeps its documented static-closure behaviour).
5. Packaging flip (§4), docs/skill text, CHANGELOG.

## 6. Tests and gates

- **No-equinox import gate:** subprocess with a meta-path finder that raises on `import equinox`;
  imports every `xtrax` submodule (walk `pkgutil`) and runs a tiny train + tile + export. Ship with a
  control: the same harness with a deliberate `import equinox` injected must fail (prove red).
- **AST sweep test:** no `import equinox`/`eqx.` outside `src/xtrax/_tree/` (the throwaway sweep, promoted).
- **Backend parity suite:** for each of the 19 classes and each `_tree` function, run under
  `XTRAX_TREE_BACKEND=equinox` and `=native`; compare leaves, treedef shape, static/data split,
  retrace counts (`jax_log_compiles`, expect equal), grads (`filter_value_and_grad`) and
  `__check_init__` errors. Equinox models (`eqx.nn.MLP`) trained through the native backend
  must match the equinox backend bit-for-bit on CPU.
- **Downstream canary:** plegadx's `RF3RunSpec` pattern (subclass + `eqx.field(static=True)`) and the
  `_find_boundaries` walk reproduced as an xtrax test under the equinox backend.
- **CI:** add job `no-equinox` (sync without the extra, `XTRAX_TREE_BACKEND` unset → must resolve
  `native`) beside `wheel-smoke`; existing jobs keep equinox. Runs on CI, never the full suite locally.
- Checkpoints: round-trip a `ResumableState` saved under the equinox backend and restored under native
  (orbax restores by leaf structure; **unverified** that treedefs match — probe in step 1).

## 7. Assumptions

| Assumption | Status |
|---|---|
| No `converter=`/`AbstractVar`/eqx serialisation in src | verified (AST sweep) |
| `register_dataclass` static metadata + no retrace on value change (jax 0.11.1) | verified (probe) |
| `eqx.tree_at` works on native structs | verified (probe) |
| Downstream subclasses/isinstance-walks xtrax modules | verified (plegadx, aminx files cited §2) |
| orbax restore across backends | unverified — step 1 probe |
| native `filter_jit` method binding matches eqx retrace behaviour for `Trainer.step` | unverified — parity suite |
| Consumers keep equinox after the flip | verified 261023: aminx, plegadx, prolix, demistify, denxity declare `equinox` directly in `pyproject.toml`; asr does not (uses xtrax only, already unbootstrappable, #5008) |

## 8. Decisions for the user

- **D1 — backend default.** Recommended: equinox when importable (zero change for current users;
  native only where equinox is absent). Alternative: native always, equinox only for passthrough of
  user models — cleaner single path, but breaks plegadx's subclasses and `_boundary.py` walker until
  they migrate.
- **D2 — packaging. DECIDED 261023 (user): optional `xtrax[equinox]` extra now**, also pulled by
  `dev`. (Rejected: hard dependency for one release with native opt-in, then flip.)

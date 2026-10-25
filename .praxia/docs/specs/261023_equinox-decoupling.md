---
title: 'Equinox decoupling: one plain-JAX implementation, Equinox as a consumer'
description: Make xtrax depend only on jax/optax for its pytree layer, with one implementation and Equinox models and downstream eqx.field subclasses still working (Part 1 of the equinox/Pallas plan)
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261023'
backlog_ids: ''
adversarial_review: ''
---
# Equinox decoupling: one plain-JAX implementation, Equinox as a consumer

## Goal

xtrax's pytree layer uses only `jax`, `optax` and the standard library. There is **one**
implementation and no backend switch. Equinox is never imported by `src/xtrax`; it appears only in
tests, as a model type xtrax must accept. Equinox users keep training Equinox models through
xtrax, and downstream subclasses written with `eqx.field(static=True)` keep working.

Part 2 (Pallas kernels, `xtrax.kernels`) is a separate spec; it depends on §3's `Struct` base for
kernel configs. Spike evidence: `research/261023_pallas-block-sparse-spike.md`.

## 1. Current coupling (origin/main 26958dc)

Inventory by AST sweep over `src/xtrax` (throwaway sweep, see §6 for the shipped version):
20 files import equinox; 19 classes subclass `eqx.Module`; API used: `field` (48), `Module` (46),
`filter_jit` (26), `is_array` (8), `filter_value_and_grad` (6), `tree_at` (5), `filter` (4),
`nn.inference_mode` (3), `nn.Linear` (2, devtools probes only), `partition`, `apply_updates`,
`filter_make_jaxpr` (1 each). No `converter=`, `AbstractVar`, `filter_vmap` or eqx serialisation.
`src/xtrax/checkpoint/` has no equinox reference.

Features the replacement must cover, per class:

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

## 2. Downstream contract

- plegadx subclasses `RunSpec` and `StageBundle` with undecorated `class X(RunSpec):` and declares
  `eqx.field(static=True)` fields (`~/projects/plegadx/src/plegadx/infra/runspec.py:43,88,238,314,353,…`).
  **Keeps working**: `eqx.field(static=True)` is a plain `dataclasses.Field` with metadata
  `{'static': True}`, and §3's `Struct` reads that key (probe 2, §7).
- plegadx finds `AxisBoundary`s by recursing only through `isinstance(obj, eqx.Module)`
  (`plegadx/_boundary.py:231,243`). **Breaks silently** (finds nothing): needs a one-function
  migration to `dataclasses.is_dataclass` / `dataclasses.fields`, landed in plegadx before the
  xtrax release that flips the base. This is the only known downstream edit.
- aminx (`training/trainer.py:175,210,800`, `ebm/training.py:505`) and plegadx
  (`adapters/aflf/equinox_aflf_sampler.py:305,442`) construct `ResumableState`; plegadx `eqx.tree_at`s
  it. **Keeps working**: `eqx.tree_at` works on any pytree, including a registered dataclass (probe 1).
- The shipped `using-xtrax` skill (`src/xtrax/agent_plugin.json`, skill sources) teaches
  `eqx.Module`/`eqx.field` for custom `RunSpec`/`AxisBoundary`; rewrite to `xtrax.Struct`/`xtrax.field`,
  noting `eqx.field(static=True)` stays accepted.
- Every equinox-using consumer (aminx, plegadx, prolix, demistify, denxity) declares `equinox`
  itself, so dropping it from xtrax's dependencies removes nothing they import.

## 3. Design: `xtrax._tree`, plain JAX only

New module `src/xtrax/_tree.py` (public names re-exported as `xtrax.Struct`, `xtrax.field`):

- **`Struct`** — base class. `__init_subclass__` applies `dataclasses.dataclass(frozen=True)` and
  `jax.tree_util.register_dataclass` to every subclass, splitting fields into data/meta by
  `field.metadata.get("static")`. `__post_init__` runs every `__check_init__` along the MRO, so
  validation is inherited. Subclasses need no decorator (plegadx style).
- **`field(*, static=False, **kw)`** — `dataclasses.field(metadata={"static": static}, **kw)`.
  Same metadata key as `eqx.field`, so either spelling works on a `Struct` subclass.
- **`partition(tree) -> (arrays, rest)` / `combine(arrays, rest)`** — `tree_map` with `None`
  placeholders; `is_array` = `jax.Array` / `np.ndarray` / `__jax_array__`.
- **`model_jit(fn)`** — the one place user models (any pytree, including `eqx.Module`) enter a
  trace: partitions the model argument, passes arrays as traced args and the non-array remainder
  as a static arg wrapped in `_Static` (hash/eq by `(treedef, leaves)`). Used by `Trainer.step`,
  `SafetyTrainStep._step_jit` and `sparse_filter_jit`; xtrax's own structs go through plain
  `jax.jit`, since their static fields already live in the treedef.
- **Gradients** — `jax.value_and_grad` w.r.t. the array partition, closing over `combine(…, rest)`.
  Updates via `optax.apply_updates` (optax is already a dependency).
- **Edits** — `dataclasses.replace` replaces `eqx.tree_at` inside xtrax.
- **Inference mode** — `inference_mode(model)`: the model's `__xtrax_inference__()` if defined;
  else, if `sys.modules.get("equinox")` is loaded and the model is one of its modules, call
  `equinox.nn.inference_mode`; else identity. Never imports equinox.
- **Methods** — `Trainer.step` / `SafetyTrainStep._step_jit` become plain methods that call a
  module-level `model_jit` function with `self` as an argument (no descriptor magic).
- **`ComposedLoss`** — keeps its public signature. Its custom `__init__` assigns through
  `object.__setattr__` and ends with `self.__post_init__()` (a class-defined `__init__` is kept by
  `dataclass` and does not call `__post_init__` itself — **unverified**, check in step 3).

User models are typed `Model = PyTree` (`training/types.py:53`, `training/state.py:10`,
`sparse/inference.py:58-61,120`).

## 4. Packaging

`equinox` leaves `[project] dependencies`. It goes to `[project.optional-dependencies].dev` (tests
use it as a fixture) and nowhere else: no `xtrax[equinox]` extra is needed, since xtrax has no
equinox code path. `devtools/gates/_performance_probes.py` replaces `eqx.nn.Linear` with a
two-array MLP.

## 5. Migration order (one PR each)

1. `_tree.py` + its tests (§6). No call sites change.
2. Static-only / plain classes: tiling iterators, `stages/*`, policies, `DataModule`,
   `SafetyManager`, `Engine` fields. Mechanical `eqx.Module` → `Struct`, `eqx.field` → `field`.
3. Training path: `Trainer`, `SafetyTrainStep`, `ResumableState`, `ComposedLoss`,
   `training/grad.py`, `engine.py:428`.
4. `RunSpec`, `sparse/inference.py`.
5. plegadx `_boundary.py` migration lands **before** this step; then packaging (§4), skill/docs
   text, CHANGELOG, release.

## 6. Tests and gates (one implementation, one suite)

No dual-backend parity suite: correctness is checked against JAX semantics directly, with Equinox
models as fixtures.

- **Struct semantics:** leaves/treedef of each of the 19 classes (static fields not leaves);
  inherited `__check_init__` raises; `dataclasses.replace`; undecorated subclass with
  `eqx.field(static=True)` and with `xtrax.field(static=True)`.
- **Retrace counts:** `jax_log_compiles` / trace counters — an array-value change does not retrace,
  a static change does; `Trainer.step` traces once over N steps.
- **Gradients:** `model_jit` + `value_and_grad` equals `jax.grad` of a hand-unrolled loss on a
  plain-dict model (no equinox needed); on an `eqx.nn.MLP` fixture it equals `eqx.filter_grad`
  once (compat check, not a parity suite); loss decreases.
- **Round-trip:** a trained `eqx.nn.MLP` comes back as `eqx.Module`; `eqx.tree_at` works on
  `ResumableState`.
- **No-equinox gate:** subprocess with a meta-path finder that raises on `import equinox`; imports
  every `xtrax` submodule (`pkgutil` walk) and runs a tiny train + tile + export on a dict model.
  Ship with a control: an injected `import equinox` must make it fail (prove red).
- **AST sweep test:** no `equinox` import or `eqx.` reference under `src/xtrax/`.
- **Checkpoint:** a `ResumableState` saved by today's xtrax (eqx.Module) restores into the new
  `Struct` version (orbax restores by leaf structure; **unverified** — probe in step 1).
- **CI:** one extra job, `no-equinox` (sync without dev's equinox; runs the gate and the dict-model
  tests). Never the full suite locally.

## 7. Assumptions

| Assumption | Status |
|---|---|
| No `converter=`/`AbstractVar`/eqx serialisation in src | verified (AST sweep) |
| `register_dataclass` static metadata; array change does not retrace, static change does (jax 0.11.1) | verified (probe 1) |
| `eqx.tree_at` works on registered dataclasses | verified (probe 1) |
| `eqx.field(static=True)` is a `dataclasses.Field` with `{'static': True}` | verified (probe 2) |
| Undecorated subclass of an `__init_subclass__` base keeps eqx.field static-ness; inherited `__check_init__` runs; `dataclasses.replace` works | verified (probe 3) |
| partition/combine + `jax.jit` (static `_Static`) + `jax.value_and_grad` + optax trains `eqx.nn.MLP`; grads equal `eqx.filter_grad`; 1 trace over 50 steps; result is `eqx.Module` again | verified (probe 3: loss 2.92 → 0.040) |
| A class cannot inherit both a native frozen base and `eqx.Module` | verified (probe 2: `TypeError`) — so no consumer may mix them; none does today |
| Downstream breakage limited to plegadx `_boundary.py` walk | verified for plegadx/aminx call sites in §2; other consumers not swept for `isinstance(<xtrax obj>, eqx.Module)` — sweep before step 5 |
| orbax restore of pre-migration checkpoints | unverified — step 1 probe |
| `_Static` hashing is stable for every non-array leaf type users hold (functions, strings, ints, enums) | unverified for unhashable leaves (lists, dicts in a model) — decide: convert to tuples, or error with the field path |
| Consumers keep equinox after the flip | verified 261023: aminx, plegadx, prolix, demistify, denxity declare `equinox` in `pyproject.toml`; asr does not (uses xtrax only, already unbootstrappable, #5008) |

Probes 1–3 were throwaway scripts run 261023 on jax 0.11.1 / equinox 0.13.8 (CPU); the shipped
tests in §6 supersede them.

## 8. Decisions

- **D1 — runtime type of xtrax's classes. DECIDED 261023 (user): plain JAX only, one
  implementation, no dual-backend parity suite.** Supersedes the earlier two-backend draft
  (equinox when importable, native fallback). Cost accepted: plegadx `_boundary.py` migrates first.
- **D2 — packaging. DECIDED 261023 (user): equinox becomes optional.** Under D1 that means a
  `dev`-only test dependency; a dedicated `xtrax[equinox]` extra has nothing to enable.

# E3-MVP spec — `xtrax run`: TrainConfig-driven training run

**task_id:** `260623_e2-run` · **epic:** #2629 (`[parent:#2605]`) · **brainstorm:** contemplex session `f300363c` (architectural; INVEST all-pass)
**Brainstorm record:** `.praxia/docs/specs/260623_hmw-design-the-xtrax-run-verb-drive-a-tr.md`
**Status:** REVISED post adversarial review (spec-challenger ∥ spec-defender + plan-auditor, session 260623). See **Adversarial Revision Log** at the bottom — 2 CRITICAL + 7 MAJOR + minors folded; 2 user design decisions (C1 checkpoint namespacing; M5 thin-observability).

## Summary

`xtrax run config.toml` drives a training run to completion from a TOML config file, reusing the existing `Engine`/`Trainer`/`DataModule`/optim stack — pure glue, never a new training loop. It is the first CLI verb to reach into the runtime layer. Winner = **A + H + I + J + M** (contemplex `f300363c`): the `[data]` factory resolves to a **dataset** that `run` *always* wraps in `DataModule` (no duck-type branch); a tiny manifest is **always** written to `.xtrax/runs/<run_id>/` so the later `resume` verb has a contract; the config object is named **`TrainConfig`** (orthogonal to E1's `RunSpec`); the config is **raw-TOML `{path, kwargs}`** per section with a mandatory `schema_version`; and a new public **`init_state`** is promoted into `xtrax.training`.

## Verified runtime anchors (source-confirmed during adversarial review)

- `Engine.__init__` (`engine/engine.py:59-61`): `trainer` (required, static) + `callbacks: tuple[...]` (**required, no default**, static) + `validation_callbacks=()`. Canonical construction: `Engine(trainer=Trainer(loss_fn, optimizer), callbacks=())` — **all-keyword**.
- `Engine.fit_sync` (`engine/engine.py:222`): bound method `(self, state, data, num_epochs, checkpoint_dir=None) -> ResumableState`; delegates to `asyncio.run(self.fit(...))`. Consumes **`num_epochs`** (not `steps`); `fit` does `range(num_epochs)` (`:114`) → `num_epochs` must be a positive `int`, never `None`.
- `Trainer` (`training/trainer.py:28-29`): `Trainer(loss_fn: LossFunction, optimizer: optax.GradientTransformation)` — both required. `LossFunction` (`training/types.py:20`) = `(predictions, targets) -> jax.Array` scalar. `Trainer.step` **does not read `state.key`** (`:49-72`).
- `ResumableState` (`training/types.py:41-55`): `(step: Array[int32], key: Array, model, opt_state, extras: dict = {})`. No `init_state` helper exists today (`training/__init__.py` `__all__` confirmed).
- `DataModule` (`data/module.py:17-23`): `(dataset, batch_size, num_epochs, seed, distributed, collate_fn=None)` — `num_epochs`/`seed`/`distributed` are **required, no defaults**. A 2-arg call raises `TypeError`.
- `make_optimizer` (`training/optim.py:25`): `(base: GradientTransformation, clip_norm)` — first arg is a **non-scalar** optimizer (a *wrapper*, not a factory) → **not constructable** from a flat `{path,kwargs}` config. `adamw_with_schedule` (`:46`): all-scalar args incl. **required `total_steps`**.
- `load_fn` (`cli/loader.py:48`): `path.rsplit(':',1)` → requires `module.path:symbol` form (left side a real importable module). Raises `CLIImportError` with the path (but **no section** context).
- `save_checkpoint` (`checkpoint/orbax.py:45-65`): keys on `int(state.step)`; `fit` saves once per epoch. ⇒ two runs sharing a checkpoint dir clobber at identical steps (the basis for the C1 fix in AC8).

## Acceptance Criteria (Given/When/Then)

**AC1 — model resolution via the existing loader.**
*Given* `[model] path="mypkg.models:MyModel"` (a real `module.path:symbol`) + a `kwargs` sub-table, *when* `run` builds the model, *then* it resolves via `load_fn(path)` and calls it with `**kwargs`. No new resolution mechanism. (Examples MUST use `module.path:symbol`; a bare `pkg:Name` whose left side is not an importable module raises `CLIImportError`.)

**AC2 — optimizer resolution (config-constructable factories only).**
*Given* `[optimizer] path=... kwargs=...`, *when* `run` builds the optimizer, *then* it resolves via `load_fn(path)(**kwargs)` and the result is an `optax.GradientTransformation`. The target MUST be a factory whose arguments are **all config scalars/kwargs** — e.g. `xtrax.training.optim:adamw_with_schedule`. **Bare `make_optimizer` is NOT a valid target** (its first arg `base` is a non-scalar `GradientTransformation`). `adamw_with_schedule`'s required `total_steps` is a **user-supplied kwarg**; the MVP does **not** auto-derive or cross-check it against `num_epochs` (documented footgun — see Pre-mortem).

**AC3 — data: factory returns a dataset, `run` ALWAYS wraps (no duck-type branch).**
*Given* `[data] factory="mypkg.data:load_train"` + a `batch_size` scalar, *when* `run` builds the data input, *then* it resolves the factory via `load_fn`, calls it to get a dataset, and **unconditionally** constructs `DataModule(dataset, batch_size, num_epochs, seed, distributed=False)` (`num_epochs`/`seed` from top-level config scalars; `distributed=False` hardcoded for MVP). *And* there is **no `isinstance`/duck-type branch on the factory's return**. **Enforcement (behavioral, not prose):** the test fixture's factory returns *an already-built `DataModule`*; the assertion is that `run` **re-wraps** it (`result.dataset is the_inner_datamodule`) — the only observable signature of an unconditional wrap. A duck-type short-circuit (`if isinstance(x, DataModule): return x`) would fail this test. (Optional belt-and-suspenders: an import-linter/grep rule forbidding `isinstance(..., DataModule)` in `cli/run.py`.)

**AC4 — `init_state` is public in `xtrax.training`, with pinned derivation.**
*Given* a resolved `model`, `optimizer`, `seed`, *when* `init_state(model, optimizer, seed)` is called, *then* it returns `ResumableState(step=jnp.asarray(0, jnp.int32), key=jax.random.PRNGKey(seed), model=model, opt_state=optimizer.init(eqx.filter(model, eqx.is_array)), extras={})`. *And* it is importable as public `xtrax.training` API (added to `__all__`); its signature carries only runtime objects (no config/path/tyro). **Test pins:** `state.step.dtype == jnp.int32`; `state.key` is reproducible from `seed`; `opt_state` matches `optimizer.init` of the filtered model.

**AC5 — `run_from_config` wires the full stack to `fit_sync` to completion.**
*Given* a parsed `TrainConfig`, *when* `run_from_config(config)` executes, *then* it builds `loss_fn` (AC-loss), `model`, `optimizer`, `data` (DataModule), `state = init_state(...)`, `engine = Engine(trainer=Trainer(loss_fn, optimizer), callbacks=())`, and calls `engine.fit_sync(state, data, num_epochs=config.num_epochs, checkpoint_dir=<run-checkpoint-dir>)` to completion, returning the final `ResumableState`. `run_from_config` lives in `cli/run.py` (cli-private glue), not the runtime layer.

**AC-loss — loss resolution from a required `[loss]` section.**
*Given* `[loss] path="mypkg.losses:mse" kwargs=...`, *when* `run` builds the loss, *then* it resolves via `load_fn(path)(**kwargs)` (or the symbol itself if it is already a `LossFunction`) yielding a callable `(predictions, targets) -> scalar`. **`[loss]` is REQUIRED**: a config missing it raises `ConfigError` at parse time (no default loss). (`Trainer` cannot be constructed without `loss_fn`.)

**AC6 — manifest is ALWAYS written with all required fields.**
*Given* any `run`, *when* it starts, *then* it writes `.xtrax/runs/<run_id>/manifest.json` containing: `run_id`, **non-optional** `model={path,kwargs}`, `optimizer={path,kwargs}`, `loss={path,kwargs}`, `data={factory,kwargs}`, `checkpoint_dir`, `config_hash` (the **un-suffixed** hash, stable for dedup/`resume` sibling-finding even when `run_id` is suffixed), and `schema_version`. *And* a test asserts `manifest["model"]["path"] is not None` (the load-bearing field `resume` needs to rebuild `state_template`).

**AC7 — run-id = config-hash with uuid-fallback on collision.**
*Given* a run, *when* `run` derives the run-id, *then* `run_id = config_hash` by default; *and if* the directory `.xtrax/runs/<config_hash>/` **already exists** (predicate = directory existence, created with `os.makedirs(exist_ok=False)`), *then* `run_id = f"{config_hash}-{short_uuid}"` so a sequential identical re-run gets a distinct dir. **MVP limitation (stated):** the check-then-create is not atomic across *concurrent* processes (TOCTOU) — two simultaneous identical runs may still race; out of MVP scope, documented.

**AC8 — checkpoints namespaced under the run dir (the real no-clobber guard) [C1].**
*Given* two sequential runs of an *identical* config, *when* each writes checkpoints, *then* checkpoints are written under the run's **unique** directory `.xtrax/runs/<run_id>/checkpoints/` (`checkpoint_dir` is derived from `run_id`, NOT a verbatim shared scalar). Because the second identical run gets a distinct `run_id` via the AC7 uuid fallback, the two runs write to **distinct** checkpoint dirs and **cannot clobber** each other. **Test:** run twice with the same config; assert the two `checkpoint_dir`s differ and both checkpoint sets exist. *(This INVERTS the brainstorm's original "checkpoint_dir independent of run-id" wording — the adversarial review proved that taking the dir verbatim from config is what CAUSES the clobber, since orbax keys on `int(state.step)`. Run-id namespacing is the only formulation that actually delivers the no-clobber guarantee. User-confirmed.)*

**AC9 — mandatory `schema_version`; deterministic config-hash.**
*Given* a config file, *when* `run` parses it, *then* a missing top-level `schema_version` raises `ConfigError` (no silent default); *and* the manifest carries the same `schema_version`. `config_hash = sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()[:12]` — `sort_keys` makes it stable under kwargs reordering and `default=str` handles TOML datetimes/non-JSON-native values. **Test:** two configs with the same keys in different order hash identically; a nested-kwargs + datetime fixture hashes without error.

**AC10 — `xtrax run` wired into REGISTRY, tyro-free.**
*Given* the CLI, *when* a user runs `xtrax run config.toml`, *then* the verb dispatches via the existing `REGISTRY` dict (`{verb:(ArgsClass, run_fn)}`); `import xtrax`/`import xtrax.cli` stay tyro-free (tyro inside `main()`); new deps (if any) land in the `cli` extra. **Test:** `assert "tyro" not in sys.modules` after `import xtrax.cli` (mirrors the E2 isolation test).

**AC11 — clear typed error on a bad/missing import-path, naming the section.**
*Given* a config whose `[model]`/`[optimizer]`/`[data]`/`[loss]` path does not resolve, *when* `run` attempts resolution, *then* it raises a typed error naming **both the section and the path**. Since `load_fn`/`CLIImportError` carries only the path, `run_from_config` MUST wrap each `load_fn` call: `try: ... except CLIImportError as e: raise CLIImportError(f"[{section}] {e}") from e`. **Do NOT modify `loader.py`** — section context lives at the call site.

**AC12 — `num_epochs` is a validated positive int.**
*Given* a config, *when* parsed, *then* `num_epochs` MUST be a positive `int` (validated at parse, raising `ConfigError` otherwise). Although `DataModule.num_epochs` permits `None` (cycle indefinitely), `fit_sync` does `range(num_epochs)` which raises `TypeError` on `None` — so the single config scalar feeding both is constrained to a positive int.

**AC13 — MVP observability limitation: no callbacks [M5].**
*Given* the run-MVP, *when* it constructs the `Engine`, *then* `callbacks=()` (empty) — the MVP fires **no callbacks**: no metric/loss logging, no progress, no early stopping. The only persisted artifacts are the orbax checkpoints + the manifest. *(Explicit, user-confirmed MVP limitation so it is not later read as a defect. Recommended first follow-up: a minimal default loss-logging callback.)*

## Decision Log (5 forks — winner + why the alternative lost)

| Fork | Winner | Rejected | Why |
|------|--------|----------|-----|
| **2 — DataModule resolution** | **A** — factory returns a dataset; `run` always wraps | E (duck-typed return), B (factory→DataModule), C (`from_config` classmethod), D (registry) | Critic [MAJOR]: E's `isinstance` on a user callable's return is a silent-behavior fork — same class as the E2 html/png no-op the epic-audit caught; A fails at one known site with one stack trace and adds zero runtime API. C/D add runtime/registry surface; B leaks `batch_size`. |
| **5 — scope** | **H** — always-write tiny manifest | F (defer manifest), G-flag variant | H gives `resume` a written contract on every run via a single code path; recorded as a **conscious seam cost**. |
| **5 — naming** | **I** — `TrainConfig` | reuse `RunSpec` | E1 owns `RunSpec`/axis-config; collision would be a real bug. No tiling/axis spec embedded in MVP. |
| **1 — schema** | **J + `schema_version`** — raw-TOML `{path,kwargs}` | K (typed dataclass now) | Smallest coherent MVP; J→K is a localized parse-boundary adapter swap *if* `schema_version` is stamped now. |
| **3 — wiring placement** | **M** — promote `init_state` to `training`, keep `run_from_config` cli-private | L (all cli-private), N (promote both) | `init_state` fixes a *documented* gap, reusable by `resume`/library users; signature carries only runtime objects (no inversion). L blocks reuse; N couples runtime to the cli `TrainConfig`. |

## Assumptions

- The user's `[data] factory` returns a dataset the existing `DataModule` accepts; `run` does not validate dataset internals, only that wrapping succeeds.
- `Engine.fit_sync` is the correct synchronous entry point (async `fit` out of MVP scope).
- `seed` is a single top-level scalar feeding both `init_state` and `DataModule.seed` (intentional coupling for MVP; splitting is a deferred option).
- TOML via stdlib `tomllib`; no new parse dependency.
- The optimizer factory is fully user-specified via `{path,kwargs}`; the user owns keeping `total_steps` consistent with run length (MVP does not derive it).

## TBDs / Deferred

- **`resume` verb** — reads the manifest, rebuilds `state_template` via `init_state` + manifest `model.path`, restores from the run's checkpoint dir.
- **`sweep` verb** — grid-over-`TrainConfig`-scalars vs bathos campaign delegate.
- **Typed `TrainConfig` dataclass (K)** — the J→K upgrade once `resume`/`sweep` justify a shared typed object; `schema_version` keeps it open.
- **Observability** — a minimal default logging callback (the AC13 follow-up), and/or an optional `[callbacks]`/`[logging]` config section.
- **Optimizer/`total_steps` auto-derivation** — cross-checking the LR schedule length against `num_epochs × steps_per_epoch`.
- **Accepting a pre-built `DataModule`** — only ever as an explicit, validated *second* config key, never a silent return-type branch.
- **Concurrent-run TOCTOU hardening** — atomic run-dir reservation.
- **Tiling/axis-spec embedding** — connecting E1's `RunSpec` to `TrainConfig` is out of MVP scope.

## Pre-mortem (invariants enforced in code/tests, not prose)

The brainstorm's pre-mortem + adversarial review traced these failure modes; each maps to a construction/write-site assertion:

1. **Manifest `model.path` non-optional** — assert non-null at the manifest write site (AC6).
2. **No-clobber actually delivered** — checkpoints namespaced under `<run_id>/`, distinct run_ids on collision (AC8); test runs twice and asserts distinct dirs + both checkpoint sets present. (The original "verbatim from config" wording would have *shipped* the clobber.)
3. **No duck-type branch — tested by double-wrap** — factory returns a `DataModule`, assert it is re-wrapped (AC3); a plain-dataset test alone would pass *with* the branch and miss the regression.
4. **`schema_version` mandatory** — missing version raises on both config parse and manifest write (AC9).
5. **Optimizer `total_steps` footgun** — documented; user-owned for MVP (AC2). A wrong `total_steps` silently desyncs the LR schedule — flagged, not auto-fixed.
6. **Section-labeled load errors** — bad path fails at the config site with the section named (AC11), not deep in `fit_sync`.

## Adversarial Revision Log (spec-challenger ∥ spec-defender + plan-auditor, 260623)

**Verdict:** NEEDS_WORK / NOT-READY → revised. 3 streams, source-verified. Folded:

- **C1 [CRITICAL]** — AC8 self-contradictory (verbatim `checkpoint_dir` + orbax keying on `int(step)` ⇒ clobber). **Resolved (user):** namespace checkpoints under `.xtrax/runs/<run_id>/checkpoints/`; collision fallback delivers no-clobber. AC8 rewritten (inverted).
- **C2 [CRITICAL]** — `make_optimizer` unconstructable from flat config (non-scalar `base`); `total_steps` unexposed. **Resolved:** AC2 restricts targets to all-scalar factories (`adamw_with_schedule`), strikes `make_optimizer`, documents the `total_steps` footgun.
- **M4 [MAJOR]** — "no duck-type branch" is an absence-of-code property; the plain-dataset test can't catch the regression. **Resolved:** AC3 enforcement switched to the **double-wrap** test (factory returns a DataModule → assert re-wrapped).
- **M5 [MAJOR]** — `callbacks=()` ⇒ silent training. **Resolved (user):** ship thin; AC13 makes it an explicit MVP limitation.
- **M1** — divergent `Engine(...)` spellings → pinned one canonical all-keyword form (anchors block + AC5). **M2** — `pkg:MyModel` examples invalid → AC1/AC3 use real `module.path:symbol`; T9 fixture must be a module-level import path, not a conftest closure. **M3** — `init_state` derivation unpinned → AC4 pins `PRNGKey(seed)`, `jnp.int32` step, `extras={}`, `opt_state` formula. **M6** — collision predicate (dir vs file) + `config_hash` stability + TOCTOU → AC7 (dir predicate, `makedirs(exist_ok=False)`, race documented) + AC6 (un-suffixed `config_hash`). **M7** — config-hash canonicalization → AC9 (`sort_keys`, `default=str`). **m1** — `num_epochs=None` crashes `range()` → AC12 (positive int). **m2/m3** — Z docstring precision; T9 fixture-as-module.
- Plan-auditor's 4 required fixes (step dtype, T6←T3 dep, T9 tyro-isolation assert, AC11 section labels) all folded.

---

**Status:** ready for revised DAG → sprint TOML (Cursor hand-off package).

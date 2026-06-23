# E3-MVP Backlog DAG — `xtrax run` (TrainConfig-driven CLI)

- **task_id:** `260623_e2-run`
- **epic:** #2629 `[parent:#2605]`
- **date:** 2026-06-23
- **spec:** `.praxia/docs/specs/260623_e3-run-verb-trainconfig-driven-cli.md` (AC1–AC11, winner A+H+I+J+M, pre-mortem invariants 1–4)
- **style:** matches `.praxia/docs/plans/260623_e2-mvp-backlog-dag.md` (waves, critical path, AC coverage map, judgment flags for Cursor hand-off)
- **provenance:** staff DAG → adversarial review (spec-challenger ∥ spec-defender + plan-auditor) → sprint TOML. Grounding anchors re-verified against source before decomposition; two under-specified seams surfaced below (loss_fn gap, DataModule arity gap) and pinned to tasks.

> **⚠️ SUPERSEDED post adversarial review (260623).** This DAG's task *structure* (T0–T9 + Z, waves, critical path) is intact, but several task *details* were revised by the adversarial pass — chiefly **AC8 inverted** (checkpoints now namespaced under `.xtrax/runs/<run_id>/checkpoints/`, NOT a verbatim config scalar — C1), **AC2** (`make_optimizer` struck as a config target — C2), the **no-duck-type test** switched to double-wrap (M4), **callbacks=()** made an explicit MVP limitation (M5), plus the `[loss]` AC and a dozen MAJOR/minor fixes. **The authoritative, fully-resolved artifacts are the revised spec (`260623_e3-run-verb-trainconfig-driven-cli.md`, see its Adversarial Revision Log) and the executable sprint TOML (`.praxia/sprint_plans/260623_e3-run-verb.toml`).** Read those, not this DAG's per-task prose, for implementation.

## Grounding re-verification (read before building — corrects the anchor)

The spec's one-line anchor `engine/engine.py:222 fit_sync` and "always wrap `DataModule(dataset, batch_size)`" each hide a seam an implementer **will** trip on. Verified against source:

1. **`fit_sync` is `Engine.fit_sync`, a bound method — NOT a free function.** `engine.py:222`. To reach it `run_from_config` must: build `Trainer(loss_fn, optimizer)` (`trainer.py`, `eqx.Module(loss_fn, optimizer)`), wrap it `Engine(trainer=..., callbacks=())` (`engine.py:46`, `trainer`/`callbacks` static, `validation_callbacks` defaults `()`), then call `engine.fit_sync(state, data, num_epochs, checkpoint_dir)`.
2. **`Trainer` needs a `loss_fn`** (`trainer.py` `loss_fn: LossFunction`). **The spec's `[model]/[optimizer]/[data]` triad omits loss.** This is a real gap: a config-resolved `loss_fn` is required to instantiate `Trainer`. → pinned to **T4** as a `[loss]` section resolved via `load_fn` (same path/kwargs shape). Flagged JUDGMENT.
3. **`DataModule(dataset, batch_size)` is NOT a 2-arg constructor.** `data/module.py:17` — `num_epochs`, `seed`, `distributed` are `eqx.field(static=True)` with **no defaults**; only `collate_fn` defaults. So "unconditional wrap" still must supply `num_epochs`/`seed`/`distributed`. → pinned to **T5**: wrap is `DataModule(dataset, batch_size=..., num_epochs=..., seed=..., distributed=False)` with `seed`/`num_epochs` from top-level config scalars, `distributed=False` for MVP. The *unconditional* invariant (no `isinstance` branch on the factory return) is unchanged and still pinned by AC3/test.
4. **`init_state` does not exist** (`training/types.py:41` — `ResumableState` is hand-assembled; `training/__init__.py` confirms no `init_state` export). T1 creates it.
5. **`fit_sync` consumes `num_epochs`** (not `steps`) — `engine.py:222`. Config duration scalar is `num_epochs` for MVP (spec Assumption "whichever fit_sync consumes" → resolved: epochs).
6. **emit/json envelope precedent:** `cli/emit.py:62` writes `{"_meta": {"schema_version": 1}, **stats}`. The manifest writer (T6) follows the same "schema_version mandatory" discipline but is a distinct file artifact (`manifest.json`), not the emit envelope.

## Architecture (one-pass sketch)

```
config.toml ──tomllib──► raw dict ──parse/validate (T3)──► TrainConfig (cli-private)
                                                              │  {schema_version, model{path,kwargs},
                                                              │   optimizer{...}, data{factory,batch_size},
                                                              │   loss{path,kwargs}, seed, num_epochs,
                                                              │   checkpoint_dir}
                                                              ▼
              run_from_config (T7, cli-private glue) ─ resolves via load_fn:
                 model = load_fn(model.path)(**kwargs)            [AC1]
                 optimizer = load_fn(optimizer.path)(**kwargs)    [AC2]
                 loss_fn = load_fn(loss.path)(**kwargs)           [seam #2]
                 dataset = load_fn(data.factory)()                [AC3]
                 data = DataModule(dataset, batch_size, num_epochs, seed, distributed=False)  ← UNCONDITIONAL
                 state = init_state(model, optimizer, seed)       [AC4, public xtrax.training]
                 Engine(Trainer(loss_fn, optimizer), callbacks=()).fit_sync(state, data, num_epochs, checkpoint_dir)  [AC5]
                 write_manifest(run_id, model, optimizer, checkpoint_dir, config_hash, schema_version)  [AC6]
                                                              ▼
              run verb (T8): RunArgs(config: str) + run_run ─► REGISTRY["run"]  [AC10]
```

- **Layer split (spec winner M):** `init_state` → **`xtrax.training`** (public, own test). Everything else (`TrainConfig` parse, `run_from_config`, manifest writer, run verb) → **`src/xtrax/cli/`** (cli-private). Verb registers in `REGISTRY`.

## Tasks

| id | size | deps | ACs / invariants | test-first | files (create/edit) | success criterion |
|----|------|------|------------------|-----------|---------------------|-------------------|
| **T0** scaffold + config-error type | S | — | AC9(part), AC11(part) | no | C `src/xtrax/cli/config.py` (stub `TrainConfig` dataclass + `ConfigError(CLIError)`); E `src/xtrax/cli/errors.py` (add `ConfigError`) | `from xtrax.cli.config import TrainConfig, ConfigError` imports; `import xtrax.cli` stays tyro-free |
| **T1** `init_state` (public) | S | — | **AC4** | yes | C `src/xtrax/training/state.py` (`init_state(model, optimizer, seed) -> ResumableState`); E `src/xtrax/training/__init__.py` (export `init_state`) | returns `ResumableState` with `step==0`, key from `jax.random.PRNGKey(seed)`, `opt_state = optimizer.init(eqx.filter(model, eqx.is_array))` |
| **T2** `init_state` test | S | T1 | **AC4** | (is test) | C `tests/training/test_init_state.py` | asserts `step==0`, key reproducible from seed, opt_state shape matches `optimizer.init`; asserts `init_state` importable from `xtrax.training` (public-API pin) |
| **T3** TOML parse → `TrainConfig` + `schema_version` guard | M | T0 | **AC9**, AC11(part), **inv#4** | yes | E `src/xtrax/cli/config.py` (`load_config(path) -> TrainConfig` via `tomllib`); C `tests/cli/test_config.py` | missing top-level `schema_version` raises `ConfigError` hard (not default); each `[model]/[optimizer]/[data]/[loss]` section parsed to `{path/factory, kwargs}`; **test pins inv#4** (missing version errors) |
| **T4** loss section + resolver wiring | S | T3 | seam#2, AC11(part) | yes | E `src/xtrax/cli/config.py` (add `[loss]` to schema); C `tests/cli/test_config.py` (loss case) | `[loss] path/kwargs` parses; **JUDGMENT: default loss if section omitted? — see Judgment Flags** |
| **T5** config-hash canonicalization | M | T3 | AC7(part), **inv#2 setup** | yes | C `src/xtrax/cli/hash.py` (`config_hash(cfg) -> str`); C `tests/cli/test_hash.py` | hash is stable under key reordering (canonical: sorted-keys JSON dump → sha256, short hex); identical configs → identical hash; **JUDGMENT: hash over raw TOML text or over canonicalized dict — see flags** |
| **T6** manifest writer (always-write) | M | T0, T5 | **AC6**, **AC8**, AC9(manifest half), **inv#1, inv#2** | yes | C `src/xtrax/cli/manifest.py` (`write_manifest(run_dir, ...)`); C `tests/cli/test_manifest.py` | writes `.xtrax/runs/<id>/manifest.json` with `run_id, model{path,kwargs}, optimizer{path,kwargs}, checkpoint_dir, config_hash, schema_version`; **test inv#1**: `model.path` non-null asserted at write site; **test inv#2**: `checkpoint_dir == config scalar`, never derived from run_id |
| **T7** `run_from_config` glue + run-id/uuid-fallback | L | T1, T3, T4, T5, T6 | **AC1, AC2, AC3, AC5, AC7, AC8, AC11**, **inv#1,2,3** | yes | C `src/xtrax/cli/run.py` (`run_from_config(cfg)`); C `tests/cli/test_run_from_config.py` | resolves model/opt/loss/data via `load_fn`; **unconditional** `DataModule` wrap (inv#3 test: no `isinstance` branch — assert wrap called on raw factory return); builds `init_state`→`Engine(Trainer(...)).fit_sync(...)`; run-id = `config_hash`, **uuid suffix iff `.xtrax/runs/<hash>/` exists** (AC7 collision test); bad path → `CLIImportError` naming section (AC11) |
| **T8** `run` verb + REGISTRY wiring | S | T7 | **AC10**, AC11(surface) | no | C `src/xtrax/cli/run_verb.py` (`RunArgs(config: str)`, `run_run`); E `src/xtrax/cli/registry.py` (+1 line `"run": (RunArgs, run_run)`) | `xtrax run config.toml` dispatches; `import xtrax.cli` + `import xtrax` stay tyro-free (tyro only in `main()`); top-level `CLIError`/`ConfigError` → clean message + nonzero exit |
| **T9** end-to-end verb test + invariant round-up | M | T8 | **AC3,6,7,8,9,10**, **inv#1–4** | (is test) | C `tests/cli/test_run_verb.py` (with tiny in-repo fixture model/loss/data factory) | `xtrax run <fixture.toml>` exits 0, writes manifest; **inv round-up asserts in one place**: (1) manifest `model.path` non-null, (2) `checkpoint_dir` == config scalar across two identical runs (no clobber), (3) data wrap unconditional (fixture factory returns plain dataset, `DataModule` still constructed), (4) missing `schema_version` → hard error; AC7 collision → second run gets uuid-suffixed run dir |
| **Z** dep + roadmap note | S | T0 | — | no | E `pyproject.toml` (confirm `cli` extra covers `run`; tomllib is stdlib → no new dep); E `src/xtrax/cli/__init__.py` (remove `run` from "Deferred" docstring, leave `sweep`/`resume`) | `lint-imports` green; no new runtime dep added (tomllib stdlib); deferred list trimmed to `sweep`/`resume` only |

## Waves & critical path

```
W0: T0 ∥ T1 ∥ Z            (T0/T1/Z need nothing; T1 is the public-API keystone)
W1: T2 (needs T1) ∥ T3 (needs T0)
W2: T4 ∥ T5 ∥ T6           (T4,T5 need T3; T6 needs T0+T5)
W3: T7                     (needs T1,T3,T4,T5,T6 — the integration keystone)
W4: T8                     (needs T7)
W5: T9                     (needs T8)
```

**Critical path:** `T0 → T3 → T5 → T6 → T7 → T8 → T9` (also `T1 → T7` joins at W3; T1 is off the longest path but is a hard blocker for T7).

**Keystone tasks:**
- **T7 `run_from_config`** — the only **L** node, sits on the critical path, owns every resolver wiring + all three construction-site invariants (inv#1/2/3) + the run-id collision seam. Riskiest single task.
- **T1 `init_state`** — public-API keystone: it is the one symbol that crosses the runtime/cli layer boundary (spec winner M). If its signature leaks a config/path/tyro concern, the layering decision is violated. Small but load-bearing.

## Pre-mortem invariant → test mapping (no invariant lives only in prose)

| invariant (spec §pre-mortem) | construction/write site | pinned by |
|------------------------------|-------------------------|-----------|
| **inv#1** manifest `model.path` non-optional | `write_manifest` (T6) | T6 test (assert non-null at write) **+** T9 round-up |
| **inv#2** `checkpoint_dir` independent of run-id | `write_manifest` (T6) reads config scalar verbatim | T6 test (== config scalar) **+** T9 two-identical-runs no-clobber |
| **inv#3** no duck-type branch in data resolution | `run_from_config` (T7) unconditional `DataModule(...)` | T7 test (wrap called on raw factory return, no `isinstance`) **+** T9 plain-dataset fixture |
| **inv#4** `schema_version` mandatory, errors hard | `load_config` (T3) parse + `write_manifest` (T6) | T3 test (missing → `ConfigError`) **+** T9 missing-version hard-error |

## AC coverage map

| AC | tasks |
|----|-------|
| AC1 model via loader | T7 (+T9) |
| AC2 optimizer via loader | T7 (+T9) |
| AC3 data: factory→dataset, unconditional wrap | T7 (inv#3 test) + T9 |
| AC4 `init_state` public in `xtrax.training` | T1 + T2 |
| AC5 `run_from_config` → `fit_sync` to completion | T7 |
| AC6 manifest always written, all fields | T6 + T9 |
| AC7 run-id = config-hash, uuid-fallback | T5 (hash) + T7 (collision) + T9 |
| AC8 `checkpoint_dir` independent of run-id | T6 (inv#2) + T9 |
| AC9 `schema_version` mandatory (config+manifest) | T3 (config) + T6 (manifest) + T9 |
| AC10 `run` in REGISTRY | T8 |
| AC11 typed error on bad import-path | T3/T4 (config-site) + T7 (resolve-site) |

## Ordering hazards / seams (an implementer can get these wrong)

1. **`fit_sync` is `Engine.fit_sync`, not free** — T7 must build `Engine(Trainer(loss_fn, optimizer), callbacks=())` first. Do not call a bare `fit_sync(...)`; it does not exist. (Anchor corrected above.)
2. **`DataModule` arity** — `num_epochs/seed/distributed` have no defaults; T5/T7 must pass them. The *unconditional* invariant is about no `isinstance` branch, NOT about a 2-arg call. Don't conflate.
3. **`init_state` must stay tyro-/config-free** — its signature is `(model, optimizer, seed)`, runtime objects only. Putting a `TrainConfig` or path string in it inverts the layer (spec rejected N for exactly this). Public-API pin lives in T2.
4. **`import xtrax.cli` tyro-free** — T8 adds `run` to `REGISTRY`; the verb `run_fn` is eager xtrax code (fine), but tyro stays inside `entrypoint.main()`. T9 should assert `'tyro' not in sys.modules` after `import xtrax.cli` (mirror the E2 isolation test).
5. **config-hash canonicalization (T5)** — hashing the raw TOML *text* makes whitespace/key-order significant → identical-meaning configs hash differently → AC7 collision logic misfires. Canonicalize (parse → sorted-key JSON → sha256) so semantically identical configs collide as intended.
6. **run-id collision path (T7)** — the uuid fallback fires **iff** `.xtrax/runs/<config_hash>/manifest.json` already exists. Test must create the first run's dir, then assert the second run writes to a *different* (uuid-suffixed) dir and does NOT overwrite the first manifest.
7. **`checkpoint_dir` vs run-dir are different directories** — `checkpoint_dir` (config scalar, passed to `fit_sync`) ≠ `.xtrax/runs/<id>/` (manifest dir). Conflating them re-introduces inv#2's clobber bug. Keep them separate variables in T7.

## Judgment flags for Cursor hand-off (everything else is mechanical)

The EXECUTE phase is intended for cheap mechanical (Cursor) execution. These tasks carry a genuine decision a mechanical agent should NOT guess — resolve in plan-audit or flag to a human:

- **T4 — loss section default.** Spec's triad omits `[loss]`. Decision: is `[loss]` **required** (cleanest, fail-loud, consistent with model/opt), or does an omitted `[loss]` get a sensible default? Recommend **required** (no magic default; matches the "fail at config site" ethos of AC11). **Decision needed before T4 is mechanical.**
- **T5 — config-hash domain & length.** Decide: hash over canonicalized **dict** (recommended — order-insensitive per hazard #5) vs raw text; and the suffix length (e.g. 12 hex chars). Recommend canonical-dict + 12 hex. **Decision needed before T5 is mechanical.**
- **T7 — `num_epochs` source & `distributed` default.** Spec Assumption leaves duration as "steps/epochs whichever fit_sync consumes" → verified **epochs**. Confirm `num_epochs` is a top-level config scalar and `distributed=False` is the MVP default for the `DataModule` wrap. Low-risk but a written decision avoids drift. **Confirm, then mechanical.**

Everything else (T0 scaffold, T1 `init_state` body, T2/T3/T6/T9 tests-from-AC, T8 +1-line REGISTRY edit, Z dep/docstring) is **mechanical** given the above three decisions.

## Status

Ready for adversarial plan-audit. 11 tasks (T0–T9 + Z), all S/M except the single L keystone (T7). No L3 tasks. Three judgment flags surfaced for the Cursor hand-off; resolving them makes the EXECUTE phase fully mechanical. `no_autonomous_push_or_merge_to_main` honored — this doc is plan-only.

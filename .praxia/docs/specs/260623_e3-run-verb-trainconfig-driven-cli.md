# E3-MVP spec — `xtrax run`: TrainConfig-driven training run

**task_id:** `260623_e2-run` · **epic:** #2629 (`[parent:#2605]`) · **brainstorm:** contemplex session `f300363c` (architectural; INVEST all-pass)
**Brainstorm record:** `.praxia/docs/specs/260623_hmw-design-the-xtrax-run-verb-drive-a-tr.md`

## Summary

`xtrax run config.toml` drives a training run to completion from a TOML config file, reusing the existing `Engine.fit_sync` / `Trainer` / `DataModule` / optim helpers — pure glue, never a new training loop. It is the first CLI verb to reach into the runtime layer. Winner = **A + H + I + J + M** (contemplex `f300363c`): the `[data]` factory resolves to a **dataset** that `run` *always* wraps in `DataModule` (no duck-type branch); a tiny manifest is **always** written to `.xtrax/runs/<id>/` so the later `resume` verb has a contract; the config object is named **`TrainConfig`** (orthogonal to E1's `RunSpec`); the config is **raw-TOML `{path, kwargs}`** per section with a mandatory `schema_version`; and a new public **`init_state`** is promoted into `xtrax.training`.

## Acceptance Criteria (Given/When/Then)

**AC1 — model resolution via the existing loader.**
*Given* a config with `[model] path="pkg:MyModel"` and a `kwargs` sub-table, *when* `run` builds the model, *then* it resolves the symbol via `cli/loader.py` `load_fn('pkg:MyModel')` and calls it with `**kwargs` — no new resolution mechanism is introduced.

**AC2 — optimizer resolution via the existing loader.**
*Given* a config with `[optimizer] path=... kwargs=...`, *when* `run` builds the optimizer, *then* it resolves and constructs it via `load_fn` + kwargs (an import-path to `make_optimizer`/`adamw_with_schedule` or any optax factory), reusing the same string→object path as the model.

**AC3 — data: factory returns a dataset, `run` ALWAYS wraps (no duck-type branch).**
*Given* a config with `[data] factory="pkg:load_train"` and a `batch_size` scalar, *when* `run` builds the data input, *then* it resolves the factory via `load_fn`, calls it to get a dataset object, and **unconditionally** constructs `DataModule(dataset, batch_size)`. *And* there is **no `isinstance`/duck-type branch on the factory's return** — the contract is exactly "factory returns a dataset." (This is the explicit guard against the E2-class silent-behavior footgun; a test asserts the wrap is unconditional.)

**AC4 — `init_state` is public in `xtrax.training`.**
*Given* a resolved `model`, `optimizer`, and `seed`, *when* `init_state(model, optimizer, seed)` is called, *then* it returns a valid `ResumableState(step, key, model, opt_state)` `eqx.Module` with `step==0` and a PRNG key derived from `seed`. *And* the symbol is importable as public `xtrax.training` API (it carries no config, no path strings, no cli/tyro concern in its signature).

**AC5 — `run_from_config` wires to `fit_sync` to completion.**
*Given* a parsed `TrainConfig`, *when* `run_from_config(config)` executes, *then* it composes resolved model+optimizer+data+`init_state` and drives `Engine.fit_sync(...)` to completion, returning the final result/state. *And* `run_from_config` lives in `cli/run.py` (cli-private glue), not in the runtime layer.

**AC6 — manifest is ALWAYS written with all required fields.**
*Given* any successful `run`, *when* it completes (or begins, as designed), *then* it writes `.xtrax/runs/<run_id>/manifest.json` containing: `run_id`, **non-optional** `model` = `{path, kwargs}`, `optimizer` = `{path, kwargs}`, `checkpoint_dir`, `config_hash`, and `schema_version`. *And* a test asserts the `model.path` field is non-null (the load-bearing field `resume` needs to rebuild `state_template`).

**AC7 — run-id = config-hash with uuid-fallback on collision.**
*Given* two runs, *when* `run` derives the run-id, *then* it is the config-hash by default; *and if* a manifest already exists at `.xtrax/runs/<config_hash>/`, *then* `run` appends a short uuid suffix so the second run does not overwrite the first's manifest.

**AC8 — `checkpoint_dir` is independent of run-id (data-loss guard).**
*Given* two runs of an *identical* config, *when* each writes checkpoints, *then* `checkpoint_dir` is taken verbatim from the config scalar and is **never derived from run-id**, so identical re-runs do not silently clobber each other's checkpoints. (Explicit AC — this is the data-loss invariant; a test pins it.)

**AC9 — mandatory `schema_version` on both config and manifest.**
*Given* a config file, *when* `run` parses it, *then* a missing top-level `schema_version` is a hard, clearly-messaged error (not a silent default); *and* the written manifest carries the same `schema_version`. (Keeps the J→K typed-config upgrade and `resume`'s versioned re-hydration open.)

**AC10 — `xtrax run` wired into REGISTRY.**
*Given* the CLI, *when* a user runs `xtrax run config.toml`, *then* the `run` verb is dispatched via the existing `REGISTRY` dict (`{verb: (ArgsClass, run_fn)}`), keeping `import xtrax`/`import xtrax.cli` tyro-free (tyro inside `main()`), with any new deps declared in the `cli` extra.

**AC11 — clear typed error on a bad/missing import-path.**
*Given* a config whose `[model]`/`[optimizer]`/`[data]` path does not resolve, *when* `run` attempts resolution, *then* it raises a clear typed error (`CLIImportError` from the loader, or a typed config error) naming the offending section and path — failing at the config site, not deep inside `fit_sync`.

## Decision Log (5 forks — winner + why the alternative lost)

| Fork | Winner | Rejected | Why |
|------|--------|----------|-----|
| **2 — DataModule resolution** | **A** — factory returns a dataset; `run` always wraps | E (duck-typed return), B (factory returns DataModule), C (`from_config` classmethod), D (registry) | Critic [MAJOR]: E's `isinstance` on a user callable's return is a silent-behavior fork — same class as the E2 html/png no-op the epic-audit caught; A fails at one known site with one stack trace and adds zero runtime API. C/D add new runtime/registry surface; B leaks `batch_size` out of the config. |
| **5 — scope** | **H** — always-write tiny manifest | F (defer manifest), G-flag variant | H gives `resume` a written contract on every run via a single code path (the `--manifest` flag variant bifurcates and leaves the seam untested); recorded as a **conscious seam cost**, not bare minimum. |
| **5 — naming** | **I** — `TrainConfig` | reuse `RunSpec` | E1 already owns `RunSpec`/axis-config; collision would be a real bug. `TrainConfig` stays orthogonal; no tiling/axis spec embedded in MVP. |
| **1 — schema** | **J + `schema_version`** — raw-TOML `{path, kwargs}` | K (typed dataclass now) | Smallest coherent MVP; J→K is a localized parse-boundary adapter swap *if* `schema_version` is stamped now (critic condition). Without `schema_version` J is a deferral trap. |
| **3 — wiring placement** | **M** — promote `init_state` to `training`, keep `run_from_config` cli-private | L (all cli-private), N (promote both) | `init_state` fixes a *documented* hand-assembly gap and is reusable by `resume` + library users (justifies one public symbol); its signature carries only runtime objects (no layering inversion). L blocks library reuse; N couples runtime to the cli-owned `TrainConfig` schema (inversion). |

## Assumptions

- The user's `[data] factory` returns a dataset object that the existing `DataModule(dataset, batch_size)` already accepts (grain/array/etc.) — `run` does not validate dataset internals, only that wrapping succeeds.
- `Engine.fit_sync` is the correct synchronous entry point (the async `fit` is out of scope for the `run` MVP).
- `seed` and `checkpoint_dir` are top-level config scalars; training duration is `steps`/`epochs` (whichever `fit_sync` consumes).
- TOML is the config format (Python stdlib `tomllib`); no new parse dependency needed.
- `config_hash` is computed over the canonicalized config contents (stable key ordering) so identical configs hash identically.

## TBDs / Deferred

- **`resume` verb** — reads the manifest, rebuilds `state_template` via `init_state` + the manifest's `model.path`, restores from `checkpoint_dir`. (Own brainstorm/spec/DAG.)
- **`sweep` verb** — grid-over-`TrainConfig`-scalars vs delegate to bathos campaign primitives. (Own brainstorm.)
- **Typed `TrainConfig` dataclass (K)** — the J→K upgrade once `resume`/`sweep` justify a shared typed object; `schema_version` keeps the migration open.
- **Accepting a pre-built `DataModule`** — only ever as an explicit, validated *second* config key, never a silent return-type branch.
- **Tiling/axis-spec embedding** — connecting E1's `RunSpec`/axis-config to `TrainConfig` is explicitly out of MVP scope.

## Pre-mortem (invariants the design must enforce in code, not prose)

The brainstorm's pre-mortem traced the most likely six-months-out failure: the manifest was written all along but its `model.path` field had been left **optional** and early runs wrote it null — so `resume` had nothing to rebuild `state_template` from. Compounded by `checkpoint_dir` later drifting to be derived from the config-hash run-id (silent clobber on identical re-runs), and a contributor "simplifying" A back into E's duck-type branch. Root cause: the load-bearing invariants lived only in prose under raw-dict J and eroded. **The spec mandates each invariant be asserted at its write/construction site with a test:**

1. **Manifest `model.path` is non-optional** — assert non-null at the manifest write site (AC6).
2. **`checkpoint_dir` independent of run-id** — assert it equals the config scalar, never the run-id (AC8).
3. **No duck-type branch in data resolution** — assert the `DataModule` wrap is unconditional (AC3).
4. **`schema_version` mandatory** — assert a missing version errors hard on both config parse and manifest write (AC9).

---

**Status:** ready for staff DAG → adversarial plan-audit → sprint TOML.

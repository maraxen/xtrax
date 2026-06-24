---
title: "xtrax resume: checkpoint-resume CLI verb"
task_id: 260624_resume-verb
date: 260624
status: finalized
brainstorm_session: true
invest_overrides: []
parent_epic: "#2605"
parent_spec: "260623_e3-run-verb-trainconfig-driven-cli.md"
---

# `xtrax resume` — checkpoint-resume CLI verb

**task_id:** `260624_resume-verb` · **epic:** #2605 (E2 auto-CLI remainder) · **brainstorm:** constrained-technical
**Parent spec:** `260623_e3-run-verb-trainconfig-driven-cli.md` (the `run` verb, which built the manifest seam this verb consumes)
**Status:** finalized — adversarial review applied

## Summary

`xtrax resume <run-id> --epochs N` resumes a previously interrupted (or completed) training run from its latest checkpoint. It reads the manifest written by `xtrax run` at `.xtrax/runs/<run-id>/manifest.json`, re-resolves all import paths to rebuild the component graph, loads the latest orbax checkpoint into a reconstructed `state_template`, wraps the dataset in a `DataModule`, and calls `Engine.fit_sync` for N additional epochs — creating a **new sibling run-dir** under a new `run_id` with a `resumed_from` pointer back to the original.

The entire correctness of this verb hinges on consuming the manifest contract hardened by the `run` verb's C1/AC6/AC7/AC8 work:
- **AC6 (always-write manifest BEFORE training):** ensures crashed-but-checkpointed runs are resumable
- **AC6 (non-optional model.path):** enables state_template reconstruction via `init_state`
- **AC8 (checkpoint_dir derived from run_id):** ensures no-clobber and enables unambiguous checkpoint discovery
- **AC7 (config_hash stable, un-suffixed):** enables sibling-finding for resumed runs

## Verified Runtime Anchors (source-confirmed)

- `load_checkpoint(manager, state_template, step=None)` (`checkpoint/orbax.py:71-100`): requires `state_template` for pytree reconstruction; `step=None` → loads `latest_step()`.
- `get_checkpoint_manager(directory, max_to_keep=5)` (`checkpoint/orbax.py:12-42`): creates `ocp.CheckpointManager` with `PyTreeCheckpointHandler`.
- `init_state(model, optimizer, seed)` (`training/state.py:8-15`): returns `ResumableState(step=0, key=PRNGKey(seed), model, opt_state, extras={})`.
- `Callback.on_resume(state)` (`training/types.py:34`): exists in protocol but **never fired by Engine** (deviation note `engine/engine.py:10`).
- `Engine.fit_sync(state, data, num_epochs, checkpoint_dir)` (`engine/engine.py:222-243`): the synchronous entry point.
- `write_manifest_dict(run_dir, cfg_dict, run_id, config_hash_val)` (new extracted helper): required to support writing manifests from `resume` (where we lack a `TrainConfig` instance).
- `REGISTRY` (`cli/registry.py:21-26`): verb dispatch dict — `resume` will be added here.

## Acceptance Criteria (Given/When/Then)

**RAC1 — manifest read from run-id.**
*Given* a run-id `abc123` and a manifest at `.xtrax/runs/abc123/manifest.json`, *when* `xtrax resume abc123 --epochs 5` is invoked, *then* the manifest is read and parsed into a dict with all required fields: `run_id`, `model.path`, `optimizer`, `loss`, `data` (including `batch_size`), `checkpoint_dir`, `config_hash`, `schema_version`, `num_epochs`, `seed`. *And* if the manifest file does not exist, a typed `ResumeError` is raised naming the expected path.

**RAC2 — optional --manifest-path override.**
*Given* `xtrax resume abc123 --epochs 5 --manifest-path /custom/path/manifest.json`, *when* the verb starts, *then* it reads from `/custom/path/manifest.json` instead of the default location. The `run_id` argument is still used for the *new* output directory derivation. *And* if the custom path does not exist, `ResumeError` is raised.

**RAC3 — schema_version validation.**
*Given* a manifest with `schema_version: N`, *when* resume reads it, *then* it validates `N == CURRENT_SCHEMA_VERSION` (a module-level constant in `cli/config.py`). *And if* mismatched, raises `ConfigError(f"manifest schema_version {N} != expected {CURRENT_SCHEMA_VERSION}")`.

**RAC4 — component re-resolution with section-labeled errors.**
*Given* a manifest with `model.path`, `optimizer.path`, `loss.path`, `data.factory`, *when* resume resolves them, *then* it calls `load_fn(path)(**kwargs)` for each via the **same `_resolve` helper** as `run_from_config`. *And if* any path fails to import, it raises `CLIImportError` naming both the section and the path (e.g., `[model] Could not import 'mypkg.models:MyModel'`). The resolve logic is extracted into a shared `resolve_components(manifest_dict) → ResolvedComponents` function.

**RAC5 — state_template reconstruction and checkpoint loading.**
*Given* resolved components (model, optimizer), *when* resume builds the state_template, *then* it calls `init_state(model, optimizer, manifest["seed"])` to get a zeroed `ResumableState`, *then* calls `load_checkpoint(get_checkpoint_manager(manifest["checkpoint_dir"]), state_template)` to load the latest checkpoint. *And if* no checkpoints exist in the dir, raises `ResumeError("No checkpoints found in {checkpoint_dir}")`. The `manifest["checkpoint_dir"]` is used exclusively to load the old state; the new CLI `run_id` derives the new output dir.

**RAC6 — new sibling run-id with `resumed_from` pointer.**
*Given* a resume of run `abc123`, *when* resume creates its output directory, *then* it generates a new `run_id` using the same `config_hash + collision-suffix` scheme as `run` (using the manifest's `config_hash`), creates `.xtrax/runs/<new_run_id>/`, and writes a **new manifest** with all original fields plus `resumed_from: "abc123"`. *And* checkpoints are written to `.xtrax/runs/<new_run_id>/checkpoints/`.

**RAC7 — Engine training with DataModule wrap and `--epochs N`.**
*Given* a loaded checkpoint state and `--epochs 5`, *when* resume calls `Engine.fit_sync`, *then* it first wraps the `dataset` in a `DataModule` (using `batch_size` and `seed` from the manifest, and `num_epochs=5` from the CLI flag), and passes `(loaded_state, data_module, num_epochs=5, checkpoint_dir=<new_run_checkpoint_dir>)`. *And* callbacks=() for MVP (same as run verb AC13).

**RAC8 — `on_resume` callback fired.**
*Given* a resume training invocation, *when* the Engine starts training, *then* it fires `on_train_start(state)` FIRST, followed immediately by `on_resume(state)` on all callbacks before the training loop begins. *Implementation:* Add a `resume: bool = False` parameter to `Engine.fit()` and `Engine.fit_sync()`. When `resume=True`, fire `on_resume` right after `on_train_start`.

**RAC9 — `--epochs` is required and validated as a positive int.**
*Given* `xtrax resume abc123`, *when* `--epochs` is omitted, *then* tyro raises a usage error. *And if* `--epochs 0` or `--epochs -1`, raises `ConfigError("--epochs must be a positive integer")`.

**RAC10 — manifest schema extension (backward-compatible).**
*Given* the manifest parser, *when* reading older manifests, *then* if `num_epochs`, `seed`, or `batch_size` are missing, raises `ResumeError("manifest missing required field: {field}. Was this run created with an older version of xtrax?")`.

**RAC11 — `xtrax resume` wired into REGISTRY, tyro-free.**
*Given* the CLI, *when* a user runs `xtrax resume abc123 --epochs 5`, *then* the verb dispatches via `REGISTRY["resume"] = (ResumeArgs, run_resume)`. *And* `import xtrax` / `import xtrax.cli` remain tyro-free (AC10 parity).

*(Note: Original brainstorm RAC11 config_hash validation was dropped as manifest dict cannot reproduce a perfect TrainConfig hash).*

## Decision Log

| # | Fork | Winner | Rejected | Rationale |
|---|------|--------|----------|-----------|
| 1 | Input contract | `<run-id>` + `--manifest-path` optional | run-dir path, config+run-id | run-id is the canonical identifier the manifest seam is built around; the manifest-path override covers relocability without polluting the common case |
| 2 | num_epochs/seed gap | Extend manifest (RAC10) | Re-read config.toml, both-with-reconciliation | Config.toml may not exist anymore (deleted, moved); manifest should be self-contained for resume correctness |
| 3 | Epoch counting | Flat `--epochs N` (user-explicit) | Derived remaining, total-target | Derivation requires `steps_per_epoch` which depends on dataset size — opaque, fragile. Flat is explicit, honest, composable (user can do math themselves) |
| 4 | Code reuse with run | Refactor into composable steps (RAC4) | Duplicate, new shared module | Composable steps minimize code duplication AND serve future `sweep` verb; extracting `resolve_components` is surgical |
| 5 | Run-id on resume | New sibling run-id with `resumed_from` (RAC6) | Same run dir (extended or segmented) | Preserves immutable-record-per-run-dir invariant; enables audit trail; avoids checkpoint step overlap edge case |
| 6 | on_resume callback | Wire via `resume: bool` parameter (RAC8) | Skip for MVP, callback-side detection | The protocol already defines the hook; not wiring it would be a deviation that compounds — and the implementation is trivial (one `if resume:` block in `fit`) |
| 7 | Schema version | Validate on read (RAC3) | Ignore for MVP | Future-proofs manifest evolution; cost is one `if` statement |

## Assumptions

- The user's model code at the import path in the manifest still produces a **structurally compatible** pytree (same leaf shapes/dtypes). If the model has changed, orbax will raise during `load_checkpoint`.
- `Engine.fit_sync` handles arbitrary starting step values correctly (orbax saves at `int(state.step)`, which will be non-zero for resumed runs — this already works since orbax keys on step number, and a resumed run's first save will be at the loaded step + steps_in_first_resumed_epoch).
- The `DataModule` constructed for a resumed run starts iteration from the beginning of the dataset (no data-position resumption). This is a known MVP limitation — the user's data pipeline must be designed to handle this (e.g., shuffle with seed).
- `seed` in the resumed run is re-used from the manifest. The loaded state's `key` (from checkpoint) is the actual PRNG state — `seed` is only used for `DataModule` construction.

## TBDs / Deferred

- **Data position resumption** — resume from the exact batch position, not just model state. Requires extending DataModule with checkpoint-aware iteration.
- **Derived epoch counting** — `remaining = total - completed` where completed is derived from checkpoint step and dataset size. Deferred because it requires coupling to dataset internals.
- **`--from-step N`** — resume from a specific checkpoint step rather than latest. Straightforward to add (pass `step=N` to `load_checkpoint`).
- **Config drift detection** — beyond config_hash comparison, detect structural pytree differences between state_template and checkpoint. Requires orbax metadata inspection.
- **`sweep` integration** — sweep may want to resume individual trials. The `resumed_from` pointer enables this; specific sweep-resume interaction deferred to sweep spec.

## Refactoring Plan (prerequisite to resume implementation)

### Step R0: Define `CURRENT_SCHEMA_VERSION`
**Files:** `cli/config.py`
Add `CURRENT_SCHEMA_VERSION = 1` as a module-level constant.

### Step R1: Refactor `write_manifest` and extend schema
**Files:** Modify `cli/manifest.py`, modify `cli/run.py` (call site).
1. Refactor `write_manifest` to extract a new `write_manifest_dict(run_dir: str, cfg_dict: dict, run_id: str, config_hash_val: str)`. Have `write_manifest` wrap it by passing `dataclasses.asdict(cfg)`.
2. Add `num_epochs`, `seed`, and `batch_size` to the serialized data.
3. Support an optional `resumed_from` field in the schema.
```python
# new data block inclusion
manifest["data"] = {
    "factory": cfg_dict["data"]["factory"],
    "kwargs": cfg_dict["data"].get("kwargs", {}),
    "batch_size": cfg_dict["data"]["batch_size"],
}
```

### Step R2: Create `read_manifest(path) → dict`
**Files:** Add to `cli/manifest.py`.
Read and validate manifest JSON. Validate required fields: `run_id`, `model.path`, `optimizer.path`, `loss.path`, `data.factory`, `data.batch_size`, `checkpoint_dir`, `config_hash`, `schema_version`, `num_epochs`, `seed`. Raise `ResumeError` for missing fields with actionable message.

### Step R3: Extract `resolve_components(manifest_dict) → ResolvedComponents`
**Files:** Refactor from `cli/run.py` into shared `cli/resolve.py`.
Extract resolution logic. The `dataset` is resolved and wrapped in a `DataModule` using the manifest's `batch_size`:
```python
def resolve_components(manifest_dict: dict, epochs: int) -> ResolvedComponents:
    # _resolve(...) for loss, model, optimizer
    dataset = load_fn(manifest_dict["data"]["factory"])(**manifest_dict["data"].get("kwargs", {}))
    data = DataModule(
        dataset,
        batch_size=manifest_dict["data"]["batch_size"],
        num_epochs=epochs,
        seed=manifest_dict["seed"],
        distributed=False,
    )
    return ResolvedComponents(model=model, optimizer=optimizer, loss_fn=loss_fn, dataset=data)
```
*(Update `run_from_config` to call this via `resolve_components(dataclasses.asdict(cfg), cfg.num_epochs)`)*

### Step R4: Add `resume: bool = False` to `Engine.fit` and `Engine.fit_sync`
**Files:** Modify `engine/engine.py`.
When `resume=True`, fire `on_resume(state)` on all callbacks immediately after `on_train_start(state)`.

## Implementation Plan

### Step I1: Resume verb and registry wiring
**Files:**
- Modify `cli/errors.py`: Add `ResumeError` inheriting from `ConfigError`.
- Create `cli/resume_verb.py`: Implement `ResumeArgs` and `run_resume(args: ResumeArgs)` fulfilling RAC1-RAC11.
- Modify `cli/registry.py`: Add `resume` to `REGISTRY`.
- Modify `cli/__init__.py`: Update module docstring to remove `resume` from the deferred list.

### Step I2: Testing
**Files:**
- Update `tests/cli/test_manifest.py` to fix any breakage from `write_manifest` signature changes.
- Create `tests/cli/test_resume_verb.py`: Add tests covering the RACs (e.g. schema_version mismatch, missing fields, empty checkpoint dir).

---

**INVEST Gate:**
- **Independent:** ✅ — resume can be built and tested independently of sweep
- **Negotiable:** ✅ — the --epochs semantics, manifest extensions, and on_resume wiring are all negotiable details
- **Valuable:** ✅ — enables recovering from crashes and extending training runs
- **Estimable:** ✅ — 5 refactoring steps + 2 implementation steps, scoped
- **Small:** ✅ — pure glue verb consuming existing infrastructure (checkpoint, manifest, Engine)
- **Testable:** ✅ — each RAC has a concrete Given/When/Then with observable behavior

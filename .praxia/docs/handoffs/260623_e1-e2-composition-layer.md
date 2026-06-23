# Handoff — E1 + E2 composition layer (#2174)

- **task_id:** `260623_e2-autocli` (prior: `260622_e1-signature-inference`)
- **session_id:** 10266de6-0cda-4054-a3e4-082cc52602da
- **date:** 2026-06-23 · **status:** in_progress (MVPs done+merged; lifecycle verbs deferred)
- **phase:** E2 composition layer (#2174 / epic #2605)

## Goal
Build the pure-JAX composition layer on xtrax via the disciplined arc: ground → contemplex brainstorm → spec → adversarial DAG → autonomous build → epic-audit → merge.

## Summary
Delivered, end-to-end and **all merged to `main`**:
- **E1-MVP** `xtrax.inference` — `infer_bundle` / `@axis_config` / `BundleSchema` / fail-loud `AxisRole` (UNKNOWN → `AmbiguousAxisError`) / `verify_structure`. Plus **#2561** hardening (reversed the `tiling→inference` dependency: `AxisRole`/`AmbiguousAxisError` now live in `xtrax.tiling.roles`; import-linter enforces `tiling ⊬ inference`) and the **using-xtrax skill** update + re-export.
- **E2-MVP** `xtrax.cli` — `xtrax plan | explain | export` via a `tyro.extras.subcommand_cli_from_dict` scaffold (REGISTRY dict; a new verb ≈ 1 line + a module), a **lazy import-path loader** (`CLIImportError`, applies lesson #145), a shapes-string parser, and the `--fmt` emit router with the **json machine-contract** (`{_meta, **PlanStatsDict}`). `cli` extra + `xtrax` console script.

Each epic was independently **epic-audited**: E1 → APPROVE; E2 → REQUEST_CHANGES that caught a real **silent html/png no-op** (+ a vacuous test), now remediated. Final main state: **341 tests passed, 4 skipped** (extra-dependent tests `importorskip` matplotlib/flatbuffers on minimal env), import-linter 2 kept/0 broken, `xtrax` script works.

**NOTHING is pushed to the remote** (no_autonomous_push invariant). The worktree branch `worktree-e1-signature-inference-epic` is fully merged to main (0 ahead).

## Next steps
1. **Push `main` to the remote** when ready — E1+E2 are local-only on main; needs explicit user go-ahead (no_autonomous_push).
2. **E2.2 — design + build the deferred verbs `run` / `sweep` / `resume`.** Each is major greenfield needing its own brainstorm → spec → DAG. Gating questions (from the E2 spec TBDs):
   - `run`: DataModule factory interface; RunSpec→Engine wiring; an `init_state(model, optimizer, seed)` helper.
   - `resume`: `RunManifest` schema (run-id, model import-path for the `state_template` rebuild, checkpoint dir); manifest location; run-id = config-hash vs UUID.
   - `sweep`: grid-over-RunSpec-scalars vs delegate to bathos campaign primitives.
3. Remove the merged worktree branch `worktree-e1-signature-inference-epic`.
4. Optional: E1 Tier-2 jaxtyping dim-name → AxisRole adapter (gated on a real aminx caller); update `docs/architecture.md` to document the new `inference` + `cli` layers.

## Immediately relevant (read first next session)
- `src/xtrax/cli/registry.py` — the REGISTRY dict; the scaffold the deferred verbs slot into (new verb ≈ 1 line + module).
- `src/xtrax/cli/explain.py` + `src/xtrax/cli/emit.py` — the `--fmt json` machine-contract + the eda extra-boundary pattern to mirror.
- `.praxia/docs/specs/260623_e2-auto-cli-for-xtrax-how-to-scope-and-d.md` — E2 spec (winner C-AMENDED); deferred verbs' gating questions in the TBD table.
- `.praxia/docs/plans/260623_e2-mvp-backlog-dag.md` — E2 DAG (7 plan-audit fixes folded); the build pattern to replay.
- `.praxia/loop_state.toml` — autonomous-loop state.

## Deferred
- **`run`/`sweep`/`resume` verbs** — full experiment lifecycle; explicitly deferred by the E2 brainstorm (roadmap note in `cli/__init__`). → E2.2, own brainstorm/spec/DAG each.
- **E1 Tier-2 jaxtyping dim-name adapter** — gated on a real caller; jaxtyping internals are private/version-fragile.

## Open questions
- `sweep`: grid-over-RunSpec vs bathos-campaign delegate? (gates the data model)
- `resume`: RunManifest as a filesystem artifact under `.xtrax/runs/<id>/` or a bathos run-record? run-id scheme?

## Lessons captured this session
- **#145** — circular imports hide behind pytest collection order; test imports in isolation + import-linter contract (the `tiling↔inference` cycle E1.6 surfaced).
- **default_ci robustness** — extra-dependent tests must `importorskip` their optional dep (matplotlib/flatbuffers), else they fail (not skip) on a minimal CI env. Surfaced at the E2 merge.
- **CLI verbs: lazy-resolve the user fn (string), eagerly import the verb run_fn (own code)** — and import tyro inside `main()` so `import xtrax`/`import xtrax.cli` stay tyro-free.

## Context
- Prior phase: `260622_e1-signature-inference` (E1-MVP + #2561). E2 consumes E1's `infer_bundle`.
- Epics: #2174 (composition layer parent), #2605 (E2 auto-CLI). E1 items #2504–2515, debt #2561 (closed).

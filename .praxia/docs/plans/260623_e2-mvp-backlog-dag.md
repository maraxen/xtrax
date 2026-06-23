# E2-MVP Backlog DAG — xtrax.cli (post adversarial plan-audit)

- **task_id:** `260623_e2-autocli`
- **date:** 2026-06-23
- **spec:** `.praxia/docs/specs/260623_e2-auto-cli-for-xtrax-how-to-scope-and-d.md` (AC1–AC8, winner C-AMENDED)
- **provenance:** staff DAG → adversarial plan-audit (Opus) verdict **NEEDS_WORK** with 7 fixes (all task-text; spine unchanged). This doc folds them in.

## Audit's load-bearing findings (empirically verified)
1. **import-linter / eda:** grimp **detects lazy (function-level) imports** — a lazy `from xtrax.eda import render` does NOT satisfy the "core must not import eda" contract. Correct resolution: `xtrax.cli` must **NOT** be added to that contract's `source_modules` (cli is *allowed* to import eda). The lazy import is purely runtime extra-isolation (avoid importing matplotlib on a non-`eda` install). Verified: adding a real `cli/emit.py` doing `from xtrax.eda import render` keeps the contract `KEPT`.
2. **`from xtrax.eda import render` itself needs no matplotlib** — `eda/__init__` is stdlib; `render()` lazily imports `viz`→matplotlib at CALL time. So the clean missing-`xtrax[eda]` error must wrap the **call site**, not the import.
3. **AC4 happy path needs a decorated fixture:** a bare fn → `synthesize_axes` UNKNOWN-role axes → `BatchPlanner.plan` **always raises `AmbiguousAxisError`** (E1 fail-loud). So `plan`/`explain` succeed only on `@axis_config`-decorated fns. The shapes-string `name` is **display-only** (synthesize_axes names axes `axis_<i>`, leading dim only).
4. Wiring CONFIRMED: `infer_bundle → (BundleSchema, list[AxisSpec])`; `BatchPlanner.plan(list[AxisSpec])`; `explain_plan(plan) → PlanStatsDict` (json-safe).

## Tasks (fixes folded; all new code under `src/xtrax/cli/`)

| id | size | deps | ACs | test-first | notes (post-audit) |
|----|------|------|-----|-----------|--------------------|
| **T0** cli-pkg-skeleton | S | — | AC2(part) | no | `cli/__init__.py` (exposes `main` via a **tyro-free wrapper**: `def main(): from xtrax.cli.entrypoint import main as _m; return _m()`) + `cli/errors.py` (`CLIError` base, `CLIImportError`, `ShapeParseError`). **FIX5:** `__init__` must be tyro-free at module load. |
| **T1** loader-lazy | M | T0 | AC3 | yes | `cli/loader.py` `load_fn('pkg:sym')` lazy via importlib at dispatch; `CLIImportError` + sys.path hint; no user import at module load (lesson #145). |
| **T2** shapes-parser | M | T0 | AC7 | yes | `cli/shapes.py` grammar `name=(d0,..)<dtype>` (f32/f64/i32/bool) → `ShapeDtypeStruct` via `build_abstract_inputs`; `ShapeParseError`. **FIX6:** docstring notes `name` is DISPLAY-ONLY (does not bind to @axis_config / planner names). |
| **T3** registry | S | T4, T5 | AC1(half) | no | `cli/registry.py` module-level `REGISTRY = {"plan":(PlanArgs,run_plan), "explain":(ExplainArgs,run_explain)}`. No `register_verb()`, no dynamic Union. **FIX7:** the verb `run_fn` is xtrax's OWN code (eager import); only the USER `--fn` is lazy-string-resolved — these are different and must not be conflated. |
| **T4** plan-verb | M | T1, T2 | AC4 | yes (fixture) | `cli/verbs/plan.py` PlanArgs(fn,shapes)+run_plan: load→parse→build_abstract_inputs→infer_bundle→`BatchPlanner().plan(axes)`→print; catch `AmbiguousAxisError`, render clean. **FIX3:** add a `@axis_config`-DECORATED fixture fn so the exit-0 happy path is reachable (bare fn always raises — that's correct E1 behavior). |
| **T5** explain+emit | L | T1, T2 | AC5, AC6 | yes | `cli/verbs/explain.py` + `cli/emit.py` --fmt router. ExplainArgs(fn,shapes,fmt=json). `explain_plan(plan)→PlanStatsDict`; json wraps `{_meta:{schema_version}, **stats}` (documented envelope). **FIX1+2:** html/png do `from xtrax.eda import render` (NOT a contract device — cli stays off the forbidden list); wrap the `render()` CALL in a clean missing-`xtrax[eda]` error (no traceback). |
| **T6** entrypoint | S | T3 | AC1(half), AC2(half) | no | `cli/entrypoint.py` `main()` = `tyro.extras.subcommand_cli_from_dict(REGISTRY)` + dispatch; **`import tyro` INSIDE `main()`** (FIX5); top-level `CLIError` handler → message + nonzero exit, no traceback. |
| **T7** packaging | S | T0 (meta), T6 (exercise) | AC2 | no | pyproject: `[project.optional-dependencies] cli = ["tyro>=0.9,<2"]` + `[project.scripts] xtrax = "xtrax.cli:main"`. **FIX1:** explicitly DO NOT add `xtrax.cli` to the importlinter EDA-forbidden `source_modules`; assert `lint-imports` stays green. |
| **T8** tests | M | T6, T7 | AC8 | (is tests) | `tests/cli/test_cli.py` (cli extra). **FIX4 — explicit tests:** `--help` lists `{plan,explain}` clean names; `explain --fmt json` round-trips → expected `PlanStatsDict` keys + `_meta` (via `json.loads(stdout)`); bad `--fn`→`CLIImportError`; **AC2 import-isolation**: `'tyro' not in sys.modules` after `import xtrax` AND after `import xtrax.cli`; **AC4 both paths**: bare fn → clean `AmbiguousAxisError`, decorated fixture → exit 0; missing-eda → clean error. |
| **Z** deferred-roadmap | S | T0 | — | no | roadmap NOTE only (docstring) listing run/sweep/resume/export = E2.2+ with gating questions (spec TBDs). NO stub code (pre-mortem rot-removal). |

## Waves & critical path
```
W0: T0
W1: T1 ∥ T2 ∥ Z       (need only T0)
W2: T4 ∥ T5           (need T1+T2)
W3: T3                (needs T4+T5 run_fns)
W4: T6                (needs T3)
W5: T7 ; then T8      (T8 needs T6+T7)
```
**Critical path:** T0 → T2 → T5(L) → T3 → T6 → T7 → T8. **Riskiest:** T5 (only L node, on the path, owns the `--fmt json` machine contract + the eda extra-boundary).

## Coverage
AC1 (T3+T6), AC2 (T7+T6, + isolation test in T8), AC3 (T1), AC4 (T4, both paths in T8), AC5 (T5), AC6 (T5), AC7 (T2), AC8 (T8). All covered; the two audit-flagged holes (AC2 isolation, AC4 message+happy) promoted to explicit T8 tests.

## Status
NEEDS_WORK → all 7 fixes folded (FIX1–7 tagged in the table); spine unchanged per auditor. Ready to load into backlog + build. Next: backlog DAG items → autonomous build (wave 0 = T0).

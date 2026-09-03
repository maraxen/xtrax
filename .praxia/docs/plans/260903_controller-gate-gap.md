---
category: plans
title: "Close the controller/ gate gap"
description: "Wire controller/ into ty and a coverage tier, install the controller extra in CI so the real-bathos surface runs, then fix the three defects that exposes"
task_id: 260903_controller-gate-gap
status: draft
---

# Sprint: close the `controller/` gate gap

## The finding

The loop-controller epic shipped **4,456 lines of source and 8,547 lines of tests
(251 test functions) into a top-level `controller/` tree, outside `src/`.** Every
quality gate in this repo is scoped to `src/xtrax`, so almost none of them see it.

| Gate | Scope | Sees `controller/`? |
|---|---|---|
| `ruff check .` | whole tree | yes |
| `ty check src/` | `[tool.ty.src] include = ["src/**/*.py"]` | **no** |
| coverage tier1 (90 line / 80 branch) | `coverage_packages = ["xtrax"]` | **no** |
| `controller` extra (`bathos>=0.13.0a1`) | — | **installed by no CI job and no Justfile recipe** |

The tests do *run* — tier1's `pytest tests/` picks up `tests/controller/` — but
nothing measures or type-checks the code under them, and every test that touches
real bathos skips.

### Measured, not inferred

- `bathos` is absent under tier1's `dev,io` extras.
  `tests/controller/test_bathos_library_wrappers_integration.py` reports
  `1 skipped` — a **collection-level** `importorskip`, so its 8 tests are never
  even collected. `test_loop_run.py:1021` skips on `bathos.capability` too.
- `ty check controller/` prints `WARN No python files found` then
  `All checks passed!` — the config's `include` filters the tree out before the
  CLI path applies. **That green is vacuous.**
- With the include widened, the hole is small and specific: **15 diagnostics**,
  of which 10 are `bathos.*` unresolved (the missing extra), 4 are eda deps
  unrelated to the controller, and **1 is a genuine type error**.
- Installing the extra takes `tests/controller/` from **259 passed / 2 skipped**
  to **268 passed / 0 skipped**. Nothing has rotted — the dark tests pass against
  real bathos `0.13.0a1`. This is the sprint's key de-risking result.
- `controller/` coverage with the extra installed: **line 89.52%, branch 77.03%**
  (816 stmts, 67 missed; 148 branches, 114 covered). Against tier1's bar that is
  0.48 and 2.97 points short.

### The defects this already exposes

1. **`controller/main_loop.py:940`** passes `higher_is_better` as
   `Mapping[str, bool] | None` into `compute_ratchet_decision`
   (`src/xtrax/loop/multi_metric_ratchet.py:148`), which requires
   `Mapping[str, bool]`. In the ratchet that decides whether a candidate wins.
2. **#4584** — `run_one_candidate_pass`'s accept path checks `.success` but never
   `run_result.exit_code`, so a failed candidate can be committed as best-so-far.
3. **#4801** — gw03's real-catalog evidence-gate integration test was never ported
   to main's `run_one_candidate_pass` signature. Only now testable, because the
   real-bathos surface finally runs.

## Phase A — wire the gates

Branch `feat/controller-gate-wiring`. Land the gates **green**, before any fix, so
each later fix has a gate that would have caught it.

**A1 — `tier4_controller` in `distribution/coverage_dag.toml`.** Mirrors the four
existing tiers:

```toml
[[tiers]]
id = "tier4_controller"
description = "Loop controller (unshipped tree; real-bathos surface)"
measure_coverage = true
uv_sync_extras = ["dev", "io", "controller"]
pytest_args = ["tests/controller/", "-q"]
coverage_packages = ["controller"]
coverage_omit = ["*/tests/*", "*/__init__.py"]
target_line_pct = 90.0
target_branch_pct = 80.0
enforce_line_pct = 90.0
enforce_branch_pct = 80.0
```

`tests/distribution/test_coverage_dag.py:42` asserts the exact tier-id list and
will fail until updated — that assertion is doing its job, not obstructing.

**A2 — `audit-coverage-tier4` recipe** in the Justfile, matching tier1/tier2's shape.

**A3 — a `controller-tests` CI job, always-on, not path-filtered.** `ci.yml`
already computes a `controller-changes` filter (line 28) but currently feeds it
only to `wheel-smoke`. The temptation is to reuse it here, as `tier3_port` does.
**Don't** — defect 1 above crosses the boundary: it lives in `controller/` but its
cause is a signature in `src/xtrax/loop/`. A filter on `controller/**` would have
sat out the very commit that introduced it. The 62-package install is the price of
a gate that actually holds.

**A4 — ty over `controller/`, scoped to the new job.** Two shapes considered:

- Widen `[tool.ty.src] include` and add `--extra controller` to
  `lint-format-type-test` — one ty invocation, but pays the 62-package install in
  a second job.
- Keep the global include at `src/`, and have the controller job run
  `ty check -c 'src.include=["controller/**/*.py"]' controller/`.

Take the second. It keeps the existing job's environment untouched and puts
everything controller-shaped in one place. Cost: the override lives in the recipe
rather than in `pyproject.toml`, so it needs a comment saying why.

There is a **fourth** ty site, easy to miss: `scripts/git-hooks/pre-push` runs its
own `ty check src/`, ratcheted against `scripts/git-hooks/ty-baseline.txt` (empty
today, so `src/` is clean), and syncs `dev,eda,io` — not `controller`. Decide
explicitly whether the hook grows to cover `controller/` or stays CI-parity-only.
Staying is defensible: the hook's stated job is to mirror `lint-format-type-test`,
and making every local push pay a 62-package sync is a real cost. If it stays,
say so in `scripts/git-hooks/README.md`, so the next person does not read the
hook's green as covering the controller.

**A5 — cap the bathos pin.** `controller = ["bathos>=0.13.0a1"]` has no upper
bound. Phase A makes CI depend on it, so an alpha-to-alpha bathos release could
turn the board red for reasons unrelated to any xtrax change. This repo already
treats unbounded pins as debt (#3641 is exactly that argument for jax). Add a cap.

**Gate for Phase A:** `just audit-coverage-tier4` green at 90/80 *is the phase's
exit criterion*, and today's measurement is 89.52/77.03 — so A includes whatever
tests close that ~0.5/3.0-point gap. The uncovered regions are named and small:
`bathos_library_wrappers.py` 199-202, 246-269; `praxia_dispatch_backend.py` 183,
187-189, 204-217, 280-281, 285-289, 312-313, 335-336; `bathos_campaign_adapter.py`
230, 269, 273-275, 350-351, 355-356, 372-373, 376-377, 463-464, 472-482.

## Phase B — fix what the gates expose

Branch `fix/controller-defects`, after A merges.

**B1 — the ratchet type error.** Establish what `higher_is_better=None` is *meant*
to mean at the call site before changing either side: a default (all-metrics-higher),
a guard, or a widened signature on `compute_ratchet_decision`. Do not silence it
with a cast — the None arrives from somewhere, and that path decides candidate
acceptance.

**B2 — #4584.** Check `run_result.exit_code` alongside `.success` on the accept
path. The test must fail first against the current code, on a candidate whose
`success` is truthy while `exit_code` is non-zero.

**B3 — #4801.** Port gw03's real-catalog evidence-gate integration test onto
main's `run_one_candidate_pass` signature, now that real bathos is installed in the
gate that runs it.

Each fix lands with a test that fails before it and passes after, and each is
covered by the Phase-A gate — that pairing is the point of the ordering.

## Phase C — housekeeping

Branch `chore/backlog-and-assets`. Independent of A and B; can run in parallel.

**C1 — backlog stale-row sweep.** 21 rows are open. **#3642 (import-linter
contracts) is verifiably done**: four contracts exist in `pyproject.toml`,
including "Base modules must not import xtrax.export", and `audit-imports` is
wired into `audit-deterministic`. **#4581 and #4582 are literal duplicates** —
same title, filed two minutes apart. Verify each remaining row against the code
and close what has already shipped.

**C2 — agent-asset refresh.** All three `agent_assets/skills/*/SKILL.md` are
stamped `xtrax_version: 0.4.0a7` (now a8), and `xtrax.export` — a subpackage
shipped across four PRs — appears in none of them; `using-xtrax` mentions only the
pre-existing `xtrax export` CLI verb. Deliberately deferred last sprint as needing
a real review rather than a mechanical bump: `using-xtrax`'s description also still
reads "xtrax v0.4.0a5 + main", contradicting its own frontmatter. Resolve the
contradiction, document the subpackage, then bump.

**C3 — release-report drift.** Commit the regenerated `READY`
`.praxia/release_readiness_report.{json,md}` so main stops carrying the pre-tag
`BLOCKED_MANUAL` snapshot.

## Verification

Per phase, before any push:

```bash
uv run --extra dev --extra io --extra controller pytest tests/controller/ -q
just audit-coverage-tier4        # >=90% line / >=80% branch
just audit-coverage-tier1        # unchanged by this sprint; prove it
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
just audit-deterministic         # stops at first failure; expect exit 0
```

`uv run` extras are not sticky across subprocess invocations — a bare `uv run
pytest` resyncs the shared venv and silently drops deps mid-run, so every call
carries its `--extra` flags explicitly. A concurrent bare `uv run` during a gate
run produces spurious failures; run gates solo.

Then CI: all jobs, with the new `controller-tests` job reporting **0 skips**. A
skip there means bathos silently failed to install and the real-toolchain claim is
void — the same failure mode `export-toolchain-tests` guards against.

## Out of scope

- #4856 (WebGPU research) — deliberately deferred; still open by design.
- Profiling trio #4586 / #4618 / #4620.
- #3641 (jax `<0.12`), beyond citing it as precedent for A5's bathos cap.
- Moving `controller/` under `src/`. It is correctly excluded from the wheel
  (`wheel-smoke` asserts this) and this sprint does not relitigate that.
- The four stale remote branches (`probe/webgpu-ci-adapter`,
  `feat/profiling-stage2-evidence`, `gw03-campaign-approval-gate`,
  `spike/iree-wasm-export`) — deletion is blocked for me by the permission
  classifier.

---
synthesis_id: 260619_synthesis_1456_coverage_dag
synthesis_version: "1.0"
backlog_id: 1456
prepared_for: xtrax Loop Sprint 32 (coverage DAG manifest phase 1)
task_id: 260617_xtrax-composition-mission
---

# Research Synthesis: Coverage DAG manifest + baseline reporter (#1456 phase 1)

**Prepared:** 2026-06-19  
**Winner:** Hybrid E→A — manifest + per-tier reporter now; CI `--cov-fail-under` enforcement deferred until tier1 is green.

---

## Executive summary

Coverage percentages from **subset pytest runs are misleading**. The `audit-deterministic` job runs only `tests/audit/` with `--cov=xtrax`; pytest-cov footers there show **~2–10% line coverage** — an artifact of measuring a tiny test surface against the full `xtrax` package, not a product health signal.

Sprint 32 delivers a **tiered coverage DAG** (`distribution/coverage_dag.toml`), a baseline reporter (`scripts/audit_coverage_dag.py`), and wiring into `just audit-deterministic` as a **non-blocking report**. CI `--cov-fail-under=90` on the monolithic job is **unchanged** this sprint.

---

## Measured baselines (2026-06-19)

| Tier | Scope | Line % | Branch % | Notes |
|------|-------|--------|----------|-------|
| **tier0_audit** | `tests/audit/` | — | — | `measure_coverage=false`; not a coverage surface |
| **tier1_core** | `tests/` minus `eda` + `port/tests`, `dev` extra | **78.5** | **66.9** | Blocking target for future gate |
| **tier2_eda** | `tests/eda/`, `dev`+`eda` extras | (subset of full) | (subset) | Optional-extra track |
| **tier3_port** | `port/tests/` | — | — | `measure_coverage=false`; path-filtered CI |
| **full+eda** | All core + eda (reference) | **87.9** | **78.1** | Monolithic reference, not a DAG tier |

**Misleading reading:** audit-deterministic pytest-cov footer (~2–10%) reflects tier0 subset only — cite tier1_core or full+eda baselines instead.

---

## Blockers

- **11 failing core tests** (tier1_core, not eda-only) block honest enforcement.
- Branch coverage materially trails line (66.9% tier1 vs 78.5% line).
- Until failures are fixed, any `--cov-fail-under` gate on tier1 would red-bar CI dishonestly.

---

## Tier contract (manifest)

| Tier | `measure_coverage` | Future enforce floors | Target |
|------|-------------------|----------------------|--------|
| tier0_audit | false | — | smoke / contracts |
| tier1_core | true | 85% line / 65% branch | 90% / 75% |
| tier2_eda | true | TBD | track separately |
| tier3_port | false | — | port CI path filter |

State is written to `.praxia/coverage_last_measured.json` on each `just audit-coverage-dag` run.

---

## Ratchet plan (phase 2 — next sprint)

1. Fix 11 tier1_core test failures.
2. Wire CI `--cov-fail-under` on **tier1_core only** with initial floors **85% line / 65% branch** (`enforce_*` in manifest).
3. Ratchet +2% per sprint toward target **90% / 75%**.
4. Promote `just audit-coverage-dag --enforce tier1_core` from report-only to blocking in `audit-deterministic` once green.
5. Child backlog: **#1456a** CI enforcement vs **#1456b** manifest (this sprint).

---

## Artifacts delivered (sprint 32)

- `distribution/coverage_dag.toml` — tier definitions, targets, deferred enforce floors
- `scripts/audit_coverage_dag.py` — measure, report table, write state; `--enforce` optional
- `tests/distribution/test_coverage_dag.py` — config load, enforce logic, mocked pytest/json
- `just audit-coverage-dag` — report-only, tier1_core default; `--all-tiers` for full DAG
- `audit-deterministic` — runs coverage DAG report **after** `tests/audit/` (exit 0)

**Explicitly deferred:** `.github/workflows/ci.yml` `--cov-fail-under=90` unchanged.

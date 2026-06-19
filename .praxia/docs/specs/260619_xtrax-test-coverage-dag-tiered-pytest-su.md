---
session_id: cbcb584a
topic: xtrax test coverage DAG: tiered pytest subsets, omit rules, CI gates, and path from current ~88% line (full+eda) / ~78% branch to a sustainable 90% floor without red-bar day-one CI
task_type: architectural
winner: Hybrid E→A: coverage DAG manifest + audit_coverage_dag.py measuring per-tier baselines this sprint; defer CI cov-fail-under until 11 test failures fixed; next sprint wire tier1 with 85/65 initial floors ratcheting to 90/75
created_at: 2026-06-19T12:34:57.741952+00:00
---

# Brainstorm: xtrax test coverage DAG: tiered pytest subsets, omit rules, CI gates, and path from current ~88% line (full+eda) / ~78% branch to a sustainable 90% floor without red-bar day-one CI

## Problem Frame
Fixed constraints:
- Coverage source of truth is pytest-cov over `xtrax` package; devtools omitted per pyproject
- CI must use fresh coverage (no committed .coverage) — #1451 gate exists
- audit-deterministic stays subset tests/audit only — not a coverage tier
- Python 3.13 only; uv + pytest; existing 824+ tests
- 11 currently failing tests in core (not eda-only) block any honest gate
- Branch coverage materially lower than line (66.9% dev-only / 78.1% full)

Negotiable:
- Tier count and naming (tier-0 smoke, tier-1 core, tier-2 eda, tier-3 port)
- Whether CI gate is line-only first then branch ratchet
- Initial floor (e.g. 85% line / 65% branch) vs target 90/80
- eda tests: optional job vs marker vs --ignore until extras installed
- Whether #1456 delivers DAG + ratchet gate script only, deferring CI --cov-fail-under change to child item
- port/ tests separate track (already path-filtered in CI)

Frame: We are designing a measurement and enforcement DAG so coverage numbers are not misleading (subset runs showing 10%) and CI does not red-bar on day one while still converging to distribution spec 90% floor.

## Idea Pool
- [ai] Tiered DAG with separate pytest paths per tier: tier0=tests/audit (no coverage gate), tier1=tests minus eda minus port (core product), tier2=tests/eda with eda extra, tier3=port/tests on path filter. Each tier has own cov floor ratchet in distribution/coverage_dag.toml. CI main job runs tier1 only.
- [ai] Tier1-first gate: fix 11 failing tests, set --cov-fail-under=85 line on tier1, branch floor 65%, ratchet +2% per sprint until 90/75. Defer eda to optional workflow.
- [ai] Single monolithic suite: uv sync --all-extras, fix all failures, one 90% gate. Simple but slow CI and couples eda/port to core releases.
- [ai] Coverage budget by package: per-subpackage floors in TOML (training, engine, tiling...), pytest-cov --cov-fail-under computed from weighted sum. Granular but high maintenance.
- [ai] Marker-based tiers: @pytest.mark.tier1 / tier2, pytest -m tier1 in CI. Requires retrofitting marks on 800+ tests.
- [ai] DAG as documentation only this sprint: write synthesis + TOML manifest + audit_coverage_dag.py validates structure and prints measured % per tier without changing CI. #1456 becomes manifest; child backlog for CI wire.
- [user] Competing approaches:
- [user] A) **Tiered path DAG** — separate pytest invocations per directory/markers: audit (no cov), core (tests/ minus eda/port), eda (optional extra), port (path-filtered). Each tier has floors in TOML; CI blocking = core tier only.
- [user] B) **Monolithic suite** — one pytest run with all extras, fix 11 failures, single 90% gate. Simple mental model, slow/flaky CI, eda matplotlib coupling.
- [user] C) **Per-package budgets** — subpackage-weighted floors. Precise but heavy upkeep on 14 subpackages.
- [user] D) **Marker retrofit** — pytest -m tier1 across 800 tests. Clean long-term, expensive migration now.
- [user] E) **DAG manifest sprint (recommended)** — this iteration delivers measured baseline per tier + TOML DAG + audit script; defer CI --cov-fail-under change until tier1 green. Split #1456 into manifest (now) vs CI enforcement (next).
- [user] My lean: E now, then A enforcement after fixing 11 test failures.
- [user] Converge on hybrid E→A: iteration 32 delivers coverage DAG TOML, synthesis doc, audit script reporting per-tier line/branch %; explicitly defer CI --cov-fail-under change. Tier1 = pytest tests/ --ignore tests/eda --ignore port/tests with dev extra. Record blockers: 11 failing tests must be fixed before any enforcement.

## Decision Log
- [REJECT] Monolithic suite with immediate 90% gate: Dominated: 11 failing tests + eda import coupling + branch gap makes day-one 90% CI red; violates pre-mortem mitigation.
- [REJECT] Per-package budgets: High maintenance for current team size; doesn't solve misleading subset coverage readings.
- [DEFER] Marker retrofit across 800 tests: Defer: good long-term but not sprint-sized; path-based tiers achieve 80% of benefit now.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Pre-mortem: Six months later the DAG TOML exists but nobody runs audit_coverage_dag.py in CI; measured baselines stale; developers still cite 10% from audit-deterministic footers; #1456 child never created so CI still has broken --cov-fail-under=90 on monolithic job. Mitigation: wire audit-coverage-dag into audit-deterministic as non-blocking report first, then blocking on tier1 once failures fixed; store last_measured.json in .praxia/; backlog child #1456a enforcement vs #1456b manifest.
**AI:** _not recorded_

## Acceptance Criteria
**Given** Fixed constraints:
- Coverage source of truth is pytest-cov over `xtrax` package; devtools omitted per pyproject
- CI must use fresh coverage (no committed .coverage) — #1451 gate exists
- audit-deterministic stays subset tests/audit only — not a coverage tier
- Python 3.13 only; uv + pytest; existing 824+ tests
- 11 currently failing tests in core (not eda-only) block any honest gate
- Branch coverage materially lower than line (66.9% dev-only / 78.1% full)

Negotiable:
- Tier count and naming (tier-0 smoke, tier-1 core, tier-2 eda, tier-3 port)
- Whether CI gate is line-only first then branch ratchet
- Initial floor (e.g. 85% line / 65% branch) vs target 90/80
- eda tests: optional job vs marker vs --ignore until extras installed
- Whether #1456 delivers DAG + ratchet gate script only, deferring CI --cov-fail-under change to child item
- port/ tests separate track (already path-filtered in CI)

Frame: We are designing a measurement and enforcement DAG so coverage numbers are not misleading (subset runs showing 10%) and CI does not red-bar on day one while still converging to distribution spec 90% floor.
**When** implementing Hybrid E→A: coverage DAG manifest + audit_coverage_dag.py measuring per-tier baselines this sprint; defer CI cov-fail-under until 11 test failures fixed; next sprint wire tier1 with 85/65 initial floors ratcheting to 90/75
**Then**
  - [ ] _add specific measurable criteria_

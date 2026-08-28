# Praxia session handoff: xtrax CSE runtime optimization

- **Date:** 2026-08-26
- **Workspace:** `xtrax-cse-impl`
- **Branch:** `cse-runtime-opt`
- **Status:** in progress
- **Task ID:** `260825_xtrax_cse_runtime_opt`
- **Session ID:** `853f6d66-4b75-4cb4-9f79-4b7c55ad0c27`

## Summary
Implemented P0 jaxpr duplicate detection and CLI reporting, plus P1 content-keyed memoization. Consolidated memo exception classes into canonical definitions in `src/xtrax/inference/errors.py`. Targeted inference and CLI tests pass: **166 passed**. The design is not complete: adversarial review found 2 blockers and 7 major objections, led by Component C dedup synthesis/collision semantics and unpinned memo hash/device/staleness assumptions.

## Verified commits
- `c42c1ed` `feat(inference): analyze_cse jaxpr duplicate-detection reporter`
- `0c14f9a` `feat(cli): explain --report cse branch (P0 wiring)`
- `93a2399` `feat(inference): memoize_jaxpr content-keyed value cache (P1, spec 260825 §4.2)`
- `9e5119a` `feat(inference): export memo symbols from package __init__`
- `99d49a2` `refactor(inference): canonical error homes in errors.py, re-exported via memo/__init__`

## Validation
Command:
`PYTHONPATH=src /home/marielle/projects/xtrax/.venv/bin/python -m pytest tests/inference tests/cli/test_explain_cse.py tests/audit/test_no_future_annotations.py -q --no-cov`

Result: `166 passed in 6.52s`.

## Critical research and review
Read:
- `.praxia/docs/research/260825_spec-challenger-r1.md`
- `.praxia/docs/research/260825_jax-cse-ecosystem.md`

Highest-priority objections:
1. **OBJ-R1-01 blocker:** sampled `DedupSpec` construction cannot provide the required full `(N,)` `index_map`; exact construction touches all rows and defeats the stated rationale, while singleton fallback under-fires.
2. **OBJ-R1-02 blocker:** collision policy requires existing caller specs, but `synthesize_dedup_spec(...)` has no existing-spec parameter or merge helper, so AC11 is unimplementable.
3. **OBJ-R1-04:** claimed memo hit regime contradicts subprocess/path-reloaded gate architecture. Surviving regime is unchanged-candidate retries.
4. **OBJ-R1-05:** empirically pin whether `hash(ClosedJaxpr)` is sensitive to closure constants before relying on salt/spot-check staleness design.
5. **OBJ-R1-06:** N5 single-device scope is not mechanically enforced and stamp lacks sharding/device-instance provenance.
6. **OBJ-R1-07:** decorator path lacks abstract inputs for D3 cost warning and stats lack wall-time/hash-cost telemetry.
7. **OBJ-R1-08:** CSE equivalence claim requires fixpoint operand rewriting, not only raw jaxpr hashes.
8. **OBJ-R1-09:** revise impurity-screen rationale and AC3 wording because `make_jaxpr` executes Python under tracers at wrap time.

Research headline: no public jaxpr-level CSE API exists in JAX; value memoization is external prior art; `unique(size=...)` and BCOO provide static-shape dedup precedents; key hazards include donation, async futures, weak_type, GPU nondeterminism, and sharding provenance.

## Current dirty tree
`git status --short` showed:
- modified `.praxia/audit_bootstrap_manifest.toml`
- modified `.praxia/release_readiness_report.json`
- modified `.praxia/release_readiness_report.md`
- untracked `tests/cli/demo.py`
- untracked `tests/demo.py`

These appear to be generated/audit or demo artifacts and were **not committed**. Review intentionally before proceeding. Do not assume the tree is clean.

## Current release gate snapshot
`.praxia/release_readiness_report.md` says `BLOCKED_AUTOMATED`. Current reported failures include ruff lint/format, ty, no-future-annotations, coverage/contract gates, and added-types-diff. Some are likely caused by the new implementation/tests; other blockers include existing distribution backlog and publish workflow markers. Do not claim overall release readiness.

## Next steps
1. Read the spec and challenge memo, then revise the spec beginning with OBJ-R1-01 and OBJ-R1-02.
2. Inspect/fix `src/xtrax/inference/memo.py`, `src/xtrax/inference/cse.py`, `src/xtrax/cli/explain.py`, and their tests for lint, format, type, annotation, and added-types-diff failures.
3. Decide whether Component C should be redesigned around exact full-batch dedup or deferred until static-shape and collision APIs are explicit.
4. Test ClosedJaxpr hash sensitivity to closure constants under the pinned JAX version.
5. Define and enforce single-device/sharding, donation, async-readiness, spot-check, and telemetry contracts.
6. Review uncommitted audit/demo artifacts and commit only intentional changes.

## Handoff tool failure
Attempted `mcp__praxia__handoff(action="create", workspace="xtrax-cse-impl", ...)`; it failed before writing because Praxia workspace initialization failed during SQLx migrations: `MCP error -32603: workspace init failed: Migration failed: Failed to run sqlx migrations`. This filesystem handoff is the fallback. Retry the MCP handoff after the Praxia database/migration issue is repaired.

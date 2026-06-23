# xtrax Release Readiness Report

- **Epic:** #1451 xtrax distribution readiness (N0-N10)
- **Generated:** 2026-06-19T21:45:55.841702+00:00
- **Verdict:** `BLOCKED_MANUAL`
- **Package version:** `0.3.0`

## Blockers
- human gate open: #1454 n9_human_oidc

## Distribution backlog (N0-N10)

| ID | Slug | Status | Gate | Blocking |
|----|------|--------|------|----------|
| 1451 | n0_coverage_hygiene | completed | PASS | yes |
| 1452 | n1_version_wheel | completed | PASS | yes |
| 1453 | n3_public_api | completed | PASS | yes |
| 1455 | n2_packaging_metadata | completed | PASS | yes |
| 1457 | n4a_docs_plumbing | completed | PASS | yes |
| 1458 | n4b_narrative_docs | completed | PASS | yes |
| 1459 | n5_output_sink_docs | completed | PASS | yes |
| 1460 | n8_project_hygiene | completed | PASS | yes |
| 1461 | n7_publish_oidc | completed | PASS | yes |
| 1454 | n9_human_oidc | open | MANUAL | yes |
| 1462 | n10_release_readiness | in_progress | META | yes |

## Automated checks

- **audit-coverage-hygiene** (backlog): PASS
- **audit-version-wheel** (backlog): PASS
- **audit-public-api** (backlog): PASS
- **audit-packaging-metadata** (backlog): PASS
- **audit-docs-build** (backlog): PASS
- **audit-narrative-docs** (backlog): PASS
- **audit-output-sink-docs** (backlog): PASS
- **audit-project-hygiene** (backlog): PASS
- **audit-publish-oidc** (backlog): PASS
- **import_cycles** (foundation): PASS
- **no_future_annotations** (foundation): PASS
- **jaxlint_performance** (foundation): PASS
- **ruff_lint** (ci_lint): PASS
- **ruff_format** (ci_lint): PASS
- **ty_check** (ci_lint): PASS
- **coverage_tier1** (coverage): PASS
- **coverage_tier2** (coverage): PASS
- **io_reexport_doctest** (docs): PASS
- **audit_contracts** (deterministic_track): PASS
- **added_types_diff** (type_hardening): PASS

## Coverage state

- `tier1_core`: line 94.9% / branch 88.9%

## Release policy

- Do **not** push release tags until this report is `READY`.
- Complete human gate #1454 (PyPI + TestPyPI Trusted Publisher) first.
- Re-run: `just audit-release-readiness`

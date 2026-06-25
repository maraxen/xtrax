# xtrax Release Readiness Report

- **Epic:** #1451 xtrax distribution readiness (N0-N10)
- **Generated:** 2026-06-25T16:37:57.072101+00:00
- **Verdict:** `BLOCKED_AUTOMATED`
- **Package version:** `0.3.0`

## Blockers
- automated check failed: coverage_tier1
- automated check failed: coverage_tier2
- automated check failed: audit_contracts
- automated check failed: added_types_diff
- human gate open: #1454 n9_human_oidc
- backlog gate failed: #1457 n4a_docs_plumbing

## Distribution backlog (N0-N10)

| ID | Slug | Status | Gate | Blocking |
|----|------|--------|------|----------|
| 1451 | n0_coverage_hygiene | completed | PASS | yes |
| 1452 | n1_version_wheel | completed | PASS | yes |
| 1453 | n3_public_api | completed | PASS | yes |
| 1455 | n2_packaging_metadata | completed | PASS | yes |
| 1457 | n4a_docs_plumbing | failed | FAIL | yes |
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
- **coverage_tier1** (coverage): FAIL
  ```
  skipped in --quick mode
  ```
- **coverage_tier2** (coverage): FAIL
  ```
  skipped in --quick mode
  ```
- **io_reexport_doctest** (docs): PASS
- **audit_contracts** (deterministic_track): FAIL
  ```
  skipped in --quick mode
  ```
- **added_types_diff** (type_hardening): FAIL
  ```
  [32m============================== [32m[1m10 passed[0m[32m in 1.77s[0m[32m ==============================[0m
merge-base=55bf118e1489080e4838bb6e80073864f1ecec2e files_checked=30 callables_checked=31

uv run ruff check src/xtrax/devtools/gates/added_types_diff.py scripts/audit_added_types_diff.py tests/audit/test_added_types_diff.py
uv run pytest tests/audit/test_added_types_diff.py -v
uv run python scripts/audit_added_types_diff.py --no-emit
FAIL: added-types diff gate
  - src/xtrax/inference/abstract.py: unable to locate callable `is_leaf`
  - src/xtrax/inference/config.py: unable to locate callable `decorator`
  - src/xtrax/training/state.py: init_state: parameter `model` missing annotation
  - src/xtrax/training/state.py: init_state: parameter `optimizer` missing annotation
error: Recipe `audit-added-types-diff` failed on line 49 with exit code 1
  ```

## Coverage state

- `tier1_core`: line 94.9% / branch 88.9%

## Release policy

- Do **not** push release tags until this report is `READY`.
- Complete human gate #1454 (PyPI + TestPyPI Trusted Publisher) first.
- Re-run: `just audit-release-readiness`

# xtrax Release Readiness Report

- **Epic:** #1451 xtrax distribution readiness (N0-N10)
- **Generated:** 2026-07-02T17:27:22.235602+00:00
- **Verdict:** `BLOCKED_AUTOMATED`
- **Package version:** `0.3.0`

## Blockers
- .github/workflows/publish.yml missing marker: 'publish-testpypi'
- .github/workflows/publish.yml missing marker: 'test.pypi.org/legacy'
- automated check failed: audit-publish-oidc
- automated check failed: coverage_tier1
- automated check failed: coverage_tier2
- automated check failed: audit_contracts
- human gate open: #1454 n9_human_oidc
- backlog gate failed: #1457 n4a_docs_plumbing
- backlog gate failed: #1461 n7_publish_oidc

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
| 1461 | n7_publish_oidc | failed | FAIL | yes |
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
- **audit-publish-oidc** (backlog): FAIL
  ```
  [31m========================= [31m[1m2 failed[0m, [32m4 passed[0m[31m in 0.10s[0m[31m ==========================[0m

uv run ruff check scripts/audit_publish_oidc.py tests/distribution/test_publish_oidc.py
uv run pytest tests/distribution/test_publish_oidc.py -v
/home/marielle/projects/xtrax/.venv/lib/python3.13/site-packages/coverage/inorout.py:561: CoverageWarning: Module xtrax was never imported. (module-not-imported); see https://coverage.readthedocs.io/en/7.14.1/messages.html#warning-module-not-imported
  self.warn(f"Module {pkg} was never imported.", slug="module-not-imported")
/home/marielle/projects/xtrax/.venv/lib/python3.13/site-packages/coverage/control.py:958: CoverageWarning: No data was collected. (no-data-collected); see https://coverage.readthedocs.io/en/7.14.1/messages.html#warning-no-data-collected
  self._warn("No data was collected.", slug="no-data-collected")
/home/marielle/projects/xtrax/.venv/lib/python3.13/site-packages/pytest_cov/plugin.py:366: CovReportWarning: Failed to generate report: No data to report.

  warnings.warn(CovReportWarning(message), stacklevel=1)
error: Recipe `audit-publish-oidc` failed on line 171 with exit code 1
  ```
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
- **added_types_diff** (type_hardening): PASS

## Coverage state

- `tier1_core`: line 93.8% / branch 87.9%

## Release policy

- Do **not** push release tags until this report is `READY`.
- Complete human gate #1454 (PyPI + TestPyPI Trusted Publisher) first.
- Re-run: `just audit-release-readiness`

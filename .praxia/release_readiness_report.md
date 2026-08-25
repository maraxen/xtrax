# xtrax Release Readiness Report

- **Epic:** #1451 xtrax distribution readiness (N0-N10)
- **Generated:** 2026-08-25T20:41:08.550750+00:00
- **Verdict:** `BLOCKED_AUTOMATED`
- **Package version:** `0.4.0a6`

## Blockers
- .github/workflows/publish.yml missing marker: 'publish-testpypi'
- .github/workflows/publish.yml missing marker: 'test.pypi.org/legacy'
- automated check failed: ruff_format
- automated check failed: ty_check
- automated check failed: coverage_tier1
- automated check failed: coverage_tier2
- automated check failed: audit_contracts
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
| 1454 | n9_human_oidc | completed | MANUAL | yes |
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
- **substrate_lock** (foundation): PASS
- **jax_pin** (foundation): PASS
- **ruff_lint** (ci_lint): PASS
- **ruff_format** (ci_lint): FAIL
  ```
  Would reformat: tests/cli/test_run_provenance_store.py
1 file would be reformatted, 483 files already formatted
  ```
- **ty_check** (ci_lint): FAIL
  ```
      |

error[unresolved-attribute]: Attribute `finalize` is not defined on `None` in union `ZarrStagingSink | None`
   --> src/xtrax/cli/run.py:120:5
    |
120 |     sink.finalize()
    |     ^^^^^^^^^^^^^
    |

Found 3 diagnostics

Installed 4 packages in 41ms
  ```
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

- `tier1_core`: line 93.5% / branch 88.0%

## Release policy

- Do **not** push release tags until this report is `READY`.
- Complete human gate #1454 (PyPI + TestPyPI Trusted Publisher) first.
- Re-run: `just audit-release-readiness`

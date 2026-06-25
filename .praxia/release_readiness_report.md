# xtrax Release Readiness Report

- **Epic:** #1451 xtrax distribution readiness (N0-N10)
- **Generated:** 2026-06-25T16:11:15.563759+00:00
- **Verdict:** `BLOCKED_AUTOMATED`
- **Package version:** `0.3.0`

## Blockers
- automated check failed: ruff_lint
- automated check failed: ruff_format
- automated check failed: ty_check
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
- **ruff_lint** (ci_lint): FAIL
  ```
  [1m[94m13 |[0m [1m[91m|[0m import pytest
[1m[94m14 |[0m [1m[91m|[0m
[1m[94m15 |[0m [1m[91m|[0m from xtrax.tiling.roles import AmbiguousAxisError, AxisRole
[1m[94m16 |[0m [1m[91m|[0m from xtrax.tiling.plan import AxisSpec, BatchPlan, BatchPlanner
   [1m[94m|[0m [1m[91m|_______________________________________________________________^[0m
[1m[94m17 |[0m
[1m[94m18 |[0m   # ---------------------------------------------------------------------------
   [1m[94m|[0m
[1m[96mhelp[0m: [1mOrganize imports[0m

Found 17 errors.
[[36m*[0m] 13 fixable with the `--fix` option.
  ```
- **ruff_format** (ci_lint): FAIL
  ```
  Would reformat: [1mtests/cli/test_loader.py[0m
Would reformat: [1mtests/cli/test_manifest.py[0m
Would reformat: [1mtests/cli/test_resume_verb.py[0m
Would reformat: [1mtests/cli/test_run_verb.py[0m
Would reformat: [1mtests/inference/test_api.py[0m
Would reformat: [1mtests/inference/test_axes.py[0m
Would reformat: [1mtests/inference/test_axis_config.py[0m
Would reformat: [1mtests/inference/test_jaxtyping_optional.py[0m
Would reformat: [1mtests/inference/test_schema.py[0m
Would reformat: [1mtests/inference/test_seam_conformance.py[0m
Would reformat: [1mtests/tiling/test_plan_unknown_guard.py[0m
28 files would be reformatted, 275 files already formatted
  ```
- **ty_check** (ci_lint): FAIL
  ```
  error[unresolved-attribute]: Unresolved attribute `tyro` on type `ModuleType`
  --> src/xtrax/cli/entrypoint.py:37:13
   |
37 |             mod.tyro = tyro
   |             ^^^^^^^^
   |

Found 1 diagnostic
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
- **added_types_diff** (type_hardening): FAIL
  ```
  [32m============================== [32m[1m10 passed[0m[32m in 1.66s[0m[32m ==============================[0m
merge-base=55bf118e1489080e4838bb6e80073864f1ecec2e files_checked=28 callables_checked=31

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

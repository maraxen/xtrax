"""Tests for distribution N6 coverage DAG manifest + baseline reporter (#1456)."""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.audit_coverage_dag import (
    ENFORCEMENT_RECIPES,
    CoverageDag,
    Tier,
    TierResult,
    audit_coverage_dag,
    build_state_payload,
    evaluate_enforce,
    format_verdict,
    load_coverage_dag,
    select_tiers,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "coverage_dag.toml"


def _discover_enforce_recipes(justfile_text: str) -> set[tuple[str, str]]:
    """Discover all (tier, recipe_name) pairs from audit_coverage_dag.py --enforce lines.

    Parses Justfile text and extracts recipes that run audit_coverage_dag.py with
    --enforce <tier>. Returns a set of (tier, recipe_name) tuples representing all
    discovered enforcement recipes, regardless of whether they are in ENFORCEMENT_RECIPES.
    """
    result = set()
    lines = justfile_text.splitlines()
    current_recipe = None

    for i, line in enumerate(lines):
        # Check if this line is a recipe header (no leading whitespace, matches pattern)
        if not line or line[0] in (" ", "\t"):
            # Indented line - part of recipe body
            if current_recipe and "audit_coverage_dag.py" in line:
                # Look for --enforce <tier> pattern
                match = re.search(r"--enforce\s+(\S+)", line)
                if match:
                    tier = match.group(1)
                    result.add((tier, current_recipe))
        else:
            # Non-indented line - check if it's a recipe header
            header_match = re.match(r"^([A-Za-z0-9_-]+)\s*:(?!=)", line)
            if header_match:
                current_recipe = header_match.group(1)

    return result


def _write_contract(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "coverage_dag.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def test_load_coverage_dag_reads_committed_toml() -> None:
    dag = load_coverage_dag(CONFIG_PATH)
    assert dag.version == "0.2.2"
    assert dag.state_path == ".praxia/coverage_last_measured.json"
    assert len(dag.tiers) == 5
    tier_ids = [tier.id for tier in dag.tiers]
    assert tier_ids == [
        "tier0_audit",
        "tier1_core",
        "tier2_eda",
        "tier3_port",
        "tier4_controller",
    ]

    tier1 = dag.tiers[1]
    assert tier1.measure_coverage is True
    assert tier1.coverage_packages == ("xtrax",)
    assert "*/xtrax/eda/*" in tier1.coverage_omit
    assert "*/xtrax/devtools/*" in tier1.coverage_omit
    assert tier1.target_line_pct == 90.0
    assert tier1.enforce_line_pct == 90.0
    assert tier1.enforce_branch_pct == 80.0
    assert "tests/eda" in " ".join(tier1.pytest_args)
    assert "tests/audit" in " ".join(tier1.pytest_args)
    assert "tests/distribution" in " ".join(tier1.pytest_args)

    tier2 = dag.tiers[2]
    assert tier2.coverage_packages == ("xtrax.eda",)
    assert tier2.enforce_line_pct == 90.0
    assert tier2.enforce_branch_pct == 75.0

    tier0 = dag.tiers[0]
    assert tier0.measure_coverage is False

    # tier4 measures controller/, a tree tier1 cannot see: tier1's coverage_packages
    # is ("xtrax",) and controller/ lives outside src/. The `controller` extra is
    # load-bearing rather than incidental -- without bathos installed,
    # test_bathos_library_wrappers_integration.py skips at collection time and the
    # wrappers it covers read as untested.
    tier4 = dag.tiers[4]
    assert tier4.measure_coverage is True
    assert tier4.coverage_packages == ("controller",)
    assert "controller" in tier4.uv_sync_extras
    assert tier4.pytest_args[0] == "tests/controller/"
    assert tier4.enforce_line_pct == 90.0
    assert tier4.enforce_branch_pct == 80.0


def test_select_tiers_defaults_to_tier1_core() -> None:
    dag = load_coverage_dag(CONFIG_PATH)
    selected = select_tiers(dag, tier_id=None, all_tiers=False)
    assert len(selected) == 1
    assert selected[0].id == "tier1_core"


def test_select_tiers_all() -> None:
    dag = load_coverage_dag(CONFIG_PATH)
    selected = select_tiers(dag, tier_id=None, all_tiers=True)
    assert [tier.id for tier in selected] == [
        "tier0_audit",
        "tier1_core",
        "tier2_eda",
        "tier3_port",
        "tier4_controller",
    ]


def test_evaluate_enforce_passes_above_floors() -> None:
    tier = Tier(
        id="tier1_core",
        description="core",
        measure_coverage=True,
        uv_sync_extras=("dev",),
        pytest_args=("tests/", "-q"),
        enforce_line_pct=85.0,
        enforce_branch_pct=65.0,
    )
    result = TierResult(
        tier_id="tier1_core",
        measure_coverage=True,
        line_pct=88.0,
        branch_pct=70.0,
        tests_run=100,
        tests_failed=0,
        pytest_exit_code=0,
    )
    evaluated = evaluate_enforce(tier, result)
    assert evaluated.enforce_passed is True
    assert evaluated.enforce_failures == ()


def test_evaluate_enforce_fails_below_line_floor() -> None:
    tier = Tier(
        id="tier1_core",
        description="core",
        measure_coverage=True,
        uv_sync_extras=("dev",),
        pytest_args=("tests/", "-q"),
        enforce_line_pct=85.0,
        enforce_branch_pct=65.0,
    )
    result = TierResult(
        tier_id="tier1_core",
        measure_coverage=True,
        line_pct=78.5,
        branch_pct=70.0,
        tests_run=100,
        tests_failed=11,
        pytest_exit_code=1,
    )
    evaluated = evaluate_enforce(tier, result)
    assert evaluated.enforce_passed is False
    assert any("line 78.5%" in item for item in evaluated.enforce_failures)


def test_evaluate_enforce_fails_below_branch_floor() -> None:
    tier = Tier(
        id="tier1_core",
        description="core",
        measure_coverage=True,
        uv_sync_extras=("dev",),
        pytest_args=("tests/", "-q"),
        enforce_line_pct=85.0,
        enforce_branch_pct=65.0,
    )
    result = TierResult(
        tier_id="tier1_core",
        measure_coverage=True,
        line_pct=90.0,
        branch_pct=64.0,
        tests_run=100,
        tests_failed=0,
        pytest_exit_code=0,
    )
    evaluated = evaluate_enforce(tier, result)
    assert evaluated.enforce_passed is False
    assert any("branch 64.0%" in item for item in evaluated.enforce_failures)


def test_build_state_payload_includes_tier_metrics() -> None:
    dag = CoverageDag(version="0.1.0", state_path=".praxia/coverage_last_measured.json", tiers=())
    results = (
        TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=78.5,
            branch_pct=66.9,
            tests_run=824,
            tests_failed=11,
            pytest_exit_code=1,
        ),
    )
    payload = build_state_payload(dag, results)
    assert payload["dag_version"] == "0.1.0"
    tier_payload = payload["tiers"]["tier1_core"]
    assert tier_payload["line_pct"] == 78.5
    assert tier_payload["branch_pct"] == 66.9
    assert tier_payload["tests_failed"] == 11


def test_audit_coverage_dag_mocks_pytest_and_writes_state(tmp_path: Path) -> None:
    config_path = _write_contract(tmp_path)
    tier = Tier(
        id="tier1_core",
        description="core",
        measure_coverage=True,
        uv_sync_extras=("dev",),
        pytest_args=("tests/", "-q"),
        enforce_line_pct=85.0,
        enforce_branch_pct=65.0,
    )

    cov_json = json.dumps(
        {
            "totals": {
                "percent_covered": 78.5,
                "percent_branches_covered": 66.9,
            }
        }
    )

    def fake_run(cmd, cwd, capture_output, text, check, env=None):  # noqa: ANN001
        if cmd[:2] == ["uv", "sync"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "--cov-report=json:" in " ".join(cmd):
            cov_arg = next(arg for arg in cmd if arg.startswith("--cov-report=json:"))
            cov_path = Path(cov_arg.split(":", 1)[1])
            cov_path.write_text(cov_json, encoding="utf-8")
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "11 failed, 813 passed in 10.0s",
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("scripts.audit_coverage_dag.subprocess.run", side_effect=fake_run):
        passed, results, failures = audit_coverage_dag(
            root=tmp_path,
            config_path=config_path,
            tiers=(tier,),
            enforce_tier=None,
        )

    assert passed is True
    assert failures == []
    assert len(results) == 1
    assert results[0].line_pct == 78.5
    assert results[0].branch_pct == 66.9
    assert results[0].tests_failed == 11

    state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
    assert state_path.is_file()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["tiers"]["tier1_core"]["line_pct"] == 78.5


def test_audit_coverage_dag_enforce_fails_with_mocked_coverage(
    tmp_path: Path,
) -> None:
    config_path = _write_contract(tmp_path)
    tier = load_coverage_dag(config_path).tiers[1]

    cov_json = json.dumps(
        {
            "totals": {
                "percent_covered": 78.5,
                "percent_branches_covered": 66.9,
            }
        }
    )

    def fake_run(cmd, cwd, capture_output, text, check, env=None):  # noqa: ANN001
        if cmd[:2] == ["uv", "sync"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        cov_arg = next(arg for arg in cmd if arg.startswith("--cov-report=json:"))
        cov_path = Path(cov_arg.split(":", 1)[1])
        cov_path.write_text(cov_json, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, "", "11 failed, 813 passed in 10.0s")

    with patch("scripts.audit_coverage_dag.subprocess.run", side_effect=fake_run):
        passed, results, failures = audit_coverage_dag(
            root=tmp_path,
            config_path=config_path,
            tiers=(tier,),
            enforce_tier="tier1_core",
        )

    assert passed is False
    assert results[0].enforce_passed is False
    assert failures


def test_main_report_only_exits_zero_with_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=78.5,
            branch_pct=66.9,
            tests_run=824,
            tests_failed=11,
            pytest_exit_code=1,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return True, (result,), []

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--config",
                str(config_path),
                "--tier",
                "tier1_core",
            ]
        )
        == 0
    )


def test_main_enforce_exits_nonzero_when_below_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        tier = load_coverage_dag(config_path).tiers[1]
        result = evaluate_enforce(
            tier,
            TierResult(
                tier_id="tier1_core",
                measure_coverage=True,
                line_pct=78.5,
                branch_pct=66.9,
                tests_run=824,
                tests_failed=11,
                pytest_exit_code=1,
            ),
        )
        return False, (result,), list(result.enforce_failures)

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--config",
                str(config_path),
                "--tier",
                "tier1_core",
                "--enforce",
                "tier1_core",
            ]
        )
        == 1
    )


def test_load_coverage_dag_rejects_missing_tiers(tmp_path: Path) -> None:
    config_path = tmp_path / "coverage_dag.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [dag]
            version = "0.1.0"
            state_path = ".praxia/coverage_last_measured.json"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one"):
        load_coverage_dag(config_path)


def test_main_report_only_labels_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-1: Report-only run with failures labels them in the verdict."""
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=78.5,
            branch_pct=66.9,
            tests_run=824,
            tests_failed=19,
            pytest_exit_code=1,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return True, (result,), []

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier1_core",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert not captured.out.startswith("PASS")
    assert (
        "REPORT (non-blocking): coverage DAG -- 19 test failures in tier1_core "
        "(pytest exit 1); enforcement lives in just audit-coverage-tier1"
    ) in captured.out


def test_main_report_only_no_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-2: Report-only run with no failures."""
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=90.0,
            branch_pct=80.0,
            tests_run=824,
            tests_failed=0,
            pytest_exit_code=0,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return True, (result,), []

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier1_core",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "REPORT (non-blocking): coverage DAG -- no test failures in tier1_core" in captured.out
    assert not captured.out.startswith("PASS")


def test_main_report_only_nonzero_exit_counts_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-3: Nonzero pytest exit (e.g. collection error) counts as failure."""
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=90.0,
            branch_pct=80.0,
            tests_run=0,
            tests_failed=0,
            pytest_exit_code=2,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return True, (result,), []

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier1_core",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert (
        "REPORT (non-blocking): coverage DAG -- 0 test failures in tier1_core "
        "(pytest exit 2); enforcement lives in just audit-coverage-tier1"
    ) in captured.out


def test_format_verdict_joins_multiple_failures() -> None:
    """AC-4: format_verdict joins multiple tier failures with pipes."""
    tier0_result = TierResult(
        tier_id="tier0_audit",
        measure_coverage=False,
        line_pct=None,
        branch_pct=None,
        tests_run=10,
        tests_failed=2,
        pytest_exit_code=1,
    )
    tier1_result = TierResult(
        tier_id="tier1_core",
        measure_coverage=True,
        line_pct=78.5,
        branch_pct=66.9,
        tests_run=824,
        tests_failed=19,
        pytest_exit_code=1,
    )

    verdict = format_verdict(
        (tier0_result, tier1_result),
        enforce_tier=None,
        passed=True,
    )
    expected = (
        "REPORT (non-blocking): coverage DAG -- "
        "2 test failures in tier0_audit (pytest exit 1); not enforced by any recipe | "
        "19 test failures in tier1_core (pytest exit 1); "
        "enforcement lives in just audit-coverage-tier1"
    )
    assert verdict == expected


def test_format_verdict_raises_on_enforce_fail() -> None:
    """AC-4: format_verdict raises ValueError when enforce_tier is set and passed=False."""
    result = TierResult(
        tier_id="tier1_core",
        measure_coverage=True,
        line_pct=78.5,
        branch_pct=66.9,
        tests_run=824,
        tests_failed=11,
        pytest_exit_code=1,
    )
    with pytest.raises(ValueError, match="enforce-fail is reported on stderr"):
        format_verdict((result,), enforce_tier="tier1_core", passed=False)


def test_main_enforce_fail_reports_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-5a: Enforce failure is reported on stderr."""
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=78.5,
            branch_pct=66.9,
            tests_run=824,
            tests_failed=11,
            pytest_exit_code=1,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return False, (result,), ["pytest failed (11 failures, exit 1)"]

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier1_core",
            "--enforce",
            "tier1_core",
        ]
    )
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "FAIL: coverage DAG enforce" in captured.err


def test_main_enforce_pass_no_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-5b: Enforce pass with no failures prints PASS: coverage DAG enforce."""
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier1_core",
            measure_coverage=True,
            line_pct=95.0,
            branch_pct=85.0,
            tests_run=824,
            tests_failed=0,
            pytest_exit_code=0,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return True, (result,), []

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier1_core",
            "--enforce",
            "tier1_core",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "PASS: coverage DAG enforce" in captured.out


def test_main_enforce_pass_with_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-5c: Enforce pass with observed failures (not enforced tier) prints REPORT not enforced."""
    config_path = _write_contract(tmp_path)

    def fake_audit(**kwargs):  # noqa: ANN003
        result = TierResult(
            tier_id="tier0_audit",
            measure_coverage=False,
            line_pct=None,
            branch_pct=None,
            tests_run=10,
            tests_failed=2,
            pytest_exit_code=1,
        )
        state_path = tmp_path / ".praxia" / "coverage_last_measured.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
        return True, (result,), []

    monkeypatch.setattr("scripts.audit_coverage_dag.audit_coverage_dag", fake_audit)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier0_audit",
            "--enforce",
            "tier0_audit",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert (
        "REPORT (not enforced): coverage DAG --enforce tier0_audit -- "
        "2 test failures in tier0_audit (pytest exit 1); not enforced by any recipe"
    ) in captured.out
    assert not captured.out.startswith("PASS")


def test_main_enforce_unknown_tier_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-5d: --enforce with unknown tier is a usage error (exits 1, stderr FAIL)."""
    config_path = _write_contract(tmp_path)

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("pytest should not run for unknown enforce tier")

    monkeypatch.setattr("scripts.audit_coverage_dag.run_tier_pytest", fake_run)

    from scripts.audit_coverage_dag import main

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--tier",
            "tier1_core",
            "--enforce",
            "",
        ]
    )
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "FAIL: coverage DAG enforce" in captured.err
    assert "unknown enforce tier" in captured.err
    assert not captured.out.startswith("PASS")


def test_enforcement_recipes_consistency() -> None:
    """AC-6: ENFORCEMENT_RECIPES constant has expected value."""
    assert ENFORCEMENT_RECIPES == {
        "tier1_core": "audit-coverage-tier1",
        "tier2_eda": "audit-coverage-tier2",
        "tier4_controller": "audit-coverage-tier4",
    }


def test_justfile_recipes_match_enforcement() -> None:
    """AC-6: Justfile enforcement recipes match ENFORCEMENT_RECIPES (bidirectional)."""
    justfile = ROOT / "Justfile"
    justfile_text = justfile.read_text(encoding="utf-8")

    discovered = _discover_enforce_recipes(justfile_text)
    expected = set(ENFORCEMENT_RECIPES.items())

    # Assert set equality (both directions)
    assert discovered == expected, (
        f"Justfile enforcement recipes mismatch:\n"
        f"Expected: {expected}\n"
        f"Discovered: {discovered}\n"
        f"Missing from Justfile: {expected - discovered}\n"
        f"Unexpected in Justfile: {discovered - expected}"
    )


def test_ci_yml_includes_enforcement_recipes() -> None:
    """AC-6: Every recipe in ENFORCEMENT_RECIPES appears in .github/workflows/ci.yml."""
    ci_yml = ROOT / ".github" / "workflows" / "ci.yml"
    ci_text = ci_yml.read_text(encoding="utf-8")

    for recipe in ENFORCEMENT_RECIPES.values():
        pattern = f"just {recipe}"
        assert pattern in ci_text, f"Recipe 'just {recipe}' not found in ci.yml"


def test_discover_enforce_recipes_detects_unmapped_recipe() -> None:
    """Positive control: _discover_enforce_recipes detects unmapped recipes."""
    synthetic_justfile = textwrap.dedent("""
        audit-coverage-tier1:
            uv run python scripts/audit_coverage_dag.py --tier tier1_core --enforce tier1_core

        audit-coverage-tier2:
            uv run python scripts/audit_coverage_dag.py --tier tier2_eda --enforce tier2_eda

        audit-coverage-tier9:
            uv run python scripts/audit_coverage_dag.py --tier tier9_x --enforce tier9_x

        audit-coverage-dag-all:
            uv run python scripts/audit_coverage_dag.py --all-tiers
    """).strip()

    discovered = _discover_enforce_recipes(synthetic_justfile)
    expected = set(ENFORCEMENT_RECIPES.items())

    # The synthetic Justfile should NOT match expected (it has an extra recipe)
    assert discovered != expected

    # The discovered set should include the three real ones plus the synthetic tier9
    assert ("tier1_core", "audit-coverage-tier1") in discovered
    assert ("tier2_eda", "audit-coverage-tier2") in discovered
    assert ("tier9_x", "audit-coverage-tier9") in discovered

    # The report-only line (no --enforce) should contribute nothing
    assert len(discovered) == 3  # Only the three --enforce lines

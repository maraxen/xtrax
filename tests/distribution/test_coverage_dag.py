"""Tests for distribution N6 coverage DAG manifest + baseline reporter (#1456)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.audit_coverage_dag import (
    CoverageDag,
    Tier,
    TierResult,
    audit_coverage_dag,
    build_state_payload,
    evaluate_enforce,
    load_coverage_dag,
    select_tiers,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "coverage_dag.toml"


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
    assert len(dag.tiers) == 4
    tier_ids = [tier.id for tier in dag.tiers]
    assert tier_ids == ["tier0_audit", "tier1_core", "tier2_eda", "tier3_port"]

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

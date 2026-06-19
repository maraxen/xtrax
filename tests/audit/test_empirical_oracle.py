"""Tests for N4.4 Empirical-oracle promotion path (#1595)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtrax.devtools.empirical_oracle import (
    BUG_LABEL,
    DEFAULT_ORACLE_PATH,
    PromotionRequest,
    ReproResult,
    attempt_bug_promotion,
    load_oracle_config,
    run_pytest_repro,
)
from xtrax.devtools.refute_promote import JudgmentCandidate

ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = ROOT / DEFAULT_ORACLE_PATH
REPRO_FIXTURE = ROOT / "tests" / "fixtures" / "audit_repro_fail.py"


def _candidate() -> JudgmentCandidate:
    return JudgmentCandidate(
        dimension="correctness",
        severity="major",
        file_line="src/xtrax/engine/loop.py:88",
        evidence="metamorphic parity suspect on zero-input edge",
        rubric_id="correctness.metamorphic",
        score=4,
        anchor_quote="oracle-class correctness claim",
    )


def _request(*, repro_test_path: Path = REPRO_FIXTURE) -> PromotionRequest:
    return PromotionRequest(
        finding_id="obs-finding-abc123",
        candidate=_candidate(),
        repro_test_path=repro_test_path,
    )


def test_load_oracle_config_loads_committed_toml() -> None:
    config = load_oracle_config(ORACLE_PATH)
    assert config.version == "0.1.0"
    assert config.max_promotions_per_run == 3
    assert config.promotion_requires == "failing_pytest"
    assert "reproducible" in config.executable_check


def test_load_oracle_config_missing_section_raises(tmp_path: Path) -> None:
    broken = tmp_path / "empirical_oracle.toml"
    broken.write_text("[other]\nversion = '0'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="\\[oracle\\] section is required"):
        load_oracle_config(broken)


@patch("xtrax.devtools.empirical_oracle.subprocess.run")
def test_run_pytest_repro_passes_gate_when_test_fails(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="F\n",
        stderr="1 failed\n",
    )
    repro = run_pytest_repro(REPRO_FIXTURE, cwd=ROOT)
    assert repro.passed_gate is True
    assert repro.exit_code == 1
    assert "failed" in repro.output_snippet
    mock_run.assert_called_once_with(
        ["uv", "run", "pytest", str(REPRO_FIXTURE), "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@patch("xtrax.devtools.empirical_oracle.subprocess.run")
def test_run_pytest_repro_fails_gate_when_test_passes(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=".\n",
        stderr="",
    )
    repro = run_pytest_repro(REPRO_FIXTURE, cwd=ROOT)
    assert repro.passed_gate is False
    assert repro.exit_code == 0


@patch("xtrax.devtools.empirical_oracle.run_pytest_repro")
def test_attempt_bug_promotion_skips_when_budget_exhausted(
    mock_repro: MagicMock,
    tmp_path: Path,
) -> None:
    mock_repro.return_value = ReproResult(
        passed_gate=True,
        exit_code=1,
        output_snippet="failed",
    )
    verdict = attempt_bug_promotion(
        _request(),
        tmp_path / "audits.jsonl",
        budget=0,
        run_id="budget-zero",
    )
    assert verdict.promoted is False
    assert verdict.budget_remaining == 0
    assert not (tmp_path / "audits.jsonl").exists()


@patch("xtrax.devtools.empirical_oracle.run_pytest_repro")
def test_attempt_bug_promotion_skips_when_repro_passes(
    mock_repro: MagicMock,
    tmp_path: Path,
) -> None:
    mock_repro.return_value = ReproResult(
        passed_gate=False,
        exit_code=0,
        output_snippet="passed",
    )
    verdict = attempt_bug_promotion(
        _request(),
        tmp_path / "audits.jsonl",
        budget=3,
        run_id="repro-pass",
    )
    assert verdict.promoted is False
    assert verdict.budget_remaining == 3
    assert not (tmp_path / "audits.jsonl").exists()


@patch("xtrax.devtools.empirical_oracle.append_finding")
@patch("xtrax.devtools.empirical_oracle.run_pytest_repro")
def test_attempt_bug_promotion_emits_bug_when_repro_fails(
    mock_repro: MagicMock,
    mock_append: MagicMock,
    tmp_path: Path,
) -> None:
    mock_repro.return_value = ReproResult(
        passed_gate=True,
        exit_code=1,
        output_snippet="AssertionError",
    )
    audits_path = tmp_path / "audits.jsonl"
    verdict = attempt_bug_promotion(
        _request(),
        audits_path,
        budget=3,
        run_id="promote-bug",
    )
    assert verdict.promoted is True
    assert verdict.label == BUG_LABEL
    assert verdict.budget_remaining == 2
    mock_append.assert_called_once()
    record = mock_append.call_args.args[0]
    assert record.source_track == "judgment"
    assert record.payload["label"] == BUG_LABEL
    assert record.payload["promoted_from"] == "obs-finding-abc123"
    assert record.payload["repro_test_path"] == str(REPRO_FIXTURE)
    assert record.payload["repro_exit_code"] == 1
    assert record.payload["protocol"] == "empirical_oracle"


@patch("xtrax.devtools.empirical_oracle.run_pytest_repro")
def test_attempt_bug_promotion_writes_jsonl_payload(
    mock_repro: MagicMock,
    tmp_path: Path,
) -> None:
    mock_repro.return_value = ReproResult(
        passed_gate=True,
        exit_code=1,
        output_snippet="failed",
    )
    audits_path = tmp_path / "audits.jsonl"
    attempt_bug_promotion(
        _request(),
        audits_path,
        budget=1,
        run_id="jsonl-bug",
    )
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["source_track"] == "judgment"
    assert payload["payload"]["label"] == BUG_LABEL
    assert payload["payload"]["promoted_from"] == "obs-finding-abc123"
    assert payload["payload"]["protocol"] == "empirical_oracle"


@patch("xtrax.devtools.empirical_oracle.run_pytest_repro")
def test_budget_cap_limits_sequential_promotions(
    mock_repro: MagicMock,
    tmp_path: Path,
) -> None:
    mock_repro.return_value = ReproResult(
        passed_gate=True,
        exit_code=1,
        output_snippet="failed",
    )
    audits_path = tmp_path / "audits.jsonl"
    budget = 2
    promoted = 0
    for index in range(4):
        verdict = attempt_bug_promotion(
            PromotionRequest(
                finding_id=f"obs-{index}",
                candidate=_candidate(),
                repro_test_path=REPRO_FIXTURE,
            ),
            audits_path,
            budget=budget,
            run_id=f"cap-{index}",
        )
        budget = verdict.budget_remaining
        if verdict.promoted:
            promoted += 1
    assert promoted == 2
    assert budget == 0
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

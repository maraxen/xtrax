"""Tests for N4.5 Documentation judgment pipeline (#1596)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.docs_judgment import (
    DEFAULT_JUDGMENT_PATH,
    StructuralScore,
    load_docs_judgment_config,
    run_docs_judgment,
    score_structural_docs,
    stub_semantic_judge,
)
from xtrax.devtools.refute_promote import OBSERVATION_LABEL
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / DEFAULT_JUDGMENT_PATH
RUBRICS_DIR = ROOT / "audit" / "rubrics"


@pytest.fixture
def documentation_rubric():
    return load_rubric(RUBRICS_DIR / "documentation.toml")


def test_load_docs_judgment_config_loads_committed_toml() -> None:
    config = load_docs_judgment_config(CONFIG_PATH)
    assert config.dimension == "documentation"
    assert config.agent_role == "reviewer"
    assert config.pass_threshold == 4
    assert config.rubric_path.name == "documentation.toml"
    assert config.structural_signals == (
        "jd_jm_errors",
        "interrogate_coverage_pct",
    )


def test_load_docs_judgment_config_missing_section_raises(tmp_path: Path) -> None:
    broken = tmp_path / "docs_judgment.toml"
    broken.write_text("[other]\nversion = '0'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="\\[judgment\\] section is required"):
        load_docs_judgment_config(broken)


@pytest.mark.parametrize(
    ("jd_count", "coverage_pct", "expected_score"),
    [
        (3, 99.0, 1),
        (1, 50.0, 1),
        (0, 75.0, 2),
        (0, 85.0, 3),
        (0, 92.0, 4),
        (0, 97.0, 5),
    ],
)
def test_score_structural_docs_mapping(
    documentation_rubric,
    jd_count: int,
    coverage_pct: float,
    expected_score: int,
) -> None:
    result = score_structural_docs(jd_count, coverage_pct, documentation_rubric)
    assert result.score == expected_score
    assert result.evidence == (
        f"jd_jm_errors={jd_count}, interrogate_coverage_pct={coverage_pct:.1f}"
    )
    assert result.anchor_quote


def test_stub_semantic_judge_returns_structural_score() -> None:
    structural = StructuralScore(
        score=4,
        anchor_quote="Good docs",
        evidence="jd_jm_errors=0, interrogate_coverage_pct=92.0",
    )
    assert stub_semantic_judge(structural, "src/xtrax/run/spec.py:1") == 4


@patch("xtrax.devtools.docs_judgment.collect_structural_signals")
def test_run_docs_judgment_passes_at_threshold(
    mock_signals,
    tmp_path: Path,
    documentation_rubric,
) -> None:
    mock_signals.return_value = (0, 92.0)
    result = run_docs_judgment(
        ROOT / "src" / "xtrax",
        tmp_path / "audits.jsonl",
        semantic_judge_fn=stub_semantic_judge,
        run_id="pass-threshold",
        config_path=CONFIG_PATH,
        root=ROOT,
    )
    assert result.structural.score == 4
    assert result.semantic_score == 4
    assert result.passed is True
    assert result.finding_emitted is True


@patch("xtrax.devtools.docs_judgment.collect_structural_signals")
def test_run_docs_judgment_fails_below_threshold(
    mock_signals,
    tmp_path: Path,
) -> None:
    mock_signals.return_value = (0, 85.0)

    def low_semantic(_: StructuralScore, __: str) -> int:
        return 3

    result = run_docs_judgment(
        ROOT / "src" / "xtrax",
        tmp_path / "audits.jsonl",
        semantic_judge_fn=low_semantic,
        run_id="fail-threshold",
        config_path=CONFIG_PATH,
        root=ROOT,
    )
    assert result.structural.score == 3
    assert result.semantic_score == 3
    assert result.passed is False


@patch("xtrax.devtools.docs_judgment.append_finding")
@patch("xtrax.devtools.docs_judgment.collect_structural_signals")
def test_run_docs_judgment_emit_payload(
    mock_signals,
    mock_append,
    tmp_path: Path,
) -> None:
    mock_signals.return_value = (0, 97.0)
    audits_path = tmp_path / "audits.jsonl"
    run_docs_judgment(
        ROOT / "src" / "xtrax",
        audits_path,
        semantic_judge_fn=stub_semantic_judge,
        run_id="emit-payload",
        config_path=CONFIG_PATH,
        root=ROOT,
    )
    mock_append.assert_called_once()
    record = mock_append.call_args.args[0]
    assert record.source_track == "judgment"
    assert record.payload["label"] == OBSERVATION_LABEL
    assert record.payload["structural_score"] == 5
    assert record.payload["semantic_score"] == 5
    assert record.payload["rubric_scorer_evidence"] == (
        "jd_jm_errors=0, interrogate_coverage_pct=97.0"
    )
    assert record.payload["protocol"] == "docs_judgment"
    assert record.payload["agent_role"] == "reviewer"


@patch("xtrax.devtools.docs_judgment.collect_structural_signals")
def test_run_docs_judgment_no_emit_skips_append(
    mock_signals,
    tmp_path: Path,
) -> None:
    mock_signals.return_value = (0, 92.0)
    result = run_docs_judgment(
        ROOT / "src" / "xtrax",
        tmp_path / "audits.jsonl",
        semantic_judge_fn=stub_semantic_judge,
        run_id="no-emit",
        config_path=CONFIG_PATH,
        root=ROOT,
        emit_finding=False,
    )
    assert result.finding_emitted is False
    assert not (tmp_path / "audits.jsonl").exists()


@patch("xtrax.devtools.docs_judgment.collect_structural_signals")
def test_run_docs_judgment_writes_jsonl(
    mock_signals,
    tmp_path: Path,
) -> None:
    mock_signals.return_value = (2, 88.0)
    audits_path = tmp_path / "audits.jsonl"
    run_docs_judgment(
        ROOT / "src" / "xtrax",
        audits_path,
        semantic_judge_fn=stub_semantic_judge,
        run_id="jsonl-write",
        config_path=CONFIG_PATH,
        root=ROOT,
    )
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["source_track"] == "judgment"
    assert payload["payload"]["protocol"] == "docs_judgment"
    assert payload["payload"]["structural_score"] == 1
    assert payload["payload"]["semantic_score"] == 1

"""Tests for D8 structure-complexity gate (N2.8 / #1588)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.audit_structure_complexity_gate import main
from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates.structure_complexity import (
    COGNITIVE_METRIC,
    RUFF_METRIC,
    ComplexityHit,
    GateResult,
    RuffHit,
    parse_complexipy_json,
    parse_ruff_complexity_json,
    run_structure_complexity_gate,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"


def test_structure_complexity_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "structure_complexity.toml")
    assert table.dimension == "structure_complexity"
    assert len(table.anchors) == 5


def test_parse_complexipy_json_fixture(tmp_path: Path) -> None:
    data = [
        {
            "path": "xtrax/foo.py",
            "function_name": "simple",
            "complexity": 3,
            "line_start": 1,
        },
        {
            "path": "xtrax/foo.py",
            "function_name": "heavy",
            "complexity": 28,
            "line_start": 10,
        },
    ]
    max_score, hits = parse_complexipy_json(data, root=tmp_path, threshold=15)
    assert max_score == 28.0
    assert len(hits) == 1
    assert hits[0].qualname == "heavy"
    assert hits[0].complexity == 28


def test_parse_ruff_complexity_json_fixture() -> None:
    data = [
        {
            "code": "C901",
            "filename": "/repo/src/xtrax/foo.py",
            "location": {"row": 12, "column": 5},
            "message": "`render` is too complex (37 > 10)",
        },
        {
            "code": "E501",
            "filename": "/repo/src/xtrax/foo.py",
            "location": {"row": 20, "column": 1},
            "message": "line too long",
        },
        {
            "code": "PLR0912",
            "filename": "/repo/src/xtrax/bar.py",
            "location": {"row": 5, "column": 5},
            "message": "Too many branches (14 > 12)",
        },
    ]
    count, hits = parse_ruff_complexity_json(data)
    assert count == 2
    assert hits[0].rule_id == "C901"
    assert hits[0].symbol == "render"
    assert hits[1].rule_id == "PLR0912"


def test_run_structure_complexity_gate_passes_at_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COGNITIVE_METRIC: MetricEntry(
                key=COGNITIVE_METRIC,
                value=50.0,
                comparator="minimize",
            ),
            RUFF_METRIC: MetricEntry(
                key=RUFF_METRIC,
                value=5.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with (
        patch(
            "xtrax.devtools.gates.structure_complexity.run_complexipy_scan",
            return_value=(30.0, []),
        ),
        patch(
            "xtrax.devtools.gates.structure_complexity.run_ruff_complexity_scan",
            return_value=(2, []),
        ),
    ):
        result = run_structure_complexity_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            write_baseline=False,
        )

    assert result.passed is True
    assert result.cognitive_complexity_max == 30.0
    assert result.ruff_violation_count == 2
    assert result.findings_emitted == 2
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "structure_complexity"
    assert record["payload"]["cognitive_complexity_max"] == 30.0


def test_run_structure_complexity_gate_fails_on_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COGNITIVE_METRIC: MetricEntry(
                key=COGNITIVE_METRIC,
                value=20.0,
                comparator="minimize",
            ),
            RUFF_METRIC: MetricEntry(
                key=RUFF_METRIC,
                value=1.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)
    cognitive_hits = (
        ComplexityHit(
            qualname="heavy",
            file_line=str(tmp_path / "heavy.py:10"),
            complexity=30,
        ),
    )
    ruff_hits = (
        RuffHit(
            rule_id="C901",
            file_line=str(tmp_path / "heavy.py:10"),
            symbol="heavy",
            message="`heavy` is too complex (30 > 10)",
        ),
    )

    with (
        patch(
            "xtrax.devtools.gates.structure_complexity.run_complexipy_scan",
            return_value=(30.0, list(cognitive_hits)),
        ),
        patch(
            "xtrax.devtools.gates.structure_complexity.run_ruff_complexity_scan",
            return_value=(3, list(ruff_hits)),
        ),
    ):
        result = run_structure_complexity_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            cognitive_ceiling=25,
            write_baseline=False,
        )

    assert result.passed is False
    assert result.findings_emitted >= 4


def test_run_structure_complexity_gate_tightens_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COGNITIVE_METRIC: MetricEntry(
                key=COGNITIVE_METRIC,
                value=50.0,
                comparator="minimize",
            ),
            RUFF_METRIC: MetricEntry(
                key=RUFF_METRIC,
                value=5.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with (
        patch(
            "xtrax.devtools.gates.structure_complexity.run_complexipy_scan",
            return_value=(30.0, []),
        ),
        patch(
            "xtrax.devtools.gates.structure_complexity.run_ruff_complexity_scan",
            return_value=(2, []),
        ),
    ):
        result = run_structure_complexity_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            write_baseline=True,
        )

    assert result.passed is True
    assert result.baseline_updated is True
    updated = load_baseline(path=baseline_path)
    assert updated.metrics[COGNITIVE_METRIC].value == 30.0
    assert updated.metrics[RUFF_METRIC].value == 2.0


def test_committed_baseline_has_structure_complexity_metrics() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert COGNITIVE_METRIC in loaded.metrics
    assert RUFF_METRIC in loaded.metrics
    assert loaded.metrics[COGNITIVE_METRIC].comparator == "minimize"
    assert loaded.metrics[RUFF_METRIC].comparator == "minimize"


def test_audit_structure_complexity_gate_cli_exits_zero_with_mock(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COGNITIVE_METRIC: MetricEntry(
                key=COGNITIVE_METRIC,
                value=100.0,
                comparator="minimize",
            ),
            RUFF_METRIC: MetricEntry(
                key=RUFF_METRIC,
                value=20.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)
    mock_result = GateResult(
        passed=True,
        cognitive_complexity_max=30.0,
        ruff_violation_count=2,
        cognitive_hits=(),
        ruff_hits=(),
        findings_emitted=2,
        baseline_updated=False,
    )

    with patch(
        "scripts.audit_structure_complexity_gate.run_structure_complexity_gate",
        return_value=mock_result,
    ):
        exit_code = main(
            [
                str(tmp_path),
                "--baseline-path",
                str(baseline_path),
                "--audits-path",
                str(audits_path),
                "--no-write-baseline",
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out

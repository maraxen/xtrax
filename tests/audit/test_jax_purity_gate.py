"""Tests for D2 JAX-purity gate (N2.2 / #1582)."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates.jax_purity import (
    METRIC_KEY,
    filter_jl_purity_errors,
    run_jax_purity_gate,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"


def test_jax_purity_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "jax_purity.toml")
    assert table.dimension == "jax_purity"
    assert len(table.anchors) == 5


def test_filter_jl_purity_errors_keeps_jl001_to_jl012_errors_only() -> None:
    findings = [
        {"rule_id": "JL001", "severity": "error", "message": "purity"},
        {"rule_id": "JL012", "severity": "error", "message": "purity edge"},
        {"rule_id": "JL013", "severity": "error", "message": "out of range"},
        {"rule_id": "JL001", "severity": "warning", "message": "warn"},
        {"rule_id": "JD001", "severity": "error", "message": "doc"},
        {"rule_id": "JL050", "severity": "error", "message": "perf"},
    ]
    errors = filter_jl_purity_errors(findings)
    assert len(errors) == 2
    assert {e["rule_id"] for e in errors} == {"JL001", "JL012"}


@pytest.mark.parametrize(
    "rule_id",
    [f"JL{i:03d}" for i in range(1, 13)],
)
def test_filter_jl_purity_errors_accepts_jl001_through_jl012(rule_id: str) -> None:
    findings = [{"rule_id": rule_id, "severity": "error", "message": "ok"}]
    assert len(filter_jl_purity_errors(findings)) == 1


def test_run_jax_purity_gate_passes_clean_tree(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with patch(
        "xtrax.devtools.gates.jax_purity._run_jaxlint_json",
        return_value=[],
    ):
        result = run_jax_purity_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is True
    assert result.violation_count == 0
    assert result.findings_emitted == 0
    assert not audits_path.exists()


def test_run_jax_purity_gate_fails_on_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    mock_findings = [
        {
            "rule_id": "JL005",
            "severity": "error",
            "message": "side effect in jit",
            "path": "src/xtrax/foo.py",
            "line": 10,
        },
        {
            "rule_id": "JL020",
            "severity": "error",
            "message": "perf rule excluded",
            "path": "src/xtrax/bar.py",
            "line": 3,
        },
    ]

    with patch(
        "xtrax.devtools.gates.jax_purity._run_jaxlint_json",
        return_value=mock_findings,
    ):
        result = run_jax_purity_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is False
    assert result.violation_count == 1
    assert result.findings_emitted == 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "jax_purity"
    assert record["payload"]["rule_id"] == "JL005"
    assert record["severity"] == "major"


def test_run_jax_purity_gate_tightens_baseline_on_improvement(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=2.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with patch(
        "xtrax.devtools.gates.jax_purity._run_jaxlint_json",
        return_value=[],
    ):
        result = run_jax_purity_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=True,
        )

    assert result.passed is True
    assert result.baseline_updated is True
    updated = load_baseline(path=baseline_path)
    assert updated.metrics[METRIC_KEY].value == 0.0


def test_committed_baseline_has_jax_purity_metric() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert METRIC_KEY in loaded.metrics
    entry = loaded.metrics[METRIC_KEY]
    assert entry.comparator == "minimize"
    assert entry.value == 0.0


def test_audit_jax_purity_gate_cli_exits_zero_on_clean_tree(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with patch(
        "xtrax.devtools.gates.jax_purity._run_jaxlint_json",
        return_value=[],
    ):
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_jax_purity_gate.py",
                "--baseline-path",
                str(baseline_path),
                "--audits-path",
                str(audits_path),
                "--no-write-baseline",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout

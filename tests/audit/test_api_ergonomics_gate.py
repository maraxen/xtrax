"""Tests for D6 API-ergonomics gate (N2.7 / #1586)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates.api_ergonomics import (
    METRIC_KEY,
    run_api_ergonomics_gate,
    scan_param_sprawl,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"


def _write_fixture_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")


def test_api_ergonomics_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "api_ergonomics.toml")
    assert table.dimension == "api_ergonomics"
    assert len(table.anchors) == 5


def test_scan_param_sprawl_flags_sprawling_public_function(tmp_path: Path) -> None:
    pkg = tmp_path / "sprawl_pkg"
    _write_fixture_module(
        pkg / "sprawl.py",
        """
        def sprawling(a, b, c, d, e, f):
            return a

        def compact(a, b, c=1):
            return a

        def _private(a, b, c, d, e, f, g):
            return a
        """,
    )
    violations = scan_param_sprawl(pkg)
    assert len(violations) == 1
    assert violations[0].qualname == "sprawling"
    assert violations[0].required_count == 6
    assert violations[0].file_line.endswith("sprawl.py:1")


def test_scan_param_sprawl_counts_kwonly_required_params(tmp_path: Path) -> None:
    pkg = tmp_path / "kwonly_pkg"
    _write_fixture_module(
        pkg / "kwonly.py",
        """
        def with_kwonly(a, b, c, d, *, e, f, g):
            return a
        """,
    )
    violations = scan_param_sprawl(pkg)
    assert len(violations) == 1
    assert violations[0].required_count == 7


def test_scan_param_sprawl_flags_public_class_method(tmp_path: Path) -> None:
    pkg = tmp_path / "class_pkg"
    _write_fixture_module(
        pkg / "service.py",
        """
        class Service:
            def run(self, a, b, c, d, e, f):
                return a

            def _hidden(self, a, b, c, d, e, f, g):
                return a
        """,
    )
    violations = scan_param_sprawl(pkg)
    assert len(violations) == 1
    assert violations[0].qualname == "Service.run"
    assert violations[0].required_count == 6


def test_scan_param_sprawl_flags_public_init(tmp_path: Path) -> None:
    pkg = tmp_path / "init_pkg"
    _write_fixture_module(
        pkg / "model.py",
        """
        class Model:
            def __init__(self, a, b, c, d, e, f):
                self.a = a
        """,
    )
    violations = scan_param_sprawl(pkg)
    assert len(violations) == 1
    assert violations[0].qualname == "Model.__init__"
    assert violations[0].required_count == 6


def test_run_api_ergonomics_gate_passes_clean_tree(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    pkg = tmp_path / "clean_pkg"
    _write_fixture_module(
        pkg / "clean.py",
        """
        def ok(a, b, c=1):
            return a
        """,
    )
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

    result = run_api_ergonomics_gate(
        target=pkg,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is True
    assert result.violation_count == 0
    assert result.findings_emitted == 0
    assert not audits_path.exists()


def test_run_api_ergonomics_gate_fails_on_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    pkg = tmp_path / "bad_pkg"
    _write_fixture_module(
        pkg / "bad.py",
        """
        def sprawling(a, b, c, d, e, f):
            return a
        """,
    )
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

    result = run_api_ergonomics_gate(
        target=pkg,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is False
    assert result.violation_count == 1
    assert result.findings_emitted == 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "api_ergonomics"
    assert record["payload"]["rule_id"] == "api_ergonomics.param_sprawl"
    assert record["payload"]["required_count"] == 6
    assert record["severity"] == "minor"


def test_run_api_ergonomics_gate_tightens_baseline_on_improvement(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    pkg = tmp_path / "improved_pkg"
    _write_fixture_module(
        pkg / "clean.py",
        """
        def ok(a, b, c=1):
            return a
        """,
    )
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

    result = run_api_ergonomics_gate(
        target=pkg,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=True,
    )

    assert result.passed is True
    assert result.baseline_updated is True
    updated = load_baseline(path=baseline_path)
    assert updated.metrics[METRIC_KEY].value == 0.0


def test_committed_baseline_has_api_ergonomics_metric() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert METRIC_KEY in loaded.metrics
    entry = loaded.metrics[METRIC_KEY]
    assert entry.comparator == "minimize"
    assert entry.value == 0.0


def test_audit_api_ergonomics_gate_cli_exits_zero_on_clean_tree(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    pkg = tmp_path / "cli_pkg"
    _write_fixture_module(
        pkg / "clean.py",
        """
        def ok(a, b, c=1):
            return a
        """,
    )
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

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_api_ergonomics_gate.py",
            str(pkg),
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

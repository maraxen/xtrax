"""Smoke: audit-deterministic recipe and CI wiring (N5.1)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_justfile_defines_audit_deterministic_recipe() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-deterministic:" in text
    for step in (
        "audit-imports",
        "audit-no-future-annotations",
        "audit-jaxlint",
        "audit-coverage-hygiene",
        "audit-version-wheel",
        "audit-packaging-metadata",
        "audit-public-api",
        "pytest tests/audit/",
        "audit-coverage-dag",
        "audit-bootstrap-dry",
        "audit-ruff-schedule",
        "validate-capability-registry",
        "validate-episodic-memory-contract",
        "audit-graph-auditor",
        "audit-port-emit-contract",
    ):
        assert step in text, f"missing expected step fragment: {step}"


def test_justfile_defines_audit_coverage_tier1() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-coverage-tier1:" in text
    assert "--tier tier1_core --enforce tier1_core" in text


def test_justfile_defines_audit_coverage_tier2() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-coverage-tier2:" in text
    assert "--tier tier2_eda --enforce tier2_eda" in text


def test_ci_yml_runs_audit_deterministic() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "audit-deterministic:" in text
    assert "just audit-deterministic" in text


def test_ci_yml_runs_tier1_coverage_gate() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "just audit-coverage-tier1" in text
    assert "--cov-fail-under=90" not in text


def test_ci_yml_runs_tier2_eda_coverage_gate() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "just audit-coverage-tier2" in text

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
        "pytest tests/audit/",
        "audit-bootstrap-dry",
        "audit-ruff-schedule",
        "validate-capability-registry",
        "validate-episodic-memory-contract",
        "audit-graph-auditor",
        "audit-port-emit-contract",
    ):
        assert step in text, f"missing expected step fragment: {step}"


def test_ci_yml_runs_audit_deterministic() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "audit-deterministic:" in text
    assert "just audit-deterministic" in text

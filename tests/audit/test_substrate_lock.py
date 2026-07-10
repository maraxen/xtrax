"""Tests for T3-02 substrate-lock grep-gate (#3022)."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_substrate_lock import audit_substrate_lock, find_forbidden_marker

CLEAN_TEMPLATE = """\
name: port_validation
nodes:
  - id: oracle_seal
    role: praxia-recon
    prompt: run oracle seal check
"""

STRICT_TEMPLATE = """\
name: sneaky_variant
nodes:
  - id: dispatch
    action_mode: strict
    tool_profile: recon_tools
"""


def test_clean_workflows_dir_passes(tmp_path: Path) -> None:
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "port_validation.yaml").write_text(CLEAN_TEMPLATE, encoding="utf-8")

    passed, hits = audit_substrate_lock(workflows_dir)

    assert passed
    assert hits == []


def test_strict_variant_fails_naming_the_file(tmp_path: Path) -> None:
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "port_validation.yaml").write_text(CLEAN_TEMPLATE, encoding="utf-8")
    strict_path = workflows_dir / "sneaky_variant_contract.yaml"
    strict_path.write_text(STRICT_TEMPLATE, encoding="utf-8")

    passed, hits = audit_substrate_lock(workflows_dir)

    assert not passed
    assert len(hits) == 1
    assert str(strict_path) in hits[0]
    assert hits[0].endswith(":4")


def test_missing_workflows_dir_passes_vacuously(tmp_path: Path) -> None:
    passed, hits = audit_substrate_lock(tmp_path / "does-not-exist")

    assert passed
    assert hits == []


def test_find_forbidden_marker_scans_only_yaml_files(tmp_path: Path) -> None:
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "dw_mapping.toml").write_text(
        '[template_to_js]\naction_mode: strict = "irrelevant-toml-key"\n',
        encoding="utf-8",
    )

    hits = find_forbidden_marker(workflows_dir)

    assert hits == []


def test_real_agent_assets_workflows_dir_is_clean() -> None:
    """Guards the actual repo asset dir the gate protects, not just fixtures."""
    real_dir = Path(__file__).resolve().parents[2] / "agent_assets" / "workflows"

    passed, hits = audit_substrate_lock(real_dir)

    assert passed, f"unexpected 'action_mode: strict' hits: {hits}"

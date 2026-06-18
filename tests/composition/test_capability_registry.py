"""Validate composition-layer capability registry v0.1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from load_capability_registry import load_capability_registry  # noqa: E402

EXPECTED_IDS = frozenset(
    {
        "composer-orchestrator",
        "jax-purity-reviewer",
        "host-prep-fixer",
        "export-bundle-inferrer",
        "graph-auditor",
    }
)


def test_registry_has_five_identities() -> None:
    registry = load_capability_registry()
    assert len(registry.identities) == 5
    assert {ident.id for ident in registry.identities} == EXPECTED_IDS


def test_registry_versions_and_node_metadata() -> None:
    registry = load_capability_registry()
    assert registry.version == "0.1.0"
    assert registry.schema_version == "1"
    assert "nl_description" in registry.node_metadata.required_slots
    assert "bathos_sidecar_ref" in registry.node_metadata.optional_slots


def test_identities_have_skills_and_profiles() -> None:
    registry = load_capability_registry()
    for ident in registry.identities:
        assert ident.skills
        assert ident.mcp_tool_profile in {"orchestration", "audit", "implement"}
        assert ident.kb_sources
        assert ident.description

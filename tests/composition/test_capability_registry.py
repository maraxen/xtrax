"""Validate composition-layer capability registry v0.2."""

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
        "reference-vendor",
        "specification-specialist",
        "test-designer",
    }
)


def test_registry_has_eight_identities() -> None:
    registry = load_capability_registry()
    assert len(registry.identities) == 8
    assert {ident.id for ident in registry.identities} == EXPECTED_IDS


def test_registry_versions_and_node_metadata() -> None:
    registry = load_capability_registry()
    assert registry.version == "0.2.0"
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


def test_port_validation_hook_registered() -> None:
    registry = load_capability_registry()
    assert "port_validation" in registry.hook_schemas
    hook = registry.hook_schemas["port_validation"]
    assert hook.schema_doc == "port/docs/hook_schema_port_validation.md"
    assert hook.workflow == "port_validation"
    assert (ROOT / hook.schema_doc).is_file()


def test_port_identities_skills_and_profiles() -> None:
    registry = load_capability_registry()
    by_id = {ident.id: ident for ident in registry.identities}

    ref_vendor = by_id["reference-vendor"]
    assert ref_vendor.skills == ("jax-port", "using-xtrax")
    assert ref_vendor.mcp_tool_profile == "implement"
    assert ref_vendor.semver == "0.2.0"

    spec_specialist = by_id["specification-specialist"]
    assert spec_specialist.skills == ("using-xtrax", "scientific-writing")
    assert spec_specialist.mcp_tool_profile == "orchestration"
    assert spec_specialist.semver == "0.2.0"

    test_designer = by_id["test-designer"]
    assert test_designer.skills == ("test-driven-development", "using-xtrax")
    assert test_designer.mcp_tool_profile == "audit"
    assert test_designer.semver == "0.2.0"

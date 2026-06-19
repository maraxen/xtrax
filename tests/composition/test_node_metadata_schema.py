"""Contract tests for graph-native node metadata schema."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from load_capability_registry import (  # noqa: E402
    load_capability_registry,
    load_node_metadata_schema,
    validate_node_metadata,
)

VALID_SAMPLE = {
    "nl_description": "Prepare host-side tensors before JIT lowering.",
    "mathjax_label": r"\mathbf{x} \in \mathbb{R}^n",
    "citations": [{"doi": "10.1234/example"}],
    "script_usage": {"language": "python", "excerpt": "x = jnp.asarray(x_host)"},
    "audit_verdict": "PASS",
    "bathos_sidecar_ref": ".praxia/experiments/run_001.toml",
}


def test_schema_loads_with_required_nl_description() -> None:
    schema = load_node_metadata_schema()
    assert schema.version == "0.1.0"
    assert schema.schema_version == "1"
    required = [slot.id for slot in schema.slots if slot.required]
    assert required == ["nl_description"]


def test_valid_sample_passes_validation() -> None:
    validate_node_metadata(VALID_SAMPLE)


def test_missing_required_slot_fails() -> None:
    metadata = {k: v for k, v in VALID_SAMPLE.items() if k != "nl_description"}
    with pytest.raises(ValueError, match="missing required slot 'nl_description'"):
        validate_node_metadata(metadata)


def test_unknown_slot_fails() -> None:
    metadata = {**VALID_SAMPLE, "extra_field": "not allowed"}
    with pytest.raises(ValueError, match="unknown metadata slots"):
        validate_node_metadata(metadata)


def test_registry_slot_ids_match_schema() -> None:
    registry = load_capability_registry()
    schema = registry.node_metadata_schema

    schema_required = {s.id for s in schema.slots if s.required}
    schema_optional = {s.id for s in schema.slots if not s.required}

    assert set(registry.node_metadata.required_slots) == schema_required
    assert set(registry.node_metadata.optional_slots) == schema_optional
    assert schema_required | schema_optional == {slot.id for slot in schema.slots}

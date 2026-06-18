#!/usr/bin/env python3
"""Load and validate the composition-layer capability registry TOML."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / ".praxia" / "composition" / "capability_registry.toml"
DEFAULT_SCHEMA = ROOT / ".praxia" / "composition" / "node_metadata_schema.toml"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[A-Za-z0-9.]+)?$")
KNOWN_KB_SOURCES = frozenset({"transduction", "knowledge", "nlm", "context7"})
KNOWN_MCP_PROFILES = frozenset({"orchestration", "audit", "implement"})
KNOWN_SLOT_TYPES = frozenset({"string", "array", "object", "enum"})


@dataclass(frozen=True)
class Identity:
    id: str
    semver: str
    skills: tuple[str, ...]
    mcp_tool_profile: str
    kb_sources: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class NodeMetadata:
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]


@dataclass(frozen=True)
class SlotDefinition:
    id: str
    required: bool
    type: str
    min_length: int | None = None
    values: tuple[str, ...] | None = None
    fields: dict[str, str] | None = None
    item_type: str | None = None


@dataclass(frozen=True)
class NodeMetadataSchema:
    version: str
    schema_version: str
    slots: tuple[SlotDefinition, ...]


@dataclass(frozen=True)
class CapabilityRegistry:
    version: str
    schema_version: str
    identities: tuple[Identity, ...]
    node_metadata: NodeMetadata
    node_metadata_schema: NodeMetadataSchema


def _require_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing or empty string field '{key}'")
    return value.strip()


def _require_str_list(data: dict[str, Any], key: str, *, context: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: '{key}' must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{context}: '{key}' entries must be non-empty strings")
        out.append(item.strip())
    return out


def _parse_identity(raw: dict[str, Any], index: int) -> Identity:
    ctx = f"identities[{index}]"
    identity_id = _require_str(raw, "id", context=ctx)
    semver = _require_str(raw, "semver", context=ctx)
    if not SEMVER_RE.match(semver):
        raise ValueError(f"{ctx}: invalid semver '{semver}'")

    skills = tuple(_require_str_list(raw, "skills", context=ctx))
    profile = _require_str(raw, "mcp_tool_profile", context=ctx)
    if profile not in KNOWN_MCP_PROFILES:
        raise ValueError(f"{ctx}: unknown mcp_tool_profile '{profile}'")

    kb_sources = tuple(_require_str_list(raw, "kb_sources", context=ctx))
    unknown = set(kb_sources) - KNOWN_KB_SOURCES
    if unknown:
        raise ValueError(f"{ctx}: unknown kb_sources {sorted(unknown)}")

    description = _require_str(raw, "description", context=ctx)
    return Identity(
        id=identity_id,
        semver=semver,
        skills=skills,
        mcp_tool_profile=profile,
        kb_sources=kb_sources,
        description=description,
    )


def _parse_slot(raw: dict[str, Any], index: int) -> SlotDefinition:
    ctx = f"slots[{index}]"
    slot_id = _require_str(raw, "id", context=ctx)
    slot_type = _require_str(raw, "type", context=ctx)
    if slot_type not in KNOWN_SLOT_TYPES:
        raise ValueError(f"{ctx}: unknown slot type '{slot_type}'")

    required = raw.get("required")
    if not isinstance(required, bool):
        raise ValueError(f"{ctx}: 'required' must be a boolean")

    min_length: int | None = None
    if "min_length" in raw:
        min_length_raw = raw["min_length"]
        if not isinstance(min_length_raw, int) or min_length_raw < 1:
            raise ValueError(f"{ctx}: 'min_length' must be a positive integer")
        min_length = min_length_raw

    values: tuple[str, ...] | None = None
    if slot_type == "enum":
        raw_values = raw.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"{ctx}: enum slot requires non-empty 'values' list")
        values = tuple(_require_str_list(raw, "values", context=ctx))

    fields: dict[str, str] | None = None
    if slot_type == "object":
        raw_fields = raw.get("fields")
        if not isinstance(raw_fields, dict) or not raw_fields:
            raise ValueError(f"{ctx}: object slot requires non-empty 'fields' table")
        fields = {}
        for field_name, field_type in raw_fields.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError(f"{ctx}: invalid object field name")
            if field_type != "string":
                raise ValueError(
                    f"{ctx}: unsupported field type '{field_type}' for '{field_name}'"
                )
            fields[field_name.strip()] = field_type

    item_type: str | None = None
    if slot_type == "array":
        item_type = _require_str(raw, "item_type", context=ctx)
        if item_type != "citation":
            raise ValueError(f"{ctx}: unknown array item_type '{item_type}'")

    return SlotDefinition(
        id=slot_id,
        required=required,
        type=slot_type,
        min_length=min_length,
        values=values,
        fields=fields,
        item_type=item_type,
    )


def load_node_metadata_schema(path: Path | None = None) -> NodeMetadataSchema:
    schema_path = path or DEFAULT_SCHEMA
    data = tomllib.loads(schema_path.read_text(encoding="utf-8"))

    schema_raw = data.get("schema")
    if not isinstance(schema_raw, dict):
        raise ValueError("missing [schema] table")

    version = _require_str(schema_raw, "version", context="schema")
    if not SEMVER_RE.match(version):
        raise ValueError(f"schema: invalid semver '{version}'")
    schema_version = _require_str(schema_raw, "schema_version", context="schema")

    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("slots must be a non-empty list")

    slots = tuple(_parse_slot(item, idx) for idx, item in enumerate(raw_slots))
    slot_ids = [slot.id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("duplicate slot ids in node_metadata_schema")

    return NodeMetadataSchema(
        version=version,
        schema_version=schema_version,
        slots=slots,
    )


def _align_registry_slots_with_schema(
    node_metadata: NodeMetadata, schema: NodeMetadataSchema
) -> None:
    schema_required = {s.id for s in schema.slots if s.required}
    schema_optional = {s.id for s in schema.slots if not s.required}
    schema_all = schema_required | schema_optional

    registry_required = set(node_metadata.required_slots)
    registry_optional = set(node_metadata.optional_slots)
    registry_all = registry_required | registry_optional

    if registry_required != schema_required:
        raise ValueError(
            "node_metadata required_slots do not match schema required slots: "
            f"registry={sorted(registry_required)} schema={sorted(schema_required)}"
        )
    if registry_optional != schema_optional:
        raise ValueError(
            "node_metadata optional_slots do not match schema optional slots: "
            f"registry={sorted(registry_optional)} schema={sorted(schema_optional)}"
        )
    if registry_all != schema_all:
        raise ValueError(
            "node_metadata slot ids drift from schema: "
            f"registry_only={sorted(registry_all - schema_all)} "
            f"schema_only={sorted(schema_all - registry_all)}"
        )


def _validate_citation(item: Any, *, context: str) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{context}: citation must be an object")
    doi = item.get("doi")
    url = item.get("url")
    text = item.get("text")
    has_doi = isinstance(doi, str) and doi.strip()
    has_url = isinstance(url, str) and url.strip()
    has_text = isinstance(text, str) and text.strip()
    if not (has_doi or has_url or has_text):
        raise ValueError(f"{context}: citation requires at least one of doi, url, text")


def _validate_slot_value(slot: SlotDefinition, value: Any, *, context: str) -> None:
    if slot.type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{context}: expected string")
        if slot.min_length is not None and len(value.strip()) < slot.min_length:
            raise ValueError(
                f"{context}: string shorter than min_length {slot.min_length}"
            )
        return

    if slot.type == "enum":
        if not isinstance(value, str) or value not in (slot.values or ()):
            allowed = ", ".join(slot.values or ())
            raise ValueError(f"{context}: expected one of [{allowed}]")
        return

    if slot.type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{context}: expected array")
        if slot.item_type == "citation":
            for idx, item in enumerate(value):
                _validate_citation(item, context=f"{context}[{idx}]")
        return

    if slot.type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{context}: expected object")
        if slot.fields:
            for field_name in slot.fields:
                field_value = value.get(field_name)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(
                        f"{context}.{field_name}: expected non-empty string"
                    )
        return

    raise ValueError(f"{context}: unsupported slot type '{slot.type}'")


def validate_node_metadata(
    metadata: dict[str, Any],
    schema: NodeMetadataSchema | None = None,
) -> None:
    """Validate a node metadata dict against the typed schema."""
    resolved = schema or load_node_metadata_schema()
    slots_by_id = {slot.id: slot for slot in resolved.slots}
    known_ids = set(slots_by_id)

    unknown = set(metadata) - known_ids
    if unknown:
        raise ValueError(f"unknown metadata slots: {sorted(unknown)}")

    for slot in resolved.slots:
        if slot.required and slot.id not in metadata:
            raise ValueError(f"missing required slot '{slot.id}'")

    for key, value in metadata.items():
        _validate_slot_value(slots_by_id[key], value, context=key)


def load_capability_registry(path: Path | None = None) -> CapabilityRegistry:
    registry_path = path or DEFAULT_REGISTRY
    data = tomllib.loads(registry_path.read_text(encoding="utf-8"))

    registry = data.get("registry")
    if not isinstance(registry, dict):
        raise ValueError("missing [registry] table")

    version = _require_str(registry, "version", context="registry")
    schema_version = _require_str(registry, "schema_version", context="registry")

    raw_identities = data.get("identities")
    if not isinstance(raw_identities, list) or not raw_identities:
        raise ValueError("identities must be a non-empty list")

    identities = tuple(
        _parse_identity(item, idx) for idx, item in enumerate(raw_identities)
    )
    ids = [ident.id for ident in identities]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate identity ids in registry")

    node_raw = data.get("node_metadata")
    if not isinstance(node_raw, dict):
        raise ValueError("missing [node_metadata] table")

    required_slots = tuple(
        _require_str_list(node_raw, "required_slots", context="node_metadata")
    )
    optional_slots = tuple(
        _require_str_list(node_raw, "optional_slots", context="node_metadata")
    )
    overlap = set(required_slots) & set(optional_slots)
    if overlap:
        raise ValueError(f"node_metadata slot overlap: {sorted(overlap)}")

    node_metadata = NodeMetadata(
        required_slots=required_slots,
        optional_slots=optional_slots,
    )
    node_metadata_schema = load_node_metadata_schema()
    _align_registry_slots_with_schema(node_metadata, node_metadata_schema)

    return CapabilityRegistry(
        version=version,
        schema_version=schema_version,
        identities=identities,
        node_metadata=node_metadata,
        node_metadata_schema=node_metadata_schema,
    )


def main() -> None:
    registry = load_capability_registry()
    print(f"registry v{registry.version} ({len(registry.identities)} identities)")
    print(
        f"node metadata schema v{registry.node_metadata_schema.version} "
        f"({len(registry.node_metadata_schema.slots)} slots)"
    )


if __name__ == "__main__":
    main()

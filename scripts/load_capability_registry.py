#!/usr/bin/env python3
"""Load and validate the composition-layer capability registry TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.composition.node_metadata import (
    SEMVER_RE,
    NodeMetadataSchema,
    SlotDefinition,
    load_node_metadata_schema,
    validate_node_metadata,
)

__all__ = [
    "NodeMetadataSchema",
    "SlotDefinition",
    "load_capability_registry",
    "load_node_metadata_schema",
    "validate_node_metadata",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / ".praxia" / "composition" / "capability_registry.toml"
KNOWN_KB_SOURCES = frozenset({"transduction", "knowledge", "nlm", "context7"})
KNOWN_MCP_PROFILES = frozenset({"orchestration", "audit", "implement"})


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
class HookSchema:
    id: str
    schema_doc: str
    workflow: str


@dataclass(frozen=True)
class CapabilityRegistry:
    version: str
    schema_version: str
    identities: tuple[Identity, ...]
    node_metadata: NodeMetadata
    node_metadata_schema: NodeMetadataSchema
    hook_schemas: dict[str, HookSchema]


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


def _parse_hook_schema(hook_id: str, raw: dict[str, Any]) -> HookSchema:
    ctx = f"hooks.subagent_stop.{hook_id}"
    schema_doc = _require_str(raw, "schema_doc", context=ctx)
    workflow = _require_str(raw, "workflow", context=ctx)
    schema_path = ROOT / schema_doc
    if not schema_path.is_file():
        raise ValueError(f"{ctx}: schema_doc path does not exist: {schema_doc}")
    return HookSchema(id=hook_id, schema_doc=schema_doc, workflow=workflow)


def _parse_hook_schemas(data: dict[str, Any]) -> dict[str, HookSchema]:
    hooks_raw = data.get("hooks")
    if hooks_raw is None:
        return {}
    if not isinstance(hooks_raw, dict):
        raise ValueError("hooks must be a table")

    subagent_stop = hooks_raw.get("subagent_stop")
    if subagent_stop is None:
        return {}
    if not isinstance(subagent_stop, dict):
        raise ValueError("hooks.subagent_stop must be a table")

    hook_schemas: dict[str, HookSchema] = {}
    for hook_id, hook_raw in subagent_stop.items():
        if not isinstance(hook_id, str) or not hook_id.strip():
            raise ValueError("hooks.subagent_stop keys must be non-empty strings")
        if not isinstance(hook_raw, dict):
            raise ValueError(f"hooks.subagent_stop.{hook_id} must be a table")
        hook_schemas[hook_id.strip()] = _parse_hook_schema(hook_id.strip(), hook_raw)

    return hook_schemas


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

    identities = tuple(_parse_identity(item, idx) for idx, item in enumerate(raw_identities))
    ids = [ident.id for ident in identities]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate identity ids in registry")

    node_raw = data.get("node_metadata")
    if not isinstance(node_raw, dict):
        raise ValueError("missing [node_metadata] table")

    required_slots = tuple(_require_str_list(node_raw, "required_slots", context="node_metadata"))
    optional_slots = tuple(_require_str_list(node_raw, "optional_slots", context="node_metadata"))
    overlap = set(required_slots) & set(optional_slots)
    if overlap:
        raise ValueError(f"node_metadata slot overlap: {sorted(overlap)}")

    node_metadata = NodeMetadata(
        required_slots=required_slots,
        optional_slots=optional_slots,
    )
    node_metadata_schema = load_node_metadata_schema()
    _align_registry_slots_with_schema(node_metadata, node_metadata_schema)
    hook_schemas = _parse_hook_schemas(data)

    return CapabilityRegistry(
        version=version,
        schema_version=schema_version,
        identities=identities,
        node_metadata=node_metadata,
        node_metadata_schema=node_metadata_schema,
        hook_schemas=hook_schemas,
    )


def main() -> None:
    registry = load_capability_registry()
    print(f"registry v{registry.version} ({len(registry.identities)} identities)")
    print(
        f"node metadata schema v{registry.node_metadata_schema.version} "
        f"({len(registry.node_metadata_schema.slots)} slots)"
    )
    if registry.hook_schemas:
        print(f"hook schemas ({len(registry.hook_schemas)}): {', '.join(registry.hook_schemas)}")


if __name__ == "__main__":
    main()

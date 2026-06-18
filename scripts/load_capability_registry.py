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
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[A-Za-z0-9.]+)?$")
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
class CapabilityRegistry:
    version: str
    schema_version: str
    identities: tuple[Identity, ...]
    node_metadata: NodeMetadata


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

    return CapabilityRegistry(
        version=version,
        schema_version=schema_version,
        identities=identities,
        node_metadata=NodeMetadata(
            required_slots=required_slots,
            optional_slots=optional_slots,
        ),
    )


def main() -> None:
    registry = load_capability_registry()
    print(f"registry v{registry.version} ({len(registry.identities)} identities)")


if __name__ == "__main__":
    main()

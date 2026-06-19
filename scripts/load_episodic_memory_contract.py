#!/usr/bin/env python3
"""Load and validate the composition-layer episodic memory contract TOML."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / ".praxia" / "composition" / "episodic_memory_contract.toml"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[A-Za-z0-9.]+)?$")
KNOWN_CHANNELS = frozenset({"recon", "plan", "audit", "research", "daily"})
CHANNEL_APPEND_TOOLS = {
    "recon": "append_recon",
    "plan": "append_plan",
    "audit": "append_audit",
    "research": "append_research",
    "daily": "append_daily",
}
KNOWN_QUERY_TOOL = "transduction_query"
KNOWN_IDENTITY_IDS = frozenset({"composer-orchestrator", "graph-auditor"})
KNOWN_KB_SOURCES = frozenset({"transduction", "knowledge", "nlm", "context7"})
KNOWN_REFRESH_POLICIES = frozenset({"epic_boundary_or_handoff", "session_handoff_only"})


@dataclass(frozen=True)
class TransductionChannel:
    id: str
    jsonl_path: str
    query_tool: str
    append_tool: str


@dataclass(frozen=True)
class NlmBinding:
    notebook_id: str
    tag_pattern: str
    refresh_policy: str


@dataclass(frozen=True)
class SessionRules:
    task_id_format: str
    handoff_path: str
    staleness_max_days: int


@dataclass(frozen=True)
class IdentityDefaults:
    identity_id: str
    kb_sources: tuple[str, ...]


@dataclass(frozen=True)
class EpisodicMemoryContract:
    version: str
    schema_version: str
    channels: tuple[TransductionChannel, ...]
    nlm_bindings: tuple[NlmBinding, ...]
    session_rules: SessionRules
    identity_defaults: tuple[IdentityDefaults, ...]


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


def _parse_channel(raw: dict[str, Any], index: int) -> TransductionChannel:
    ctx = f"channels[{index}]"
    channel_id = _require_str(raw, "id", context=ctx)
    if channel_id not in KNOWN_CHANNELS:
        raise ValueError(f"{ctx}: unknown channel id '{channel_id}'")

    jsonl_path = _require_str(raw, "jsonl_path", context=ctx)
    query_tool = _require_str(raw, "query_tool", context=ctx)
    if query_tool != KNOWN_QUERY_TOOL:
        raise ValueError(f"{ctx}: unknown query_tool '{query_tool}'")

    append_tool = _require_str(raw, "append_tool", context=ctx)
    expected_append = CHANNEL_APPEND_TOOLS[channel_id]
    if append_tool != expected_append:
        raise ValueError(
            f"{ctx}: append_tool must be '{expected_append}', got '{append_tool}'"
        )

    return TransductionChannel(
        id=channel_id,
        jsonl_path=jsonl_path,
        query_tool=query_tool,
        append_tool=append_tool,
    )


def _parse_nlm_binding(raw: dict[str, Any], index: int) -> NlmBinding:
    ctx = f"nlm_bindings[{index}]"
    notebook_id = _require_str(raw, "notebook_id", context=ctx)
    tag_pattern = _require_str(raw, "tag_pattern", context=ctx)
    refresh_policy = _require_str(raw, "refresh_policy", context=ctx)
    if refresh_policy not in KNOWN_REFRESH_POLICIES:
        raise ValueError(f"{ctx}: unknown refresh_policy '{refresh_policy}'")
    return NlmBinding(
        notebook_id=notebook_id,
        tag_pattern=tag_pattern,
        refresh_policy=refresh_policy,
    )


def _parse_session_rules(raw: dict[str, Any]) -> SessionRules:
    ctx = "session_rules"
    task_id_format = _require_str(raw, "task_id_format", context=ctx)
    handoff_path = _require_str(raw, "handoff_path", context=ctx)
    staleness_raw = raw.get("staleness_max_days")
    if not isinstance(staleness_raw, int) or staleness_raw < 1:
        raise ValueError(f"{ctx}: staleness_max_days must be a positive integer")
    return SessionRules(
        task_id_format=task_id_format,
        handoff_path=handoff_path,
        staleness_max_days=staleness_raw,
    )


def _parse_identity_defaults(raw: dict[str, Any], index: int) -> IdentityDefaults:
    ctx = f"identity_defaults[{index}]"
    identity_id = _require_str(raw, "identity_id", context=ctx)
    if identity_id not in KNOWN_IDENTITY_IDS:
        raise ValueError(f"{ctx}: unknown identity_id '{identity_id}'")

    kb_sources = tuple(_require_str_list(raw, "kb_sources", context=ctx))
    unknown = set(kb_sources) - KNOWN_KB_SOURCES
    if unknown:
        raise ValueError(f"{ctx}: unknown kb_sources {sorted(unknown)}")

    return IdentityDefaults(identity_id=identity_id, kb_sources=kb_sources)


def load_episodic_memory_contract(path: Path | None = None) -> EpisodicMemoryContract:
    contract_path = path or DEFAULT_CONTRACT
    data = tomllib.loads(contract_path.read_text(encoding="utf-8"))

    contract_raw = data.get("contract")
    if not isinstance(contract_raw, dict):
        raise ValueError("missing [contract] table")

    version = _require_str(contract_raw, "version", context="contract")
    if not SEMVER_RE.match(version):
        raise ValueError(f"contract: invalid semver '{version}'")
    schema_version = _require_str(contract_raw, "schema_version", context="contract")

    raw_channels = data.get("channels")
    if not isinstance(raw_channels, list) or not raw_channels:
        raise ValueError("channels must be a non-empty list")

    channels = tuple(_parse_channel(item, idx) for idx, item in enumerate(raw_channels))
    channel_ids = [channel.id for channel in channels]
    if len(channel_ids) != len(set(channel_ids)):
        raise ValueError("duplicate channel ids in contract")
    missing_channels = KNOWN_CHANNELS - set(channel_ids)
    if missing_channels:
        raise ValueError(f"missing required channels: {sorted(missing_channels)}")

    raw_nlm_bindings = data.get("nlm_bindings")
    if not isinstance(raw_nlm_bindings, list) or not raw_nlm_bindings:
        raise ValueError("nlm_bindings must be a non-empty list")
    nlm_bindings = tuple(
        _parse_nlm_binding(item, idx) for idx, item in enumerate(raw_nlm_bindings)
    )

    session_rules_raw = data.get("session_rules")
    if not isinstance(session_rules_raw, dict):
        raise ValueError("missing [session_rules] table")
    session_rules = _parse_session_rules(session_rules_raw)

    raw_identity_defaults = data.get("identity_defaults")
    if not isinstance(raw_identity_defaults, list) or not raw_identity_defaults:
        raise ValueError("identity_defaults must be a non-empty list")
    identity_defaults = tuple(
        _parse_identity_defaults(item, idx)
        for idx, item in enumerate(raw_identity_defaults)
    )
    identity_ids = [item.identity_id for item in identity_defaults]
    if len(identity_ids) != len(set(identity_ids)):
        raise ValueError("duplicate identity_defaults ids in contract")
    missing_identities = KNOWN_IDENTITY_IDS - set(identity_ids)
    if missing_identities:
        raise ValueError(
            f"missing required identity_defaults: {sorted(missing_identities)}"
        )

    return EpisodicMemoryContract(
        version=version,
        schema_version=schema_version,
        channels=channels,
        nlm_bindings=nlm_bindings,
        session_rules=session_rules,
        identity_defaults=identity_defaults,
    )


def main() -> None:
    contract = load_episodic_memory_contract()
    print(
        f"episodic memory contract v{contract.version} "
        f"({len(contract.channels)} channels, "
        f"{len(contract.nlm_bindings)} nlm bindings)"
    )
    print(
        f"session rules: task_id={contract.session_rules.task_id_format}, "
        f"staleness={contract.session_rules.staleness_max_days}d"
    )
    for ident in contract.identity_defaults:
        print(f"identity {ident.identity_id}: kb_sources={list(ident.kb_sources)}")


if __name__ == "__main__":
    main()

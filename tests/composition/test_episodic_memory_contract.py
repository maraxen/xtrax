"""Validate composition-layer episodic memory contract v0.1."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from load_episodic_memory_contract import load_episodic_memory_contract  # noqa: E402

EXPECTED_CHANNEL_IDS = frozenset({"recon", "plan", "audit", "research", "daily"})
INVALID_CONTRACT_TOML = """\
[contract]
version = "not-semver"
schema_version = "1"

[[channels]]
id = "recon"
jsonl_path = ""
query_tool = "transduction_query"
append_tool = "append_recon"
"""


def test_contract_loads_with_five_channels() -> None:
    contract = load_episodic_memory_contract()
    assert contract.version == "0.1.0"
    assert contract.schema_version == "1"
    assert {channel.id for channel in contract.channels} == EXPECTED_CHANNEL_IDS


def test_channels_have_tools_and_jsonl_paths() -> None:
    contract = load_episodic_memory_contract()
    for channel in contract.channels:
        assert channel.jsonl_path
        assert channel.query_tool == "transduction_query"
        assert channel.append_tool.startswith("append_")


def test_session_rules_and_nlm_binding() -> None:
    contract = load_episodic_memory_contract()
    assert contract.session_rules.task_id_format == "YYMMDD_<slug>"
    assert contract.session_rules.handoff_path == ".praxia/handoffs/"
    assert contract.session_rules.staleness_max_days == 30
    assert len(contract.nlm_bindings) == 1
    assert contract.nlm_bindings[0].refresh_policy == "epic_boundary_or_handoff"


def test_identity_defaults_match_capability_registry() -> None:
    contract = load_episodic_memory_contract()
    by_id = {item.identity_id: item for item in contract.identity_defaults}
    assert set(by_id) == {"composer-orchestrator", "graph-auditor"}
    assert by_id["composer-orchestrator"].kb_sources == ("transduction", "knowledge")
    assert by_id["graph-auditor"].kb_sources == ("transduction",)


def test_committed_jsonl_paths_exist() -> None:
    contract = load_episodic_memory_contract()
    for channel in contract.channels:
        assert (ROOT / channel.jsonl_path).is_file(), channel.jsonl_path


def test_rejects_invalid_fixture(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_episodic_memory_contract.toml"
    invalid_path.write_text(INVALID_CONTRACT_TOML, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid semver"):
        load_episodic_memory_contract(invalid_path)

"""Tests for the scope-expansion gate (T2-31, #2181, AC-24)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xtrax.loop.scope_expansion_gate import (
    ApprovalExpiredError,
    NoMatchingApprovalError,
    ScopeExpansionNotApprovedError,
    assert_scope_expansion_approved,
)
from xtrax.run.freshness import Attestation


def _fresh_attestation(now: datetime | None = None) -> dict:
    """Build a fresh attestation dict for inclusion in TOML."""
    attested_at = (now or datetime.now(UTC)).isoformat()
    return {
        "attested_at": attested_at,
        "ttl_days": 30.0,
        "attested_by": "Marielle Russo",
        "note": "Test approval",
    }


def _stale_attestation() -> dict:
    """Build a stale attestation dict for inclusion in TOML."""
    return {
        "attested_at": "2000-01-01T00:00:00+00:00",
        "ttl_days": 1.0,
        "attested_by": "Marielle Russo",
        "note": "Test approval (stale)",
    }


def _write_gates_toml(path: Path, gates: list[dict]) -> None:
    """Helper to write a gates TOML file. Minimal TOML format."""
    lines = []
    for gate in gates:
        lines.append("[[gates]]")
        for key, value in gate.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, float):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    path.write_text("\n".join(lines))


class TestAssertScopeExpansionApproved:
    def test_returns_attestation_when_matching_fresh_entry_exists(self, tmp_path: Path) -> None:
        """Matching entry with fresh attestation → returns Attestation."""
        gates_toml = tmp_path / "gates.toml"
        fresh_att = _fresh_attestation()
        gates = [
            {
                "id": "T2-31",
                "event_ref": "network:pypi.org",
                **fresh_att,
            }
        ]
        _write_gates_toml(gates_toml, gates)

        result = assert_scope_expansion_approved("network:pypi.org", toml_path=gates_toml)

        assert isinstance(result, Attestation)
        assert result.attested_at == fresh_att["attested_at"]
        assert result.ttl_days == fresh_att["ttl_days"]
        assert result.attested_by == fresh_att["attested_by"]
        assert result.note == fresh_att["note"]

    def test_raises_no_matching_approval_error_when_event_ref_differs(self, tmp_path: Path) -> None:
        """Entry exists for id="T2-31" but event_ref doesn't match → NoMatchingApprovalError."""
        gates_toml = tmp_path / "gates.toml"
        gates = [
            {
                "id": "T2-31",
                "event_ref": "network:pypi.org",
                **_fresh_attestation(),
            }
        ]
        _write_gates_toml(gates_toml, gates)

        with pytest.raises(NoMatchingApprovalError, match="no matching T2-31 approval found"):
            assert_scope_expansion_approved("network:different.com", toml_path=gates_toml)

    def test_raises_approval_expired_error_when_ttl_expired(self, tmp_path: Path) -> None:
        """Matching entry but TTL expired → ApprovalExpiredError."""
        gates_toml = tmp_path / "gates.toml"
        gates = [
            {
                "id": "T2-31",
                "event_ref": "tool:new_verb",
                **_stale_attestation(),
            }
        ]
        _write_gates_toml(gates_toml, gates)

        with pytest.raises(ApprovalExpiredError, match="approval.*is stale/expired"):
            assert_scope_expansion_approved("tool:new_verb", toml_path=gates_toml)

    def test_raises_no_matching_approval_when_no_t2_31_entries(self, tmp_path: Path) -> None:
        """No id="T2-31" entries at all → NoMatchingApprovalError."""
        gates_toml = tmp_path / "gates.toml"
        gates = [
            {
                "id": "T2-28",  # Different gate
                "event_ref": "constitution",
                **_fresh_attestation(),
            }
        ]
        _write_gates_toml(gates_toml, gates)

        with pytest.raises(NoMatchingApprovalError, match="no matching T2-31 approval found"):
            assert_scope_expansion_approved("network:anything.org", toml_path=gates_toml)

    def test_raises_no_matching_approval_when_toml_file_not_found(self, tmp_path: Path) -> None:
        """TOML file doesn't exist → NoMatchingApprovalError (not FileNotFoundError)."""
        nonexistent = tmp_path / "nonexistent.toml"

        with pytest.raises(NoMatchingApprovalError, match="gates TOML not found"):
            assert_scope_expansion_approved("network:pypi.org", toml_path=nonexistent)

    def test_raises_no_matching_approval_on_toml_parse_error(self, tmp_path: Path) -> None:
        """Invalid TOML → NoMatchingApprovalError (parse error is not a caller error)."""
        gates_toml = tmp_path / "gates.toml"
        gates_toml.write_text("invalid [[ toml syntax [[")

        with pytest.raises(NoMatchingApprovalError, match="failed to parse gates TOML"):
            assert_scope_expansion_approved("network:pypi.org", toml_path=gates_toml)

    def test_exception_hierarchy(self) -> None:
        """NoMatchingApprovalError and ApprovalExpiredError are both subclasses of base."""
        assert issubclass(NoMatchingApprovalError, ScopeExpansionNotApprovedError)
        assert issubclass(ApprovalExpiredError, ScopeExpansionNotApprovedError)

    def test_multiple_t2_31_entries_finds_correct_one(self, tmp_path: Path) -> None:
        """Multiple T2-31 entries → finds the one matching event_ref."""
        gates_toml = tmp_path / "gates.toml"
        gates = [
            {
                "id": "T2-31",
                "event_ref": "network:pypi.org",
                **_stale_attestation(),
            },
            {
                "id": "T2-31",
                "event_ref": "tool:new_verb",
                **_fresh_attestation(),
            },
            {
                "id": "T2-31",
                "event_ref": "network:github.com",
                **_stale_attestation(),
            },
        ]
        _write_gates_toml(gates_toml, gates)

        # Should find and return the fresh one for "tool:new_verb"
        result = assert_scope_expansion_approved("tool:new_verb", toml_path=gates_toml)
        assert result.attested_by == "Marielle Russo"

    def test_missing_optional_note_field(self, tmp_path: Path) -> None:
        """Attestation without optional 'note' field → defaults to empty string."""
        gates_toml = tmp_path / "gates.toml"
        att = _fresh_attestation()
        del att["note"]
        gates = [
            {
                "id": "T2-31",
                "event_ref": "network:pypi.org",
                **att,
            }
        ]
        _write_gates_toml(gates_toml, gates)

        result = assert_scope_expansion_approved("network:pypi.org", toml_path=gates_toml)
        assert result.note == ""

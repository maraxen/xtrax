"""Tests for the campaign-approval gate (T2-32, #2181, AC-25)."""

from datetime import UTC, datetime

import pytest

from xtrax.devtools.freshness import Attestation
from xtrax.loop.campaign_approval_gate import (
    ApprovalExpiredError,
    CampaignNotApprovedError,
    NoMatchingApprovalError,
    assert_campaign_approved,
)


def _fresh_attestation(now: datetime | None = None) -> dict:
    """Build a fresh gate entry dict for use in TOML."""
    ref_time = now or datetime.now(UTC)
    return {
        "id": "T2-32",
        "event_ref": "campaign_001_initial",
        "attested_at": ref_time.isoformat(),
        "ttl_days": 30.0,
        "attested_by": "Marielle Russo",
        "note": "Approval granted for campaign start.",
    }


def _expired_attestation() -> dict:
    """Build an expired gate entry dict."""
    return {
        "id": "T2-32",
        "event_ref": "campaign_001_initial",
        "attested_at": "2000-01-01T00:00:00Z",
        "ttl_days": 1.0,
        "attested_by": "Marielle Russo",
        "note": "Old approval (expired).",
    }


class TestAssertCampaignApprovedHappyPath:
    """Matching fresh entry returns Attestation, no raise."""

    def test_fresh_approval_returns_attestation(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        entry = _fresh_attestation()
        toml_content = f"""
[[gates]]
id = "T2-32"
event_ref = "{entry["event_ref"]}"
attested_at = "{entry["attested_at"]}"
ttl_days = {entry["ttl_days"]}
attested_by = "{entry["attested_by"]}"
note = "{entry["note"]}"
"""
        gates_file.write_text(toml_content)

        attestation = assert_campaign_approved(
            "campaign_001_initial", toml_path=gates_file, now=datetime.now(UTC)
        )

        assert isinstance(attestation, Attestation)
        assert attestation.attested_by == "Marielle Russo"
        assert attestation.ttl_days == 30.0


class TestAssertCampaignApprovedDifferentEventRef:
    """Entry exists for id=T2-32 but different event_ref -> NoMatchingApprovalError."""

    def test_different_event_ref_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        entry = _fresh_attestation()
        toml_content = f"""
[[gates]]
id = "T2-32"
event_ref = "{entry["event_ref"]}"
attested_at = "{entry["attested_at"]}"
ttl_days = {entry["ttl_days"]}
attested_by = "{entry["attested_by"]}"
note = "{entry["note"]}"
"""
        gates_file.write_text(toml_content)

        with pytest.raises(NoMatchingApprovalError, match="no T2-32 approval found") as exc_info:
            assert_campaign_approved(
                "campaign_002_different", toml_path=gates_file, now=datetime.now(UTC)
            )
        assert "campaign_002_different" in str(exc_info.value)


class TestAssertCampaignApprovedExpiredApproval:
    """Matching entry but TTL expired -> ApprovalExpiredError with reasons."""

    def test_expired_approval_raises_approval_expired_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        entry = _expired_attestation()
        toml_content = f"""
[[gates]]
id = "T2-32"
event_ref = "{entry["event_ref"]}"
attested_at = "{entry["attested_at"]}"
ttl_days = {entry["ttl_days"]}
attested_by = "{entry["attested_by"]}"
note = "{entry["note"]}"
"""
        gates_file.write_text(toml_content)

        with pytest.raises(ApprovalExpiredError, match="approval expired or not fresh") as exc_info:
            assert_campaign_approved(
                "campaign_001_initial", toml_path=gates_file, now=datetime.now(UTC)
            )
        # Verify the error message includes attestation age info
        assert "past TTL" in str(exc_info.value)


class TestAssertCampaignApprovedNoT2_32Entries:
    """No id=T2-32 entries at all -> NoMatchingApprovalError."""

    def test_no_t2_32_entries_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        toml_content = """
[[gates]]
id = "T2-28"
attested_at = "2026-07-14T00:00:00Z"
ttl_days = 365
attested_by = "Marielle Russo"
note = "Constitution authorship gate (different gate, not T2-32)"
"""
        gates_file.write_text(toml_content)

        with pytest.raises(NoMatchingApprovalError, match="no T2-32 approval found"):
            assert_campaign_approved(
                "campaign_001_initial", toml_path=gates_file, now=datetime.now(UTC)
            )


class TestAssertCampaignApprovedFileMissing:
    """TOML file doesn't exist -> NoMatchingApprovalError (not FileNotFoundError)."""

    def test_missing_gates_file_raises_no_matching_approval_error(self, tmp_path) -> None:
        missing_file = tmp_path / "nonexistent.toml"

        with pytest.raises(NoMatchingApprovalError, match="gates file not found"):
            assert_campaign_approved("campaign_001_initial", toml_path=missing_file)


class TestAssertCampaignApprovedErrorHierarchy:
    """All errors are CampaignNotApprovedError subclasses."""

    def test_no_matching_approval_error_is_subclass(self) -> None:
        assert issubclass(NoMatchingApprovalError, CampaignNotApprovedError)

    def test_approval_expired_error_is_subclass(self) -> None:
        assert issubclass(ApprovalExpiredError, CampaignNotApprovedError)

    def test_catching_base_class_catches_all_subclasses(self, tmp_path) -> None:
        missing_file = tmp_path / "nonexistent.toml"
        with pytest.raises(CampaignNotApprovedError):
            assert_campaign_approved("campaign_001_initial", toml_path=missing_file)


class TestAssertCampaignApprovedEdgeCases:
    """Edge cases: malformed TOML, empty gates list, etc."""

    def test_malformed_toml_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "bad.toml"
        gates_file.write_text("this is not valid toml [[[")

        with pytest.raises(NoMatchingApprovalError, match="failed to parse gates file"):
            assert_campaign_approved("campaign_001_initial", toml_path=gates_file)

    def test_empty_gates_list_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        gates_file.write_text("# Empty gates file\n")

        with pytest.raises(NoMatchingApprovalError, match="no T2-32 approval found"):
            assert_campaign_approved("campaign_001_initial", toml_path=gates_file)

    def test_gates_not_a_list_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        gates_file.write_text("gates = {}")

        with pytest.raises(NoMatchingApprovalError, match="gates TOML entry is not a list"):
            assert_campaign_approved("campaign_001_initial", toml_path=gates_file)


class TestAssertCampaignApprovedMultipleEntries:
    """Multiple T2-32 entries: only the matching event_ref counts."""

    def test_selects_correct_entry_among_multiple_t2_32_entries(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        now = datetime.now(UTC)
        ref_time_iso = now.isoformat()
        toml_content = f"""
[[gates]]
id = "T2-32"
event_ref = "campaign_aaa"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "First campaign approval"

[[gates]]
id = "T2-32"
event_ref = "campaign_bbb"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Second campaign approval"

[[gates]]
id = "T2-32"
event_ref = "campaign_ccc"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Third campaign approval"
"""
        gates_file.write_text(toml_content)

        attestation = assert_campaign_approved("campaign_bbb", toml_path=gates_file, now=now)

        assert attestation.note == "Second campaign approval"

    def test_ignores_other_gate_types(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        now = datetime.now(UTC)
        ref_time_iso = now.isoformat()
        toml_content = f"""
[[gates]]
id = "T2-28"
attested_at = "{ref_time_iso}"
ttl_days = 365
attested_by = "Marielle Russo"
note = "Constitution gate"

[[gates]]
id = "T2-32"
event_ref = "campaign_approve_this"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Campaign gate"
"""
        gates_file.write_text(toml_content)

        attestation = assert_campaign_approved(
            "campaign_approve_this", toml_path=gates_file, now=now
        )

        assert attestation.note == "Campaign gate"

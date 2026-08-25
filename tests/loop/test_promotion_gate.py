"""Tests for the promotion-to-main gate (T2-30, #2181, AC-23)."""

from datetime import UTC, datetime

import pytest

from xtrax.loop.promotion_gate import (
    ApprovalExpiredError,
    NoMatchingApprovalError,
    PromotionNotApprovedError,
    assert_promotion_approved,
)
from xtrax.run.freshness import Attestation


def _fresh_attestation(now: datetime | None = None) -> dict:
    """Build a fresh gate entry dict for use in TOML."""
    ref_time = now or datetime.now(UTC)
    return {
        "id": "T2-30",
        "event_ref": "abc123def456",
        "attested_at": ref_time.isoformat(),
        "ttl_days": 30.0,
        "attested_by": "Marielle Russo",
        "note": "Approval granted after engineering review.",
    }


def _expired_attestation() -> dict:
    """Build an expired gate entry dict."""
    return {
        "id": "T2-30",
        "event_ref": "abc123def456",
        "attested_at": "2000-01-01T00:00:00Z",
        "ttl_days": 1.0,
        "attested_by": "Marielle Russo",
        "note": "Old approval (expired).",
    }


class TestAssertPromotionApprovedHappyPath:
    """Matching fresh entry returns Attestation, no raise."""

    def test_fresh_approval_returns_attestation(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        entry = _fresh_attestation()
        toml_content = f"""
[[gates]]
id = "T2-30"
event_ref = "{entry["event_ref"]}"
attested_at = "{entry["attested_at"]}"
ttl_days = {entry["ttl_days"]}
attested_by = "{entry["attested_by"]}"
note = "{entry["note"]}"
"""
        gates_file.write_text(toml_content)

        attestation = assert_promotion_approved(
            "abc123def456", toml_path=gates_file, now=datetime.now(UTC)
        )

        assert isinstance(attestation, Attestation)
        assert attestation.attested_by == "Marielle Russo"
        assert attestation.ttl_days == 30.0


class TestAssertPromotionApprovedDifferentEventRef:
    """Entry exists for id=T2-30 but event_ref is a different SHA -> NoMatchingApprovalError."""

    def test_different_event_ref_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        entry = _fresh_attestation()
        toml_content = f"""
[[gates]]
id = "T2-30"
event_ref = "{entry["event_ref"]}"
attested_at = "{entry["attested_at"]}"
ttl_days = {entry["ttl_days"]}
attested_by = "{entry["attested_by"]}"
note = "{entry["note"]}"
"""
        gates_file.write_text(toml_content)

        with pytest.raises(NoMatchingApprovalError, match="no T2-30 approval found") as exc_info:
            assert_promotion_approved(
                "different_sha_xyz789", toml_path=gates_file, now=datetime.now(UTC)
            )
        assert "different_sha_xyz789" in str(exc_info.value)


class TestAssertPromotionApprovedExpiredApproval:
    """Matching entry but TTL expired -> ApprovalExpiredError with reasons."""

    def test_expired_approval_raises_approval_expired_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        entry = _expired_attestation()
        toml_content = f"""
[[gates]]
id = "T2-30"
event_ref = "{entry["event_ref"]}"
attested_at = "{entry["attested_at"]}"
ttl_days = {entry["ttl_days"]}
attested_by = "{entry["attested_by"]}"
note = "{entry["note"]}"
"""
        gates_file.write_text(toml_content)

        with pytest.raises(ApprovalExpiredError, match="approval expired or not fresh") as exc_info:
            assert_promotion_approved("abc123def456", toml_path=gates_file, now=datetime.now(UTC))
        # Verify the error message includes attestation age info
        assert "past TTL" in str(exc_info.value)


class TestAssertPromotionApprovedNoT2_30Entries:
    """No id=T2-30 entries at all -> NoMatchingApprovalError."""

    def test_no_t2_30_entries_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        toml_content = """
[[gates]]
id = "T2-28"
attested_at = "2026-07-14T00:00:00Z"
ttl_days = 365
attested_by = "Marielle Russo"
note = "Constitution authorship gate (different gate, not T2-30)"
"""
        gates_file.write_text(toml_content)

        with pytest.raises(NoMatchingApprovalError, match="no T2-30 approval found"):
            assert_promotion_approved("abc123def456", toml_path=gates_file, now=datetime.now(UTC))


class TestAssertPromotionApprovedFileMissing:
    """TOML file doesn't exist -> NoMatchingApprovalError (not FileNotFoundError)."""

    def test_missing_gates_file_raises_no_matching_approval_error(self, tmp_path) -> None:
        missing_file = tmp_path / "nonexistent.toml"

        with pytest.raises(NoMatchingApprovalError, match="gates file not found"):
            assert_promotion_approved("abc123def456", toml_path=missing_file)


class TestAssertPromotionApprovedErrorHierarchy:
    """All errors are PromotionNotApprovedError subclasses."""

    def test_no_matching_approval_error_is_subclass(self) -> None:
        assert issubclass(NoMatchingApprovalError, PromotionNotApprovedError)

    def test_approval_expired_error_is_subclass(self) -> None:
        assert issubclass(ApprovalExpiredError, PromotionNotApprovedError)

    def test_catching_base_class_catches_all_subclasses(self, tmp_path) -> None:
        missing_file = tmp_path / "nonexistent.toml"
        with pytest.raises(PromotionNotApprovedError):
            assert_promotion_approved("abc123def456", toml_path=missing_file)


class TestAssertPromotionApprovedEdgeCases:
    """Edge cases: malformed TOML, empty gates list, etc."""

    def test_malformed_toml_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "bad.toml"
        gates_file.write_text("this is not valid toml [[[")

        with pytest.raises(NoMatchingApprovalError, match="failed to parse gates file"):
            assert_promotion_approved("abc123def456", toml_path=gates_file)

    def test_empty_gates_list_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        gates_file.write_text("# Empty gates file\n")

        with pytest.raises(NoMatchingApprovalError, match="no T2-30 approval found"):
            assert_promotion_approved("abc123def456", toml_path=gates_file)

    def test_gates_not_a_list_raises_no_matching_approval_error(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        gates_file.write_text("gates = {}")

        with pytest.raises(NoMatchingApprovalError, match="gates TOML entry is not a list"):
            assert_promotion_approved("abc123def456", toml_path=gates_file)


class TestAssertPromotionApprovedMultipleEntries:
    """Multiple T2-30 entries: only the matching event_ref counts."""

    def test_selects_correct_entry_among_multiple_t2_30_entries(self, tmp_path) -> None:
        gates_file = tmp_path / "gates.toml"
        now = datetime.now(UTC)
        ref_time_iso = now.isoformat()
        toml_content = f"""
[[gates]]
id = "T2-30"
event_ref = "sha_aaa111"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "First approval"

[[gates]]
id = "T2-30"
event_ref = "sha_bbb222"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Second approval"

[[gates]]
id = "T2-30"
event_ref = "sha_ccc333"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Third approval"
"""
        gates_file.write_text(toml_content)

        attestation = assert_promotion_approved("sha_bbb222", toml_path=gates_file, now=now)

        assert attestation.note == "Second approval"

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
id = "T2-30"
event_ref = "sha_promote_this"
attested_at = "{ref_time_iso}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Promotion gate"
"""
        gates_file.write_text(toml_content)

        attestation = assert_promotion_approved("sha_promote_this", toml_path=gates_file, now=now)

        assert attestation.note == "Promotion gate"

"""Tests for the promotion_gate CI check script (audit_promotion_gate.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from subprocess import run


def _fresh_attestation(sha: str = "abc123def456", now: datetime | None = None) -> str:
    """Build a TOML string with a fresh gate entry."""
    ref_time = now or datetime.now(UTC)
    return f"""
[[gates]]
id = "T2-30"
event_ref = "{sha}"
attested_at = "{ref_time.isoformat()}"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Approval granted after engineering review."
"""


def _expired_attestation(sha: str = "abc123def456") -> str:
    """Build a TOML string with an expired gate entry."""
    return f"""
[[gates]]
id = "T2-30"
event_ref = "{sha}"
attested_at = "2000-01-01T00:00:00Z"
ttl_days = 1.0
attested_by = "Marielle Russo"
note = "Old approval (expired)."
"""


class TestPromotionGateCIHappyPath:
    """Approved and fresh commit SHA -> exit 0."""

    def test_approved_sha_exits_zero(self, tmp_path) -> None:
        """Fresh approval entry -> exit 0, attestation printed."""
        gates_file = tmp_path / "gates.toml"
        sha = "abc123def456"
        gates_file.write_text(_fresh_attestation(sha))

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                sha,
                "--toml-path",
                str(gates_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Promotion approved" in result.stdout
        assert sha in result.stdout
        assert "Marielle Russo" in result.stdout


class TestPromotionGateCINoMatchingApproval:
    """No matching entry for the given SHA -> exit 1."""

    def test_no_matching_sha_exits_one(self, tmp_path) -> None:
        """Missing entry for requested SHA -> exit 1, clear error message."""
        gates_file = tmp_path / "gates.toml"
        gates_file.write_text(_fresh_attestation("abc123def456"))

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                "different_sha_xyz789",
                "--toml-path",
                str(gates_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "No matching approval found" in result.stderr
        assert "different_sha_xyz789" in result.stderr


class TestPromotionGateCIExpiredApproval:
    """Matching entry but TTL expired -> exit 1."""

    def test_expired_approval_exits_one(self, tmp_path) -> None:
        """Expired attestation -> exit 1, freshness error in stderr."""
        gates_file = tmp_path / "gates.toml"
        sha = "abc123def456"
        gates_file.write_text(_expired_attestation(sha))

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                sha,
                "--toml-path",
                str(gates_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Approval expired or not fresh" in result.stderr


class TestPromotionGateCIMissingToml:
    """TOML file doesn't exist -> exit 1."""

    def test_missing_toml_exits_one(self, tmp_path) -> None:
        """Missing gates file -> exit 1, clear error message."""
        missing_file = tmp_path / "nonexistent.toml"

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                "abc123def456",
                "--toml-path",
                str(missing_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "No matching approval found" in result.stderr


class TestPromotionGateCIMalformedToml:
    """TOML file is syntactically invalid -> exit 1."""

    def test_malformed_toml_exits_one(self, tmp_path) -> None:
        """Invalid TOML syntax -> exit 1, parse error in stderr."""
        gates_file = tmp_path / "bad.toml"
        gates_file.write_text("this is not valid toml [[[")

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                "abc123def456",
                "--toml-path",
                str(gates_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "No matching approval found" in result.stderr or "failed" in result.stderr.lower()


class TestPromotionGateCIEmptyGates:
    """Empty gates list -> exit 1."""

    def test_empty_gates_exits_one(self, tmp_path) -> None:
        """No [[gates]] entries at all -> exit 1."""
        gates_file = tmp_path / "gates.toml"
        gates_file.write_text("# Empty gates file\n")

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                "abc123def456",
                "--toml-path",
                str(gates_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "No matching approval found" in result.stderr


class TestPromotionGateCIMultipleEntries:
    """Multiple T2-30 entries; selects the one with matching event_ref."""

    def test_selects_correct_entry_among_many(self, tmp_path) -> None:
        """Multiple [[gates]] with different SHAs; selects the matching one."""
        gates_file = tmp_path / "gates.toml"
        now = datetime.now(UTC)
        ref_time_iso = now.isoformat()
        content = f"""
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
        gates_file.write_text(content)

        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
                "--commit-sha",
                "sha_bbb222",
                "--toml-path",
                str(gates_file),
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Promotion approved" in result.stdout
        assert "sha_bbb222" in result.stdout
        assert "Second approval" in result.stdout


class TestPromotionGateCIRequiredArg:
    """--commit-sha is required."""

    def test_missing_commit_sha_exits_with_error(self) -> None:
        """Missing --commit-sha argument -> exit 2 (argparse error)."""
        result = run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_promotion_gate.py",
            ],
            cwd="/home/marielle/projects/xtrax/.claude/worktrees/wt-20260811-233351",
            capture_output=True,
            text=True,
        )

        # argparse exits with 2 on usage error
        assert result.returncode == 2
        assert "required" in result.stderr.lower()

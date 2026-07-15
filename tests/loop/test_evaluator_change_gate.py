"""Tests for the evaluator-change gate (T2-29, #2181, AC-22, gate b)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from xtrax.loop.closure_lock import ClosureManifest, build_closure_manifest
from xtrax.loop.evaluator_change_gate import (
    ApprovalExpiredError,
    NoMatchingApprovalError,
    assert_evaluator_change_approved,
)


@pytest.fixture
def closure_dir(tmp_path: Path) -> Path:
    (tmp_path / "evaluator.py").write_text("def evaluate(): return 1\n", encoding="utf-8")
    (tmp_path / "splits.json").write_text('{"train": [1, 2]}', encoding="utf-8")
    (tmp_path / "metric_defs.json").write_text('{"metric": "accuracy"}', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("# pinned deps v1\n", encoding="utf-8")
    return tmp_path


def _manifest(
    closure_dir: Path, *, evaluator_body: str = "def evaluate(): return 1\n"
) -> ClosureManifest:
    (closure_dir / "evaluator.py").write_text(evaluator_body, encoding="utf-8")
    return build_closure_manifest(
        evaluator_paths=(closure_dir / "evaluator.py",),
        split_paths=(closure_dir / "splits.json",),
        metric_def_paths=(closure_dir / "metric_defs.json",),
        config={"lr": 0.1},
        pinned_deps_source=closure_dir / "uv.lock",
    )


def _write_gates_toml(
    path: Path,
    *,
    gate_id: str = "T2-29",
    event_ref: str,
    attested_at: str,
    ttl_days: float = 30,
    attested_by: str = "Marielle Russo",
    note: str = "approved via direct conversation",
) -> Path:
    path.write_text(
        f"""
[[gates]]
id = "{gate_id}"
event_ref = "{event_ref}"
attested_at = "{attested_at}"
ttl_days = {ttl_days}
attested_by = "{attested_by}"
note = "{note}"
""",
        encoding="utf-8",
    )
    return path


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stale_iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


class TestAssertEvaluatorChangeApproved:
    def test_matching_fresh_attestation_returns_attestation(
        self, closure_dir: Path, tmp_path: Path
    ) -> None:
        manifest = _manifest(closure_dir)
        gates_toml = _write_gates_toml(
            tmp_path / "gates.toml",
            event_ref=manifest.closure_hash,
            attested_at=_now_iso(),
            ttl_days=30,
        )

        attestation = assert_evaluator_change_approved(manifest, toml_path=gates_toml)

        assert attestation.attested_by == "Marielle Russo"
        assert attestation.ttl_days == 30

    def test_entry_for_different_event_ref_raises_no_matching_approval(
        self, closure_dir: Path, tmp_path: Path
    ) -> None:
        manifest = _manifest(closure_dir)
        gates_toml = _write_gates_toml(
            tmp_path / "gates.toml",
            event_ref="a-completely-different-closure-hash",
            attested_at=_now_iso(),
            ttl_days=30,
        )

        with pytest.raises(NoMatchingApprovalError, match="no sign-off found"):
            assert_evaluator_change_approved(manifest, toml_path=gates_toml)

    def test_stale_evaluator_change_is_not_covered_by_prior_approval(
        self, closure_dir: Path, tmp_path: Path
    ) -> None:
        """Regression guard for the constitution's 'not once -- no standing blanket approval':
        an approval scoped to an earlier evaluator-change event must not cover a later, different
        change, even though both are id="T2-29" entries in the same file.
        """
        old_manifest = _manifest(closure_dir, evaluator_body="def evaluate(): return 1\n")
        gates_toml = _write_gates_toml(
            tmp_path / "gates.toml",
            event_ref=old_manifest.closure_hash,
            attested_at=_now_iso(),
            ttl_days=30,
        )

        new_manifest = _manifest(closure_dir, evaluator_body="def evaluate(): return 2\n")
        assert new_manifest.closure_hash != old_manifest.closure_hash

        with pytest.raises(NoMatchingApprovalError):
            assert_evaluator_change_approved(new_manifest, toml_path=gates_toml)

    def test_expired_ttl_raises_approval_expired(self, closure_dir: Path, tmp_path: Path) -> None:
        manifest = _manifest(closure_dir)
        gates_toml = _write_gates_toml(
            tmp_path / "gates.toml",
            event_ref=manifest.closure_hash,
            attested_at=_stale_iso(400),
            ttl_days=30,
        )

        with pytest.raises(ApprovalExpiredError, match="expired"):
            assert_evaluator_change_approved(manifest, toml_path=gates_toml)

    def test_no_t2_29_entries_at_all_raises_no_matching_approval(
        self, closure_dir: Path, tmp_path: Path
    ) -> None:
        manifest = _manifest(closure_dir)
        gates_toml = tmp_path / "gates.toml"
        gates_toml.write_text(
            """
[[gates]]
id = "T2-28"
event_ref = "unrelated"
attested_at = "2026-07-14T00:00:00Z"
ttl_days = 365
attested_by = "Marielle Russo"
note = "constitution authorship, not an evaluator-change approval"
""",
            encoding="utf-8",
        )

        with pytest.raises(NoMatchingApprovalError):
            assert_evaluator_change_approved(manifest, toml_path=gates_toml)

    def test_missing_toml_file_raises_no_matching_approval_not_file_not_found(
        self, closure_dir: Path, tmp_path: Path
    ) -> None:
        manifest = _manifest(closure_dir)
        missing = tmp_path / "does-not-exist.toml"
        assert not missing.exists()

        with pytest.raises(NoMatchingApprovalError):
            assert_evaluator_change_approved(manifest, toml_path=missing)

    def test_multiple_matching_entries_uses_most_recent(
        self, closure_dir: Path, tmp_path: Path
    ) -> None:
        manifest = _manifest(closure_dir)
        gates_toml = tmp_path / "gates.toml"
        older = _stale_iso(10)
        newer = _now_iso()
        gates_toml.write_text(
            f"""
[[gates]]
id = "T2-29"
event_ref = "{manifest.closure_hash}"
attested_at = "{older}"
ttl_days = 1
attested_by = "Marielle Russo"
note = "an older, now-expired approval for the same event"

[[gates]]
id = "T2-29"
event_ref = "{manifest.closure_hash}"
attested_at = "{newer}"
ttl_days = 30
attested_by = "Marielle Russo"
note = "the most recent approval"
""",
            encoding="utf-8",
        )

        attestation = assert_evaluator_change_approved(manifest, toml_path=gates_toml)
        assert attestation.note == "the most recent approval"

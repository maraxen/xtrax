"""Contract tests for xtrax.tombstone ledger (#1580)."""

import json
from pathlib import Path

import pytest

from xtrax.tombstone import (
    TombstoneEntry,
    append_tombstone,
    filter_tombstoned,
    is_tombstoned,
    load_tombstones,
)
from xtrax.findings import append_finding, emit_metric_finding

GOLDEN_RUN_ID = "00000000-0000-4000-8000-000000000001"

SAMPLE_FINDING = emit_metric_finding(
    dim="documentation",
    severity="minor",
    file_line="src/xtrax/training/trainer.py:42",
    evidence="JD001: public symbol missing docstring",
    rule_id="JD001",
    symbol_qualname="xtrax.training.trainer.Trainer",
    payload={"tool": "jaxlint"},
    run_id=GOLDEN_RUN_ID,
)


def test_append_tombstone_and_is_tombstoned_round_trip(tmp_path: Path) -> None:
    ledger_path = tmp_path / "audit_tombstones.jsonl"
    entry = TombstoneEntry(
        finding_id=SAMPLE_FINDING.finding_id,
        reason="accepted risk for v1",
        actor="maintainer",
        recorded_at="2026-06-19T00:00:00+00:00",
        disposition="accepted",
    )
    append_tombstone(entry, path=ledger_path)
    assert is_tombstoned(SAMPLE_FINDING.finding_id, path=ledger_path)
    assert load_tombstones(path=ledger_path) == {SAMPLE_FINDING.finding_id}


def test_governance_fields_persisted_in_jsonl(tmp_path: Path) -> None:
    ledger_path = tmp_path / "audit_tombstones.jsonl"
    entry = TombstoneEntry(
        finding_id="abc123",
        reason="out of audit scope",
        actor="reviewer",
        recorded_at="2026-06-19T12:00:00+00:00",
        disposition="out_of_scope",
    )
    append_tombstone(entry, path=ledger_path)
    payload = json.loads(ledger_path.read_text(encoding="utf-8").strip())
    assert payload == {
        "actor": "reviewer",
        "disposition": "out_of_scope",
        "finding_id": "abc123",
        "reason": "out of audit scope",
        "recorded_at": "2026-06-19T12:00:00+00:00",
    }


def test_append_finding_skips_tombstoned_finding_id(tmp_path: Path) -> None:
    audits_path = tmp_path / "audits.jsonl"
    ledger_path = tmp_path / "audit_tombstones.jsonl"
    append_tombstone(
        TombstoneEntry(
            finding_id=SAMPLE_FINDING.finding_id,
            reason="wontfix",
            actor="maintainer",
            recorded_at="2026-06-19T00:00:00+00:00",
            disposition="wontfix",
        ),
        path=ledger_path,
    )
    append_finding(
        SAMPLE_FINDING,
        audits_path=audits_path,
        tombstone_path=ledger_path,
    )
    assert not audits_path.is_file()


def test_append_finding_writes_non_tombstoned_findings(tmp_path: Path) -> None:
    audits_path = tmp_path / "audits.jsonl"
    ledger_path = tmp_path / "audit_tombstones.jsonl"
    append_finding(
        SAMPLE_FINDING,
        audits_path=audits_path,
        tombstone_path=ledger_path,
    )
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["finding_id"] == SAMPLE_FINDING.finding_id


def test_filter_tombstoned_drops_suppressed(tmp_path: Path) -> None:
    ledger_path = tmp_path / "audit_tombstones.jsonl"
    other = emit_metric_finding(
        dim="structure",
        severity="info",
        file_line="src/xtrax/__init__.py:1",
        evidence="other finding",
        rule_id="F2",
        symbol_qualname="",
        run_id=GOLDEN_RUN_ID,
    )
    append_tombstone(
        TombstoneEntry(
            finding_id=SAMPLE_FINDING.finding_id,
            reason="accepted",
            actor="maintainer",
            recorded_at="2026-06-19T00:00:00+00:00",
            disposition="accepted",
        ),
        path=ledger_path,
    )
    kept = filter_tombstoned(
        [SAMPLE_FINDING, other],
        path=ledger_path,
    )
    assert kept == [other]


def test_append_tombstone_rejects_empty_finding_id(tmp_path: Path) -> None:
    ledger_path = tmp_path / "audit_tombstones.jsonl"
    with pytest.raises(ValueError, match="finding_id must be non-empty"):
        append_tombstone(
            TombstoneEntry(
                finding_id="  ",
                reason="bad",
                actor="maintainer",
                recorded_at="2026-06-19T00:00:00+00:00",
                disposition="accepted",
            ),
            path=ledger_path,
        )

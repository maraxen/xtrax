"""Merge-blocking contract tests for xtrax.devtools.emit (N1.1 / AC-1.1)."""

import json
from pathlib import Path

import pytest

from xtrax.devtools.emit import (
    SCHEMA_VERSION,
    SchemaVersionMismatchError,
    append_finding,
    assert_schema_version_compatible,
    compute_finding_id,
    emit_judgment_finding,
    emit_metric_finding,
    finding_from_dict,
    finding_to_dict,
    round_trip_finding,
)

GOLDEN_RUN_ID = "00000000-0000-4000-8000-000000000001"

GOLDEN_METRIC = emit_metric_finding(
    dim="documentation",
    severity="minor",
    file_line="src/xtrax/training/trainer.py:42",
    evidence="JD001: public symbol missing docstring",
    rule_id="JD001",
    symbol_qualname="xtrax.training.trainer.Trainer",
    payload={"tool": "jaxlint"},
    run_id=GOLDEN_RUN_ID,
)

GOLDEN_JUDGMENT = emit_judgment_finding(
    dim="api_ergonomics",
    severity="info",
    file_line="src/xtrax/run/spec.py:18",
    evidence="Param count heuristic flagged 6 parameters on RunSpec.__init__",
    rubric_id="param_count_heuristic",
    score=3,
    anchor_quote="≤4–5 params is the committed heuristic ceiling",
    symbol_qualname="xtrax.run.spec.RunSpec.__init__",
    run_id=GOLDEN_RUN_ID,
)


def test_metric_golden_round_trip() -> None:
    restored = round_trip_finding(GOLDEN_METRIC)
    assert restored == GOLDEN_METRIC
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.source_track == "deterministic"
    assert restored.payload["rule_id"] == "JD001"


def test_judgment_golden_round_trip() -> None:
    restored = round_trip_finding(GOLDEN_JUDGMENT)
    assert restored == GOLDEN_JUDGMENT
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.source_track == "judgment"
    assert restored.payload["rubric_id"] == "param_count_heuristic"


def test_schema_version_mismatch_loud_fails() -> None:
    with pytest.raises(SchemaVersionMismatchError, match="schema_version mismatch"):
        assert_schema_version_compatible(
            record_schema_version=1,
            baseline_schema_version=2,
        )

    stale = finding_from_dict({**finding_to_dict(GOLDEN_METRIC), "schema_version": 99})
    with pytest.raises(SchemaVersionMismatchError):
        round_trip_finding(stale)


def test_judgment_finding_id_stable_for_same_structural_key() -> None:
    shared = dict(
        dim="documentation",
        severity="major",
        file_line="src/xtrax/io/__init__.py:7",
        evidence="Different prose evidence should not change the structural id",
        rubric_id="semantic_accuracy",
        symbol_qualname="xtrax.io.async_indexed_stream",
        run_id=GOLDEN_RUN_ID,
    )
    first = emit_judgment_finding(score=2, anchor_quote="quote A", **shared)
    second = emit_judgment_finding(score=4, anchor_quote="quote B", **shared)
    assert first.finding_id == second.finding_id
    assert first.finding_id == compute_finding_id(
        shared["dim"],
        shared["rubric_id"],
        symbol_qualname=shared["symbol_qualname"],
    )


def test_content_hash_fallback_when_symbol_absent() -> None:
    evidence = "module-level import cycle detected"
    first_id = compute_finding_id(
        "structure",
        "F1",
        symbol_qualname="",
        evidence=evidence,
    )
    second_id = compute_finding_id(
        "structure",
        "F1",
        symbol_qualname="",
        evidence="module-level  import   cycle detected",
    )
    assert first_id == second_id

    record = emit_metric_finding(
        dim="structure",
        severity="critical",
        file_line="src/xtrax/__init__.py:1",
        evidence=evidence,
        rule_id="F1",
        symbol_qualname="",
        run_id=GOLDEN_RUN_ID,
    )
    assert record.finding_id == first_id


def test_append_finding_writes_jsonl(tmp_path: Path) -> None:
    audits_path = tmp_path / "audits.jsonl"
    append_finding(GOLDEN_METRIC, audits_path=audits_path)
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["finding_id"] == GOLDEN_METRIC.finding_id
    assert payload["schema_version"] == SCHEMA_VERSION

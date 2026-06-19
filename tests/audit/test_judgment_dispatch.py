"""Tests for N4.1 judgment dispatch wiring (#1590)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.judgment import (
    DEFAULT_DISPATCH_PATH,
    load_judgment_dispatch,
    run_judgment_dispatch,
    validate_judgment_wiring,
)
from xtrax.devtools.routing import DEFAULT_ROUTING_PATH
from xtrax.devtools.rubrics import DEFAULT_RUBRICS_DIR

ROOT = Path(__file__).resolve().parents[2]
DISPATCH_PATH = ROOT / DEFAULT_DISPATCH_PATH
RUBRICS_DIR = ROOT / DEFAULT_RUBRICS_DIR
ROUTING_PATH = ROOT / DEFAULT_ROUTING_PATH

EXPECTED_DIMENSIONS = frozenset(
    {
        "correctness",
        "jax_purity",
        "type_hardening",
        "performance",
        "documentation",
        "api_ergonomics",
        "test_rigor",
        "structure_complexity",
    }
)


def test_load_judgment_dispatch_returns_eight_entries() -> None:
    entries = load_judgment_dispatch(DISPATCH_PATH)
    dimensions = {entry.dimension for entry in entries}
    assert dimensions == EXPECTED_DIMENSIONS
    assert len(entries) == 8
    for entry in entries:
        assert entry.agent_role
        assert entry.label == "observation"
        assert entry.default_severity == "info"
        assert entry.rubric_path.name == f"{entry.dimension}.toml"


def test_validate_judgment_wiring_passes_for_repo_dispatch() -> None:
    validate_judgment_wiring(
        dispatch_path=DISPATCH_PATH,
        rubrics_dir=RUBRICS_DIR,
        routing_path=ROUTING_PATH,
    )


def test_validate_judgment_wiring_fails_on_empty_agent_role(tmp_path: Path) -> None:
    broken = tmp_path / "judgment_dispatch.toml"
    broken.write_text(
        """
[dispatch]
schema_version = 1
version = "0.0.0"

[[dimensions]]
dimension = "correctness"
agent_role = ""
rubric_path = "audit/rubrics/correctness.toml"
default_severity = "info"
label = "observation"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agent_role is empty"):
        validate_judgment_wiring(
            dispatch_path=broken,
            rubrics_dir=RUBRICS_DIR,
            routing_path=ROUTING_PATH,
        )


def test_run_judgment_dispatch_validate_only_no_append(tmp_path: Path) -> None:
    audits_path = tmp_path / "audits.jsonl"
    result = run_judgment_dispatch(
        audits_path,
        emit_observations=False,
        run_id="judgment-dispatch-test",
        dispatch_path=DISPATCH_PATH,
        rubrics_dir=RUBRICS_DIR,
        routing_path=ROUTING_PATH,
    )
    assert result.passed is True
    assert len(result.entries) == 8
    assert result.findings_emitted == 0
    assert not audits_path.exists()
    for entry in result.entries:
        assert entry.destination == "found_issues"
        assert entry.anchor_quote
        assert entry.finding_emitted is False
    assert set(result.destinations) == EXPECTED_DIMENSIONS


@patch("xtrax.devtools.judgment.append_finding")
def test_run_judgment_dispatch_emits_judgment_observations(
    mock_append: object,
    tmp_path: Path,
) -> None:
    audits_path = tmp_path / "audits.jsonl"
    result = run_judgment_dispatch(
        audits_path,
        emit_observations=True,
        run_id="judgment-emit-test",
        dispatch_path=DISPATCH_PATH,
        rubrics_dir=RUBRICS_DIR,
        routing_path=ROUTING_PATH,
    )
    assert result.passed is True
    assert result.findings_emitted == 8
    assert len(result.entries) == 8
    assert mock_append.call_count == 8
    for call in mock_append.call_args_list:
        record = call.args[0]
        assert record.source_track == "judgment"
        assert record.severity == "info"
        assert record.dim in EXPECTED_DIMENSIONS
        assert record.payload["score"] == 0
        assert record.payload["anchor_quote"]
        assert "judgment dispatch armed for" in record.evidence
        assert result.destinations[record.dim] == "found_issues"


def test_run_judgment_dispatch_emit_writes_jsonl(tmp_path: Path) -> None:
    audits_path = tmp_path / "audits.jsonl"
    run_judgment_dispatch(
        audits_path,
        emit_observations=True,
        run_id="judgment-jsonl-test",
        dispatch_path=DISPATCH_PATH,
        rubrics_dir=RUBRICS_DIR,
        routing_path=ROUTING_PATH,
    )
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    for line in lines:
        payload = json.loads(line)
        assert payload["source_track"] == "judgment"
        assert payload["severity"] == "info"
        assert payload["dim"] in EXPECTED_DIMENSIONS
        assert payload["payload"]["score"] == 0

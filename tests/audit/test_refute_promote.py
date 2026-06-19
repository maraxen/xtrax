"""Tests for N4.3 Refute-or-Promote protocol (#1593)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.refute_promote import (
    DEFAULT_PROTOCOL_PATH,
    JudgmentCandidate,
    load_refute_promote_protocol,
    promote_and_emit,
    resolve_persona_pair,
    run_refute_or_promote,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / DEFAULT_PROTOCOL_PATH


def _candidate(*, dimension: str = "type_hardening") -> JudgmentCandidate:
    return JudgmentCandidate(
        dimension=dimension,
        severity="info",
        file_line="src/xtrax/training/trainer.py:10",
        evidence="annotation coverage below rubric anchor",
        rubric_id="type_hardening.coverage",
        score=3,
        anchor_quote="whole-baseline annotation coverage ratchet",
    )


def test_load_refute_promote_protocol_loads_committed_toml() -> None:
    pairs = load_refute_promote_protocol(PROTOCOL_PATH)
    dimensions = {pair.dimension for pair in pairs}
    assert "*" in dimensions
    assert "correctness" in dimensions
    assert len(pairs) >= 8


def test_resolve_persona_pair_uses_default_for_type_hardening() -> None:
    assert_role, refute_role = resolve_persona_pair(
        "type_hardening",
        path=PROTOCOL_PATH,
    )
    assert assert_role == "auditor"
    assert refute_role == "oracle"


def test_resolve_persona_pair_overrides_for_correctness() -> None:
    assert_role, refute_role = resolve_persona_pair(
        "correctness",
        path=PROTOCOL_PATH,
    )
    assert assert_role == "oracle"
    assert refute_role == "auditor"


def test_resolve_persona_pair_overrides_for_jax_purity() -> None:
    assert_role, refute_role = resolve_persona_pair(
        "jax_purity",
        path=PROTOCOL_PATH,
    )
    assert assert_role == "jax-purity-reviewer"
    assert refute_role == "oracle"


def test_refute_kills_drops_without_emit(tmp_path: Path) -> None:
    candidate = _candidate(dimension="correctness")
    verdict = run_refute_or_promote(
        candidate,
        assert_fn=lambda _: True,
        refute_fn=lambda _: True,
        assert_role="oracle",
        refute_role="auditor",
    )
    assert verdict.promoted is False
    assert verdict.dropped is True
    assert verdict.phase_log == ("assert:oracle:pass", "refute:auditor:kill")

    audits_path = tmp_path / "audits.jsonl"
    record = promote_and_emit(
        candidate,
        verdict,
        audits_path,
        run_id="refute-kill-test",
    )
    assert record is None
    assert not audits_path.exists()


def test_assert_fails_drops_without_emit(tmp_path: Path) -> None:
    candidate = _candidate()
    verdict = run_refute_or_promote(
        candidate,
        assert_fn=lambda _: False,
        refute_fn=lambda _: False,
        assert_role="auditor",
        refute_role="oracle",
    )
    assert verdict.promoted is False
    assert verdict.dropped is True
    assert verdict.phase_log == ("assert:auditor:fail",)

    audits_path = tmp_path / "audits.jsonl"
    record = promote_and_emit(
        candidate,
        verdict,
        audits_path,
        run_id="assert-fail-test",
    )
    assert record is None
    assert not audits_path.exists()


@patch("xtrax.devtools.refute_promote.append_finding")
def test_survives_promotes_observation_emit(
    mock_append: object,
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    verdict = run_refute_or_promote(
        candidate,
        assert_fn=lambda _: True,
        refute_fn=lambda _: False,
        assert_role="auditor",
        refute_role="oracle",
    )
    assert verdict.promoted is True
    assert verdict.dropped is False
    assert verdict.label == "observation"
    assert verdict.phase_log == ("assert:auditor:pass", "refute:oracle:survive")

    audits_path = tmp_path / "audits.jsonl"
    record = promote_and_emit(
        candidate,
        verdict,
        audits_path,
        run_id="survive-emit-test",
    )
    assert record is not None
    assert record.source_track == "judgment"
    assert record.payload["label"] == "observation"
    assert record.payload["assert_role"] == "auditor"
    assert record.payload["refute_role"] == "oracle"
    assert record.payload["protocol"] == "refute_or_promote"
    assert record.payload["phase_log"] == list(verdict.phase_log)
    mock_append.assert_called_once()


def test_survives_writes_jsonl_with_label_payload(tmp_path: Path) -> None:
    candidate = _candidate()
    verdict = run_refute_or_promote(
        candidate,
        assert_fn=lambda _: True,
        refute_fn=lambda _: False,
        assert_role="auditor",
        refute_role="oracle",
    )
    audits_path = tmp_path / "audits.jsonl"
    promote_and_emit(
        candidate,
        verdict,
        audits_path,
        run_id="jsonl-label-test",
    )
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["source_track"] == "judgment"
    assert payload["payload"]["label"] == "observation"
    assert payload["payload"]["protocol"] == "refute_or_promote"


def test_resolve_persona_pair_missing_dimension_and_default_raises(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "refute_or_promote.toml"
    broken.write_text(
        """
[protocol]
schema_version = 1
version = "0.0.0"
ceiling_note = "test"

[[persona_pairs]]
dimension = "correctness"
assert_role = "oracle"
refute_role = "auditor"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no persona pair"):
        resolve_persona_pair("type_hardening", path=broken)

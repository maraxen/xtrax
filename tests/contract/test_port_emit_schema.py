"""Contract tests for port/emit/port_emit.py (AC-6, AC-9)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_EMIT_PATH = ROOT / "port" / "emit" / "port_emit.py"
_spec = importlib.util.spec_from_file_location("port_emit", _EMIT_PATH)
assert _spec and _spec.loader
port_emit = importlib.util.module_from_spec(_spec)
sys.modules["port_emit"] = port_emit
_spec.loader.exec_module(port_emit)

GOLDEN_TASK_ID = "260617_xtrax-composition-mission"
GOLDEN_ORACLE = "ref:port/reference/example:v0.1.0:sha256:abc123"
GOLDEN_QUALNAME = "xtrax.transforms.map.apply_map"
GOLDEN_TIER = "parity_tier_3"
GOLDEN_TOLERANCE = "rtol=1e-4,atol=1e-4,matmul_precision=highest"


def _golden_verdict(*, status: port_emit.TierStatus = "FAIL") -> port_emit.TierVerdict:
    return port_emit.TierVerdict(
        status=status,
        tolerance_policy=GOLDEN_TOLERANCE,
        error_taxonomy_class="numeric_drift",
        max_discrepancy=0.0023,
    )


def _golden_evidence() -> port_emit.Evidence:
    return port_emit.Evidence(
        pytest_nodeid="port/tests/test_parity_example.py::test_tier_3",
        traceback_excerpt="AssertionError: max abs diff 0.0023 > rtol",
    )


def test_finding_id_includes_tolerance_policy() -> None:
    base = port_emit.compute_port_finding_id(
        port_emit.PORT_DIM,
        GOLDEN_QUALNAME,
        GOLDEN_TIER,
        GOLDEN_TOLERANCE,
    )
    shifted = port_emit.compute_port_finding_id(
        port_emit.PORT_DIM,
        GOLDEN_QUALNAME,
        GOLDEN_TIER,
        "rtol=1e-5,atol=1e-5,matmul_precision=highest",
    )
    assert base != shifted


def test_port_record_sketch_fields() -> None:
    record = port_emit._build_port_record(
        task_id=GOLDEN_TASK_ID,
        symbol_qualname=GOLDEN_QUALNAME,
        port_parity_tier=GOLDEN_TIER,
        oracle_id=GOLDEN_ORACLE,
        tier_verdict=_golden_verdict(),
        evidence=_golden_evidence(),
    )
    port_emit._validate_port_record(record)
    data = port_emit.port_record_to_dict(record)
    assert data["domain"] == "port"
    assert data["port_parity_tier"] == GOLDEN_TIER
    assert data["oracle_id"] == GOLDEN_ORACLE
    assert data["tier_verdict"]["status"] == "FAIL"
    assert data["tier_verdict"]["tolerance_policy"] == GOLDEN_TOLERANCE
    assert data["finding_id"] == port_emit.compute_port_finding_id(
        port_emit.PORT_DIM,
        GOLDEN_QUALNAME,
        GOLDEN_TIER,
        GOLDEN_TOLERANCE,
    )


def test_emit_tier_verdict_writes_jsonl(tmp_path: Path) -> None:
    audits_path = tmp_path / "audits.jsonl"
    finding_id = port_emit.emit_tier_verdict(
        task_id=GOLDEN_TASK_ID,
        symbol_qualname=GOLDEN_QUALNAME,
        port_parity_tier=GOLDEN_TIER,
        oracle_id=GOLDEN_ORACLE,
        tier_verdict=_golden_verdict(),
        evidence=_golden_evidence(),
        audits_path=audits_path,
    )
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["finding_id"] == finding_id
    if port_emit.devtools_emit_available():
        assert payload["dim"] == "port"
        assert payload["payload"]["domain"] == "port"
        assert payload["payload"]["port_parity_tier"] == GOLDEN_TIER
        assert payload["payload"]["oracle_id"] == GOLDEN_ORACLE
        tier_payload = payload["payload"]["tier_verdict"]
        assert tier_payload["tolerance_policy"] == GOLDEN_TOLERANCE
    else:
        assert payload["domain"] == "port"
        assert payload["port_parity_tier"] == GOLDEN_TIER
        assert payload["oracle_id"] == GOLDEN_ORACLE


@pytest.mark.skipif(
    not port_emit.devtools_emit_available(),
    reason="xtrax.findings (#1577) not importable",
)
def test_dual_validation_against_n11_envelope(tmp_path: Path) -> None:
    from xtrax.findings import finding_from_dict, round_trip_finding

    audits_path = tmp_path / "audits.jsonl"
    port_emit.emit_tier_verdict(
        task_id=GOLDEN_TASK_ID,
        symbol_qualname=GOLDEN_QUALNAME,
        port_parity_tier=GOLDEN_TIER,
        oracle_id=GOLDEN_ORACLE,
        tier_verdict=_golden_verdict(status="PASS"),
        evidence=_golden_evidence(),
        audits_path=audits_path,
    )
    payload = json.loads(audits_path.read_text(encoding="utf-8").strip())
    restored = round_trip_finding(finding_from_dict(payload))
    assert restored.dim == "port"
    assert restored.source_track == "deterministic"
    assert restored.payload["domain"] == "port"
    assert restored.payload["port_parity_tier"] == GOLDEN_TIER
    assert restored.finding_id == port_emit.compute_port_finding_id(
        port_emit.PORT_DIM,
        GOLDEN_QUALNAME,
        GOLDEN_TIER,
        GOLDEN_TOLERANCE,
    )

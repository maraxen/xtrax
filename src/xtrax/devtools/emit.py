"""Contract-tested audit finding emit seam (N1.1 substrate)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1

Severity = Literal["info", "minor", "major", "critical"]
SourceTrack = Literal["deterministic", "judgment"]


class SchemaVersionMismatchError(RuntimeError):
    """Raised when finding or baseline schema_version does not match expected."""


@dataclass(frozen=True, slots=True)
class AuditFinding:
    schema_version: int
    run_id: str
    dim: str
    severity: Severity
    file_line: str
    evidence: str
    source_track: SourceTrack
    finding_id: str
    payload: dict[str, Any]


def _stable_hash(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _normalize_evidence(evidence: str) -> str:
    return " ".join(evidence.split())


def compute_finding_id(
    dim: str,
    rule_or_rubric_id: str,
    *,
    symbol_qualname: str = "",
    evidence: str = "",
) -> str:
    """Structural dedup key: qualname+rule when present, else content-hash fallback."""
    if symbol_qualname:
        return _stable_hash(dim, symbol_qualname, rule_or_rubric_id)
    return _stable_hash(dim, _normalize_evidence(evidence))


def assert_schema_version_compatible(
    record_schema_version: int,
    baseline_schema_version: int,
) -> None:
    """LOUD-FAIL on schema drift (pre-mortem #3); never silently skip."""
    if record_schema_version != baseline_schema_version:
        msg = (
            f"schema_version mismatch: record={record_schema_version} "
            f"baseline={baseline_schema_version}"
        )
        raise SchemaVersionMismatchError(msg)


def finding_to_dict(record: AuditFinding) -> dict[str, Any]:
    return asdict(record)


def finding_from_dict(data: dict[str, Any]) -> AuditFinding:
    return AuditFinding(
        schema_version=int(data["schema_version"]),
        run_id=str(data["run_id"]),
        dim=str(data["dim"]),
        severity=data["severity"],
        file_line=str(data["file_line"]),
        evidence=str(data["evidence"]),
        source_track=data["source_track"],
        finding_id=str(data["finding_id"]),
        payload=dict(data["payload"]),
    )


def round_trip_finding(record: AuditFinding) -> AuditFinding:
    """Serialize to JSON and back; validates the envelope contract."""
    payload = json.loads(json.dumps(finding_to_dict(record)))
    restored = finding_from_dict(payload)
    assert_schema_version_compatible(restored.schema_version, SCHEMA_VERSION)
    return restored


def emit_metric_finding(
    dim: str,
    severity: Severity,
    file_line: str,
    evidence: str,
    rule_id: str,
    symbol_qualname: str,
    payload: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
) -> AuditFinding:
    extra = dict(payload or {})
    finding_id = compute_finding_id(
        dim,
        rule_id,
        symbol_qualname=symbol_qualname,
        evidence=evidence,
    )
    return AuditFinding(
        schema_version=SCHEMA_VERSION,
        run_id=run_id or str(uuid.uuid4()),
        dim=dim,
        severity=severity,
        file_line=file_line,
        evidence=evidence,
        source_track="deterministic",
        finding_id=finding_id,
        payload={
            "rule_id": rule_id,
            "symbol_qualname": symbol_qualname,
            **extra,
        },
    )


def emit_judgment_finding(
    dim: str,
    severity: Severity,
    file_line: str,
    evidence: str,
    rubric_id: str,
    score: int,
    anchor_quote: str,
    symbol_qualname: str,
    *,
    run_id: str | None = None,
) -> AuditFinding:
    finding_id = compute_finding_id(
        dim,
        rubric_id,
        symbol_qualname=symbol_qualname,
        evidence=evidence,
    )
    return AuditFinding(
        schema_version=SCHEMA_VERSION,
        run_id=run_id or str(uuid.uuid4()),
        dim=dim,
        severity=severity,
        file_line=file_line,
        evidence=evidence,
        source_track="judgment",
        finding_id=finding_id,
        payload={
            "rubric_id": rubric_id,
            "score": score,
            "anchor_quote": anchor_quote,
            "symbol_qualname": symbol_qualname,
        },
    )


def append_finding(
    record: AuditFinding,
    audits_path: Path = Path(".praxia/audits.jsonl"),
    tombstone_path: Path | None = None,
) -> None:
    """Append one JSONL finding record (port/#2180 delegation target)."""
    from xtrax.devtools.tombstone import DEFAULT_TOMBSTONE_PATH, is_tombstoned

    resolved_tombstone_path = (
        DEFAULT_TOMBSTONE_PATH if tombstone_path is None else tombstone_path
    )
    if is_tombstoned(record.finding_id, path=resolved_tombstone_path):
        return
    audits_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(finding_to_dict(record), sort_keys=True)
    with audits_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")

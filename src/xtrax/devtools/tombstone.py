"""Tombstone ledger for suppressed audit findings (N1.4 / #1580)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from xtrax.findings import AuditFinding

DEFAULT_TOMBSTONE_PATH = Path(".praxia/audit_tombstones.jsonl")

Disposition = Literal["accepted", "wontfix", "out_of_scope"]


@dataclass(frozen=True, slots=True)
class TombstoneEntry:
    finding_id: str
    reason: str
    actor: str
    recorded_at: str
    disposition: Disposition


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _validate_disposition(disposition: str) -> Disposition:
    allowed: frozenset[Disposition] = frozenset({"accepted", "wontfix", "out_of_scope"})
    if disposition not in allowed:
        msg = f"disposition must be one of {sorted(allowed)}, got {disposition!r}"
        raise ValueError(msg)
    return disposition


def load_tombstones(path: Path = DEFAULT_TOMBSTONE_PATH) -> set[str]:
    """Return the set of tombstoned finding_ids."""
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    finding_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        finding_ids.add(str(payload["finding_id"]))
    return finding_ids


def is_tombstoned(
    finding_id: str,
    path: Path = DEFAULT_TOMBSTONE_PATH,
) -> bool:
    return finding_id in load_tombstones(path)


def append_tombstone(
    entry: TombstoneEntry,
    path: Path = DEFAULT_TOMBSTONE_PATH,
) -> None:
    """Append one tombstone record (atomic line write with parent mkdir)."""
    if not entry.finding_id.strip():
        msg = "finding_id must be non-empty"
        raise ValueError(msg)
    _validate_disposition(entry.disposition)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry), sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")
        handle.flush()


def filter_tombstoned(
    findings: list[AuditFinding],
    path: Path = DEFAULT_TOMBSTONE_PATH,
) -> list[AuditFinding]:
    tombstoned = load_tombstones(path)
    if not tombstoned:
        return list(findings)
    return [finding for finding in findings if finding.finding_id not in tombstoned]

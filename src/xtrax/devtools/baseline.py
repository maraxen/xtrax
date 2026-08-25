"""Baseline-JSON ratchet substrate (N1.2 / #1578)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from xtrax.findings import assert_schema_version_compatible

BASELINE_SCHEMA_VERSION = 1

Comparator = Literal["minimize", "maximize", "best_ever"]

DEFAULT_BASELINE_PATH = Path(".praxia/audit_baseline.json")


@dataclass(frozen=True, slots=True)
class MetricEntry:
    key: str
    value: float
    comparator: Comparator


@dataclass(frozen=True, slots=True)
class AuditBaseline:
    schema_version: int
    updated_at: str
    metrics: dict[str, MetricEntry]


def assert_baseline_schema_compatible(record_version: int) -> None:
    """LOUD-FAIL on schema drift; delegates to emit envelope helper."""
    assert_schema_version_compatible(record_version, BASELINE_SCHEMA_VERSION)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _should_tighten(
    baseline_value: float,
    observed: float,
    comparator: Comparator,
) -> bool:
    if comparator == "minimize":
        return observed < baseline_value
    if comparator == "maximize":
        return observed > baseline_value
    # best_ever: record-only ratchet; tighten on strict improvement (lower is better).
    return observed < baseline_value


def compare_metric(
    baseline_value: float,
    observed: float,
    comparator: Comparator,
) -> bool:
    """Return True when observed passes the gate for this comparator."""
    if comparator == "minimize":
        return observed <= baseline_value
    if comparator == "maximize":
        return observed >= baseline_value
    # best_ever: recorded-not-blocked; gate passes; baseline tightens on improvement.
    return True


def baseline_to_dict(baseline: AuditBaseline) -> dict[str, object]:
    return {
        "schema_version": baseline.schema_version,
        "updated_at": baseline.updated_at,
        "metrics": {key: asdict(entry) for key, entry in sorted(baseline.metrics.items())},
    }


def _as_comparator(value: object) -> Comparator:
    if value not in ("minimize", "maximize", "best_ever"):
        msg = f"comparator must be minimize|maximize|best_ever, got {value!r}"
        raise ValueError(msg)
    return cast(Comparator, value)


def baseline_from_dict(data: dict[str, object]) -> AuditBaseline:
    raw_schema = data["schema_version"]
    if not isinstance(raw_schema, (int, str)):
        msg = "schema_version must be an int or str"
        raise ValueError(msg)
    schema_version = int(raw_schema)
    assert_baseline_schema_compatible(schema_version)
    raw_metrics = data.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        msg = "metrics must be a mapping"
        raise ValueError(msg)
    metrics: dict[str, MetricEntry] = {}
    for key, payload in raw_metrics.items():
        if not isinstance(payload, dict):
            msg = f"metric {key!r} must be a mapping"
            raise ValueError(msg)
        metric = cast(dict[str, object], payload)
        raw_key = metric.get("key", key)
        raw_value = metric.get("value")
        raw_comparator = metric.get("comparator")
        if raw_value is None or raw_comparator is None:
            msg = f"metric {key!r} must include value and comparator"
            raise ValueError(msg)
        if not isinstance(raw_key, str):
            raw_key = str(key)
        if not isinstance(raw_value, (int, float, str)):
            msg = f"metric {key!r} value must be numeric"
            raise ValueError(msg)
        metrics[str(key)] = MetricEntry(
            key=raw_key,
            value=float(raw_value),
            comparator=_as_comparator(raw_comparator),
        )
    return AuditBaseline(
        schema_version=schema_version,
        updated_at=str(data["updated_at"]),
        metrics=metrics,
    )


def load_baseline(
    path: Path = DEFAULT_BASELINE_PATH,
) -> AuditBaseline:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "baseline JSON root must be an object"
        raise ValueError(msg)
    return baseline_from_dict(payload)


def save_baseline(
    baseline: AuditBaseline,
    path: Path = DEFAULT_BASELINE_PATH,
) -> None:
    """Persist baseline via atomic tmp-rename write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(baseline_to_dict(baseline), indent=2, sort_keys=True)
    serialized = f"{serialized}\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def evaluate_metric(
    baseline: AuditBaseline,
    key: str,
    observed: float,
) -> tuple[bool, bool]:
    """Return (passes_gate, should_update_baseline)."""
    entry = baseline.metrics.get(key)
    if entry is None:
        return True, True
    passes_gate = compare_metric(entry.value, observed, entry.comparator)
    should_update = _should_tighten(entry.value, observed, entry.comparator)
    return passes_gate, should_update


def update_metric(
    baseline: AuditBaseline,
    key: str,
    observed: float,
    comparator: Comparator,
) -> AuditBaseline:
    """Bootstrap or tighten a metric; returns a new baseline snapshot."""
    entry = baseline.metrics.get(key)
    if entry is None:
        new_entry = MetricEntry(key=key, value=observed, comparator=comparator)
    elif _should_tighten(entry.value, observed, entry.comparator):
        new_entry = replace(entry, value=observed)
    else:
        new_entry = entry
    metrics = dict(baseline.metrics)
    metrics[key] = new_entry
    return replace(
        baseline,
        updated_at=_utc_now_iso(),
        metrics=metrics,
    )

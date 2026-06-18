"""D1 correctness gate: jaxlint JL count + baseline ratchet (N2.1 / #1581)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.devtools.baseline import (
    DEFAULT_BASELINE_PATH,
    evaluate_metric,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.devtools.emit import Severity, append_finding, emit_metric_finding

METRIC_KEY = "correctness.jl_violation_count"
DIMENSION = "correctness"


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    violation_count: int
    findings_emitted: int
    baseline_updated: bool
    metric_key: str = METRIC_KEY


def _map_severity(jaxlint_level: str) -> Severity:
    level = jaxlint_level.lower()
    if level == "error":
        return "major"
    if level == "warning":
        return "minor"
    return "info"


def _run_jaxlint_json(target: Path, *, root: Path) -> list[dict[str, Any]]:
    cmd = [
        "uv",
        "run",
        "jaxlint",
        "check",
        "--format",
        "json",
        "--no-doc",
        str(target),
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if not proc.stdout.strip():
        return []
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(findings, list):
        return []
    return [f for f in findings if isinstance(f, dict)]


def filter_jl_errors(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep jaxlint error findings whose rule_id starts with JL."""
    errors: list[dict[str, Any]] = []
    for finding in findings:
        rule_id = str(finding.get("rule_id", ""))
        severity = str(finding.get("severity", "")).lower()
        if rule_id.startswith("JL") and severity == "error":
            errors.append(finding)
    return errors


def _file_line(finding: dict[str, Any]) -> str:
    path = str(finding.get("path", ""))
    line = finding.get("line", 0)
    return f"{path}:{line}"


def run_correctness_gate(
    target: Path,
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    *,
    root: Path | None = None,
    run_id: str | None = None,
    write_baseline: bool = True,
) -> GateResult:
    """Run jaxlint JL error count, emit findings, evaluate baseline ratchet."""
    resolved_root = root or Path.cwd()
    raw_findings = _run_jaxlint_json(target.resolve(), root=resolved_root)
    jl_errors = filter_jl_errors(raw_findings)
    violation_count = len(jl_errors)

    emitted = 0
    for finding in jl_errors:
        rule_id = str(finding.get("rule_id", ""))
        message = str(finding.get("message", ""))
        record = emit_metric_finding(
            dim=DIMENSION,
            severity=_map_severity(str(finding.get("severity", "error"))),
            file_line=_file_line(finding),
            evidence=message,
            rule_id=rule_id,
            symbol_qualname="",
            payload={"violation_kind": "jaxlint_jl"},
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    baseline = load_baseline(path=baseline_path)
    passes_gate, should_update = evaluate_metric(
        baseline,
        METRIC_KEY,
        float(violation_count),
    )
    baseline_updated = False
    if passes_gate and should_update and write_baseline:
        tightened = update_metric(
            baseline,
            METRIC_KEY,
            float(violation_count),
            "minimize",
        )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passes_gate,
        violation_count=violation_count,
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
    )

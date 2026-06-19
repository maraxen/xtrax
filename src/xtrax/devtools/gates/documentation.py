"""D5 documentation gate: interrogate coverage% + jaxlint JD/JM (N2.6 / #1585)."""

from __future__ import annotations

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
from xtrax.devtools.emit import append_finding, emit_metric_finding
from xtrax.devtools.gates._interrogate import run_interrogate_coverage
from xtrax.devtools.gates._jaxlint import (
    file_line,
    map_severity,
)
from xtrax.devtools.gates._jaxlint import (
    run_jaxlint_json as _run_jaxlint_json,
)

COVERAGE_METRIC_KEY = "documentation.interrogate_coverage_pct"
JD_METRIC_KEY = "documentation.jd_violation_count"
DIMENSION = "documentation"


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    interrogate_coverage_pct: float
    jd_violation_count: int
    findings_emitted: int
    baseline_updated: bool
    coverage_metric_key: str = COVERAGE_METRIC_KEY
    jd_metric_key: str = JD_METRIC_KEY


def filter_jd_jm_errors(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep jaxlint error findings for documentation rules JD* and JM*."""
    errors: list[dict[str, Any]] = []
    for finding in findings:
        rule_id = str(finding.get("rule_id", ""))
        severity = str(finding.get("severity", "")).lower()
        if (rule_id.startswith("JD") or rule_id.startswith("JM")) and severity == "error":
            errors.append(finding)
    return errors


def run_documentation_gate(
    target: Path,
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    *,
    root: Path | None = None,
    run_id: str | None = None,
    write_baseline: bool = True,
) -> GateResult:
    """Run interrogate + jaxlint JD/JM, emit findings, evaluate baseline ratchet."""
    resolved_root = root or Path.cwd()
    resolved_target = target.resolve()

    coverage_pct = run_interrogate_coverage(resolved_target, resolved_root)
    raw_findings = _run_jaxlint_json(
        resolved_target,
        root=resolved_root,
        performance_only=False,
    )
    jd_errors = filter_jd_jm_errors(raw_findings)
    jd_violation_count = len(jd_errors)

    emitted = 0
    record = emit_metric_finding(
        dim=DIMENSION,
        severity="info",
        file_line=str(resolved_target),
        evidence=f"interrogate coverage {coverage_pct:.1f}%",
        rule_id="documentation.interrogate_coverage",
        symbol_qualname="",
        payload={
            "violation_kind": "interrogate_coverage",
            "interrogate_coverage_pct": coverage_pct,
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)
    emitted += 1

    for finding in jd_errors:
        rule_id = str(finding.get("rule_id", ""))
        message = str(finding.get("message", ""))
        record = emit_metric_finding(
            dim=DIMENSION,
            severity=map_severity(str(finding.get("severity", "error"))),
            file_line=file_line(finding),
            evidence=message,
            rule_id=rule_id,
            symbol_qualname="",
            payload={"violation_kind": "jaxlint_jd_jm"},
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    baseline = load_baseline(path=baseline_path)
    passes_coverage, update_coverage = evaluate_metric(
        baseline,
        COVERAGE_METRIC_KEY,
        coverage_pct,
    )
    passes_jd, update_jd = evaluate_metric(
        baseline,
        JD_METRIC_KEY,
        float(jd_violation_count),
    )
    passed = passes_coverage and passes_jd

    baseline_updated = False
    if passed and write_baseline and (update_coverage or update_jd):
        tightened = baseline
        if update_coverage:
            tightened = update_metric(
                tightened,
                COVERAGE_METRIC_KEY,
                coverage_pct,
                "maximize",
            )
        if update_jd:
            tightened = update_metric(
                tightened,
                JD_METRIC_KEY,
                float(jd_violation_count),
                "minimize",
            )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passed,
        interrogate_coverage_pct=coverage_pct,
        jd_violation_count=jd_violation_count,
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
    )

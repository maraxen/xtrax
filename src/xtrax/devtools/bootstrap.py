"""N3.1 audit bootstrap — orchestrate gates and emit backlog manifest (#1592)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from xtrax.devtools.baseline import AuditBaseline, load_baseline
from xtrax.devtools.gates.api_ergonomics import (
    METRIC_KEY as API_METRIC_KEY,
)
from xtrax.devtools.gates.api_ergonomics import (
    run_api_ergonomics_gate,
)
from xtrax.devtools.gates.correctness import (
    METRIC_KEY as CORRECTNESS_METRIC_KEY,
)
from xtrax.devtools.gates.correctness import (
    run_correctness_gate,
)
from xtrax.devtools.gates.documentation import (
    COVERAGE_METRIC_KEY as DOC_COVERAGE_METRIC_KEY,
)
from xtrax.devtools.gates.documentation import (
    JD_METRIC_KEY as DOC_JD_METRIC_KEY,
)
from xtrax.devtools.gates.documentation import (
    run_documentation_gate,
)
from xtrax.devtools.gates.jax_purity import (
    METRIC_KEY as JAX_PURITY_METRIC_KEY,
)
from xtrax.devtools.gates.jax_purity import (
    run_jax_purity_gate,
)
from xtrax.devtools.gates.performance import (
    DEFAULT_TARGETS_PATH,
    WALL_TIME_METRIC_KEY,
    run_performance_gate,
)
from xtrax.devtools.gates.performance import (
    METRIC_KEY as PERFORMANCE_TRACE_METRIC_KEY,
)
from xtrax.devtools.gates.structure_complexity import (
    COGNITIVE_METRIC,
    RUFF_METRIC,
    run_structure_complexity_gate,
)
from xtrax.devtools.gates.test_rigor import (
    BRANCH_METRIC,
    LINE_METRIC,
    run_test_rigor_gate,
)
from xtrax.devtools.gates.type_hardening import (
    COVERAGE_METRIC_KEY as TYPE_COVERAGE_METRIC_KEY,
)
from xtrax.devtools.gates.type_hardening import (
    SHAPE_METRIC_KEY as TYPE_SHAPE_METRIC_KEY,
)
from xtrax.devtools.gates.type_hardening import (
    run_type_hardening_gate,
)

DEFAULT_MANIFEST_PATH = Path(".praxia/audit_bootstrap_manifest.toml")

_FAILURE_HINTS: dict[str, str] = {
    "correctness": "Reduce jaxlint JL error count below the baseline ratchet.",
    "jax_purity": "Fix JL001-JL012 purity violations at transform boundaries.",
    "type_hardening": "Raise annotation coverage and jaxtyping shape specificity.",
    "performance": "Eliminate excess trace/recompile counts in performance probes.",
    "documentation": "Improve interrogate docstring coverage and fix JD/JM violations.",
    "api_ergonomics": "Refactor param-sprawl public APIs to fewer required parameters.",
    "test_rigor": "Increase pytest line and branch coverage above baseline.",
    "structure_complexity": (
        "Refactor high cognitive-complexity and ruff C901/PLR091x hotspots."
    ),
}

_DEBT_HINTS: dict[str, str] = {
    "correctness": "Ratchet ceiling debt: JL violations remain at baseline allowance.",
    "jax_purity": "Ratchet ceiling debt: purity JL hits remain at baseline allowance.",
    "type_hardening": (
        "Ratchet floor debt: annotation or shape metrics stuck at baseline."
    ),
    "performance": (
        "Ratchet ceiling debt: trace violations remain at baseline allowance."
    ),
    "documentation": "Ratchet floor/ceiling debt: doc coverage or JD hits at baseline.",
    "api_ergonomics": "Ratchet ceiling debt: param-sprawl count remains at baseline.",
    "test_rigor": "Ratchet floor debt: coverage percentages remain at baseline.",
    "structure_complexity": (
        "Ratchet ceiling debt: complexity metrics remain at baseline allowance."
    ),
}


@dataclass(frozen=True, slots=True)
class DimensionRun:
    dimension: str
    passed: bool
    metrics: dict[str, float]
    findings_emitted: int
    baseline_updated: bool


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    passed: bool
    runs: tuple[DimensionRun, ...]
    manifest_path: Path


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _metric_at_debt(
    key: str,
    observed: float,
    baseline: AuditBaseline,
) -> bool:
    entry = baseline.metrics.get(key)
    if entry is None:
        return False
    if entry.comparator == "minimize":
        return observed >= entry.value and entry.value > 0
    if entry.comparator == "maximize":
        return observed <= entry.value and observed < 100.0
    return observed >= entry.value and entry.value > 0


def _backlog_seed(
    dimension: str,
    passed: bool,
    metrics: dict[str, float],
    baseline: AuditBaseline,
) -> str:
    if not passed:
        return _FAILURE_HINTS[dimension]
    if any(_metric_at_debt(key, value, baseline) for key, value in metrics.items()):
        return _DEBT_HINTS[dimension]
    return ""


def _format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6g}"
        if "." not in text:
            return f"{text}.0"
        return text
    if isinstance(value, int):
        return f"{value}.0"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    msg = f"unsupported TOML value type: {type(value).__name__}"
    raise TypeError(msg)


def _format_inline_table(metrics: dict[str, float]) -> str:
    if not metrics:
        return "{}"
    parts = [
        f'"{key}" = {_format_toml_value(value)}'
        for key, value in sorted(metrics.items())
    ]
    return "{ " + ", ".join(parts) + " }"


def write_bootstrap_manifest(
    path: Path,
    *,
    run_id: str,
    updated_at: str,
    all_passed: bool,
    runs: tuple[DimensionRun, ...],
    baseline: AuditBaseline,
) -> None:
    """Write per-dimension bootstrap manifest TOML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[bootstrap]",
        f"run_id = {_format_toml_value(run_id)}",
        f"updated_at = {_format_toml_value(updated_at)}",
        f"all_passed = {_format_toml_value(all_passed)}",
        "",
    ]
    for run in runs:
        seed = _backlog_seed(run.dimension, run.passed, run.metrics, baseline)
        lines.extend(
            [
                "[[dimensions]]",
                f"dimension = {_format_toml_value(run.dimension)}",
                f"passed = {_format_toml_value(run.passed)}",
                f"metrics = {_format_inline_table(run.metrics)}",
                f"backlog_seed = {_format_toml_value(seed)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit_bootstrap(
    audits_path: Path,
    baseline_path: Path,
    root: Path,
    *,
    write_baseline: bool = True,
    test_rigor_quick: bool = True,
    run_id: str | None = None,
    manifest_path: Path | None = None,
) -> BootstrapResult:
    """Run all eight dimension gates, optionally tighten baseline, emit manifest."""
    resolved_root = root.resolve()
    resolved_audits = audits_path.resolve()
    resolved_baseline = baseline_path.resolve()
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else (resolved_root / DEFAULT_MANIFEST_PATH).resolve()
    )
    resolved_run_id = run_id or str(uuid.uuid4())
    target = resolved_root / "src" / "xtrax"
    tests_path = (
        resolved_root / "tests" / "audit"
        if test_rigor_quick
        else resolved_root / "tests"
    )
    performance_targets = resolved_root / DEFAULT_TARGETS_PATH

    correctness = run_correctness_gate(
        target=target,
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        root=resolved_root,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    jax_purity = run_jax_purity_gate(
        target=target,
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        root=resolved_root,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    type_hardening = run_type_hardening_gate(
        target=target,
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    performance = run_performance_gate(
        targets_path=performance_targets,
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    documentation = run_documentation_gate(
        target=target,
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        root=resolved_root,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    api_ergonomics = run_api_ergonomics_gate(
        target=target,
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    test_rigor = run_test_rigor_gate(
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        root=resolved_root,
        tests_path=tests_path,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )
    structure_complexity = run_structure_complexity_gate(
        audits_path=resolved_audits,
        baseline_path=resolved_baseline,
        root=resolved_root,
        run_id=resolved_run_id,
        write_baseline=write_baseline,
    )

    runs = (
        DimensionRun(
            dimension="correctness",
            passed=correctness.passed,
            metrics={CORRECTNESS_METRIC_KEY: float(correctness.violation_count)},
            findings_emitted=correctness.findings_emitted,
            baseline_updated=correctness.baseline_updated,
        ),
        DimensionRun(
            dimension="jax_purity",
            passed=jax_purity.passed,
            metrics={JAX_PURITY_METRIC_KEY: float(jax_purity.violation_count)},
            findings_emitted=jax_purity.findings_emitted,
            baseline_updated=jax_purity.baseline_updated,
        ),
        DimensionRun(
            dimension="type_hardening",
            passed=type_hardening.passed,
            metrics={
                TYPE_COVERAGE_METRIC_KEY: type_hardening.annotation_coverage_pct,
                TYPE_SHAPE_METRIC_KEY: type_hardening.shape_specificity_pct,
            },
            findings_emitted=type_hardening.findings_emitted,
            baseline_updated=type_hardening.baseline_updated,
        ),
        DimensionRun(
            dimension="performance",
            passed=performance.passed,
            metrics=_performance_metrics(
                performance.trace_violation_count,
                performance.wall_time_median_ms,
            ),
            findings_emitted=performance.findings_emitted,
            baseline_updated=performance.baseline_updated,
        ),
        DimensionRun(
            dimension="documentation",
            passed=documentation.passed,
            metrics={
                DOC_COVERAGE_METRIC_KEY: documentation.interrogate_coverage_pct,
                DOC_JD_METRIC_KEY: float(documentation.jd_violation_count),
            },
            findings_emitted=documentation.findings_emitted,
            baseline_updated=documentation.baseline_updated,
        ),
        DimensionRun(
            dimension="api_ergonomics",
            passed=api_ergonomics.passed,
            metrics={API_METRIC_KEY: float(api_ergonomics.violation_count)},
            findings_emitted=api_ergonomics.findings_emitted,
            baseline_updated=api_ergonomics.baseline_updated,
        ),
        DimensionRun(
            dimension="test_rigor",
            passed=test_rigor.passed,
            metrics={
                LINE_METRIC: test_rigor.line_coverage_pct,
                BRANCH_METRIC: test_rigor.branch_coverage_pct,
            },
            findings_emitted=test_rigor.findings_emitted,
            baseline_updated=test_rigor.baseline_updated,
        ),
        DimensionRun(
            dimension="structure_complexity",
            passed=structure_complexity.passed,
            metrics={
                COGNITIVE_METRIC: structure_complexity.cognitive_complexity_max,
                RUFF_METRIC: float(structure_complexity.ruff_violation_count),
            },
            findings_emitted=structure_complexity.findings_emitted,
            baseline_updated=structure_complexity.baseline_updated,
        ),
    )

    all_passed = all(run.passed for run in runs)
    baseline = load_baseline(path=resolved_baseline)
    write_bootstrap_manifest(
        resolved_manifest,
        run_id=resolved_run_id,
        updated_at=_utc_now_iso(),
        all_passed=all_passed,
        runs=runs,
        baseline=baseline,
    )

    return BootstrapResult(
        passed=all_passed,
        runs=runs,
        manifest_path=resolved_manifest,
    )


def _performance_metrics(
    trace_violation_count: int,
    wall_time_median_ms: float | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        PERFORMANCE_TRACE_METRIC_KEY: float(trace_violation_count),
    }
    if wall_time_median_ms is not None:
        metrics[WALL_TIME_METRIC_KEY] = wall_time_median_ms
    return metrics


__all__ = [
    "BootstrapResult",
    "DEFAULT_MANIFEST_PATH",
    "DimensionRun",
    "run_audit_bootstrap",
    "write_bootstrap_manifest",
]

"""D4 performance gate: trace-count blocking + recorded wall-time (N2.5 / #1584)."""

from __future__ import annotations

import statistics
import time
import tomllib
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
from xtrax.devtools.gates._dispatch_probe import measure_dispatch_counts
from xtrax.devtools.gates._trace_probe import (
    ProbeResult,
    import_probe,
    import_qualname,
    run_trace_gate,
    run_trace_probe,
)
from xtrax.profiling.emitters import emit_probe_record

METRIC_KEY = "performance.trace_violation_count"
WALL_TIME_METRIC_KEY = "performance.wall_time_median_ms"
DISPATCH_METRIC_KEY = "performance.dispatch_violation_count"
DIMENSION = "performance"
DEFAULT_TARGETS_PATH = Path("audit/performance_targets.toml")
WALL_TIME_SAMPLES = 3
# D6-style anchoring: records land under the REPOSITORY's outputs/, never the
# caller's cwd.
DEFAULT_PROBE_RECORD_DIR = (
    Path(__file__).resolve().parents[4] / "outputs" / "profiling" / "gate"
)


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    qualname: str
    max_traces: int
    trace_probe: str | None = None
    # Phase C profiler-backed probe kinds -- all opt-in; absent fields leave
    # gate behavior byte-identical to the pre-profiling configuration.
    max_compilations: int | None = None
    max_jit_traces: int | None = None
    emit_probe_record: bool = False

    @property
    def has_dispatch_ceilings(self) -> bool:
        return self.max_compilations is not None or self.max_jit_traces is not None


@dataclass(frozen=True, slots=True)
class PerformanceTargets:
    schema: str
    version: str
    max_traces_default: int
    probes: tuple[ProbeSpec, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    trace_violation_count: int
    wall_time_median_ms: float | None
    findings_emitted: int
    baseline_updated: bool
    metric_key: str = METRIC_KEY
    dispatch_violation_count: int = 0


def load_performance_targets(path: Path) -> PerformanceTargets:
    """Load audit/performance_targets.toml probe configuration."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    gate = data.get("gate", {})
    if not isinstance(gate, dict):
        msg = "performance_targets.toml [gate] must be a table"
        raise ValueError(msg)

    schema = str(gate.get("schema", ""))
    version = str(gate.get("version", ""))
    max_traces_default = gate.get("max_traces_default", 1)
    if not isinstance(max_traces_default, int) or max_traces_default < 1:
        msg = f"gate.max_traces_default must be a positive int, got {max_traces_default!r}"
        raise ValueError(msg)

    raw_probes = data.get("probes", [])
    if not isinstance(raw_probes, list):
        msg = "probes must be a list of tables"
        raise ValueError(msg)

    probes: list[ProbeSpec] = []
    for entry in raw_probes:
        if not isinstance(entry, dict):
            continue
        qualname = entry.get("qualname")
        if not isinstance(qualname, str) or not qualname:
            continue
        max_traces = entry.get("max_traces", max_traces_default)
        if not isinstance(max_traces, int) or max_traces < 1:
            msg = f"probe max_traces must be a positive int for {qualname!r}"
            raise ValueError(msg)
        trace_probe = entry.get("trace_probe")
        probe_value = trace_probe if isinstance(trace_probe, str) else None

        def _optional_ceiling(name: str) -> int | None:
            raw = entry.get(name)
            if raw is None:
                return None
            if not isinstance(raw, int) or raw < 1:
                msg = f"probe {name} must be a positive int for {qualname!r}, got {raw!r}"
                raise ValueError(msg)
            return raw

        emit_record_raw = entry.get("emit_probe_record", False)
        if not isinstance(emit_record_raw, bool):
            msg = f"probe emit_probe_record must be a bool for {qualname!r}"
            raise ValueError(msg)
        probes.append(
            ProbeSpec(
                qualname=qualname,
                max_traces=max_traces,
                trace_probe=probe_value,
                max_compilations=_optional_ceiling("max_compilations"),
                max_jit_traces=_optional_ceiling("max_jit_traces"),
                emit_probe_record=emit_record_raw,
            )
        )

    return PerformanceTargets(
        schema=schema,
        version=version,
        max_traces_default=max_traces_default,
        probes=tuple(probes),
    )


def _emit_trace_failure(
    result: ProbeResult,
    *,
    audits_path: Path,
    run_id: str | None,
) -> None:
    record = emit_metric_finding(
        dim=DIMENSION,
        severity="major",
        file_line=f"probe:{result.qualname}",
        evidence=result.reason or "trace count exceeded max_traces",
        rule_id="performance.trace_count",
        symbol_qualname=result.qualname,
        payload={
            "violation_kind": "trace_count",
            "max_traces": result.max_traces,
            "trace_probe": result.trace_probe,
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)


def _measure_wall_time_median_ms(
    spec: ProbeSpec,
    *,
    samples: int = WALL_TIME_SAMPLES,
) -> float | None:
    if not spec.trace_probe:
        return None

    target = import_qualname(spec.qualname)
    probe_runner = import_probe(spec.trace_probe)
    timings_ms: list[float] = []

    for _ in range(samples):
        start = time.perf_counter()

        def _invoke(guarded: Any) -> None:
            probe_runner(guarded)

        run_trace_gate(target, _invoke, max_traces=spec.max_traces)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return float(statistics.median(timings_ms))


def _emit_dispatch_failure(
    spec: ProbeSpec,
    counter: str,
    observed: int,
    ceiling: int,
    *,
    audits_path: Path,
    run_id: str | None,
) -> None:
    record = emit_metric_finding(
        dim=DIMENSION,
        severity="major",
        file_line=f"probe:{spec.qualname}",
        evidence=(
            f"{counter}={observed} exceeded max_{counter}={ceiling} "
            f"for {spec.qualname}"
        ),
        rule_id="performance.dispatch_count",
        symbol_qualname=spec.qualname,
        payload={
            "violation_kind": "dispatch_count",
            "counter": counter,
            "observed": observed,
            "ceiling": ceiling,
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)


def _run_dispatch_probes(
    targets: PerformanceTargets,
    *,
    audits_path: Path,
    run_id: str | None,
    probe_record_dir: Path,
) -> tuple[int, int]:
    """Opt-in dispatch tripwires + ProbeRecord emission; returns (violations, emitted)."""
    violations = 0
    emitted = 0
    for spec in targets.probes:
        if not (spec.has_dispatch_ceilings or spec.emit_probe_record):
            continue
        try:
            result = measure_dispatch_counts(spec.qualname, spec.trace_probe)
        except Exception as exc:  # noqa: BLE001 — a broken probe reports; it must not crash the gate
            finding = emit_metric_finding(
                dim=DIMENSION,
                severity="major",
                file_line=f"probe:{spec.qualname}",
                evidence=(
                    f"dispatch probe for {spec.qualname} failed to run: {exc}"
                ),
                rule_id="performance.dispatch_probe_error",
                symbol_qualname=spec.qualname,
                payload={"violation_kind": "probe_error"},
                run_id=run_id,
            )
            append_finding(finding, audits_path=audits_path)
            violations += 1
            emitted += 1
            continue
        if result.skipped:
            continue
        for counter, ceiling in (
            ("compilations", spec.max_compilations),
            ("jit_traces", spec.max_jit_traces),
        ):
            if ceiling is None:
                continue
            observed = result.counts.get(f"n_{counter}", 0)
            if observed > ceiling:
                _emit_dispatch_failure(
                    spec,
                    f"n_{counter}",
                    observed,
                    ceiling,
                    audits_path=audits_path,
                    run_id=run_id,
                )
                violations += 1
                emitted += 1
        if spec.emit_probe_record:
            safe = spec.qualname.replace(".", "_")
            emit_probe_record(
                path=probe_record_dir / f"gate_{safe}.json",
                probe_id=f"perf_gate_{safe}",
                stage=1,
                n_atoms=1,
                platform="cpu",
                metrics=dict(result.counts),
                config={
                    "qualname": spec.qualname,
                    "source": "performance_gate",
                    "axis_note": (
                        "scale-free gate artifact; n_atoms placeholder by contract"
                    ),
                },
            )
    return violations, emitted


def run_performance_gate(
    targets_path: Path,
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    *,
    run_id: str | None = None,
    write_baseline: bool = True,
    probe_record_dir: Path | None = None,
) -> GateResult:
    """Run trace-count gate (blocking) and record wall-time median (non-blocking)."""
    targets = load_performance_targets(targets_path)
    probe_results: list[ProbeResult] = []
    for spec in targets.probes:
        probe_results.append(
            run_trace_probe(
                spec.qualname,
                max_traces=spec.max_traces,
                trace_probe=spec.trace_probe,
            )
        )

    failures = [result for result in probe_results if not result.skipped and not result.passed]
    trace_violation_count = len(failures)

    emitted = 0
    for result in failures:
        _emit_trace_failure(result, audits_path=audits_path, run_id=run_id)
        emitted += 1

    wall_time_median_ms: float | None = None
    representative = next((spec for spec in targets.probes if spec.trace_probe), None)
    if representative is not None:
        wall_time_median_ms = _measure_wall_time_median_ms(representative)
        record = emit_metric_finding(
            dim=DIMENSION,
            severity="info",
            file_line=f"probe:{representative.qualname}",
            evidence=(
                f"wall_time_median_ms={wall_time_median_ms:.3f} for {representative.qualname}"
            ),
            rule_id=WALL_TIME_METRIC_KEY,
            symbol_qualname=representative.qualname,
            payload={
                "recorded_only": True,
                "wall_time_median_ms": wall_time_median_ms,
                "trace_probe": representative.trace_probe,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    # --- Phase C: opt-in dispatch tripwires + ProbeRecord emission --------
    record_dir = (
        probe_record_dir if probe_record_dir is not None else DEFAULT_PROBE_RECORD_DIR
    )
    dispatch_violation_count, extra_emitted = _run_dispatch_probes(
        targets,
        audits_path=audits_path,
        run_id=run_id,
        probe_record_dir=record_dir,
    )
    emitted += extra_emitted

    baseline = load_baseline(path=baseline_path)
    passes_gate, should_update = evaluate_metric(
        baseline,
        METRIC_KEY,
        float(trace_violation_count),
    )
    dispatch_ceilings_configured = any(
        spec.has_dispatch_ceilings for spec in targets.probes
    )
    if dispatch_ceilings_configured:
        # Opt-in ratchet: only evaluated when a probe actually configures
        # ceilings -- otherwise the bootstrap-on-missing-key semantics would
        # stamp a 0.0 entry into every repo's baseline uninvited.
        dispatch_passes, dispatch_should_update = evaluate_metric(
            baseline,
            DISPATCH_METRIC_KEY,
            float(dispatch_violation_count),
        )
        passes_gate = passes_gate and dispatch_passes
        should_update = should_update or dispatch_should_update
    baseline_updated = False
    if passes_gate and should_update and write_baseline:
        tightened = update_metric(
            baseline,
            METRIC_KEY,
            float(trace_violation_count),
            "minimize",
        )
        if wall_time_median_ms is not None:
            _, wall_should_update = evaluate_metric(
                tightened,
                WALL_TIME_METRIC_KEY,
                wall_time_median_ms,
            )
            if wall_should_update:
                tightened = update_metric(
                    tightened,
                    WALL_TIME_METRIC_KEY,
                    wall_time_median_ms,
                    "best_ever",
                )
        if dispatch_ceilings_configured:
            _, dispatch_tighten = evaluate_metric(
                tightened,
                DISPATCH_METRIC_KEY,
                float(dispatch_violation_count),
            )
            if dispatch_tighten:
                tightened = update_metric(
                    tightened,
                    DISPATCH_METRIC_KEY,
                    float(dispatch_violation_count),
                    "minimize",
                )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passes_gate,
        trace_violation_count=trace_violation_count,
        wall_time_median_ms=wall_time_median_ms,
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
        dispatch_violation_count=dispatch_violation_count,
    )

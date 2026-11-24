"""Operational telemetry, shadow heuristic calibration, and algedonic detection.

Records optional ``cisternal`` M1 operational events about the profiling and
comparison loop, evaluates ``cisternal.shadow`` calibration gates for
provisional GPU heuristic rules, detects algedonic failure signals, and appends
optional flock-guarded findings into existing ``.praxia/audits.jsonl`` ledgers.

Omits future annotations and keeps zero ``bathos`` imports, zero
tracker dispatch strings, and zero module-level ``jax`` or ``cisternal``
imports so core installs and CI audits stay completely isolated.
"""

import contextlib
import dataclasses
import enum
import hashlib
import importlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from xtrax.findings import AuditFinding, Severity, emit_metric_finding, finding_to_dict
from xtrax.profiling.compare import DeltaVerdict, VerdictStatus
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.triage import BottleneckFinding, triage_probe_bundle
from xtrax.telemetry.record import telemetry_opted_out
from xtrax.tombstone import DEFAULT_TOMBSTONE_PATH, is_tombstoned

try:
    import fcntl

    _HAVE_FLOCK = True
except ImportError:  # pragma: no cover - non-POSIX fallback
    _HAVE_FLOCK = False

PROVISIONAL_TRIAGE_CATEGORIES: tuple[str, ...] = (
    "fp64_pipe_saturation",
    "register_spill_pressure",
    "spatial_dead_work",
    "atomic_contention",
)

TRIAGE_SHADOW_ENFORCER = (
    "tests/telemetry/test_profiling_events.py::test_triage_shadow_writes_isolated"
)
_LOG_FILE_RE = re.compile(r"(?P<base>.+\.jsonl)(?:\.(?P<rot>\d+))?")
_LOG_ENV_VARS: tuple[str, ...] = (
    "CISTERNAL_TELEMETRY_LOG_DIR",
    "CISTERNAL_LOG_DIR",
    "BTH_LOG_DIR",
    "CTXP_LOG_DIR",
)
_CLI_INVOCATION_EVENTS = frozenset(
    {
        "xtrax.profile.measure",
        "xtrax.profile.check",
        "xtrax.profile.compare",
        "xtrax.profile.triage",
        "xtrax.profile.not_measured",
    }
)
_HARD_BLOCK_PREFIXES = frozenset(
    {
        "precision_deficit",
        "precision_lower_bound_deficit",
        "fpr_exceeded",
        "recall_deficit",
        "recall_lower_bound_deficit",
    }
)
_EROSION_REASONS = frozenset(
    {
        "parity_gate_bypassed",
        "hold_fixed_bypassed",
        "dirty_git_sha_advisory_only",
    }
)


class TelemetryReadError(ValueError):
    """Raised when operational telemetry JSONL or verdict sidecars are corrupt or unreadable."""


@dataclasses.dataclass(slots=True)
class TelemetrySessionStatus:
    """Lifecycle outcome for :func:`profiling_telemetry_session`."""

    active: bool
    log_dir: Path | None = None
    reason: str = ""
    init_owned: bool = False
    retargeted_from_active: bool = False
    degraded_reason: str | None = None


class AlgedonicSignalKind(enum.Enum):
    """Data-driven operational failure modes surfaced by :func:`detect_algedonic_signals`."""

    INSTRUMENT_DEGRADED = "instrument_degraded"
    NOISE_FLOOR_BLOWN = "noise_floor_blown"
    GUARDRAIL_EROSION = "guardrail_erosion"
    PARITY_ESCAPE = "parity_escape"
    HEURISTIC_MISCALIBRATED = "heuristic_miscalibrated"
    TRIAGE_GAP = "triage_gap"


@dataclasses.dataclass(frozen=True, slots=True)
class AlgedonicThresholds:
    """Detector thresholds for operational telemetry analysis."""

    min_degraded_dual_pass_per_env: int = 2
    max_consecutive_revert_noise: int = 3
    max_noise_floor_x: float = 1.10
    min_bypass_count: int = 2
    min_compare_for_rate: int = 5
    max_bypass_rate: float = 0.25
    min_invocations_for_error_rate: int = 5
    max_not_measured_rate: float = 0.30
    min_triage_gap_count: int = 2
    min_shadow_labels_for_demote: int = 10


@dataclasses.dataclass(frozen=True, slots=True)
class AlgedonicSignal:
    """Actionable operational escalation signal detected in telemetry logs."""

    kind: AlgedonicSignalKind
    severity: str
    summary: str
    evidence: dict[str, Any]
    recommended_action: str
    demote_feature: str | None = None


def _active_exporter_dirs(pipeline: Any) -> set[Path]:
    """Return resolved parent directories for active JSONL exporters."""
    dirs: set[Path] = set()
    if pipeline is None:
        return dirs
    for exp in getattr(pipeline, "_exporters", ()):
        exp_path = getattr(exp, "_path", None)
        if exp_path is not None:
            dirs.add(Path(exp_path).resolve().parent)
    return dirs


def _resolve_layers_2_to_4() -> Path | None:
    """Resolve telemetry log dir across Layers 2-4 without triggering Layer-5 default."""
    try:
        res_mod = importlib.import_module("cisternal.shadow.resolution")
        resolved = res_mod.resolve_log_dir(cli_arg=None)
        return Path(resolved.path).resolve()
    except ImportError:
        for name in _LOG_ENV_VARS:
            raw = os.getenv(name)
            if raw is not None:
                stripped = raw.strip()
                if not stripped or stripped.lower() == "none":
                    return None
                return Path(stripped).resolve()
        return None
    except Exception:
        return None


def _start_owned_pipeline(
    cisternal_mod: Any,
    pipeline_mod: Any,
    target_dir: Path | None,
    retargeted: bool,
) -> TelemetrySessionStatus:
    """Initialize a new ``cisternal`` pipeline without raising on failure."""
    try:
        cisternal_mod.init(log_dir=target_dir, heartbeat_interval=30.0)
        actual_dir = (
            target_dir
            if target_dir is not None
            else next(iter(_active_exporter_dirs(pipeline_mod.get_pipeline())), None)
        )
        return TelemetrySessionStatus(
            active=True,
            log_dir=actual_dir,
            reason="initialized",
            init_owned=True,
            retargeted_from_active=retargeted,
        )
    except Exception as exc:  # noqa: BLE001 - never-raise telemetry contract
        return TelemetrySessionStatus(
            active=False,
            log_dir=target_dir,
            reason="init_failed",
            init_owned=False,
            retargeted_from_active=retargeted,
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )


def _resolve_implicit_session_target(
    active: Any,
    active_dirs: set[Path],
) -> tuple[Path | None, TelemetrySessionStatus | None]:
    """Resolve target directory or early reused/disabled status when ``log_dir`` is omitted."""
    if active is not None:
        reused = next(iter(active_dirs), None)
        return None, TelemetrySessionStatus(active=True, log_dir=reused, reason="reused_active")
    resolved_dir = _resolve_layers_2_to_4()
    if resolved_dir is None and os.getenv("XTRAX_PROFILE_TELEMETRY") != "1":
        return None, TelemetrySessionStatus(active=False, reason="log_dir_off")
    return resolved_dir, None


def _resolve_explicit_session_target(
    log_dir: str | Path,
    active: Any,
    active_dirs: set[Path],
    pipeline_mod: Any,
) -> tuple[Path | None, bool, TelemetrySessionStatus | None]:
    """Resolve target directory or early status when ``log_dir`` is explicitly requested."""
    target_dir = Path(log_dir).resolve()
    if active is not None and target_dir in active_dirs:
        return (
            None,
            False,
            TelemetrySessionStatus(active=True, log_dir=target_dir, reason="reused_matching_dir"),
        )
    if active is None:
        return target_dir, False, None
    try:
        pipeline_mod.shutdown_pipeline()
        return target_dir, True, None
    except Exception as exc:  # noqa: BLE001 - best-effort shutdown
        return (
            None,
            False,
            TelemetrySessionStatus(
                active=False,
                log_dir=target_dir,
                reason="retarget_shutdown_failed",
                degraded_reason=f"{type(exc).__name__}: {exc}",
            ),
        )


@contextlib.contextmanager
def profiling_telemetry_session(
    log_dir: str | Path | None = None,
    *,
    emit_events: bool = True,
) -> Iterator[TelemetrySessionStatus]:
    """Optionally initialize and flush a ``cisternal`` M1 pipeline around profiling work."""
    if not emit_events or telemetry_opted_out():
        reason = "opted_out" if telemetry_opted_out() else "emit_disabled"
        yield TelemetrySessionStatus(active=False, reason=reason)
        return

    try:
        cisternal_mod = importlib.import_module("cisternal")
        pipeline_mod = importlib.import_module("cisternal.telemetry.pipeline")
    except ImportError:
        yield TelemetrySessionStatus(active=False, reason="cisternal_not_installed")
        return

    active = pipeline_mod.get_pipeline()
    active_dirs = _active_exporter_dirs(active)

    if log_dir is None:
        target_dir, early_status = _resolve_implicit_session_target(active, active_dirs)
        retargeted = False
    else:
        target_dir, retargeted, early_status = _resolve_explicit_session_target(
            log_dir, active, active_dirs, pipeline_mod
        )

    if early_status is not None:
        yield early_status
        return

    status = _start_owned_pipeline(cisternal_mod, pipeline_mod, target_dir, retargeted)
    try:
        yield status
    finally:
        if status.init_owned:
            try:
                pipeline_mod.shutdown_pipeline()
            except Exception as exc:  # noqa: BLE001 - never crash caller on flush
                status.degraded_reason = f"shutdown: {type(exc).__name__}: {exc}"


def _emit_m1(name: str, *, level: str = "INFO", **fields: Any) -> bool:
    """Emit a ``cisternal`` event when an active pipeline exists; never raises."""
    try:
        cisternal_mod = importlib.import_module("cisternal")
        pipeline_mod = importlib.import_module("cisternal.telemetry.pipeline")
        if pipeline_mod.get_pipeline() is None:
            return False
        cisternal_mod.emit_event(name, level=level, **fields)
        return True
    except Exception:
        return False


def emit_measure_telemetry(
    fn: str,
    platform: str,
    device_kind: str | None = None,
    *,
    cardinality: int = 1,
    n_flags: int = 0,
    exit_code: int = 0,
    duration_ms: float = 0.0,
) -> bool:
    """Emit ``xtrax.profile.measure`` operational telemetry."""
    return _emit_m1(
        "xtrax.profile.measure",
        level="INFO",
        mode="fn",
        fn=fn,
        platform=platform,
        device_kind=device_kind,
        cardinality=cardinality,
        n_flags=n_flags,
        exit_code=exit_code,
        duration_ms=float(duration_ms),
    )


def emit_check_telemetry(
    record_path: str | Path,
    exit_code: int,
    *,
    stale: bool,
    n_flags: int,
) -> bool:
    """Emit ``xtrax.profile.check`` operational telemetry."""
    return _emit_m1(
        "xtrax.profile.check",
        level="WARNING" if stale else "INFO",
        mode="check",
        record_path=str(record_path),
        exit_code=exit_code,
        stale=stale,
        n_flags=n_flags,
    )


def emit_not_measured_telemetry(mode: str, reason_code: str, reason: str) -> bool:
    """Emit ``xtrax.profile.not_measured`` operational telemetry."""
    return _emit_m1(
        "xtrax.profile.not_measured",
        level="WARNING",
        mode=mode,
        reason_code=reason_code,
        reason=reason,
        exit_code=2,
    )


def emit_dual_pass_telemetry(result: Any) -> bool:
    """Emit ``xtrax.profile.dual_pass`` capturing Pass B and Pass C health."""
    skip = getattr(result, "diag_skipped_reason", None)
    if getattr(result, "diag_record", None) is not None:
        pass_b_status = "ok"
    elif skip in ("diag_disabled", "no_importable_spec"):
        pass_b_status = str(skip)
    else:
        pass_b_status = "failed"

    pass_c_status = str(getattr(result, "ptx_capture_status", "disabled"))
    ptxas_warnings = tuple(getattr(result, "ptxas_warnings", ()))
    filter_status = str(getattr(result, "ptx_filter_status", "not_applicable"))
    hlo_skipped = getattr(result, "hlo_attribution_skipped", None)
    degraded = (
        pass_b_status == "failed"
        or pass_c_status == "failed"
        or pass_c_status.startswith("skipped:")
        or len(ptxas_warnings) > 0
        or filter_status == "unfiltered_name_mismatch"
        or bool(hlo_skipped)
    )
    prod = result.prod_record
    return _emit_m1(
        "xtrax.profile.dual_pass",
        level="WARNING" if degraded else "INFO",
        prod_probe_id=prod.probe_id,
        device_kind=prod.device_kind or "",
        jax_version=prod.jax_version,
        pass_b_status=pass_b_status,
        pass_c_status=pass_c_status,
        ptx_spill_source=str(getattr(result, "ptx_spill_source", "none")),
        ptx_filter_status=filter_status,
        hlo_attribution_skipped=hlo_skipped,
        ptxas_warning_count=len(ptxas_warnings),
        diag_skipped_reason=skip,
    )


def emit_compare_telemetry(
    verdict: DeltaVerdict,
    *,
    bypass_parity: bool = False,
    bypass_corruption: bool = False,
    bypass_hold_fixed: bool = False,
    allow_dirty: bool = False,
) -> bool:
    """Emit ``xtrax.profile.compare`` operational telemetry."""
    level = "INFO" if verdict.status is VerdictStatus.KEEP else "WARNING"
    return _emit_m1(
        "xtrax.profile.compare",
        level=level,
        mode="compare",
        baseline_probe_id=verdict.baseline_probe_id,
        candidate_probe_id=verdict.candidate_probe_id,
        baseline_sha=verdict.baseline_sha,
        candidate_sha=verdict.candidate_sha,
        verdict_status=verdict.status.value,
        promotable=verdict.promotable,
        primary_metric=verdict.primary_metric,
        speedup_x=verdict.speedup_x,
        noise_floor_x=verdict.noise_floor_x,
        reasons=list(verdict.reasons),
        bypass_parity=bypass_parity,
        bypass_corruption=bypass_corruption,
        bypass_hold_fixed=bypass_hold_fixed,
        allow_dirty=allow_dirty,
    )


def _derive_top_kernel_pct(diag: ProbeRecord | None) -> float:
    """Compute top-scope exclusive share aligned with ``report.py``."""
    if diag is None or not diag.scopes:
        return 0.0
    durations = [float(val[0]) for val in diag.scopes.values() if val is not None]
    total_exc = sum(durations)
    if total_exc <= 0.0:
        return 0.0
    return 100.0 * max(durations) / total_exc


def emit_triage_telemetry(
    prod: ProbeRecord,
    findings: Sequence[BottleneckFinding],
    *,
    diag: ProbeRecord | None = None,
    top_kernel_pct: float | None = None,
    target_missed: bool = False,
) -> bool:
    """Emit ``xtrax.profile.triage`` recording rule firings and unclassified gaps."""
    eff_top_pct = _derive_top_kernel_pct(diag) if top_kernel_pct is None else float(top_kernel_pct)
    unclassified = len(findings) == 0 and (eff_top_pct >= 30.0 or target_missed)
    return _emit_m1(
        "xtrax.profile.triage",
        level="WARNING" if unclassified else "INFO",
        mode="triage",
        prod_probe_id=prod.probe_id,
        shadow_subject_key=make_triage_subject_key(prod),
        has_diag=diag is not None,
        categories=[f.category.value for f in findings],
        n_findings=len(findings),
        top_kernel_pct=eff_top_pct,
        target_missed=target_missed,
        unclassified_slowdown=unclassified,
    )


def record_parity_escape(
    probe_id: str,
    commit_sha: str,
    reason: str,
    *,
    control_added: str = "",
) -> bool:
    """Emit ``CRITICAL`` ``xtrax.profile.parity_escape`` when a defect bypasses gating."""
    return _emit_m1(
        "xtrax.profile.parity_escape",
        level="CRITICAL",
        probe_id=probe_id,
        commit_sha=commit_sha,
        reason=reason,
        control_added=control_added,
    )


def _triage_inputs_digest(prod: ProbeRecord) -> str:
    """Full SHA-256 hex digest over canonical triage-relevant ``ProbeRecord`` fields."""
    payload = {
        "probe_id": prod.probe_id,
        "git_sha": prod.git_sha,
        "timestamp": prod.timestamp,
        "stage": prod.stage,
        "n_atoms": prod.n_atoms,
        "platform": prod.platform,
        "device_kind": prod.device_kind,
        "x64_enabled": prod.x64_enabled,
        "xla_flags": prod.xla_flags,
        "metrics": prod.metrics,
        "config": prod.config,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_triage_subject_key(prod: ProbeRecord) -> str:
    """Return canonical per-measurement ``probe:<slug>:<sha12>:<digest12>`` key."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", prod.probe_id).strip("-") or "probe"
    sha12 = re.sub(r"[^0-9a-fA-F-]+", "", prod.git_sha)[:12] or "nosha"
    digest12 = _triage_inputs_digest(prod)[:12]
    return f"probe:{slug}:{sha12}:{digest12}"


def resolve_triage_subject_key(probe_or_key: str | Path | ProbeRecord) -> str:
    """Accept a ``probe:...`` key string, ``ProbeRecord``, or path to a ``ProbeRecord``."""
    if isinstance(probe_or_key, ProbeRecord):
        return make_triage_subject_key(probe_or_key)
    if (
        isinstance(probe_or_key, str)
        and probe_or_key.startswith("probe:")
        and len(probe_or_key) > len("probe:")
    ):
        return probe_or_key
    return make_triage_subject_key(ProbeRecord.read(Path(probe_or_key)))


def emit_triage_shadow_verdicts(
    prod: ProbeRecord,
    findings: Sequence[BottleneckFinding],
    *,
    stage: str = "shadow",
    feature_version: str = "0.1.0",
) -> int:
    """Emit ``shadow.verdict`` events for all four provisional triage rules."""
    try:
        events_mod = importlib.import_module("cisternal.shadow.events")
        pipeline_mod = importlib.import_module("cisternal.telemetry.pipeline")
    except ImportError:
        return 0
    if pipeline_mod.get_pipeline() is None:
        return 0

    subject_key = make_triage_subject_key(prod)
    inputs_sha256 = _triage_inputs_digest(prod)
    fired = {f.category.value for f in findings}
    count = 0
    for cat in PROVISIONAL_TRIAGE_CATEGORIES:
        ok = events_mod.emit_verdict(
            feature=f"xtrax.triage.{cat}",
            subject_key=subject_key,
            verdict=(cat in fired),
            stage=stage,
            feature_version=feature_version,
            inputs_sha256=inputs_sha256,
        )
        if ok:
            count += 1
    return count


def emit_triage_shadow_label(
    probe_or_key: str | Path | ProbeRecord,
    category: str,
    label: bool,
    *,
    label_source: str = "at_work",
    labeller: str = "agent",
) -> bool:
    """Emit a single targeted ``shadow.label`` event for one provisional triage rule."""
    norm_category = category.strip().removeprefix("xtrax.triage.")
    if norm_category not in PROVISIONAL_TRIAGE_CATEGORIES:
        raise ValueError(
            f"Unknown triage shadow category {category!r}; expected one of "
            f"{PROVISIONAL_TRIAGE_CATEGORIES}"
        )
    subject_key = resolve_triage_subject_key(probe_or_key)
    try:
        events_mod = importlib.import_module("cisternal.shadow.events")
        pipeline_mod = importlib.import_module("cisternal.telemetry.pipeline")
    except ImportError:
        return False
    if pipeline_mod.get_pipeline() is None:
        return False
    return bool(
        events_mod.emit_label(
            feature=f"xtrax.triage.{norm_category}",
            subject_key=subject_key,
            label=label,
            label_source=label_source,
            labeller=labeller,
        )
    )


def _run_isolation_audit(
    prod: ProbeRecord,
    *,
    live_root: Path,
    shadow_log_dir: Path,
    diag: ProbeRecord | None = None,
    stage: str = "shadow",
    enforced_by: str = TRIAGE_SHADOW_ENFORCER,
    verdict_feature_version: str = "__isolation_audit__",
    inject_leak_fn: Any = None,
) -> dict[str, bool]:
    """Run ``cisternal.shadow.isolation.prove_isolation`` across triage shadow verdicts."""
    iso_mod = importlib.import_module("cisternal.shadow.isolation")
    events_mod = importlib.import_module("cisternal.shadow.events")

    def _canonical(findings: list[BottleneckFinding]) -> tuple[tuple[str, str], ...]:
        return tuple((f.category.value, f.severity) for f in findings)

    def _live_fn() -> tuple[tuple[str, str], ...]:
        return _canonical(triage_probe_bundle(prod, diag=diag))

    def _shadow_fn() -> tuple[tuple[str, str], ...]:
        findings = triage_probe_bundle(prod, diag=diag)
        emit_triage_shadow_verdicts(
            prod,
            findings,
            stage=stage,
            feature_version=verdict_feature_version,
        )
        if inject_leak_fn is not None:
            inject_leak_fn(live_root)
        return _canonical(findings)

    result = iso_mod.prove_isolation(
        _live_fn,
        _shadow_fn,
        live_roots=[live_root],
        shadow_log_dir=shadow_log_dir,
        quiescence_wait_s=0.05,
        check_roots=[live_root, shadow_log_dir],
        feature="",
        enforced_by=enforced_by,
    )
    leaked_strs = [str(p) for p in result.leaked_paths]
    out: dict[str, bool] = {}
    for cat in PROVISIONAL_TRIAGE_CATEGORIES:
        feature = f"xtrax.triage.{cat}"
        events_mod.emit_isolation(
            feature=feature,
            writes_isolated=result.passed,
            leaked_paths=leaked_strs,
            enforced_by=enforced_by,
        )
        out[feature] = bool(result.passed)
    return out


def attest_triage_shadow_isolation(
    log_dir: str | Path | None = None,
    *,
    stage: str = "shadow",
) -> dict[str, bool]:
    """Emit real ``shadow.isolation`` attestations for all four provisional triage rules."""
    prod = ProbeRecord(
        probe_id="isolation-audit-probe",
        stage=1,
        n_atoms=64,
        platform="cpu",
        git_sha="a" * 40,
        metrics={"total_step_seconds": 1.0, "active_work_ratio": 0.10, "host_seconds": 0.0},
    )
    with tempfile.TemporaryDirectory(prefix="xtrax_triage_live_") as tmp_live:
        with profiling_telemetry_session(log_dir=log_dir) as status:
            if not status.active or status.log_dir is None:
                return {}
            return _run_isolation_audit(
                prod,
                live_root=Path(tmp_live),
                shadow_log_dir=status.log_dir,
                stage=stage,
                enforced_by=TRIAGE_SHADOW_ENFORCER,
                verdict_feature_version="__isolation_audit__",
            )


def _log_file_order_key(path: Path) -> tuple[str, int]:
    match = _LOG_FILE_RE.fullmatch(path.name)
    assert match is not None
    return (match.group("base"), -int(match.group("rot") or 0))


def _parse_jsonl_events(path: Path) -> list[dict[str, Any]]:
    """Read and parse one JSONL telemetry file strictly."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TelemetryReadError(f"Cannot read telemetry log {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise TelemetryReadError(f"Corrupt JSONL at {path}:{lineno}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TelemetryReadError(f"Telemetry JSONL line {path}:{lineno} must be a JSON object")
        events.append(parsed)
    return events


def read_telemetry_events(log_dir: str | Path) -> list[dict[str, Any]]:
    """Read ``events.*.jsonl*`` rotation logs strictly, raising on any parse/IO defect."""
    root = Path(log_dir)
    if not root.is_dir():
        raise TelemetryReadError(f"Telemetry log directory does not exist: {root}")
    candidates = [
        p for p in root.iterdir() if p.is_file() and _LOG_FILE_RE.fullmatch(p.name) is not None
    ]
    events: list[dict[str, Any]] = []
    for path in sorted(candidates, key=_log_file_order_key):
        events.extend(_parse_jsonl_events(path))
    return events


def read_rejected_verdicts(rejected_dir: str | Path) -> list[dict[str, Any]]:
    """Read ``*.verdict.json`` sidecars strictly and normalize ``verdict_status``."""
    root = Path(rejected_dir)
    if not root.is_dir():
        raise TelemetryReadError(f"Rejected verdicts directory does not exist: {root}")
    paths = [p for p in root.iterdir() if p.is_file() and p.name.endswith(".verdict.json")]
    paths.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    out: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TelemetryReadError(f"Corrupt verdict sidecar {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise TelemetryReadError(f"Verdict sidecar {path} must be a JSON object")
        normalized = dict(raw)
        normalized["verdict_status"] = str(
            raw.get("verdict_status") or raw.get("status") or ""
        ).lower()
        out.append(normalized)
    return out


def _default_shadow_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "audit" / "shadow").is_dir():
        return cwd
    return Path(__file__).resolve().parents[3]


def collect_triage_shadow_checks(
    log_dir: Path | None,
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all ``audit/shadow/triage_*.toml`` gates against ``log_dir``."""
    if log_dir is None:
        return []
    try:
        mode_state_mod = importlib.import_module("cisternal.shadow.mode_state")
    except ImportError:
        return []

    root = _default_shadow_repo_root(repo_root)
    shadow_dir = root / "audit" / "shadow"
    if not shadow_dir.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for toml_path in sorted(shadow_dir.glob("triage_*.toml")):
        mode_data = mode_state_mod.load_mode_state(toml_path)
        feat_name = str(mode_data.get("feature", {}).get("name", toml_path.stem))
        try:
            res = mode_state_mod.evaluate_gate(mode_data, Path(log_dir), repo_root=root)
        except mode_state_mod.GateCheckError as exc:
            if getattr(exc, "exit_code", 3) == 2 and "REFUSAL:" in str(exc):
                results.append(
                    {
                        "feature": feat_name,
                        "status": "uncalibrated",
                        "promotion_permitted": False,
                        "blocking_reasons": [],
                        "metrics": {"verdicts_total": 0, "labels_total": 0},
                        "detail": str(exc),
                    }
                )
                continue
            raise
        results.append(
            {
                "feature": feat_name,
                "status": "ready" if res.promotion_permitted else "blocked",
                "promotion_permitted": res.promotion_permitted,
                "blocking_reasons": list(res.blocking_reasons),
                "metrics": dict(res.metrics),
                "detail": "",
            }
        )
    return results


def _normalize_ts(raw_ts: Any) -> float:
    if isinstance(raw_ts, int | float) and not isinstance(raw_ts, bool):
        return float(raw_ts)
    if isinstance(raw_ts, str) and raw_ts.strip():
        try:
            return datetime.fromisoformat(raw_ts.strip()).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _unwrap_event(ev: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    name = str(ev.get("name", ""))
    ts = _normalize_ts(ev.get("ts") if "ts" in ev else ev.get("timestamp"))
    raw_fields = ev.get("fields")
    fields = dict(raw_fields) if isinstance(raw_fields, dict) else dict(ev)
    return name, ts, fields


def _compare_dedup_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("baseline_probe_id", "")),
        str(row.get("candidate_probe_id", "")),
        str(row.get("baseline_sha", "")),
        str(row.get("candidate_sha", "")),
        str(row.get("primary_metric", "")),
        str(row.get("verdict_status") or row.get("status") or "").lower(),
    )


def _merge_compare_timeline(
    norm_events: list[tuple[int, str, float, dict[str, Any]]],
    verdicts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build chronologically ordered, deduplicated comparison records."""
    rows: list[tuple[float, int, int, dict[str, Any]]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for idx, name, ts, fields in norm_events:
        if name != "xtrax.profile.compare":
            continue
        row = dict(fields)
        row["verdict_status"] = str(row.get("verdict_status") or row.get("status") or "").lower()
        key = _compare_dedup_key(row)
        seen.add(key)
        rows.append((ts, 0, idx, row))
    for v_idx, sidecar in enumerate(verdicts):
        row = dict(sidecar)
        row["verdict_status"] = str(row.get("verdict_status") or row.get("status") or "").lower()
        key = _compare_dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        ts = _normalize_ts(row.get("recorded_at") or row.get("ts"))
        rows.append((ts, 1, v_idx, row))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in rows]


def _detect_instrument_degraded(
    norm_events: list[tuple[int, str, float, dict[str, Any]]],
    thresholds: AlgedonicThresholds,
) -> AlgedonicSignal | None:
    by_env: dict[tuple[str, str], int] = {}
    for _idx, name, _ts, fields in norm_events:
        if name != "xtrax.profile.dual_pass":
            continue
        pass_b = str(fields.get("pass_b_status", ""))
        pass_c = str(fields.get("pass_c_status", ""))
        degraded = (
            pass_b == "failed"
            or pass_c == "failed"
            or pass_c.startswith("skipped:")
            or int(fields.get("ptxas_warning_count") or 0) > 0
            or fields.get("ptx_filter_status") == "unfiltered_name_mismatch"
            or bool(fields.get("hlo_attribution_skipped"))
        )
        if degraded:
            env_key = (str(fields.get("device_kind") or ""), str(fields.get("jax_version") or ""))
            by_env[env_key] = by_env.get(env_key, 0) + 1
    failing = {
        f"{dev or 'unknown'}@{ver or 'unknown'}": count
        for (dev, ver), count in by_env.items()
        if count >= thresholds.min_degraded_dual_pass_per_env
    }
    if not failing:
        return None
    return AlgedonicSignal(
        kind=AlgedonicSignalKind.INSTRUMENT_DEGRADED,
        severity="error",
        summary=f"Degraded dual-pass GPU telemetry observed repeatedly in {failing}.",
        evidence={"degraded_counts_by_env": failing},
        recommended_action=(
            "Stop optimization cycle via algedonic-escalation; inspect CUDA/ptxas toolchain, "
            "cmdbuf=off HLO/trace capture, and Triton kernel dump filters before trusting timings."
        ),
    )


def _detect_noise_streak(
    compare_rows: list[dict[str, Any]],
    thresholds: AlgedonicThresholds,
) -> AlgedonicSignal | None:
    streak = 0
    max_streak = 0
    max_nf = 0.0
    for row in compare_rows:
        if row.get("verdict_status") == "revert_noise":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
        nf = row.get("noise_floor_x")
        if isinstance(nf, int | float) and not isinstance(nf, bool):
            max_nf = max(max_nf, float(nf))
    if not (
        max_streak >= thresholds.max_consecutive_revert_noise
        or max_nf > thresholds.max_noise_floor_x
    ):
        return None
    return AlgedonicSignal(
        kind=AlgedonicSignalKind.NOISE_FLOOR_BLOWN,
        severity="warning",
        summary=(
            f"Noise floor exceeded (longest REVERT_NOISE streak={max_streak}, "
            f"max noise_floor_x={max_nf:.3f})."
        ),
        evidence={"longest_revert_noise_streak": max_streak, "max_noise_floor_x": max_nf},
        recommended_action=(
            "Increase n_wall repetitions, lock GPU clocks/exclusive node, and re-measure "
            "before proposing further micro-optimizations."
        ),
    )


def _is_erosion_compare_row(row: dict[str, Any]) -> bool:
    """Return True if a compare event or verdict sidecar bypassed a guardrail."""
    if bool(row.get("bypass_parity") or row.get("bypass_hold_fixed") or row.get("allow_dirty")):
        return True
    reasons = row.get("reasons")
    return isinstance(reasons, list | tuple) and bool(_EROSION_REASONS.intersection(reasons))


def _detect_guardrail_erosion(
    norm_events: list[tuple[int, str, float, dict[str, Any]]],
    compare_rows: list[dict[str, Any]],
    thresholds: AlgedonicThresholds,
) -> AlgedonicSignal | None:
    erosion_count = sum(1 for r in compare_rows if _is_erosion_compare_row(r))
    optout_count = sum(1 for r in compare_rows if bool(r.get("bypass_corruption")))
    n_compare = len(compare_rows)
    n_cli = sum(1 for _i, name, _t, _f in norm_events if name in _CLI_INVOCATION_EVENTS)
    n_not_meas = sum(1 for _i, name, _t, _f in norm_events if name == "xtrax.profile.not_measured")
    erosion_hit = erosion_count >= thresholds.min_bypass_count or (
        n_compare >= thresholds.min_compare_for_rate
        and (erosion_count / n_compare) > thresholds.max_bypass_rate
    )
    error_rate_hit = (
        n_cli >= thresholds.min_invocations_for_error_rate
        and (n_not_meas / n_cli) > thresholds.max_not_measured_rate
    )
    if not (erosion_hit or error_rate_hit):
        return None
    return AlgedonicSignal(
        kind=AlgedonicSignalKind.GUARDRAIL_EROSION,
        severity="error",
        summary=(
            f"Guardrail erosion or repeated measurement failure "
            f"(bypasses={erosion_count}/{n_compare}, not_measured={n_not_meas}/{n_cli})."
        ),
        evidence={
            "erosion_bypass_count": erosion_count,
            "corruption_optout_count": optout_count,
            "n_compare": n_compare,
            "n_cli_invocations": n_cli,
            "n_not_measured": n_not_meas,
        },
        recommended_action=(
            "Halt autonomous promotion; restore clean-SHA parity/hold-fixed contracts "
            "and resolve measurement errors before continuing."
        ),
    )


def _is_shadow_miscalibrated(entry: dict[str, Any], min_labels: int) -> bool:
    if entry.get("status") == "uncalibrated":
        return False
    prefixes = {str(r).split(":", 1)[0] for r in entry.get("blocking_reasons", ())}
    if "isolation_violation_detected" in prefixes:
        return True
    metrics = entry.get("metrics") or {}
    labels_total = int(metrics.get("labels_total") or 0)
    if labels_total < min_labels:
        return False
    if bool(prefixes & _HARD_BLOCK_PREFIXES):
        return True
    fp = int(metrics.get("fp") or 0)
    tn = int(metrics.get("tn") or 0)
    tp = int(metrics.get("tp") or 0)
    fn = int(metrics.get("fn") or 0)
    return "precision_lower_bound_below_base_rate" in prefixes and (fp + tn) > 0 and (tp + fn) > 0


def _detect_shadow_and_triage_signals(
    norm_events: list[tuple[int, str, float, dict[str, Any]]],
    shadow_checks: Sequence[dict[str, Any]],
    thresholds: AlgedonicThresholds,
) -> list[AlgedonicSignal]:
    signals: list[AlgedonicSignal] = []
    escapes = [
        fields for _i, name, _t, fields in norm_events if name == "xtrax.profile.parity_escape"
    ]
    if escapes:
        signals.append(
            AlgedonicSignal(
                kind=AlgedonicSignalKind.PARITY_ESCAPE,
                severity="critical",
                summary=f"{len(escapes)} parity escape event(s) recorded.",
                evidence={"count": len(escapes), "escapes": escapes},
                recommended_action=(
                    "Demote autonomous ratchet to supervised/advisory mode immediately, add a "
                    "mandatory ParityContract negative control, and run bth blast-radius assess "
                    "when bathos is active."
                ),
            )
        )

    for entry in shadow_checks:
        if not _is_shadow_miscalibrated(entry, thresholds.min_shadow_labels_for_demote):
            continue
        feat = str(entry.get("feature") or "") or None
        labels_total = int((entry.get("metrics") or {}).get("labels_total") or 0)
        signals.append(
            AlgedonicSignal(
                kind=AlgedonicSignalKind.HEURISTIC_MISCALIBRATED,
                severity="warning",
                summary=(
                    f"Triage shadow gate miscalibrated or isolated violation for "
                    f"{feat or '<unknown>'}."
                ),
                evidence={
                    "feature": feat,
                    "blocking_reasons": list(entry.get("blocking_reasons", ())),
                    "labels_total": labels_total,
                },
                recommended_action=(
                    "Recalibrate heuristic thresholds or inspect isolation failure before "
                    "advancing the triage rule beyond shadow stage."
                ),
                demote_feature=feat,
            )
        )

    gap_events = [
        fields
        for _i, name, _t, fields in norm_events
        if name == "xtrax.profile.triage" and bool(fields.get("unclassified_slowdown"))
    ]
    if len(gap_events) >= thresholds.min_triage_gap_count:
        signals.append(
            AlgedonicSignal(
                kind=AlgedonicSignalKind.TRIAGE_GAP,
                severity="warning",
                summary=f"{len(gap_events)} unclassified slowdown(s) observed in triage.",
                evidence={"count": len(gap_events), "events": gap_events},
                recommended_action=(
                    "File an issue-tracker backlog/ideas item with the paired .diag.json "
                    "fixture to add the missing bottleneck classification rule."
                ),
            )
        )
    return signals


def detect_algedonic_signals(
    events: Sequence[dict[str, Any]],
    *,
    verdicts: Sequence[dict[str, Any]] = (),
    shadow_checks: Sequence[dict[str, Any]] = (),
    thresholds: AlgedonicThresholds | None = None,
) -> list[AlgedonicSignal]:
    """Evaluate operational events, rejected verdicts, and shadow gates for algedonic signals."""
    eff_thr = thresholds if thresholds is not None else AlgedonicThresholds()
    unwrapped = [(idx, *_unwrap_event(ev)) for idx, ev in enumerate(events)]
    unwrapped.sort(key=lambda item: (item[2], item[0]))
    compare_rows = _merge_compare_timeline(unwrapped, verdicts)

    signals: list[AlgedonicSignal] = []
    for cand in (
        _detect_instrument_degraded(unwrapped, eff_thr),
        _detect_noise_streak(compare_rows, eff_thr),
        _detect_guardrail_erosion(unwrapped, compare_rows, eff_thr),
    ):
        if cand is not None:
            signals.append(cand)
    signals.extend(_detect_shadow_and_triage_signals(unwrapped, shadow_checks, eff_thr))
    return signals


def append_algedonic_finding(
    signal: AlgedonicSignal,
    *,
    audits_path: Path | None = None,
    tombstone_path: Path = DEFAULT_TOMBSTONE_PATH,
    repo_root: Path | None = None,
    run_id: str | None = None,
) -> AuditFinding | None:
    """Append an algedonic signal as an unlocked-safe ``AuditFinding`` JSONL row when opted in."""
    if audits_path is None:
        tracker_dir = (repo_root or Path.cwd()) / ".praxia"
        if not tracker_dir.is_dir():
            return None
        target_path = tracker_dir / "audits.jsonl"
    else:
        target_path = Path(audits_path)

    sev_map: dict[str, Severity] = {
        "warning": "minor",
        "error": "major",
        "critical": "critical",
    }
    severity: Severity = sev_map.get(signal.severity, "minor")
    finding = emit_metric_finding(
        dim="xtrax.profiling",
        severity=severity,
        file_line="src/xtrax/telemetry/profiling_events.py",
        evidence=signal.summary,
        rule_id=f"xtrax.algedonic.{signal.kind.value}",
        symbol_qualname=signal.demote_feature or "",
        payload={
            "kind": signal.kind.value,
            "recommended_action": signal.recommended_action,
            "demote_feature": signal.demote_feature,
            "evidence": signal.evidence,
        },
        run_id=run_id or str(uuid.uuid4()),
    )
    if is_tombstoned(finding.finding_id, path=tombstone_path):
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(finding_to_dict(finding), sort_keys=True) + "\n"
    with target_path.open("a", encoding="utf-8") as handle:
        if _HAVE_FLOCK:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if _HAVE_FLOCK:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return finding


__all__ = [
    "AlgedonicSignal",
    "AlgedonicSignalKind",
    "AlgedonicThresholds",
    "PROVISIONAL_TRIAGE_CATEGORIES",
    "TRIAGE_SHADOW_ENFORCER",
    "TelemetryReadError",
    "TelemetrySessionStatus",
    "append_algedonic_finding",
    "attest_triage_shadow_isolation",
    "collect_triage_shadow_checks",
    "detect_algedonic_signals",
    "emit_check_telemetry",
    "emit_compare_telemetry",
    "emit_dual_pass_telemetry",
    "emit_measure_telemetry",
    "emit_not_measured_telemetry",
    "emit_triage_shadow_label",
    "emit_triage_shadow_verdicts",
    "emit_triage_telemetry",
    "make_triage_subject_key",
    "profiling_telemetry_session",
    "read_rejected_verdicts",
    "read_telemetry_events",
    "record_parity_escape",
    "resolve_triage_subject_key",
]

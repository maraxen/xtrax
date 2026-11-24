"""Tests for xtrax.telemetry.profiling_events, shadow calibration, and algedonic detection."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from xtrax.profiling.compare import DeltaVerdict, VerdictStatus
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.triage import BottleneckCategory, BottleneckFinding
from xtrax.telemetry.profiling_events import (
    PROVISIONAL_TRIAGE_CATEGORIES,
    TRIAGE_SHADOW_ENFORCER,
    AlgedonicSignal,
    AlgedonicSignalKind,
    AlgedonicThresholds,
    TelemetryReadError,
    _run_isolation_audit,
    append_algedonic_finding,
    attest_triage_shadow_isolation,
    collect_triage_shadow_checks,
    detect_algedonic_signals,
    emit_check_telemetry,
    emit_compare_telemetry,
    emit_dual_pass_telemetry,
    emit_measure_telemetry,
    emit_not_measured_telemetry,
    emit_triage_shadow_label,
    emit_triage_shadow_verdicts,
    emit_triage_telemetry,
    make_triage_subject_key,
    profiling_telemetry_session,
    read_rejected_verdicts,
    read_telemetry_events,
    record_parity_escape,
    resolve_triage_subject_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 40


def _sample_prod(
    probe_id: str = "xtrax-profile:mod:step",
    *,
    scopes: dict[str, tuple[float, int] | None] | None = None,
    attribution_method: dict[str, str] | None = None,
    **extra_metrics: float,
) -> ProbeRecord:
    metrics: dict[str, float] = {"total_step_seconds": 1.0, "active_work_ratio": 0.1}
    metrics.update(extra_metrics)
    return ProbeRecord(
        probe_id=probe_id,
        stage=1,
        n_atoms=64,
        platform="cpu",
        git_sha=SHA_A,
        metrics=metrics,
        scopes=scopes,
        attribution_method=attribution_method,
    )


def _sample_finding(cat: BottleneckCategory) -> BottleneckFinding:
    return BottleneckFinding(
        category=cat,
        severity="high",
        target_kernel="k",
        evidence={"x": 1.0},
        summary="summary",
        playbook_ref="ref",
        threshold_status="provisional_gpu_heuristic",
    )


def _install_fake_cisternal(
    monkeypatch,
    default_dir: Path,
    *,
    resolution_mode: str = "ok",
    init_raises: bool = False,
    shutdown_raises: bool = False,
    emit_raises: bool = False,
):
    state: dict[str, object] = {
        "pipeline": None,
        "events": [],
        "verdicts": [],
        "labels": [],
        "isolations": [],
        "init_calls": [],
        "shutdown_calls": 0,
    }

    def _init(*, log_dir=None, heartbeat_interval=30.0):
        state["init_calls"].append((log_dir, heartbeat_interval))
        if init_raises:
            raise RuntimeError("init failed")
        actual = Path(log_dir) if log_dir is not None else default_dir
        exporter = SimpleNamespace(_path=actual / "events.host.jsonl")
        state["pipeline"] = SimpleNamespace(_exporters=(exporter,))

    def _emit_event(name: str, *, level: str = "INFO", **fields):
        if emit_raises:
            raise RuntimeError("emit boom")
        state["events"].append({"name": name, "level": level, "fields": dict(fields)})

    def _get_pipeline():
        return state["pipeline"]

    def _shutdown_pipeline():
        state["shutdown_calls"] = int(state["shutdown_calls"]) + 1
        if shutdown_raises:
            raise RuntimeError("shutdown boom")
        state["pipeline"] = None

    def _emit_verdict(**kwargs):
        state["verdicts"].append(dict(kwargs))
        return True

    def _emit_label(**kwargs):
        state["labels"].append(dict(kwargs))
        return True

    def _emit_isolation(**kwargs):
        state["isolations"].append(dict(kwargs))
        return True

    def _resolve_log_dir(*, cli_arg=None):
        if resolution_mode == "missing":
            raise ImportError("no cisternal.shadow.resolution")
        if resolution_mode == "boom":
            raise RuntimeError("resolution error")
        return SimpleNamespace(path=cli_arg or default_dir)

    def _prove_isolation(live_fn, shadow_fn, *, live_roots, shadow_log_dir, **_kwargs):
        live_out = live_fn()
        shadow_out = shadow_fn()
        leaked = [p for root in live_roots for p in Path(root).glob("*") if p.is_file()]
        return SimpleNamespace(passed=(live_out == shadow_out and not leaked), leaked_paths=leaked)

    fake_mods = {
        "cisternal": SimpleNamespace(init=_init, emit_event=_emit_event),
        "cisternal.telemetry.pipeline": SimpleNamespace(
            get_pipeline=_get_pipeline,
            shutdown_pipeline=_shutdown_pipeline,
        ),
        "cisternal.shadow.events": SimpleNamespace(
            emit_verdict=_emit_verdict,
            emit_label=_emit_label,
            emit_isolation=_emit_isolation,
        ),
        "cisternal.shadow.isolation": SimpleNamespace(prove_isolation=_prove_isolation),
    }
    if resolution_mode != "missing":
        fake_mods["cisternal.shadow.resolution"] = SimpleNamespace(resolve_log_dir=_resolve_log_dir)

    orig_import = importlib.import_module

    def _fake_import(name: str, package=None):
        if name in fake_mods:
            return fake_mods[name]
        if name.startswith("cisternal"):
            raise ImportError(f"missing {name}")
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    return state


def test_importing_profiling_events_is_lazy_and_isolated():
    src_dir = REPO_ROOT / "src"
    code = (
        "import sys; import xtrax.telemetry.profiling_events; "
        "leaks = [m for m in ('jax', 'cisternal', 'bathos') if m in sys.modules]; "
        "assert not leaks, leaks"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src_dir), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_layer5_off_is_silent_with_zero_writes(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("CISTERNAL_TELEMETRY_LOG_DIR", raising=False)
    monkeypatch.delenv("CISTERNAL_LOG_DIR", raising=False)
    monkeypatch.delenv("BTH_LOG_DIR", raising=False)
    monkeypatch.delenv("CTXP_LOG_DIR", raising=False)
    monkeypatch.delenv("XTRAX_PROFILE_TELEMETRY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    with profiling_telemetry_session(log_dir=None) as status:
        assert status.active is False
        assert status.reason in ("log_dir_off", "cisternal_not_installed")
        assert not emit_measure_telemetry(
            "fn", "cpu", None, cardinality=8, n_flags=0, exit_code=0, duration_ms=1.0
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert list(tmp_path.rglob("*.jsonl*")) == []


def test_session_fallbacks_when_cisternal_missing_or_opted_out(monkeypatch, tmp_path):
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    with profiling_telemetry_session(log_dir=tmp_path) as st_opt:
        assert st_opt.active is False
        assert st_opt.reason == "opted_out"
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT")

    orig_import = importlib.import_module

    def _no_cisternal(name: str, package=None):
        if name == "cisternal" or name.startswith("cisternal."):
            raise ImportError("no cisternal")
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _no_cisternal)
    with profiling_telemetry_session(log_dir=tmp_path) as st_imp:
        assert st_imp.active is False
        assert st_imp.reason == "cisternal_not_installed"
    assert emit_triage_shadow_verdicts(_sample_prod(), []) == 0
    assert emit_triage_shadow_label(_sample_prod(), "atomic_contention", True) is False


def test_mocked_session_lifecycle_all_emitters_and_fallbacks(monkeypatch, tmp_path):
    default_dir = tmp_path / "default_logs"
    log_a = tmp_path / "logs_a"
    log_b = tmp_path / "logs_b"

    # Pre-0.1.11 fallback env var inspection + "none" disable
    state = _install_fake_cisternal(monkeypatch, default_dir, resolution_mode="missing")
    monkeypatch.setenv("CISTERNAL_TELEMETRY_LOG_DIR", "none")
    monkeypatch.delenv("XTRAX_PROFILE_TELEMETRY", raising=False)
    with profiling_telemetry_session(log_dir=None) as st_none:
        assert st_none.active is False
        assert st_none.reason == "log_dir_off"

    # Layer-5 opt-in populates actual log_dir via active exporter
    monkeypatch.delenv("CISTERNAL_TELEMETRY_LOG_DIR")
    monkeypatch.setenv("XTRAX_PROFILE_TELEMETRY", "1")
    with profiling_telemetry_session(log_dir=None) as st_l5:
        assert st_l5.active is True
        assert st_l5.log_dir == default_dir
        assert st_l5.init_owned is True
    assert state["shutdown_calls"] == 1

    # Resolution via cisternal.shadow.resolution + exception fallback
    _install_fake_cisternal(monkeypatch, default_dir, resolution_mode="boom")
    monkeypatch.delenv("XTRAX_PROFILE_TELEMETRY")
    with profiling_telemetry_session(log_dir=None) as st_boom:
        assert st_boom.active is False
        assert st_boom.reason == "log_dir_off"

    # Owned lifecycle, reuse, retargeting, and all 7 event emitters
    state = _install_fake_cisternal(monkeypatch, default_dir, resolution_mode="ok")
    with profiling_telemetry_session(log_dir=log_a) as st_a:
        assert st_a.active is True
        assert st_a.init_owned is True
        with profiling_telemetry_session(log_dir=None) as st_reused:
            assert st_reused.active is True
            assert st_reused.init_owned is False
            assert st_reused.log_dir == log_a.resolve()
        with profiling_telemetry_session(log_dir=log_a) as st_same:
            assert st_same.init_owned is False
            assert st_same.reason == "reused_matching_dir"
        with profiling_telemetry_session(log_dir=log_b) as st_retgt:
            assert st_retgt.retargeted_from_active is True
            assert st_retgt.log_dir == log_b.resolve()
        # Re-open log_a via retargeting to emit all 7 events cleanly
        with profiling_telemetry_session(log_dir=log_a):
            assert emit_measure_telemetry(
                "m:f", "cpu", None, cardinality=8, n_flags=0, exit_code=0, duration_ms=1.5
            )
            assert emit_check_telemetry(tmp_path / "r.json", 1, stale=True, n_flags=1)
            assert emit_not_measured_telemetry("compare", "bad", "msg")
            prod = _sample_prod()
            diag = _sample_prod(
                "diag",
                scopes={"k1": (0.7, 1), "k2": (0.3, 1), "absent": None},
                attribution_method={"k1": "op_name", "k2": "op_name", "absent": "named_scope"},
            )
            diag_zero = _sample_prod(
                "diag0",
                scopes={"k1": (0.0, 1)},
                attribution_method={"k1": "op_name"},
            )
            assert emit_triage_telemetry(
                prod, [_sample_finding(BottleneckCategory.SPATIAL_DEAD_WORK)], diag=diag
            )
            assert emit_triage_telemetry(prod, [], diag=diag_zero, target_missed=True)
            assert emit_dual_pass_telemetry(
                SimpleNamespace(
                    prod_record=prod,
                    diag_record=diag,
                    diag_skipped_reason=None,
                    ptx_capture_status="ok",
                    ptx_spill_source="ptxas",
                    ptx_filter_status="matched",
                    hlo_attribution_skipped=None,
                    ptxas_warnings=(),
                )
            )
            assert emit_dual_pass_telemetry(
                SimpleNamespace(
                    prod_record=prod,
                    diag_record=None,
                    diag_skipped_reason="diag_disabled",
                    ptx_capture_status="disabled",
                    ptx_spill_source="none",
                    ptx_filter_status="not_applicable",
                    hlo_attribution_skipped=None,
                    ptxas_warnings=(),
                )
            )
            assert emit_dual_pass_telemetry(
                SimpleNamespace(
                    prod_record=prod,
                    diag_record=None,
                    diag_skipped_reason="diag_subprocess_failed:exit_1",
                    ptx_capture_status="skipped:exit_1",
                    ptx_spill_source="none",
                    ptx_filter_status="unfiltered_name_mismatch",
                    hlo_attribution_skipped="unjittable_wrapper",
                    ptxas_warnings=("warn",),
                )
            )
            verdict = DeltaVerdict(
                status=VerdictStatus.KEEP,
                promotable=True,
                primary_metric="total_step_seconds",
                baseline_value=1.0,
                candidate_value=0.5,
                speedup_x=2.0,
                rel_delta_pct=-50.0,
                noise_floor_x=1.02,
                baseline_probe_id="b",
                candidate_probe_id="c",
                baseline_sha=SHA_A,
                candidate_sha="b" * 40,
                reasons=(),
                metric_deltas={"total_step_seconds": -0.5},
            )
            assert emit_compare_telemetry(verdict)
            assert record_parity_escape("p1", SHA_A, "missed edge", control_added="c1")

            att = attest_triage_shadow_isolation(log_dir=log_a)
            assert len(att) == 4 and all(att.values())

    names = [ev["name"] for ev in state["events"]]
    assert {
        "xtrax.profile.measure",
        "xtrax.profile.check",
        "xtrax.profile.not_measured",
        "xtrax.profile.dual_pass",
        "xtrax.profile.compare",
        "xtrax.profile.triage",
        "xtrax.profile.parity_escape",
    } <= set(names)
    triage_evs = [ev["fields"] for ev in state["events"] if ev["name"] == "xtrax.profile.triage"]
    assert triage_evs[0]["top_kernel_pct"] == pytest.approx(70.0)
    assert triage_evs[1]["top_kernel_pct"] == 0.0
    assert triage_evs[1]["unclassified_slowdown"] is True

    # Failure containment: init error, retarget shutdown error, emit error, shutdown error
    _install_fake_cisternal(monkeypatch, default_dir, init_raises=True)
    with profiling_telemetry_session(log_dir=log_a) as st_init_err:
        assert st_init_err.active is False
        assert st_init_err.reason == "init_failed"
        assert "RuntimeError" in (st_init_err.degraded_reason or "")

    state_sd = _install_fake_cisternal(monkeypatch, default_dir, shutdown_raises=True)
    with profiling_telemetry_session(log_dir=log_a) as st_sd:
        assert st_sd.active is True
        with profiling_telemetry_session(log_dir=log_b) as st_ret_err:
            assert st_ret_err.active is False
            assert st_ret_err.reason == "retarget_shutdown_failed"
    assert "shutdown:" in (st_sd.degraded_reason or "")
    assert state_sd["shutdown_calls"] == 2

    _install_fake_cisternal(monkeypatch, default_dir, emit_raises=True)
    with profiling_telemetry_session(log_dir=log_a):
        assert (
            emit_measure_telemetry(
                "fn", "cpu", None, cardinality=1, n_flags=0, exit_code=0, duration_ms=1.0
            )
            is False
        )


def test_mocked_shadow_labels_and_collect_checks(monkeypatch, tmp_path):
    p1 = _sample_prod(probe_id="xtrax-profile:m:f", total_step_seconds=1.0)
    p2 = _sample_prod(probe_id="xtrax-profile:m:f", total_step_seconds=2.0)
    k1 = make_triage_subject_key(p1)
    k2 = make_triage_subject_key(p2)
    assert k1 != k2
    assert k1.count(":") == 3
    assert resolve_triage_subject_key(k1) == k1
    p1_path = tmp_path / "p1.json"
    p1.write(p1_path)
    assert resolve_triage_subject_key(p1_path) == k1

    default_dir = tmp_path / "shadow_logs"
    state = _install_fake_cisternal(monkeypatch, default_dir)
    # Inactive pipeline returns 0 / False without crashing
    assert emit_triage_shadow_verdicts(p1, []) == 0
    assert emit_triage_shadow_label(k1, "atomic_contention", True) is False

    with profiling_telemetry_session(log_dir=default_dir):
        n_v = emit_triage_shadow_verdicts(
            p1, [_sample_finding(BottleneckCategory.ATOMIC_CONTENTION)]
        )
        assert n_v == len(PROVISIONAL_TRIAGE_CATEGORIES)
        assert emit_triage_shadow_label(k1, "xtrax.triage.atomic_contention", True) is True
        assert emit_triage_shadow_label(k1, "register_spill_pressure", False) is True

    assert [lbl["feature"] for lbl in state["labels"]] == [
        "xtrax.triage.atomic_contention",
        "xtrax.triage.register_spill_pressure",
    ]
    with pytest.raises(ValueError, match="Unknown triage shadow category"):
        emit_triage_shadow_label(k1, "unknown_rule", True)

    # collect_triage_shadow_checks: None/ImportError/ready/blocked/uncalibrated/exit 3
    assert collect_triage_shadow_checks(None) == []
    assert collect_triage_shadow_checks(default_dir) == []

    class _FakeGateErr(Exception):
        def __init__(self, msg: str, exit_code: int) -> None:
            super().__init__(msg)
            self.exit_code = exit_code

    def _make_fake_mode_state(mode: str):
        def _load(toml_path: Path):
            return {"feature": {"name": f"feat.{toml_path.stem}"}}

        def _eval(mode_data, _log_dir, *, repo_root=None):
            name = mode_data["feature"]["name"]
            if mode == "corrupt":
                raise _FakeGateErr("bad schema", 3)
            if name.endswith("fp64_pipe_saturation"):
                raise _FakeGateErr("REFUSAL: 0 verdicts and 0 labels", 2)
            if name.endswith("atomic_contention"):
                return SimpleNamespace(
                    promotion_permitted=False,
                    blocking_reasons=["precision_deficit:0.4/0.7"],
                    metrics={"labels_total": 12},
                )
            return SimpleNamespace(
                promotion_permitted=True,
                blocking_reasons=[],
                metrics={"labels_total": 12},
            )

        return SimpleNamespace(
            GateCheckError=_FakeGateErr, load_mode_state=_load, evaluate_gate=_eval
        )

    orig_import = importlib.import_module
    fake_ms = _make_fake_mode_state("ok")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, pkg=None: (
            fake_ms if name == "cisternal.shadow.mode_state" else orig_import(name, pkg)
        ),
    )
    checks = collect_triage_shadow_checks(default_dir, repo_root=REPO_ROOT)
    statuses = {c["feature"]: c["status"] for c in checks}
    assert statuses["feat.triage_fp64_pipe_saturation"] == "uncalibrated"
    assert statuses["feat.triage_atomic_contention"] == "blocked"
    assert statuses["feat.triage_spatial_dead_work"] == "ready"
    assert collect_triage_shadow_checks(default_dir, repo_root=tmp_path) == []

    fake_ms_bad = _make_fake_mode_state("corrupt")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, pkg=None: (
            fake_ms_bad if name == "cisternal.shadow.mode_state" else orig_import(name, pkg)
        ),
    )
    with pytest.raises(_FakeGateErr):
        collect_triage_shadow_checks(default_dir, repo_root=REPO_ROOT)


def test_detect_algedonic_signals_all_six_positive_and_negative_controls(tmp_path):
    clean_events = [
        {
            "name": "xtrax.profile.dual_pass",
            "ts": 1.0,
            "fields": {
                "device_kind": "l40s",
                "jax_version": "0.10.2",
                "pass_b_status": "disabled",
                "pass_c_status": "disabled",
                "ptxas_warning_count": 0,
                "ptx_filter_status": "not_applicable",
                "hlo_attribution_skipped": None,
            },
        },
        {
            "name": "xtrax.profile.compare",
            "ts": 2.0,
            "fields": {
                "baseline_probe_id": "b",
                "candidate_probe_id": "c",
                "baseline_sha": SHA_A,
                "candidate_sha": "b" * 40,
                "primary_metric": "total_step_seconds",
                "verdict_status": "keep",
                "noise_floor_x": 1.02,
                "bypass_corruption": True,
                "bypass_parity": False,
                "bypass_hold_fixed": False,
                "allow_dirty": False,
            },
        },
        {
            "name": "xtrax.profile.compare",
            "ts": 3.0,
            "fields": {
                "baseline_probe_id": "b",
                "candidate_probe_id": "c2",
                "baseline_sha": "b" * 40,
                "candidate_sha": "c" * 40,
                "primary_metric": "total_step_seconds",
                "verdict_status": "keep",
                "noise_floor_x": 1.01,
                "bypass_corruption": True,
                "bypass_parity": False,
                "bypass_hold_fixed": False,
                "allow_dirty": False,
            },
        },
    ]
    safe_shadow = [
        {
            "feature": "xtrax.triage.spatial_dead_work",
            "status": "uncalibrated",
            "blocking_reasons": [],
        },
        {
            "feature": "xtrax.triage.atomic_contention",
            "status": "blocked",
            "blocking_reasons": ["precision_lower_bound_below_base_rate"],
            "metrics": {"labels_total": 12, "tp": 12, "fp": 0, "tn": 0, "fn": 0},
        },
        {
            "feature": "xtrax.triage.register_spill_pressure",
            "status": "blocked",
            "blocking_reasons": ["precision_deficit:0.4/0.7"],
            "metrics": {"labels_total": 4},
        },
    ]
    assert detect_algedonic_signals(clean_events, shadow_checks=safe_shadow) == []

    rej_dir = tmp_path / "rej"
    rej_dir.mkdir()
    (rej_dir / "a.verdict.json").write_text(
        json.dumps(
            {
                "status": "revert_noise",
                "recorded_at": "2026-11-23T10:00:02+00:00",
                "baseline_probe_id": "p",
                "candidate_probe_id": "n1",
                "baseline_sha": SHA_A,
                "candidate_sha": "c" * 40,
                "primary_metric": "total_step_seconds",
                "noise_floor_x": 1.05,
            }
        )
    )
    verdicts = read_rejected_verdicts(rej_dir)
    bad_events = [
        {
            "name": "xtrax.profile.dual_pass",
            "ts": 10.0,
            "fields": {
                "device_kind": "l40s",
                "jax_version": "0.10.2",
                "pass_b_status": "failed",
                "pass_c_status": "disabled",
                "ptxas_warning_count": 0,
            },
        },
        {
            "name": "xtrax.profile.dual_pass",
            "ts": 11.0,
            "fields": {
                "device_kind": "l40s",
                "jax_version": "0.10.2",
                "pass_b_status": "ok",
                "pass_c_status": "skipped:exit_1",
                "ptxas_warning_count": 0,
            },
        },
        {
            "name": "xtrax.profile.compare",
            "ts": "2026-11-23T10:00:00+00:00",
            "fields": {
                "baseline_probe_id": "p",
                "candidate_probe_id": "n0",
                "baseline_sha": SHA_A,
                "candidate_sha": "b" * 40,
                "primary_metric": "total_step_seconds",
                "verdict_status": "revert_noise",
                "noise_floor_x": 1.04,
                "bypass_parity": True,
            },
        },
        {
            "name": "xtrax.profile.compare",
            "ts": "2026-11-23T10:00:04+00:00",
            "fields": {
                "baseline_probe_id": "p",
                "candidate_probe_id": "n2",
                "baseline_sha": SHA_A,
                "candidate_sha": "d" * 40,
                "primary_metric": "total_step_seconds",
                "verdict_status": "revert_noise",
                "noise_floor_x": 1.15,
                "allow_dirty": True,
            },
        },
        {"name": "xtrax.profile.parity_escape", "ts": 15.0, "fields": {"probe_id": "p"}},
        {"name": "xtrax.profile.triage", "ts": 16.0, "fields": {"unclassified_slowdown": True}},
        {"name": "xtrax.profile.triage", "ts": 17.0, "fields": {"unclassified_slowdown": True}},
    ]
    bad_shadow = [
        {
            "feature": "xtrax.triage.fp64_pipe_saturation",
            "status": "blocked",
            "blocking_reasons": ["precision_deficit:0.40/0.70"],
            "metrics": {"labels_total": 12, "tp": 4, "fp": 6, "tn": 2, "fn": 0},
        }
    ]
    detected = detect_algedonic_signals(bad_events, verdicts=verdicts, shadow_checks=bad_shadow)
    assert {s.kind for s in detected} == set(AlgedonicSignalKind)

    # Low-sample negative control on custom thresholds + isolation_violation immediate fire
    iso_bad = [
        {
            "feature": "xtrax.triage.spatial_dead_work",
            "status": "blocked",
            "blocking_reasons": ["isolation_violation_detected:1"],
            "metrics": {"labels_total": 0},
        }
    ]
    sig_iso = detect_algedonic_signals([], shadow_checks=iso_bad, thresholds=AlgedonicThresholds())
    assert [s.kind for s in sig_iso] == [AlgedonicSignalKind.HEURISTIC_MISCALIBRATED]

    # GUARDRAIL_EROSION triggered purely by error_rate_hit (zero compare bypasses)
    err_rate_events = [
        {"name": "xtrax.profile.measure", "ts": 1.0, "fields": {}},
        {"name": "xtrax.profile.measure", "ts": 2.0, "fields": {}},
        {"name": "xtrax.profile.measure", "ts": 3.0, "fields": {}},
        {"name": "xtrax.profile.not_measured", "ts": 4.0, "fields": {}},
        {"name": "xtrax.profile.not_measured", "ts": 5.0, "fields": {}},
    ]
    sig_err = detect_algedonic_signals(err_rate_events, verdicts=[], shadow_checks=[])
    assert [s.kind for s in sig_err] == [AlgedonicSignalKind.GUARDRAIL_EROSION]
    assert sig_err[0].evidence["erosion_bypass_count"] == 0
    assert sig_err[0].evidence["n_not_measured"] == 2
    assert sig_err[0].evidence["n_cli_invocations"] == 5

    ok_rate_events = [
        {"name": "xtrax.profile.measure", "ts": 1.0, "fields": {}},
        {"name": "xtrax.profile.measure", "ts": 2.0, "fields": {}},
        {"name": "xtrax.profile.measure", "ts": 3.0, "fields": {}},
        {"name": "xtrax.profile.measure", "ts": 4.0, "fields": {}},
        {"name": "xtrax.profile.not_measured", "ts": 5.0, "fields": {}},
    ]
    assert detect_algedonic_signals(ok_rate_events, verdicts=[], shadow_checks=[]) == []


def test_read_rotation_ordering_errors_and_append_algedonic_finding(tmp_path):
    rot_dir = tmp_path / "rot"
    rot_dir.mkdir()
    (rot_dir / "events.h.jsonl").write_text("\n" + json.dumps({"name": "third", "ts": 3}) + "\n")
    (rot_dir / "events.h.jsonl.1").write_text(json.dumps({"name": "second", "ts": 2}) + "\n")
    (rot_dir / "events.h.jsonl.2").write_text(json.dumps({"name": "first", "ts": 1}) + "\n")
    (rot_dir / "ignored.json").write_text("{not_jsonl")
    assert [e["name"] for e in read_telemetry_events(rot_dir)] == ["first", "second", "third"]

    missing = tmp_path / "does_not_exist"
    with pytest.raises(TelemetryReadError):
        read_telemetry_events(missing)
    with pytest.raises(TelemetryReadError):
        read_rejected_verdicts(missing)

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "events.host.1.jsonl").write_text("{not json\n")
    with pytest.raises(TelemetryReadError, match="Corrupt JSONL"):
        read_telemetry_events(corrupt_dir)

    non_dict_dir = tmp_path / "non_dict"
    non_dict_dir.mkdir()
    (non_dict_dir / "events.host.jsonl").write_text("[1, 2]\n")
    with pytest.raises(TelemetryReadError, match="JSON object"):
        read_telemetry_events(non_dict_dir)

    (corrupt_dir / "bad.verdict.json").write_text("[1, 2, 3]")
    with pytest.raises(TelemetryReadError, match="JSON object"):
        read_rejected_verdicts(corrupt_dir)

    sig = AlgedonicSignal(
        kind=AlgedonicSignalKind.TRIAGE_GAP,
        severity="warning",
        summary="2 unclassified slowdowns",
        evidence={"count": 2},
        recommended_action="Add rule",
    )
    assert append_algedonic_finding(sig, repo_root=tmp_path) is None
    assert not (tmp_path / ".praxia").exists()

    # Existing .praxia dir writes flock-protected JSONL row; tombstone suppresses re-emission
    (tmp_path / ".praxia").mkdir()
    tomb = tmp_path / "tombstones.jsonl"
    finding = append_algedonic_finding(sig, repo_root=tmp_path, tombstone_path=tomb)
    assert finding is not None and (tmp_path / ".praxia" / "audits.jsonl").is_file()
    row = json.loads((tmp_path / ".praxia" / "audits.jsonl").read_text().strip())
    assert row["dim"] == "xtrax.profiling"
    assert row["payload"]["rule_id"] == "xtrax.algedonic.triage_gap"

    tomb.write_text(json.dumps({"finding_id": finding.finding_id}) + "\n")
    assert (
        append_algedonic_finding(
            sig, audits_path=tmp_path / "suppressed.jsonl", tombstone_path=tomb
        )
        is None
    )


def test_triage_shadow_writes_isolated(tmp_path: Path) -> None:
    """Enforcer attested in audit/shadow/triage_*.toml (keeps importorskip inside body)."""
    pytest.importorskip("cisternal.shadow")

    live_root = tmp_path / "live"
    live_root.mkdir()
    shadow_dir = tmp_path / "shadow"
    prod = _sample_prod()
    with profiling_telemetry_session(log_dir=shadow_dir):
        clean_res = _run_isolation_audit(
            prod,
            live_root=live_root,
            shadow_log_dir=shadow_dir,
            enforced_by=TRIAGE_SHADOW_ENFORCER,
        )
        assert all(clean_res.values())

        leaky_res = _run_isolation_audit(
            prod,
            live_root=live_root,
            shadow_log_dir=shadow_dir,
            enforced_by=TRIAGE_SHADOW_ENFORCER,
            inject_leak_fn=lambda root: (root / "leaked.txt").write_text("leak"),
        )
        assert not all(leaky_res.values())


def test_live_cisternal_session_and_shadow_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("cisternal")
    pytest.importorskip("cisternal.shadow")

    log_a = tmp_path / "logs_a"
    log_b = tmp_path / "logs_b"
    with profiling_telemetry_session(log_dir=log_a) as st_a:
        assert st_a.active is True
        prod = _sample_prod()
        emit_triage_shadow_verdicts(prod, [_sample_finding(BottleneckCategory.ATOMIC_CONTENTION)])
        assert emit_triage_shadow_label(make_triage_subject_key(prod), "atomic_contention", True)
        with profiling_telemetry_session(log_dir=log_b) as st_b:
            assert st_b.retargeted_from_active is True
            assert emit_measure_telemetry(
                "m:b", "cpu", None, cardinality=4, n_flags=0, exit_code=0, duration_ms=1.0
            )
    assert any(ev["name"] == "shadow.label" for ev in read_telemetry_events(log_a))
    assert any(ev["name"] == "xtrax.profile.measure" for ev in read_telemetry_events(log_b))


def test_live_cisternal_shadow_tomls_and_isolation_attestation(tmp_path):
    pytest.importorskip("cisternal.shadow")
    from cisternal.shadow.mode_state import GateCheckError, evaluate_gate, load_mode_state

    toml_files = sorted((REPO_ROOT / "audit" / "shadow").glob("triage_*.toml"))
    assert len(toml_files) == 4
    for toml_path in toml_files:
        loaded = load_mode_state(toml_path)
        assert loaded["shadow"]["writes_isolated_enforced_by"] == TRIAGE_SHADOW_ENFORCER

    shadow_dir = tmp_path / "shadow_eval"
    att = attest_triage_shadow_isolation(log_dir=shadow_dir)
    assert len(att) == 4 and all(att.values())

    empty_dir = tmp_path / "empty_eval"
    empty_dir.mkdir()
    checks = collect_triage_shadow_checks(empty_dir, repo_root=REPO_ROOT)
    assert len(checks) == 4
    assert all(c["status"] == "uncalibrated" for c in checks)

    bad_repo = tmp_path / "bad_repo"
    (bad_repo / "audit" / "shadow").mkdir(parents=True)
    (bad_repo / "audit" / "shadow" / "triage_bad.toml").write_text("[unknown_table]\nx = 1\n")
    with pytest.raises(GateCheckError) as exc_info:
        collect_triage_shadow_checks(empty_dir, repo_root=bad_repo)
    assert exc_info.value.exit_code == 3

    mode_data = load_mode_state(REPO_ROOT / "audit" / "shadow" / "triage_spatial_dead_work.toml")
    with pytest.raises(GateCheckError, match="0 verdicts and 0 labels"):
        evaluate_gate(mode_data, shadow_dir, repo_root=REPO_ROOT)

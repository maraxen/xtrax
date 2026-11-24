"""Tests for the ``xtrax profile`` verb (spec 261016 X4).

Exit codes: 0 pass, 1 flags raised, 2 could not measure. Exit 2 must never read
as a pass. Fixtures live in ``tests/cli/_profile_fixtures.py``.
"""

import pytest

from xtrax.profiling.record import ProbeRecord

HOST = "tests.cli._profile_fixtures:host_loop"
VMAP = "tests.cli._profile_fixtures:vmapped_twin"
FRESH = "tests.cli._profile_fixtures:fresh_jit_per_call"
IDENTITY = "tests.cli._profile_fixtures:identity"
RAISES = "tests.cli._profile_fixtures:raises"
MISSING = "tests.cli._profile_fixtures:no_such_symbol"
FACTORY = "tests.cli._profile_fixtures:make_inputs"
SHAPES = "x=(64,128)f32"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No ambient resolver layers: no env var, empty user config, cwd outside the repo."""
    for env_name in (
        "XTRAX_PROFILE_DIR",
        "CISTERNAL_TELEMETRY_LOG_DIR",
        "CISTERNAL_LOG_DIR",
        "BTH_LOG_DIR",
        "CTXP_LOG_DIR",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)


def _run(capsys, **overrides):
    from xtrax.cli.profile_verb import ProfileArgs, run_profile

    fields = {
        "fn": VMAP,
        "shapes": SHAPES,
        "extent_axis": "x",
        "n_steady": 2,
        "n_wall": 2,
    }
    fields.update(overrides)
    try:
        run_profile(ProfileArgs(**fields))
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_profile_verb_is_registered():
    from xtrax.cli.registry import REGISTRY

    assert "profile" in REGISTRY
    args_cls, run_fn = REGISTRY["profile"]
    assert args_cls.__name__ == "ProfileArgs"
    assert callable(run_fn)


def test_host_loop_is_flagged_by_dispatch_ratio(capsys):
    code, out, _ = _run(capsys, fn=HOST)
    assert code == 1
    assert "FLAG" in out
    assert "exec/element" in out


def test_vmapped_twin_passes(capsys):
    code, out, _ = _run(capsys, fn=VMAP, n_units=10)
    assert code == 0
    assert "PASS" in out
    assert "CPU-relative" in out


def test_fresh_jit_per_call_is_flagged_by_steady_recompiles(capsys):
    code, out, _ = _run(capsys, fn=FRESH)
    assert code == 1
    assert "steady" in out.lower()


def test_bad_fn_path_exits_2(capsys):
    code, out, err = _run(capsys, fn=MISSING)
    assert code == 2
    assert "could not measure" in out.lower()
    assert "no_such_symbol" in err


def test_fn_that_raises_exits_2(capsys):
    code, out, _ = _run(capsys, fn=RAISES)
    assert code == 2
    assert "could not measure" in out.lower()


def test_zero_executions_exits_2_not_pass(capsys):
    code, out, _ = _run(capsys, fn=IDENTITY)
    assert code == 2
    assert "PASS" not in out


def test_extent_axis_must_name_a_shape_input(capsys):
    code, out, err = _run(capsys, extent_axis="nope")
    assert code == 2
    assert "nope" in err


def test_exactly_one_of_shapes_or_inputs(capsys):
    code, _, err = _run(capsys, shapes=None, inputs=None)
    assert code == 2
    assert "--shapes" in err and "--inputs" in err


def test_inputs_factory_mode_reads_extent_from_kwarg(capsys):
    code, out, _ = _run(
        capsys,
        fn=HOST,
        shapes=None,
        inputs=FACTORY,
        scale=64,
        extent_axis="x",
    )
    assert code == 1
    assert "exec/element" in out


def test_inputs_factory_requires_scale(capsys):
    code, _, err = _run(capsys, shapes=None, inputs=FACTORY, scale=None)
    assert code == 2
    assert "--scale" in err


def test_record_written_to_out_dir_with_expected_metrics(capsys, tmp_path):
    out_dir = tmp_path / "profiles"
    code, out, _ = _run(capsys, fn=VMAP, out=str(out_dir))
    assert code == 0
    assert f"{out_dir}" in out and "source: arg" in out

    records = sorted(out_dir.glob("*.json"))
    assert len(records) == 1
    record = ProbeRecord.read(records[0])
    assert record.stage == 1
    assert record.n_atoms == 64
    expected = {
        "n_executions",
        "n_compilations",
        "n_jit_traces",
        "total_step_seconds",
        "compile_seconds",
        "device_seconds",
        "host_seconds",
        "exec_per_element",
    }
    assert expected <= set(record.metrics)


def test_nothing_written_when_resolver_is_off(capsys, tmp_path):
    code, out, _ = _run(capsys, fn=VMAP)
    assert code == 0
    assert "off" in out
    assert list(tmp_path.rglob("*.json")) == []


# --- X6: fit pre-check and extrapolation ------------------------------------

CLEAN_SHA = "a" * 40


def _small_device(monkeypatch, limit: int | None = 1024):
    """Pretend the device has ``limit`` bytes; the real CPU reports none."""
    import xtrax.profiling.fit as fit_mod

    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (limit, "cpu"))


def test_fit_false_without_target_extent_exits_2(capsys, monkeypatch):
    _small_device(monkeypatch)
    code, out, err = _run(capsys, fn=VMAP)
    assert code == 2
    assert "fits: no" in out
    assert "--target-extent" in err
    assert "PASS" not in out


def test_fit_false_with_target_extent_still_exits_2(capsys, monkeypatch):
    # The unit does not fit at the MEASURED extent, so --target-extent cannot rescue it.
    _small_device(monkeypatch)
    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    code, out, err = _run(capsys, fn=VMAP, target_extent=128)
    assert code == 2
    assert "fits: no" in out
    assert "PASS" not in out
    assert "does not fit device memory" in err


def _fake_fit(limit_extent: int):
    """Fit stand-in: a unit fits while its leading input extent is below ``limit_extent``."""
    from xtrax.profiling.fit import FitCheck

    def fake(fn, args, kwargs=None):
        extent = int(args[0].shape[0])
        fits = extent < limit_extent
        return FitCheck(
            fits=fits,
            required_bytes=extent,
            limit_bytes=limit_extent,
            reason=f"fake: extent {extent}",
        )

    return fake


def test_target_extent_that_does_not_fit_withholds_estimate(capsys, monkeypatch):
    from xtrax.cli import profile_verb

    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    monkeypatch.setattr(profile_verb, "fit_precheck", _fake_fit(limit_extent=256))
    code, out, _ = _run(capsys, fn=VMAP, target_extent=256, n_units=10)
    assert code == 0, "a target that does not fit must not change the verdict"
    assert "does not fit at target extent" in out
    assert "wall estimate withheld" in out
    assert "est. wall per unit" not in out


def test_target_extent_that_fits_prints_fit_row_and_estimate(capsys, monkeypatch):
    from xtrax.cli import profile_verb

    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    monkeypatch.setattr(profile_verb, "fit_precheck", _fake_fit(limit_extent=1024))
    code, out, _ = _run(capsys, fn=VMAP, target_extent=256)
    assert code == 0
    assert "fits at target extent: yes" in out
    assert "est. wall per unit" in out


def test_target_extent_with_unknown_fit_says_so(capsys, monkeypatch):
    from xtrax.cli import profile_verb
    from xtrax.profiling.fit import FitCheck

    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)

    def unknown(fn, args, kwargs=None):
        if int(args[0].shape[0]) == 256:
            return FitCheck(None, None, None, "fake: no limit")
        return FitCheck(True, 1, 2, "fake: measured fits")

    monkeypatch.setattr(profile_verb, "fit_precheck", unknown)
    code, out, _ = _run(capsys, fn=VMAP, target_extent=256)
    assert code == 0
    assert "fits at target extent: unknown" in out


# --- measure-path exceptions map to exit 2 (never exit 1) -------------------


def test_probe_exception_maps_to_exit_2(capsys, monkeypatch):
    from xtrax.cli import profile_verb

    def boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(profile_verb, "probe_unit", boom)
    code, out, err = _run(capsys, fn=VMAP)
    assert code == 2
    assert "could not measure" in out.lower()
    assert "RuntimeError" in err and "probe exploded" in err


def test_uncaught_runtime_error_in_zero_args_maps_to_exit_2(capsys, monkeypatch):
    # The jnp.zeros allocation at a huge --shapes raises JaxRuntimeError; it must not
    # escape as an uncaught exception (which the process reports as exit 1).
    from xtrax.cli import profile_verb

    def oom(specs, extent_axis, extent):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    monkeypatch.setattr(profile_verb, "_zero_args", oom)
    code, out, err = _run(capsys, fn=VMAP)
    assert code == 2
    assert "could not measure" in out.lower()
    assert "RESOURCE_EXHAUSTED" in err


def test_fit_row_is_printed_when_it_fits(capsys, monkeypatch):
    _small_device(monkeypatch, limit=None)
    code, out, _ = _run(capsys, fn=VMAP)
    assert code == 0
    assert "fits: unknown" in out


def test_extrapolated_estimate_printed_when_claim_passes(capsys, monkeypatch):
    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    code, out, _ = _run(capsys, fn=VMAP, target_extent=256, n_units=10)
    assert code == 0
    assert "extrapolated, CPU-relative" in out
    assert "wall estimate withheld" not in out


def test_extrapolated_target_is_recorded_as_float_metric(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    out_dir = tmp_path / "profiles"
    code, _, _ = _run(capsys, fn=VMAP, target_extent=256, out=str(out_dir))
    assert code == 0
    (record_path,) = sorted(out_dir.glob("*.json"))
    record = ProbeRecord.read(record_path)
    assert record.metrics["target_extent"] == 256.0
    assert isinstance(record.metrics["target_extent"], float)


def test_wall_estimate_withheld_on_unverifiable_sha(capsys, monkeypatch):
    # The tmp cwd is not a git repo and no override is set, so the sha is "unknown".
    monkeypatch.delenv("XTRAX_GIT_SHA", raising=False)
    code, out, _ = _run(capsys, fn=VMAP, target_extent=256, n_units=10)
    assert code == 0, "withholding the wall estimate must not change the exit code"
    assert "wall estimate withheld" in out
    assert "extrapolated" not in out


def test_wall_estimate_withheld_beyond_scale_limit(capsys, monkeypatch):
    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    code, out, _ = _run(capsys, fn=VMAP, target_extent=1000)
    assert code == 0
    assert "wall estimate withheld" in out
    assert "SCALE_EXTRAPOLATION_LIMIT" in out


def test_target_not_above_measured_extent_is_not_extrapolated(capsys, monkeypatch):
    monkeypatch.setenv("XTRAX_GIT_SHA", CLEAN_SHA)
    code, out, _ = _run(capsys, fn=VMAP, target_extent=64)
    assert code == 0
    assert "no extrapolation" in out
    assert "extrapolated" not in out


# --- X9: --check ------------------------------------------------------------


def _write_check_record(path, *, git_sha, exec_pe=0.01, n_compilations=0.0, n_executions=None):
    from xtrax.profiling.emitters import emit_probe_record

    # exec_per_element is n_executions / n_atoms by construction in run_profile, so
    # the default keeps the record internally consistent (n_atoms is 64 here).
    if n_executions is None:
        n_executions = exec_pe * 64
    emit_probe_record(
        path=path,
        probe_id="xtrax-profile:test",
        stage=1,
        n_atoms=64,
        platform="cpu",
        metrics={
            "n_executions": n_executions,
            "n_compilations": n_compilations,
            "n_jit_traces": 1.0,
            "total_step_seconds": 0.001,
            "compile_seconds": 0.1,
            "device_seconds": 0.0005,
            "host_seconds": 0.0005,
            "exec_per_element": exec_pe,
        },
        config={"fn": "test"},
        git_sha=git_sha,
    )
    return path


def _git_head(repo):
    import subprocess

    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        check=True,
    )
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def test_check_current_passing_record_exits_0(capsys, tmp_path):
    head = _git_head(tmp_path)
    record = _write_check_record(tmp_path / "r.json", git_sha=head)
    code, out, _ = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(record), sha="HEAD"
    )
    assert code == 0
    assert "PASS" in out


def test_check_mismatched_sha_is_stale_exit_1(capsys, tmp_path):
    _git_head(tmp_path)
    record = _write_check_record(tmp_path / "r.json", git_sha="0" * 40)
    code, out, _ = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(record), sha="HEAD"
    )
    assert code == 1
    assert "STALE" in out


def test_check_flagged_metrics_exit_1(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40, exec_pe=3.0)
    code, out, _ = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 1
    assert "FLAGGED" in out
    assert "exec/element" in out


def test_check_recompiling_record_is_flagged(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40, n_compilations=2.0)
    code, out, _ = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 1
    assert "n_compilations" in out


def test_check_record_missing_n_compilations_exits_2(capsys, tmp_path):
    import json

    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40)
    data = json.loads(record.read_text())
    del data["metrics"]["n_compilations"]
    record.write_text(json.dumps(data))
    code, _, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "n_compilations" in err


def test_check_garbage_file_exits_2(capsys, tmp_path):
    garbage = tmp_path / "garbage.json"
    garbage.write_text("this is not a record")
    code, out, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(garbage))
    assert code == 2
    assert "could not measure" in out.lower()
    assert "could not read" in err


def test_check_unresolvable_sha_exits_2(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40)
    code, _, err = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(record), sha="HEAD"
    )
    assert code == 2
    assert "resolve" in err


def test_check_unverifiable_sha_exits_2(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="unknown")
    code, _, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "unverifiable" in err


def test_fn_and_check_together_exit_2(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40)
    code, _, err = _run(capsys, check=str(record))
    assert code == 2
    assert "exactly one" in err


def test_sha_without_check_exits_2(capsys):
    code, _, err = _run(capsys, sha="HEAD")
    assert code == 2
    assert "--check" in err


# --- --check hardening: consistency, sha freshness --------------------------


def test_check_without_sha_cannot_pass(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40)
    code, out, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "PASS" not in out
    assert "pass --sha <rev> to check freshness" in err


def test_check_without_sha_still_reports_a_flag_as_exit_1(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40, n_compilations=2.0)
    code, out, _ = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 1
    assert "FLAGGED" in out


def test_check_short_sha_equal_to_head_passes(capsys, tmp_path):
    head = _git_head(tmp_path)
    record = _write_check_record(tmp_path / "r.json", git_sha=head[:7])
    code, out, _ = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(record), sha="HEAD"
    )
    assert code == 0
    assert "PASS" in out


def test_check_short_sha_of_another_commit_is_stale(capsys, tmp_path):
    import subprocess

    head = _git_head(tmp_path)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "second",
        ],
        check=True,
    )
    record = _write_check_record(tmp_path / "r.json", git_sha=head[:7])
    code, out, _ = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(record), sha="HEAD"
    )
    assert code == 1
    assert "STALE" in out


def test_check_unresolvable_short_sha_exits_2_not_stale(capsys, tmp_path):
    _git_head(tmp_path)
    record = _write_check_record(tmp_path / "r.json", git_sha="fffffff")
    code, out, err = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(record), sha="HEAD"
    )
    assert code == 2
    assert "unknown sha" in err
    assert "STALE" not in out


def test_check_zero_executions_exits_2(capsys, tmp_path):
    record = _write_check_record(
        tmp_path / "r.json", git_sha="b" * 40, exec_pe=0.0, n_executions=0.0
    )
    code, out, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "zero executions" in err
    assert "PASS" not in out


def test_check_missing_n_executions_exits_2(capsys, tmp_path):
    import json

    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40)
    data = json.loads(record.read_text())
    del data["metrics"]["n_executions"]
    record.write_text(json.dumps(data))
    code, _, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "n_executions" in err


def test_check_inconsistent_exec_per_element_exits_2(capsys, tmp_path):
    # 193 executions over 64 elements is about 3.0 per element, not 0.01.
    record = _write_check_record(
        tmp_path / "r.json", git_sha="b" * 40, exec_pe=0.01, n_executions=193.0
    )
    code, out, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "inconsistent" in err
    assert "PASS" not in out


def test_check_negative_n_compilations_exits_2(capsys, tmp_path):
    record = _write_check_record(tmp_path / "r.json", git_sha="b" * 40, n_compilations=-1.0)
    code, out, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "n_compilations" in err
    assert "PASS" not in out


def test_check_negative_exec_per_element_exits_2(capsys, tmp_path):
    record = _write_check_record(
        tmp_path / "r.json", git_sha="b" * 40, exec_pe=-0.5, n_executions=-32.0
    )
    code, _, err = _run(capsys, fn=None, shapes=None, extent_axis=None, check=str(record))
    assert code == 2
    assert "negative" in err


# --- --compare / --triage / --telemetry-summary ----------------------------


def _write_full_record(
    path,
    *,
    probe_id="xtrax-profile:mod:step",
    git_sha="a" * 40,
    total_step_seconds=1.0,
    config_extra=None,
    metrics_extra=None,
):
    from xtrax.profiling.emitters import emit_probe_record

    cfg = {
        "fn": "mod:step",
        "step_time_basis": "batch",
        "non_degeneracy_status": "pass",
        "corruption_control_status": "pass",
    }
    if config_extra:
        cfg.update(config_extra)
    met = {
        "n_executions": 10.0,
        "n_compilations": 0.0,
        "n_jit_traces": 10.0,
        "total_step_seconds": total_step_seconds,
        "wall_mad_seconds": 0.005,
        "wall_iqr_seconds": 0.01,
        "peak_vram_bytes": 1024.0,
        "output_l2_norm": 3.0,
        "parity_rel_err": 1e-6,
        "corruption_rel_err": 0.5,
        "work_active_count": 100.0,
    }
    if metrics_extra:
        met.update(metrics_extra)
    return emit_probe_record(
        path=path,
        probe_id=probe_id,
        stage=1,
        n_atoms=64,
        platform="cpu",
        metrics=met,
        config=cfg,
        git_sha=git_sha,
    )


def test_compare_keep_ratchet_hold_fixed_and_bypass(capsys, tmp_path):
    p_base = tmp_path / "base.json"
    p_cand = tmp_path / "cand.json"
    p_noise = tmp_path / "noise.json"
    p_drift = tmp_path / "drift.json"
    ptr = tmp_path / "ptr.json"
    rej_dir = tmp_path / "rejected"

    _write_full_record(p_base, git_sha="a" * 40, total_step_seconds=1.0)
    _write_full_record(p_cand, git_sha="b" * 40, total_step_seconds=0.5)
    _write_full_record(p_noise, git_sha="c" * 40, total_step_seconds=0.495)
    _write_full_record(
        p_drift,
        git_sha="d" * 40,
        total_step_seconds=0.5,
        config_extra={"extra_knob": "changed"},
    )

    code, out, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        ratchet_pointer=str(ptr),
    )
    assert code == 0
    assert "status | keep" in out
    assert ptr.is_file()

    # Default inferred hold-fixed rejects extra_knob drift -> exit 2
    code_drift, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_drift)),
    )
    assert code_drift == 2

    # --bypass-hold-fixed caps KEEP at ADVISORY_ONLY -> exit 1
    code_byp, out_byp, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_drift)),
        bypass_hold_fixed=True,
    )
    assert code_byp == 1
    assert "advisory_only" in out_byp

    # --rejected-dir writes sidecar on REVERT_NOISE -> exit 1
    code_noise, out_noise, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_cand), str(p_noise)),
        ratchet_pointer=str(ptr),
        rejected_dir=str(rej_dir),
    )
    assert code_noise == 1
    assert "revert_noise" in out_noise
    assert len(list(rej_dir.glob("*.verdict.json"))) == 1

    # Opt-out of corruption control when record has not_measured status
    p_nocorr = tmp_path / "nocorr.json"
    _write_full_record(
        p_nocorr,
        git_sha="e" * 40,
        total_step_seconds=0.5,
        config_extra={"corruption_control_status": "not_measured"},
        metrics_extra={"total_step_seconds": 0.5},
    )
    # Remove corruption_rel_err to simulate unmeasured corruption control
    import json

    raw = json.loads(p_nocorr.read_text())
    del raw["metrics"]["corruption_rel_err"]
    p_nocorr.write_text(json.dumps(raw))
    code_nc, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_nocorr)),
        require_corruption_control=False,
    )
    assert code_nc == 0


def test_triage_compare_and_telemetry_summary_core(capsys, tmp_path):
    import json

    p_triage = tmp_path / "prod_triage.json"
    _write_full_record(
        p_triage,
        git_sha="a" * 40,
        total_step_seconds=1.0,
        metrics_extra={"active_work_ratio": 0.10, "ctas_per_atomic_target": 128.0},
    )
    code_t, out_t, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        triage=str(p_triage),
    )
    assert code_t == 1
    assert "shadow_subject_key: probe:" in out_t
    assert "spatial_dead_work" in out_t
    assert "atomic_contention" in out_t

    p_clean = tmp_path / "clean.json"
    _write_full_record(p_clean, git_sha="f" * 40, total_step_seconds=1.0)
    code_t0, out_t0, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        triage=str(p_clean),
        target_missed=True,
    )
    assert code_t0 == 0
    assert "- none" in out_t0

    # --from-triage-probe + --triage-category validation works across all 4 categories
    p_cand = tmp_path / "cand.json"
    _write_full_record(p_cand, git_sha="b" * 40, total_step_seconds=0.5)
    for cat in (
        "atomic_contention",
        "xtrax.triage.register_spill_pressure",
        "spatial_dead_work",
        "fp64_pipe_saturation",
    ):
        code_c, _, _ = _run(
            capsys,
            fn=None,
            shapes=None,
            extent_axis=None,
            compare=(str(p_triage), str(p_cand)),
            from_triage_probe=str(p_triage),
            triage_category=cat,
        )
        assert code_c == 0

    code_bad_cat, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_triage), str(p_cand)),
        from_triage_probe=str(p_triage),
        triage_category="invalid_cat",
    )
    assert code_bad_cat == 2

    # --telemetry-summary works on pure synthetic JSONL + .verdict.json without cisternal
    log_dir = tmp_path / "synthetic_logs"
    log_dir.mkdir()
    (log_dir / "events.host.jsonl").write_text(
        json.dumps({"name": "xtrax.profile.measure", "ts": 1.0, "fields": {"exit_code": 0}}) + "\n",
        encoding="utf-8",
    )
    code_sum0, out_sum0, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        telemetry_log_dir=str(log_dir),
    )
    assert code_sum0 == 0
    assert "Algedonic Signals" in out_sum0

    # Two unclassified slowdowns fire TRIAGE_GAP -> exit 1
    with (log_dir / "events.host.jsonl").open("a", encoding="utf-8") as handle:
        for ts in (2.0, 3.0):
            handle.write(
                json.dumps(
                    {
                        "name": "xtrax.profile.triage",
                        "ts": ts,
                        "fields": {"unclassified_slowdown": True},
                    }
                )
                + "\n"
            )
    code_sum1, out_sum1, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        telemetry_log_dir=str(log_dir),
    )
    assert code_sum1 == 1
    assert "triage_gap" in out_sum1


def test_triage_compare_live_cisternal_shadow_roundtrip(capsys, tmp_path):
    pytest.importorskip("cisternal.shadow")
    from xtrax.telemetry.profiling_events import read_telemetry_events

    log_dir = tmp_path / "telemetry"
    p_triage = tmp_path / "prod_triage.json"
    _write_full_record(
        p_triage,
        git_sha="a" * 40,
        total_step_seconds=1.0,
        metrics_extra={"active_work_ratio": 0.10, "ctas_per_atomic_target": 128.0},
    )
    code_t, out_t, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        triage=str(p_triage),
        telemetry_log_dir=str(log_dir),
    )
    assert code_t == 1
    assert "shadow_subject_key: probe:" in out_t

    events = read_telemetry_events(log_dir)
    assert any(ev["name"] == "shadow.verdict" for ev in events)

    # Targeted feedback across all 4 provisional rules
    p_cand = tmp_path / "cand.json"
    _write_full_record(p_cand, git_sha="b" * 40, total_step_seconds=0.5)
    for cat in (
        "atomic_contention",
        "xtrax.triage.register_spill_pressure",
        "spatial_dead_work",
        "fp64_pipe_saturation",
    ):
        code_c, _, _ = _run(
            capsys,
            fn=None,
            shapes=None,
            extent_axis=None,
            compare=(str(p_triage), str(p_cand)),
            from_triage_probe=str(p_triage),
            triage_category=cat,
            telemetry_log_dir=str(log_dir),
        )
        assert code_c == 0

    events_after = read_telemetry_events(log_dir)
    label_events = [ev for ev in events_after if ev["name"] == "shadow.label"]
    labeled = {ev["fields"]["feature"] for ev in label_events}
    assert all(ev["fields"]["label"] is True for ev in label_events)
    assert labeled == {
        "xtrax.triage.atomic_contention",
        "xtrax.triage.register_spill_pressure",
        "xtrax.triage.spatial_dead_work",
        "xtrax.triage.fp64_pipe_saturation",
    }

    # Regression feedback emits label=False; ADVISORY_ONLY skips shadow.label emission
    p_reg = tmp_path / "reg.json"
    _write_full_record(p_reg, git_sha="c" * 40, total_step_seconds=2.0)
    code_reg, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_triage), str(p_reg)),
        from_triage_probe=str(p_triage),
        triage_category="spatial_dead_work",
        telemetry_log_dir=str(log_dir),
    )
    assert code_reg == 1

    p_dirty = tmp_path / "dirty.json"
    _write_full_record(p_dirty, git_sha="d" * 40 + "-dirty", total_step_seconds=0.5)
    code_adv, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_triage), str(p_dirty)),
        allow_dirty=True,
        from_triage_probe=str(p_triage),
        triage_category="atomic_contention",
        telemetry_log_dir=str(log_dir),
    )
    assert code_adv == 1

    labels_final = [ev for ev in read_telemetry_events(log_dir) if ev["name"] == "shadow.label"]
    assert len(labels_final) == len(label_events) + 1
    assert labels_final[-1]["fields"]["label"] is False
    assert labels_final[-1]["fields"]["feature"] == "xtrax.triage.spatial_dead_work"

    # --telemetry-summary on clean log + warm-up shadow gates returns exit 0
    code_sum0, out_sum0, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        telemetry_log_dir=str(log_dir),
    )
    assert code_sum0 == 0
    assert "Algedonic Signals" in out_sum0

    # Trigger --target-missed twice to fire TRIAGE_GAP algedonic signal -> exit 1
    p_clean = tmp_path / "clean.json"
    _write_full_record(p_clean, git_sha="f" * 40, total_step_seconds=1.0)
    for _ in range(2):
        code_tm, _, _ = _run(
            capsys,
            fn=None,
            shapes=None,
            extent_axis=None,
            triage=str(p_clean),
            target_missed=True,
            telemetry_log_dir=str(log_dir),
        )
        assert code_tm == 0

    code_sum1, out_sum1, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        telemetry_log_dir=str(log_dir),
    )
    assert code_sum1 == 1
    assert "triage_gap" in out_sum1


def test_cross_mode_validation_errors_exit_2(capsys, tmp_path):
    p = tmp_path / "r.json"
    _write_full_record(p, git_sha="a" * 40)

    # Zero modes active
    code, _, err = _run(capsys, fn=None, shapes=None, extent_axis=None)
    assert code == 2
    assert "exactly one" in err

    # Two modes active
    code2, _, err2 = _run(
        capsys, fn=None, shapes=None, extent_axis=None, check=str(p), telemetry_summary=True
    )
    assert code2 == 2
    assert "exactly one" in err2

    # --target-missed outside --triage
    code3, _, _ = _run(capsys, fn=VMAP, target_missed=True)
    assert code3 == 2

    # --telemetry-summary with nonexistent log dir -> exit 2
    code4, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        telemetry_log_dir=str(tmp_path / "missing_dir"),
    )
    assert code4 == 2

    # Corrupt JSONL inside log dir -> exit 2
    bad_dir = tmp_path / "bad_logs"
    bad_dir.mkdir()
    (bad_dir / "events.h.1.jsonl").write_text("{corrupt\n")
    code5, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        telemetry_log_dir=str(bad_dir),
    )
    assert code5 == 2

    # Unpaired --from-triage-probe without --triage-category -> exit 2
    code6, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p), str(p)),
        from_triage_probe=str(p),
    )
    assert code6 == 2

    # Empty or whitespace --primary-metric -> exit 2
    for blank in ("", "   "):
        code_pm, _, err_pm = _run(
            capsys,
            fn=None,
            shapes=None,
            extent_axis=None,
            compare=(str(p), str(p)),
            primary_metric=blank,
        )
        assert code_pm == 2
        assert "--primary-metric cannot be empty" in err_pm

    # --higher-is-better outside --compare -> exit 2
    code_hib, _, _ = _run(capsys, fn=VMAP, higher_is_better=True)
    assert code_hib == 2


def test_compare_hold_fixed_parity_and_error_paths(capsys, tmp_path):
    p_base = tmp_path / "base.json"
    p_cand = tmp_path / "cand.json"
    _write_full_record(p_base, git_sha="a" * 40, total_step_seconds=1.0)
    _write_full_record(p_cand, git_sha="b" * 40, total_step_seconds=0.5)

    # Valid explicit --hold-fixed key -> PASS (exit 0)
    code_ok, out_ok, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        hold_fixed="fn",
    )
    assert code_ok == 0
    assert "keep" in out_ok

    # --higher-is-better inverts primary_metric direction for throughput metrics
    p_tp_base = tmp_path / "tp_base.json"
    p_tp_fast = tmp_path / "tp_fast.json"
    p_tp_slow = tmp_path / "tp_slow.json"
    _write_full_record(p_tp_base, git_sha="a" * 40, metrics_extra={"throughput": 100.0})
    _write_full_record(p_tp_fast, git_sha="b" * 40, metrics_extra={"throughput": 120.0})
    _write_full_record(p_tp_slow, git_sha="c" * 40, metrics_extra={"throughput": 80.0})
    code_tp_ok, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_tp_base), str(p_tp_fast)),
        primary_metric="throughput",
        higher_is_better=True,
    )
    assert code_tp_ok == 0
    code_tp_reg, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_tp_base), str(p_tp_slow)),
        primary_metric="throughput",
        higher_is_better=True,
    )
    assert code_tp_reg == 1

    # Empty token inside --hold-fixed -> exit 2
    code_empty, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        hold_fixed=",",
    )
    assert code_empty == 2

    # Mutually exclusive --hold-fixed + --bypass-hold-fixed -> exit 2
    code_conflict, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        hold_fixed="fn",
        bypass_hold_fixed=True,
    )
    assert code_conflict == 2

    # --no-require-parity caps verdict at ADVISORY_ONLY (exit 1)
    code_adv, out_adv, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        require_parity=False,
    )
    assert code_adv == 1
    assert "advisory_only" in out_adv

    # Combining --no-require-parity with --max-rel-err -> exit 2
    code_pconflict, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        require_parity=False,
        max_rel_err=1e-4,
    )
    assert code_pconflict == 2

    # Unreadable JSON in --compare -> exit 2
    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not valid json")
    code_bad_cmp, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(bad_json), str(p_cand)),
    )
    assert code_bad_cmp == 2

    # --compare with --rejected-dir writes sidecar on non-promotable verdict
    rej_dir = tmp_path / "rejected"
    rej_dir.mkdir()
    code_rej, out_rej, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        compare=(str(p_base), str(p_cand)),
        require_parity=False,
        rejected_dir=str(rej_dir),
    )
    assert code_rej == 1
    assert "sidecar" in out_rej
    assert any(rej_dir.glob("*.verdict.json"))


def test_triage_and_telemetry_summary_edge_errors(capsys, tmp_path):
    bad_json = tmp_path / "bad_triage.json"
    bad_json.write_text("{not valid")
    code_tbad, _, _ = _run(capsys, fn=None, shapes=None, extent_axis=None, triage=str(bad_json))
    assert code_tbad == 2

    p_valid = tmp_path / "valid.json"
    _write_full_record(p_valid, git_sha="a" * 40, total_step_seconds=1.0)
    code_prec_bad, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        triage=str(p_valid),
        precision_baseline=str(p_valid),
    )
    assert code_prec_bad == 2

    code_rej_missing, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
        rejected_dir=str(tmp_path / "no_such_rej"),
    )
    assert code_rej_missing == 2

    # With cwd=tmp_path and XDG_CONFIG_HOME isolated, no log/rejected dirs resolve -> exit 2
    code_none, _, _ = _run(
        capsys,
        fn=None,
        shapes=None,
        extent_axis=None,
        telemetry_summary=True,
    )
    assert code_none == 2

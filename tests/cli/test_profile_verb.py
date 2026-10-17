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
    monkeypatch.delenv("XTRAX_PROFILE_DIR", raising=False)
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

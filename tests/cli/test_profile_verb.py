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

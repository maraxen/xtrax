"""Bench-stats -> ProbeRecord bridge pins (xtrax.profiling.bench).

The bridge is opt-in (XTRAX_BENCH_RECORD_DIR) and fail-closed: undeclared
benches are never recorded under guessed semantics, an unknown stats field
aborts the recording path loudly, and the subprocess test at the bottom
runs the REAL benchmarks/conftest.py hook to prove records land on disk
with their declared stage/scale/config intact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax.profiling.bench import (
    bench_metrics_from_stats,
    build_bench_record_plan,
    parse_bench_extra_info,
    record_from_plan,
    sanitize_bench_fullname,
)
from xtrax.profiling.claims import ClaimValidityError
from xtrax.profiling.record import ProbeRecord

_REPO_ROOT = Path(__file__).resolve().parents[2]

# All 16 pinned Stats.fields with plausible values (durations in seconds).
FULL_STATS = {
    "min": 0.001,
    "max": 0.002,
    "mean": 0.0015,
    "stddev": 0.0002,
    "rounds": 25,
    "median": 0.0014,
    "iqr": 0.0003,
    "q1": 0.0013,
    "q3": 0.0016,
    "iqr_outliers": 0,
    "stddev_outliers": 1,
    "outliers": "1;1",  # display-only composite string, dropped by design
    "ld15iqr": 0.0012,
    "hd15iqr": 0.0018,
    "ops": 50,
    "total": 0.0375,
}

DECLARATION = {
    "xtrax_stage": 1,
    "xtrax_n_atoms": 32,
    "xtrax_scale_basis": "batch_rows",
}


def test_sanitize_fullname_keeps_param_drops_path():
    stem = sanitize_bench_fullname(
        "benchmarks/bench_tiling.py::test_tiling_dispatch_overhead[vmap]"
    )
    assert stem == "bench_tiling.py_test_tiling_dispatch_overhead[vmap]"


def test_sanitize_fullname_plain_test_is_stable():
    assert (
        sanitize_bench_fullname("benchmarks/bench_training_step.py::test_trainer_step_throughput")
        == "bench_training_step.py_test_trainer_step_throughput"
    )


def test_parse_extra_info_happy_path():
    extra = dict(DECLARATION)
    extra["xtrax_dispatch_entry"] = "axis_dispatch"
    extra["unrelated_plugin_key"] = "ignored"
    stage, n_atoms, config = parse_bench_extra_info(extra)
    assert stage == 1
    assert n_atoms == 32
    assert config == {"scale_basis": "batch_rows", "dispatch_entry": "axis_dispatch"}


def test_parse_missing_stage_names_the_missing_key():
    with pytest.raises(ClaimValidityError, match="xtrax_stage"):
        parse_bench_extra_info({"xtrax_n_atoms": 32})


def test_parse_missing_n_atoms_names_the_missing_key():
    with pytest.raises(ClaimValidityError, match="xtrax_n_atoms"):
        parse_bench_extra_info({"xtrax_stage": 1})


def test_parse_non_int_stage_raises():
    with pytest.raises(ClaimValidityError, match="not coercible"):
        parse_bench_extra_info({"xtrax_stage": "micro", "xtrax_n_atoms": 32})


def test_metrics_convert_seconds_to_ms_and_pass_counts_through():
    metrics = bench_metrics_from_stats(FULL_STATS)
    assert metrics["mean_ms"] == pytest.approx(1.5)
    assert metrics["total_ms"] == pytest.approx(37.5)
    assert metrics["ld15iqr_ms"] == pytest.approx(1.2)
    # Counts are unsuffixed and float-coerced (contract stores floats only).
    assert metrics["rounds"] == 25.0
    assert metrics["ops"] == 50.0
    assert metrics["stddev_outliers"] == 1.0
    # The display-only composite never enters float-only metrics.
    assert "outliers" not in metrics


def test_metrics_unknown_field_raises_instead_of_dropping():
    drifted = dict(FULL_STATS)
    drifted["p50"] = 0.0014  # hypothetical renamed median in a plugin upgrade
    with pytest.raises(ClaimValidityError, match=r"\['p50'\]"):
        bench_metrics_from_stats(drifted)


@pytest.fixture()
def _platform_cpu(monkeypatch):
    import xtrax.profiling.bench as bench_mod

    monkeypatch.setattr(bench_mod, "_capture_platform", lambda: "cpu")


def test_plan_to_record_round_trip(_platform_cpu, tmp_path):
    plan = build_bench_record_plan(
        fullname="benchmarks/bench_grad_accum.py::test_accumulate_grads_scaling[4]",
        params={"n_microbatches": 4},
        extra_info=dict(DECLARATION),
        stats_dict=FULL_STATS,
    )
    assert plan.probe_id == "bench_grad_accum.py_test_accumulate_grads_scaling[4]"
    # Params win the config over same-named declarations; distinct keys merge.
    assert plan.config["n_microbatches"] == "4"
    assert plan.config["scale_basis"] == "batch_rows"

    record = record_from_plan(plan)
    record.write(tmp_path / "rec.json")
    loaded = ProbeRecord.read(tmp_path / "rec.json")
    assert loaded.probe_id == plan.probe_id
    assert loaded.stage == 1
    assert loaded.n_atoms == 32
    assert loaded.metrics["mean_ms"] == pytest.approx(1.5)
    # No trace was captured by a wall-clock bench: scopes stay None (not {}).
    assert loaded.scopes is None
    assert loaded.attribution_method is None


def test_stage2_declaration_fails_closed_on_cpu(_platform_cpu):
    declared_gpu_only = dict(DECLARATION, xtrax_stage=2)
    with pytest.raises(ClaimValidityError, match="requires platform='gpu'"):
        record_from_plan(
            build_bench_record_plan(
                fullname="benchmarks/bench_x.py::test_x",
                params=None,
                extra_info=declared_gpu_only,
                stats_dict=FULL_STATS,
            )
        )


def test_nonfinite_stat_rejected_by_contract(_platform_cpu):
    diverged = dict(FULL_STATS, mean=float("nan"))
    with pytest.raises(ClaimValidityError, match="not finite"):
        record_from_plan(
            build_bench_record_plan(
                fullname="benchmarks/bench_x.py::test_x",
                params=None,
                extra_info=dict(DECLARATION),
                stats_dict=diverged,
            )
        )


def test_end_to_end_real_conftest_writes_declared_records(tmp_path):
    """Run the repo's own benchmarks/ under the emission env var.

    This is the closed loop: plugin collects fixtures, benches declare via
    extra_info, sessionfinish writes one validated record per bench into
    XTRAX_BENCH_RECORD_DIR. Runs only bench_tiling.py to bound jax startup
    cost while still exercising parametrized identities ([vmap]/[safe_map]
    /[dedup]).
    """
    env = dict(os.environ)
    # Coverage-inheritance scrub (review of CI tier1 failure): pytest-cov
    # exports COV_CORE_* so spawned pytest processes auto-resume measuring
    # with the PARENT'S source spec; the child's data then merges back
    # un-omitted and pollutes the tier's aggregate with every unexecuted
    # module. This test measures nothing itself -- the child must run bare.
    for key in [k for k in env if k.startswith("COV_CORE") or k == "COVERAGE_PROCESS_START"]:
        env.pop(key)
    env["XTRAX_BENCH_RECORD_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_REPO_ROOT / "benchmarks" / "bench_tiling.py"),
            "--benchmark-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        # cwd=tmp_path: the child writes its own default .coverage into
        # its cwd; pointing it at the repo root let an unfiltered,
        # source-scanned child database merge into tier1 measurements
        # (CI coverage-gate failure). Nothing here needs repo-root cwd.
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(written) == 3, proc.stdout
    strategies = set()
    for name in written:
        rec = ProbeRecord.read(tmp_path / name)
        assert rec.probe_id.startswith("bench_tiling.py_test_tiling_dispatch_overhead")
        assert rec.stage == 1
        assert rec.config["scale_basis"] == "batch_rows"
        assert rec.metrics["mean_ms"] > 0.0
        assert json.dumps(rec.config)  # config survives round-trip as JSON
        strategies.add(rec.config["strategy_name"])
    assert strategies == {"vmap", "safe_map", "dedup"}

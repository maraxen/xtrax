"""Tests for D4 performance gate (N2.5 / #1584)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates._trace_probe import ProbeResult, run_trace_probe
from xtrax.devtools.gates.performance import (
    METRIC_KEY,
    WALL_TIME_METRIC_KEY,
    load_performance_targets,
    run_performance_gate,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"
TARGETS_PATH = ROOT / "audit" / "performance_targets.toml"
PROBE_STABLE = "xtrax.devtools.gates._performance_probes:probe_stable_jnp_kernel"


def test_performance_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "performance.toml")
    assert table.dimension == "performance"
    assert len(table.anchors) == 5


def test_load_performance_targets_repo_config() -> None:
    targets = load_performance_targets(TARGETS_PATH)
    assert targets.schema == "performance-gate-v0"
    assert targets.max_traces_default == 1
    assert len(targets.probes) == 1
    assert targets.probes[0].qualname.endswith("filter_jit_linear_kernel")
    assert targets.probes[0].trace_probe is not None


def test_run_trace_probe_skips_without_trace_probe() -> None:
    result = run_trace_probe(
        "xtrax.devtools.gates._performance_probes.filter_jit_linear_kernel",
        max_traces=1,
        trace_probe=None,
    )
    assert result.skipped is True
    assert result.passed is False


def test_run_trace_probe_stable_jnp_kernel_passes(tmp_path: Path) -> None:
    probe_module = tmp_path / "trace_kernel_fixture.py"
    probe_module.write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def kernel(x):
                return x + 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))

    result = run_trace_probe(
        "trace_kernel_fixture.kernel",
        max_traces=1,
        trace_probe=PROBE_STABLE,
    )
    assert result.passed is True
    assert not result.skipped


def test_run_performance_gate_fails_on_mock_trace_failure(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "xtrax.fake.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    mock_failure = ProbeResult(
        qualname="xtrax.fake.kernel",
        passed=False,
        reason="trace count exceeded",
        trace_probe=PROBE_STABLE,
        max_traces=1,
    )

    with (
        patch(
            "xtrax.devtools.gates.performance.run_trace_probe",
            return_value=mock_failure,
        ),
        patch(
            "xtrax.devtools.gates.performance._measure_wall_time_median_ms",
            return_value=1.0,
        ),
    ):
        result = run_performance_gate(
            targets_path=targets_path,
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is False
    assert result.trace_violation_count == 1
    assert result.findings_emitted >= 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "performance"
    assert record["payload"]["violation_kind"] == "trace_count"
    assert record["severity"] == "major"


def test_run_performance_gate_passes_with_stable_probe(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    probe_module = tmp_path / "stable_kernel_fixture.py"
    probe_module.write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def kernel(x):
                return x + 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))

    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "stable_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is True
    assert result.trace_violation_count == 0
    assert result.wall_time_median_ms is not None
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    wall_record = json.loads(lines[-1])
    assert wall_record["severity"] == "info"
    assert wall_record["payload"]["recorded_only"] is True
    assert "wall_time_median_ms" in wall_record["payload"]


def test_run_performance_gate_emits_recorded_wall_time(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    mock_pass = ProbeResult(
        qualname="xtrax.devtools.gates._performance_probes.filter_jit_linear_kernel",
        passed=True,
        trace_probe="xtrax.devtools.gates._performance_probes:probe_filter_jit_linear_kernel",
        max_traces=1,
    )

    with (
        patch(
            "xtrax.devtools.gates.performance.run_trace_probe",
            return_value=mock_pass,
        ),
        patch(
            "xtrax.devtools.gates.performance._measure_wall_time_median_ms",
            return_value=1.0,
        ),
    ):
        result = run_performance_gate(
            targets_path=TARGETS_PATH,
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is True
    assert result.wall_time_median_ms == 1.0
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["payload"]["recorded_only"] is True
    assert record["payload"]["wall_time_median_ms"] == 1.0


def test_run_performance_gate_integration_filter_jit_linear(tmp_path: Path) -> None:
    """End-to-end trace gate against repo performance_targets (no mocks)."""
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=TARGETS_PATH,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is True
    assert result.trace_violation_count == 0
    assert result.wall_time_median_ms is not None


def test_committed_baseline_has_performance_trace_metric() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert METRIC_KEY in loaded.metrics
    entry = loaded.metrics[METRIC_KEY]
    assert entry.comparator == "minimize"
    assert entry.value == 0.0


def test_audit_performance_gate_cli_exits_zero(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_performance_gate.py",
            "--baseline-path",
            str(baseline_path),
            "--audits-path",
            str(audits_path),
            "--no-write-baseline",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout
    assert WALL_TIME_METRIC_KEY in result.stdout


# --- Phase C: profiler-backed dispatch tripwires + ProbeRecord emission -----


def test_repo_targets_have_no_dispatch_config() -> None:
    """No-behavior-change guard: the repo's own TOML stays pre-Phase-C."""
    from xtrax.devtools.gates.performance import load_performance_targets

    targets = load_performance_targets(TARGETS_PATH)
    for spec in targets.probes:
        assert spec.max_compilations is None
        assert spec.max_jit_traces is None
        assert spec.emit_probe_record is False


def test_load_performance_targets_parses_dispatch_fields(tmp_path: Path) -> None:
    from xtrax.devtools.gates.performance import load_performance_targets

    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "stable_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            max_compilations = 2
            max_jit_traces = 8
            emit_probe_record = true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = load_performance_targets(targets_path).probes[0]
    assert spec.max_compilations == 2
    assert spec.max_jit_traces == 8
    assert spec.emit_probe_record is True


def test_load_performance_targets_rejects_bad_ceiling(tmp_path: Path) -> None:
    from xtrax.devtools.gates.performance import load_performance_targets

    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "stable_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            max_compilations = 0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_compilations"):
        load_performance_targets(targets_path)


def _write_kernel_module(tmp_path: Path, name: str) -> None:
    import sys

    module_path = tmp_path / f"{name}.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def kernel(x):
                return x + 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))


def test_dispatch_tripwire_fails_on_tight_ceiling(tmp_path: Path) -> None:
    """Real traced run: any sane kernel emits >=1 jit-trace (D9), so a
    ceiling of 1 is always exceeded -- deterministic fail-closed check.

    Seeding DISPATCH_METRIC_KEY=0.0 mirrors the existing trace-count tests:
    under the baseline ratchet's bootstrap-on-missing-key semantics
    (evaluate_metric returns (True, True) for an untracked metric), blocking
    starts once the committed baseline anchors the metric -- first-ever runs
    bootstrap instead of failing.
    """
    from xtrax.devtools.gates.performance import (
        DISPATCH_METRIC_KEY,
        run_performance_gate,
    )

    _write_kernel_module(tmp_path, "dispatch_kernel_fixture")
    audits_path = tmp_path / "audits.jsonl"
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "dispatch_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            max_jit_traces = 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(key=METRIC_KEY, value=0.0, comparator="minimize"),
            DISPATCH_METRIC_KEY: MetricEntry(
                key=DISPATCH_METRIC_KEY, value=0.0, comparator="minimize"
            ),
        },
    )
    baseline_path = tmp_path / "audit_baseline.json"
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is False
    assert result.trace_violation_count == 0
    assert result.dispatch_violation_count == 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    violations = [
        json.loads(line)
        for line in lines
        if json.loads(line)["payload"].get("violation_kind") == "dispatch_count"
    ]
    assert len(violations) == 1
    assert violations[0]["payload"]["violation_kind"] == "dispatch_count"
    assert violations[0]["severity"] == "major"
    assert violations[0]["payload"]["counter"] == "n_jit_traces"


def test_dispatch_generous_ceiling_passes_and_emits_probe_record(
    tmp_path: Path,
) -> None:
    from xtrax.devtools.gates.performance import (
        DISPATCH_METRIC_KEY,
        run_performance_gate,
    )
    from xtrax.profiling.record import ProbeRecord

    _write_kernel_module(tmp_path, "generous_kernel_fixture")
    audits_path = tmp_path / "audits.jsonl"
    records_dir = tmp_path / "probe_records"
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "generous_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            max_compilations = 2
            max_jit_traces = 64
            emit_probe_record = true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "audit_baseline.json"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(key=METRIC_KEY, value=0.0, comparator="minimize"),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=True,
        probe_record_dir=records_dir,
    )

    assert result.passed is True
    assert result.dispatch_violation_count == 0

    written = sorted(records_dir.glob("gate_*.json"))
    assert len(written) == 1
    record = ProbeRecord.read(written[0])
    assert record.metrics["n_executions"] >= 1.0

    # Opt-in ratchet bootstrapped its baseline entry on first passing use.
    reloaded = load_baseline(path=baseline_path)
    assert DISPATCH_METRIC_KEY in reloaded.metrics


def test_dispatch_probe_crash_becomes_finding_not_gate_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review finding: a broken dispatch probe (bad qualname/import) must
    surface as a major finding -- never crash the whole gate with a raw
    traceback after the caller already did real work."""
    import xtrax.devtools.gates.performance as perf_mod

    def _boom(qualname: str, trace_probe: str):
        raise RuntimeError(f"no module named {qualname!r}")

    monkeypatch.setattr(perf_mod, "measure_dispatch_counts", _boom)

    _write_kernel_module(tmp_path, "crashy_kernel_fixture")
    audits_path = tmp_path / "audits.jsonl"
    records_dir = tmp_path / "probe_records"
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "crashy_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            max_compilations = 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "audit_baseline.json"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(key=METRIC_KEY, value=0.0, comparator="minimize"),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=True,
        probe_record_dir=records_dir,
    )

    # Containment contract: the crash becomes a counted major finding
    # (dispatch_probe_error) instead of an uncaught exception. Whether a
    # dispatch finding flips overall pass/fail is existing gate policy
    # (trace violations drive `passed`) -- unchanged here.
    assert result.dispatch_violation_count == 1
    assert result.findings_emitted >= 2
    assert "dispatch_probe_error" in audits_path.read_text()


def test_no_dispatch_config_leaves_dispatch_metric_absent(tmp_path: Path) -> None:
    from xtrax.devtools.gates.performance import (
        DISPATCH_METRIC_KEY,
        run_performance_gate,
    )

    _write_kernel_module(tmp_path, "plain_kernel_fixture")
    audits_path = tmp_path / "audits.jsonl"
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "plain_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "audit_baseline.json"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(key=METRIC_KEY, value=0.0, comparator="minimize"),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=True,
    )

    assert result.passed is True
    assert result.dispatch_violation_count == 0
    reloaded = load_baseline(path=baseline_path)
    assert DISPATCH_METRIC_KEY not in reloaded.metrics


def test_boolean_toml_ceiling_rejected(tmp_path: Path) -> None:
    """Review finding: TOML `max_compilations = true` must not become ceiling 1."""
    import pytest as _pytest

    from xtrax.devtools.gates.performance import load_performance_targets

    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            """
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"

            [[probes]]
            qualname = "some.kernel"
            max_traces = 1
            max_compilations = true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with _pytest.raises(ValueError, match="must be a positive int"):
        load_performance_targets(targets_path)


# --- X8: dispatch probe on probe_unit keeps the pre-refactor counts -----------

_KERNEL_QUALNAME = "xtrax.devtools.gates._performance_probes.filter_jit_linear_kernel"
_KERNEL_PROBE = "xtrax.devtools.gates._performance_probes:probe_filter_jit_linear_kernel"


def _legacy_dispatch_counts(qualname: str, trace_probe: str) -> dict[str, int]:
    """Oracle: the pre-X8 measure_dispatch_counts body, reproduced inline.

    One warm-up outside the window, one traced call, newest trace file parsed
    with parse_dispatch_counts (no fn_name filter).
    """
    import gzip
    import json
    import tempfile

    import jax

    from xtrax.devtools.gates._trace_probe import (
        _guard_for_trace_count,
        import_probe,
        import_qualname,
    )
    from xtrax.profiling.trace import parse_dispatch_counts

    target = import_qualname(qualname)
    probe_runner = import_probe(trace_probe)
    guarded = _guard_for_trace_count(target, max_traces=10**6)
    probe_runner(guarded)
    with tempfile.TemporaryDirectory() as tmp:
        with jax.profiler.trace(tmp, create_perfetto_trace=True):
            probe_runner(guarded)
        trace_files = sorted(Path(tmp).rglob("*.trace.json.gz"))
        with gzip.open(trace_files[-1], "rt") as fh:
            events = json.load(fh)["traceEvents"]
    return parse_dispatch_counts(events)


def test_dispatch_probe_counts_match_pre_refactor_oracle() -> None:
    """X8: the probe_unit-backed gate yields the same integer counts as the old
    direct-trace implementation on the same fixture kernel."""
    from xtrax.devtools.gates._dispatch_probe import measure_dispatch_counts

    refactored = measure_dispatch_counts(_KERNEL_QUALNAME, _KERNEL_PROBE)
    legacy = _legacy_dispatch_counts(_KERNEL_QUALNAME, _KERNEL_PROBE)

    assert refactored.skipped is False
    assert refactored.passed is True
    assert refactored.counts == legacy
    assert set(refactored.counts) == {"n_executions", "n_compilations", "n_jit_traces"}
    assert refactored.counts["n_compilations"] == 0
    assert refactored.counts["n_executions"] >= 1


def test_dispatch_probe_skips_without_trace_probe() -> None:
    from xtrax.devtools.gates._dispatch_probe import measure_dispatch_counts

    result = measure_dispatch_counts(_KERNEL_QUALNAME, None)
    assert result.skipped is True
    assert result.passed is False
    assert result.counts == {}
    assert result.reason == "no trace_probe configured for kernel"


# --- model_jit kernel under the trace-count gate (equinox decoupling, step 3) ----------------


def test_model_jit_kernel_passes_and_a_retracing_probe_fails(tmp_path, monkeypatch) -> None:
    """The kernel is a ``model_jit`` function over a dict MLP; a probe that changes a static
    leaf on every call retraces it, and the gate must report that."""
    (tmp_path / "retrace_probe_fixture.py").write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def probe(guarded):
                base = {
                    "w1": jnp.ones((4, 4)),
                    "b1": jnp.zeros((4,)),
                    "w2": jnp.ones((4, 2)),
                }
                x = jnp.ones((4,))
                for act in ("tanh", "relu", "gelu"):  # a new static leaf each call
                    guarded({**base, "act": act}, x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    stable = run_trace_probe(_KERNEL_QUALNAME, max_traces=1, trace_probe=_KERNEL_PROBE)
    assert stable.passed is True and not stable.skipped

    retracing = run_trace_probe(
        _KERNEL_QUALNAME, max_traces=1, trace_probe="retrace_probe_fixture:probe"
    )
    assert retracing.passed is False
    assert "traced > 1 times" in retracing.reason


def test_model_jit_kernel_is_not_wrapped_in_a_second_jit() -> None:
    """The gate leaves a ``model_jit`` target alone (a plain jax.jit would reject its str leaf)."""
    from xtrax.devtools.gates._trace_probe import _guard_for_trace_count, import_qualname

    target = import_qualname(_KERNEL_QUALNAME)
    assert _guard_for_trace_count(target, max_traces=1) is target

    def plain(x):
        return x + 1

    assert _guard_for_trace_count(plain, max_traces=1) is not plain


# --- X5: per-kernel max_executions ceiling ------------------------------------


def _write_multicall_fixture(tmp_path: Path, name: str, calls: int) -> str:
    """A kernel plus a probe that calls it ``calls`` times in one unit.

    Returns the ``module:probe`` trace_probe string.
    """
    import sys

    module_path = tmp_path / f"{name}.py"
    module_path.write_text(
        textwrap.dedent(
            f"""
            import jax.numpy as jnp

            def kernel(x):
                return x + 1

            def probe(guarded):
                x = jnp.zeros(3)
                for _ in range({calls}):
                    guarded(x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    return f"{name}:probe"


def _executions_targets(tmp_path: Path, qualname: str, probe: str, ceiling: str) -> Path:
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "{qualname}"
            max_traces = 1
            trace_probe = "{probe}"
            max_executions = {ceiling}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return targets_path


def _seed_dispatch_baseline(path: Path) -> None:
    from xtrax.devtools.gates.performance import DISPATCH_METRIC_KEY

    save_baseline(
        AuditBaseline(
            schema_version=BASELINE_SCHEMA_VERSION,
            updated_at="2026-10-16T00:00:00+00:00",
            metrics={
                DISPATCH_METRIC_KEY: MetricEntry(
                    key=DISPATCH_METRIC_KEY, value=0.0, comparator="minimize"
                ),
            },
        ),
        path=path,
    )


def test_load_performance_targets_parses_max_executions(tmp_path: Path) -> None:
    targets_path = _executions_targets(tmp_path, "some.kernel", "some.mod:probe", "3")
    spec = load_performance_targets(targets_path).probes[0]
    assert spec.max_executions == 3
    assert spec.max_compilations is None
    assert spec.max_jit_traces is None
    assert spec.has_dispatch_ceilings is True


def test_absent_max_executions_is_not_checked(tmp_path: Path) -> None:
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            """
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"

            [[probes]]
            qualname = "some.kernel"
            max_traces = 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = load_performance_targets(targets_path).probes[0]
    assert spec.max_executions is None
    assert spec.has_dispatch_ceilings is False


def test_boolean_toml_max_executions_rejected(tmp_path: Path) -> None:
    """X5: TOML `max_executions = true` must not become ceiling 1."""
    targets_path = _executions_targets(tmp_path, "some.kernel", "some.mod:probe", "true")
    with pytest.raises(ValueError, match="max_executions"):
        load_performance_targets(targets_path)


def test_max_executions_ceiling_below_measured_count_fails(tmp_path: Path) -> None:
    """Four guarded calls per unit: measured n_executions >= 2, so ceiling 1 must fail."""
    from xtrax.devtools.gates._dispatch_probe import measure_dispatch_counts

    qualname = "executions_fixture_low.kernel"
    probe = _write_multicall_fixture(tmp_path, "executions_fixture_low", calls=4)
    measured = measure_dispatch_counts(qualname, probe).counts["n_executions"]
    assert measured >= 2, "fixture must execute more than once per unit"

    audits_path = tmp_path / "audits.jsonl"
    baseline_path = tmp_path / "audit_baseline.json"
    _seed_dispatch_baseline(baseline_path)
    targets_path = _executions_targets(tmp_path, qualname, probe, "1")

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is False
    assert result.dispatch_violation_count == 1
    entries = [json.loads(line) for line in audits_path.read_text(encoding="utf-8").splitlines()]
    dispatch = [e for e in entries if e["payload"].get("violation_kind") == "dispatch_count"]
    assert len(dispatch) == 1
    assert dispatch[0]["payload"]["counter"] == "n_executions"
    assert dispatch[0]["payload"]["ceiling"] == 1
    assert dispatch[0]["payload"]["observed"] == measured


def test_max_executions_ceiling_above_measured_count_passes(tmp_path: Path) -> None:
    qualname = "executions_fixture_high.kernel"
    probe = _write_multicall_fixture(tmp_path, "executions_fixture_high", calls=4)

    audits_path = tmp_path / "audits.jsonl"
    baseline_path = tmp_path / "audit_baseline.json"
    _seed_dispatch_baseline(baseline_path)
    targets_path = _executions_targets(tmp_path, qualname, probe, "64")

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is True
    assert result.dispatch_violation_count == 0

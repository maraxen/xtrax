"""Tests for the unit-probe measurement core (spec 261016 X1).

Covers the pure interval-union parser in ``xtrax.profiling.trace`` and the
``probe_unit`` procedure: a host-loop vs vmap negative control, recompile
detection, the accounting invariants of the device/host split, validation,
and import hygiene (no jax at module scope).
"""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax.profiling.trace import interval_union_seconds
from xtrax.profiling.unit_probe import UnitProbeResult, probe_unit

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _op(ts: float, dur: float, name: str = "op") -> dict:
    return {"ph": "X", "name": name, "ts": ts, "dur": dur, "args": {"hlo_op": name}}


# --- interval_union_seconds --------------------------------------------------


def test_union_disjoint_intervals_sum():
    events = [_op(0, 1_000_000), _op(2_000_000, 500_000)]
    assert interval_union_seconds(events) == pytest.approx(1.5)


def test_union_overlapping_intervals_counted_once():
    # [0, 2s] and [1s, 3s] cover 3 s, not 4 s.
    events = [_op(0, 2_000_000), _op(1_000_000, 2_000_000)]
    assert interval_union_seconds(events) == pytest.approx(3.0)


def test_union_nested_interval_counted_once():
    # [1s, 2s] lies inside [0, 4s]; the union is 4 s.
    events = [_op(0, 4_000_000), _op(1_000_000, 1_000_000)]
    assert interval_union_seconds(events) == pytest.approx(4.0)


def test_union_ignores_events_without_hlo_op():
    events = [
        _op(0, 1_000_000),
        {"ph": "X", "name": "python", "ts": 0, "dur": 9_000_000, "args": {}},
        {"ph": "X", "name": "bare", "ts": 0, "dur": 9_000_000},
    ]
    assert interval_union_seconds(events) == pytest.approx(1.0)


def test_union_ignores_non_complete_phases():
    events = [
        {"ph": "B", "ts": 0, "dur": 5_000_000, "args": {"hlo_op": "x"}},
        {"ph": "i", "ts": 0, "dur": 5_000_000, "args": {"hlo_op": "x"}},
        _op(0, 1_000_000),
    ]
    assert interval_union_seconds(events) == pytest.approx(1.0)


def test_union_ignores_events_without_numeric_ts_or_dur():
    events = [
        {"ph": "X", "ts": None, "dur": 1_000, "args": {"hlo_op": "a"}},
        {"ph": "X", "ts": 0, "args": {"hlo_op": "b"}},
        {"ph": "X", "ts": "0", "dur": 1_000, "args": {"hlo_op": "c"}},
        _op(0, 2_000_000),
    ]
    assert interval_union_seconds(events) == pytest.approx(2.0)


def test_union_empty_is_zero():
    assert interval_union_seconds([]) == 0.0


# --- probe_unit: validation --------------------------------------------------


def test_n_steady_zero_raises():
    with pytest.raises(ValueError):
        probe_unit(lambda x: x, (1.0,), n_steady=0)


def test_n_wall_zero_raises():
    with pytest.raises(ValueError):
        probe_unit(lambda x: x, (1.0,), n_wall=0)


def test_executions_per_element_zero_cardinality_raises():
    result = UnitProbeResult(
        n_steady=1,
        warmup_compile_count=0,
        warmup_compile_seconds=0.0,
        steady_compile_count=0,
        executions_per_unit=4.0,
        jit_traces_per_unit=0.0,
        wall_seconds_per_unit=0.0,
        traced_seconds_per_unit=0.0,
        device_seconds_per_unit=0.0,
        host_seconds_per_unit=0.0,
    )
    with pytest.raises(ValueError):
        result.executions_per_element(0)
    assert result.executions_per_element(4) == pytest.approx(1.0)


# --- probe_unit: negative control (host loop vs vmap) -------------------------


def _twin_functions(batch: int, feat: int):
    import jax
    import jax.numpy as jnp

    w = jnp.eye(feat) * 0.5

    def _unit(xb):
        h = xb
        for _ in range(4):
            h = jnp.tanh(h @ w)
        return h.sum()

    unit = jax.jit(_unit)
    vmapped = jax.jit(jax.vmap(_unit))

    def looped(x):
        return jnp.stack([unit(x[b]) for b in range(batch)])

    return looped, vmapped


def test_negative_control_host_loop_fails_and_vmap_passes():
    import jax

    batch, feat = 16, 32
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, feat, feat))
    looped_fn, vmapped_fn = _twin_functions(batch, feat)

    looped = probe_unit(looped_fn, (x,), n_steady=2, n_wall=3)
    vmapped = probe_unit(vmapped_fn, (x,), n_steady=2, n_wall=3)

    assert looped.executions_per_element(batch) >= 2.0
    assert vmapped.executions_per_element(batch) <= 0.5
    assert looped.executions_per_unit >= 10 * vmapped.executions_per_unit
    assert looped.steady_compile_count == 0
    assert vmapped.steady_compile_count == 0


# --- probe_unit: recompile detection ------------------------------------------


def test_fresh_jit_per_call_is_flagged_as_steady_recompile():
    import jax

    n_steady = 3

    def fresh(x):
        return jax.jit(lambda y: y + 1)(x)

    result = probe_unit(fresh, (1.0,), n_steady=n_steady, n_wall=2)
    assert result.steady_compile_count >= n_steady


# --- probe_unit: accounting invariants ----------------------------------------


def _check_invariants(result: UnitProbeResult) -> None:
    assert result.device_seconds_per_unit <= result.traced_seconds_per_unit + 1e-6
    assert result.host_seconds_per_unit >= 0.0
    if result.traced_seconds_per_unit >= result.device_seconds_per_unit:
        total = result.device_seconds_per_unit + result.host_seconds_per_unit
        assert abs(total - result.traced_seconds_per_unit) < 1e-6
    metrics = result.as_metrics()
    expected_keys = {
        "n_executions",
        "n_compilations",
        "n_jit_traces",
        "total_step_seconds",
        "compile_seconds",
        "device_seconds",
        "host_seconds",
        "wall_mad_seconds",
        "wall_iqr_seconds",
    }
    if result.peak_vram_bytes is not None:
        expected_keys.add("peak_vram_bytes")
    assert set(metrics) == expected_keys
    for value in metrics.values():
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_invariants_hold_for_host_loop_and_vmap():
    import jax

    batch, feat = 16, 32
    x = jax.random.normal(jax.random.PRNGKey(1), (batch, feat, feat))
    looped_fn, vmapped_fn = _twin_functions(batch, feat)
    _check_invariants(probe_unit(looped_fn, (x,), n_steady=2, n_wall=3))
    _check_invariants(probe_unit(vmapped_fn, (x,), n_steady=2, n_wall=3))


def test_invariants_hold_for_recompiling_callable():
    import jax

    def fresh(x):
        return jax.jit(lambda y: y * 2)(x)

    _check_invariants(probe_unit(fresh, (1.0,), n_steady=2, n_wall=2))


# --- import hygiene -----------------------------------------------------------


def test_importing_unit_probe_does_not_import_jax():
    code = (
        "import sys; import xtrax.profiling.unit_probe; "
        "assert 'jax' not in sys.modules, 'jax imported at module scope'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC_DIR), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# --- probe_unit: raw dispatch_counts field (X8) -------------------------------


def test_dispatch_counts_field_matches_per_unit_and_defaults_empty():
    import jax
    import jax.numpy as jnp

    n_steady = 2
    unit = jax.jit(lambda x: jnp.sin(x) + 1.0)
    result = probe_unit(unit, (jnp.ones(8),), n_steady=n_steady, n_wall=1)

    assert set(result.dispatch_counts) == {"n_executions", "n_compilations", "n_jit_traces"}
    assert all(isinstance(v, int) for v in result.dispatch_counts.values())
    # Per-unit values are the window totals divided by n_steady.
    assert result.executions_per_unit == pytest.approx(
        result.dispatch_counts["n_executions"] / n_steady
    )
    assert result.jit_traces_per_unit == pytest.approx(
        result.dispatch_counts["n_jit_traces"] / n_steady
    )

    # Backward compatible: direct construction without the field still works.
    bare = UnitProbeResult(
        n_steady=1,
        warmup_compile_count=0,
        warmup_compile_seconds=0.0,
        steady_compile_count=0,
        executions_per_unit=1.0,
        jit_traces_per_unit=0.0,
        wall_seconds_per_unit=0.0,
        traced_seconds_per_unit=0.0,
        device_seconds_per_unit=0.0,
        host_seconds_per_unit=0.0,
    )
    assert bare.dispatch_counts == {}
    assert bare.wall_mad_seconds == 0.0
    assert bare.wall_iqr_seconds == 0.0
    assert bare.peak_vram_bytes is None


def test_wall_dispersion_stats_and_iqr_threshold():
    import statistics

    from xtrax.profiling.unit_probe import _compute_wall_stats

    med1, mad1, iqr1 = _compute_wall_stats([0.5])
    assert med1 == pytest.approx(0.5)
    assert mad1 == 0.0
    assert iqr1 == 0.0

    med2, mad2, iqr2 = _compute_wall_stats([1.0, 3.0])
    assert med2 == pytest.approx(2.0)
    assert mad2 == pytest.approx(1.0)
    assert iqr2 == 0.0

    med3, mad3, iqr3 = _compute_wall_stats([1.0, 2.0, 5.0])
    assert med3 == pytest.approx(2.0)
    assert mad3 == pytest.approx(1.0)
    assert iqr3 == 0.0  # must NOT substitute max-min range when n_wall < 4

    walls5 = [1.0, 2.0, 3.0, 4.0, 10.0]
    med5, mad5, iqr5 = _compute_wall_stats(walls5)
    q1, _, q3 = statistics.quantiles(walls5, n=4)
    assert med5 == pytest.approx(3.0)
    assert mad5 == pytest.approx(1.0)
    assert iqr5 == pytest.approx(q3 - q1)


def test_peak_vram_bytes_capture_and_failure_containment():
    import jax

    from xtrax.profiling.unit_probe import _capture_peak_vram_bytes

    class _FakeGpu:
        platform = "gpu"

        def __init__(self, peak: int | None, *, boom: bool = False) -> None:
            self._peak = peak
            self._boom = boom

        def memory_stats(self):
            if self._boom:
                raise RuntimeError("stats failed")
            return {"peak_bytes_in_use": self._peak}

    class _Leaf:
        def __init__(self, dev):
            self._dev = dev

        def devices(self):
            return {self._dev}

    peak = _capture_peak_vram_bytes(_Leaf(_FakeGpu(123456)), (), {}, jax)
    assert peak == 123456

    assert _capture_peak_vram_bytes(_Leaf(_FakeGpu(0, boom=True)), (), {}, jax) is None

    res = UnitProbeResult(
        n_steady=1,
        warmup_compile_count=0,
        warmup_compile_seconds=0.0,
        steady_compile_count=0,
        executions_per_unit=1.0,
        jit_traces_per_unit=0.0,
        wall_seconds_per_unit=0.1,
        traced_seconds_per_unit=0.1,
        device_seconds_per_unit=0.05,
        host_seconds_per_unit=0.05,
        peak_vram_bytes=65536,
    )
    m = res.as_metrics()
    assert m["peak_vram_bytes"] == 65536.0
    assert isinstance(m["peak_vram_bytes"], float)


def test_devices_and_peak_vram_edge_cases_and_trace_dir(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp

    from xtrax.profiling.unit_probe import (
        _capture_peak_vram_bytes,
        _devices_from_leaf,
        _read_gpu_peak_bytes,
    )

    class _FakeGpu:
        platform = "gpu"

        def __init__(self, stats):
            self._stats = stats

        def memory_stats(self):
            return self._stats

    class _LeafDevsRaisesThenCallableDev:
        def devices(self):
            raise RuntimeError("no devices()")

        def device(self):
            return _FakeGpu({"peak_bytes_in_use": 4096.0})

    class _LeafCallableDevRaises:
        def device(self):
            raise RuntimeError("no device()")

    class _LeafAttrDev:
        device = _FakeGpu({"peak_bytes_in_use": 8192})

    assert len(_devices_from_leaf(_LeafDevsRaisesThenCallableDev())) == 1
    assert _devices_from_leaf(_LeafCallableDevRaises()) == []
    assert len(_devices_from_leaf(_LeafAttrDev())) == 1

    assert _read_gpu_peak_bytes(_FakeGpu(None)) is None
    assert _read_gpu_peak_bytes(_FakeGpu({"peak_bytes_in_use": True})) is None
    assert _read_gpu_peak_bytes(_FakeGpu({"peak_bytes_in_use": -5})) is None
    assert _read_gpu_peak_bytes(_FakeGpu({"peak_bytes_in_use": 2048})) == 2048

    # Falls back from empty output to args/kwargs GPU device.
    assert _capture_peak_vram_bytes(None, (_LeafAttrDev(),), {}, jax) == 8192

    # Falls back to jax.devices() when neither out nor args carry GPU leaves, and survives error.
    monkeypatch.setattr(jax, "devices", lambda: [_FakeGpu({"peak_bytes_in_use": 1024})])
    assert _capture_peak_vram_bytes(None, (), {}, jax) == 1024

    def _boom_devices():
        raise RuntimeError("devices failed")

    monkeypatch.setattr(jax, "devices", _boom_devices)
    assert _capture_peak_vram_bytes(None, (), {}, jax) is None

    res = probe_unit(
        lambda x: x + 1.0,
        (jnp.zeros((4,)),),
        n_steady=1,
        n_wall=1,
        trace_dir=tmp_path,
    )
    assert res.n_steady == 1
    assert any(p.name.startswith("unit_probe_") for p in tmp_path.iterdir())

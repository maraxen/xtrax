"""Measure one unit of work at its real shape: recompiles, dispatches, time split.

A **unit** is one call of a callable at production shape: one bootstrap
replicate, one sampler step, one chunk, one element of a loop. ``probe_unit``
runs the unit in three phases and reports what each phase shows:

1. Warm-up: one call under ``count_backend_compiles``. Reports the compiles
   and compile seconds a first call pays (``warmup_compile_*``).
2. Untraced wall: ``n_wall`` calls, each timed around call plus
   ``block_until_ready``. ``wall_seconds_per_unit`` is the median.
3. Steady traced: ``n_steady`` calls under ``jax.profiler.trace`` and under
   ``count_backend_compiles``. Everything derived from the trace comes from
   this window.

What the numbers mean, all per unit (steady calls divided by ``n_steady``):

- ``executions_per_unit``: XLA executable invocations (``n_executions`` in
  the trace). A host loop over B elements gives roughly 3 per element; a
  ``vmap`` gives 1 for the whole batch. This is the dispatch-count
  discriminator: steady-state compiles are zero for both.
- ``steady_compile_count``: backend compiles during the steady window. A
  non-zero value means the unit recompiles each call (for example a fresh
  ``jax.jit`` built inside the callable).
- ``jit_traces_per_unit``: Python-level jit trace events (relative only).
- ``device_seconds_per_unit``: the union of ``hlo_op`` device intervals,
  not their sum, since concurrent CPU threads overlap.
- ``host_seconds_per_unit``: the traced window minus device time, floored at
  zero. The split is two-way: device versus host plus Python. The trace does
  not separate Python from C++ dispatch reliably, so they are not split.

CPU timings are CPU-relative only: they rank and compare units on the same
machine and do not predict accelerator wall time. On GPU, ``peak_vram_bytes``
reads XLA's process-lifetime ``peak_bytes_in_use`` high-water mark, so exact
per-unit VRAM isolation requires subprocess execution (see
``xtrax.profiling.kernel_probe.probe_kernel_dual_pass``).

jax is imported lazily inside the functions, so importing this module stays
free of jax (the ``xtrax.profiling`` leaf contract).
"""

import statistics
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xtrax.profiling.compile_count import count_backend_compiles
from xtrax.profiling.trace import (
    interval_union_seconds,
    load_trace_events,
    parse_dispatch_counts,
)


@dataclass(frozen=True, slots=True)
class UnitProbeResult:
    """Per-unit measurements from :func:`probe_unit`.

    Attributes:
        n_steady: Number of steady traced calls the per-unit values divide by.
        warmup_compile_count: Backend compiles during the warm-up call.
        warmup_compile_seconds: Summed compile duration during the warm-up call.
        steady_compile_count: Backend compiles during the steady traced window
            (total, not per unit).
        executions_per_unit: XLA executable invocations per unit.
        jit_traces_per_unit: Python-level jit trace events per unit.
        wall_seconds_per_unit: Median untraced wall time of one unit.
        traced_seconds_per_unit: Traced window divided by ``n_steady``.
        device_seconds_per_unit: Device interval union divided by ``n_steady``.
        host_seconds_per_unit: ``max(traced - device, 0)`` per unit.
        dispatch_counts: Raw totals from ``parse_dispatch_counts`` over the
            steady traced window (keys ``n_executions``, ``n_compilations``,
            ``n_jit_traces``). These are window totals, not per unit, and
            ``n_compilations`` here counts trace events, which is a different
            source from ``steady_compile_count``. Empty when constructed
            directly.
        wall_mad_seconds: Median absolute deviation of untraced wall timings
            (``0.0`` when ``n_wall < 2``).
        wall_iqr_seconds: Interquartile range (``q3 - q1``) of untraced wall
            timings via ``statistics.quantiles(..., n=4)`` (``0.0`` when
            ``n_wall < 4``).
        peak_vram_bytes: Maximum GPU ``peak_bytes_in_use`` reported by device
            ``memory_stats()``, or ``None`` on CPU / unavailable stats. Note
            that XLA's ``peak_bytes_in_use`` is a process-lifetime high-water
            mark; exact per-unit isolation requires subprocess execution.
    """

    n_steady: int
    warmup_compile_count: int
    warmup_compile_seconds: float
    steady_compile_count: int
    executions_per_unit: float
    jit_traces_per_unit: float
    wall_seconds_per_unit: float
    traced_seconds_per_unit: float
    device_seconds_per_unit: float
    host_seconds_per_unit: float
    dispatch_counts: dict[str, int] = field(default_factory=dict)
    wall_mad_seconds: float = 0.0
    wall_iqr_seconds: float = 0.0
    peak_vram_bytes: int | None = None

    def executions_per_element(self, cardinality: int) -> float:
        """Executions per unit divided by the scale-axis cardinality.

        Args:
            cardinality: Number of elements the unit processes (for example B
                for a batch of B). Must be at least 1.

        Raises:
            ValueError: ``cardinality`` is below 1.
        """
        if cardinality < 1:
            raise ValueError(f"cardinality must be >= 1, got {cardinality}")
        return self.executions_per_unit / cardinality

    def as_metrics(self) -> dict[str, float]:
        """Finite float metrics for ``ProbeRecord.metrics``, all per unit.

        Keys: ``n_executions``, ``n_compilations`` (steady compiles per unit),
        ``n_jit_traces``, ``total_step_seconds`` (untraced median wall),
        ``compile_seconds`` (warm-up compile seconds), ``device_seconds``,
        ``host_seconds``, ``wall_mad_seconds``, ``wall_iqr_seconds``, and
        ``peak_vram_bytes`` (omitted when ``peak_vram_bytes`` is ``None``).
        """
        metrics: dict[str, float] = {
            "n_executions": float(self.executions_per_unit),
            "n_compilations": float(self.steady_compile_count) / self.n_steady,
            "n_jit_traces": float(self.jit_traces_per_unit),
            "total_step_seconds": float(self.wall_seconds_per_unit),
            "compile_seconds": float(self.warmup_compile_seconds),
            "device_seconds": float(self.device_seconds_per_unit),
            "host_seconds": float(self.host_seconds_per_unit),
            "wall_mad_seconds": float(self.wall_mad_seconds),
            "wall_iqr_seconds": float(self.wall_iqr_seconds),
        }
        if self.peak_vram_bytes is not None:
            metrics["peak_vram_bytes"] = float(self.peak_vram_bytes)
        return metrics


def _compute_wall_stats(walls: list[float]) -> tuple[float, float, float]:
    """Return ``(median, mad, iqr)`` seconds over untraced wall timings."""
    wall_median = float(statistics.median(walls))
    wall_mad = (
        float(statistics.median([abs(w - wall_median) for w in walls])) if len(walls) >= 2 else 0.0
    )
    if len(walls) >= 4:
        q1, _, q3 = statistics.quantiles(walls, n=4)
        wall_iqr = max(0.0, float(q3 - q1))
    else:
        wall_iqr = 0.0
    return wall_median, wall_mad, wall_iqr


def _devices_from_leaf(leaf: Any) -> list[Any]:
    """Extract attached JAX device handles from a single pytree leaf."""
    devs_fn = getattr(leaf, "devices", None)
    if callable(devs_fn):
        try:
            leaf_devs = devs_fn()
        except Exception:
            leaf_devs = None
        if leaf_devs is not None:
            return list(leaf_devs)
    dev_attr = getattr(leaf, "device", None)
    if callable(dev_attr):
        try:
            dev = dev_attr()
        except Exception:
            dev = None
        return [dev] if dev is not None else []
    return [dev_attr] if dev_attr is not None else []


def _extract_leaf_devices(tree: Any, jax_mod: Any) -> list[Any]:
    """Collect devices attached to leaves of ``tree``."""
    found: list[Any] = []
    for leaf in jax_mod.tree_util.tree_leaves(tree):
        found.extend(_devices_from_leaf(leaf))
    return found


def _read_gpu_peak_bytes(dev: Any) -> int | None:
    """Return ``peak_bytes_in_use`` from a GPU device handle, or ``None``."""
    try:
        stats = dev.memory_stats()
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    raw_peak = stats.get("peak_bytes_in_use")
    if not isinstance(raw_peak, bool) and isinstance(raw_peak, int | float) and raw_peak >= 0:
        return int(raw_peak)
    return None


def _capture_peak_vram_bytes(
    out: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    jax_mod: Any,
) -> int | None:
    """Query GPU ``peak_bytes_in_use`` across execution devices.

    Note that XLA ``peak_bytes_in_use`` reflects the process-lifetime device
    high-water mark rather than a single isolated kernel invocation.
    """
    candidates = _extract_leaf_devices(out, jax_mod)
    if not any(getattr(dev, "platform", None) == "gpu" for dev in candidates):
        candidates = _extract_leaf_devices((args, kwargs), jax_mod)
    if not any(getattr(dev, "platform", None) == "gpu" for dev in candidates):
        try:
            candidates = list(jax_mod.devices()[:1])
        except Exception:
            candidates = []

    seen: set[int] = set()
    peaks: list[int] = []
    for dev in candidates:
        if getattr(dev, "platform", None) != "gpu" or id(dev) in seen:
            continue
        seen.add(id(dev))
        peak = _read_gpu_peak_bytes(dev)
        if peak is not None:
            peaks.append(peak)
    return max(peaks) if peaks else None


def probe_unit(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    n_steady: int = 3,
    n_wall: int = 5,
    trace_dir: str | Path | None = None,
) -> UnitProbeResult:
    """Run ``fn(*args, **kwargs)`` as a unit and measure it.

    Args:
        fn: Callable for one unit. Its output is passed through
            ``jax.block_until_ready`` as a whole pytree after every call.
        args: Positional arguments for one call.
        kwargs: Keyword arguments for one call.
        n_steady: Traced steady calls (at least 1).
        n_wall: Untraced timed calls for the wall median (at least 1).
        trace_dir: Parent directory for the trace. A fresh subdirectory is
            created inside it, so earlier captures are never re-read. Default:
            a temporary directory removed on return.

    Returns:
        A :class:`UnitProbeResult`.

    Raises:
        ValueError: ``n_steady`` or ``n_wall`` is below 1.
    """
    if n_steady < 1:
        raise ValueError(f"n_steady must be >= 1, got {n_steady}")
    if n_wall < 1:
        raise ValueError(f"n_wall must be >= 1, got {n_wall}")

    import jax

    call_kwargs = dict(kwargs or {})

    def call() -> Any:
        return jax.block_until_ready(fn(*args, **call_kwargs))

    with count_backend_compiles() as warmup:
        call()

    walls: list[float] = []
    last_out: Any = None
    for _ in range(n_wall):
        t0 = time.perf_counter()
        last_out = call()
        walls.append(time.perf_counter() - t0)
    wall_median, wall_mad, wall_iqr = _compute_wall_stats(walls)
    peak_vram_bytes = _capture_peak_vram_bytes(last_out, args, call_kwargs, jax)

    trace_kwargs: dict[str, Any] = {"create_perfetto_trace": True}
    if hasattr(jax.profiler, "ProfileOptions"):
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        trace_kwargs["profiler_options"] = options

    with ExitStack() as stack:
        if trace_dir is None:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            parent = Path(trace_dir)
            parent.mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix="unit_probe_", dir=parent))
        with count_backend_compiles() as steady:
            with jax.profiler.trace(str(root), **trace_kwargs):
                t0 = time.perf_counter()
                for _ in range(n_steady):
                    call()
                traced_window = time.perf_counter() - t0
        events = load_trace_events(root)

    counts = parse_dispatch_counts(events)
    device_window = interval_union_seconds(events)

    traced_per_unit = traced_window / n_steady
    device_per_unit = device_window / n_steady
    host_per_unit = max(traced_per_unit - device_per_unit, 0.0)

    return UnitProbeResult(
        n_steady=n_steady,
        warmup_compile_count=warmup.count,
        warmup_compile_seconds=warmup.seconds,
        steady_compile_count=steady.count,
        executions_per_unit=counts["n_executions"] / n_steady,
        jit_traces_per_unit=counts["n_jit_traces"] / n_steady,
        wall_seconds_per_unit=wall_median,
        traced_seconds_per_unit=traced_per_unit,
        device_seconds_per_unit=device_per_unit,
        host_seconds_per_unit=host_per_unit,
        dispatch_counts=dict(counts),
        wall_mad_seconds=wall_mad,
        wall_iqr_seconds=wall_iqr,
        peak_vram_bytes=peak_vram_bytes,
    )

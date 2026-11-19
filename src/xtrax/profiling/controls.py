"""Shipped negative controls for the unit probe (spec 261016 X7).

The probe must discriminate a host loop from a batched unit on the platform
it runs on, because JAX profiler event names change across versions. This
module ships the spike-B workload in two forms:

- ``host_loop(x)``: a Python loop over the leading axis, one jitted step per
  row. Several XLA executions per element.
- ``vmapped_twin(x)``: one jitted ``vmap`` over the batch. One execution for
  the whole batch.

The workload is ``4 x tanh(h @ W)`` on ``(B, 128)`` float32 rows, with a
fixed ``W`` (128 x 128).

:func:`run_negative_controls` runs :func:`~xtrax.profiling.unit_probe.probe_unit`
on both at each batch size. A control *holds* when, at every batch size:

- host_loop executions per element is at least ``threshold``;
- vmapped_twin executions per element is below ``threshold``;
- vmapped_twin has zero steady-state compiles.

The smallest batch size is the sensitivity check: the host loop must still be
flagged there.

This module is not re-exported from ``xtrax.profiling``: it imports jax at
module scope, and the package docstring promises a jax-free import. Import it
as ``xtrax.profiling.controls``.

CPU timings are relative only. The dispatch ratio is scale-free, which is why
it is the discriminator.
"""

import functools
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from xtrax.profiling.rules import EXEC_PER_ELEMENT_FLAG
from xtrax.profiling.unit_probe import probe_unit

FEATURES = 128
N_LAYERS = 4
HOST_LOOP = "host_loop"
VMAPPED_TWIN = "vmapped_twin"


@dataclass(frozen=True, slots=True)
class ControlMeasurement:
    """Probe numbers for one (variant, batch size) pair.

    Attributes:
        variant: ``"host_loop"`` or ``"vmapped_twin"``.
        batch_size: B, the scale-axis cardinality.
        executions_per_unit: XLA executions per call of the unit.
        exec_per_element: ``executions_per_unit / batch_size``.
        steady_compile_count: Backend compiles in the steady window (total).
        wall_seconds_per_unit: Median untraced wall time of one call (CPU
            relative).
    """

    variant: str
    batch_size: int
    executions_per_unit: float
    exec_per_element: float
    steady_compile_count: int
    wall_seconds_per_unit: float


@dataclass(frozen=True, slots=True)
class ControlsReport:
    """Outcome of :func:`run_negative_controls`.

    Attributes:
        threshold: The exec/element threshold the controls were judged against.
        batch_sizes: Batch sizes measured, in the order given.
        measurements: One entry per (variant, batch size), host_loop first at
            each batch size.
        held: True when every check passed at every batch size.
        reasons: Human-readable reason for each failed check. Empty when held.
    """

    threshold: float
    batch_sizes: tuple[int, ...]
    measurements: tuple[ControlMeasurement, ...]
    held: bool
    reasons: tuple[str, ...]

    def measurement(self, variant: str, batch_size: int) -> ControlMeasurement:
        """Return the measurement for ``variant`` at ``batch_size``.

        Raises:
            KeyError: No such (variant, batch size) pair was measured.
        """
        for m in self.measurements:
            if m.variant == variant and m.batch_size == batch_size:
                return m
        raise KeyError((variant, batch_size))


@functools.lru_cache(maxsize=1)
def _kernels() -> tuple[Any, Any, Any]:
    """Build the jitted row step, the jitted batched step and W, once."""
    key = jax.random.PRNGKey(0)
    weights = jax.random.normal(key, (FEATURES, FEATURES), jnp.float32)
    weights = weights / math.sqrt(FEATURES)

    def workload(row, w):
        h = row
        for _ in range(N_LAYERS):
            h = jnp.tanh(h @ w)
        return h

    row_step = jax.jit(workload)
    batched_step = jax.jit(jax.vmap(workload, in_axes=(0, None)))
    return row_step, batched_step, weights


def host_loop(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Host loop over the leading axis: one jitted step per row."""
    row_step, _, w = _kernels()
    return jnp.stack([row_step(x[b], w) for b in range(x.shape[0])])


def vmapped_twin(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Batched twin of :func:`host_loop`: one jitted vmap for the batch."""
    _, batched_step, w = _kernels()
    return batched_step(x, w)


def _inputs(batch_size: int) -> jax.Array:
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (batch_size, FEATURES), jnp.float32)


def run_negative_controls(
    batch_sizes: tuple[int, ...] = (8, 64),
    threshold: float = EXEC_PER_ELEMENT_FLAG,
    n_steady: int = 2,
    n_wall: int = 2,
) -> ControlsReport:
    """Run the host-loop and vmapped-twin controls at each batch size.

    Args:
        batch_sizes: Batch sizes B to measure. Each must be at least 1. The
            smallest is the sensitivity check.
        threshold: Executions-per-element threshold. Host loops must be at or
            above it, the vmapped twin below it. Defaults to
            ``rules.EXEC_PER_ELEMENT_FLAG``, the same value the verb uses.
        n_steady: Traced steady calls per measurement (passed to ``probe_unit``).
        n_wall: Untraced wall calls per measurement (passed to ``probe_unit``).

    Returns:
        A :class:`ControlsReport`. Nothing is printed.

    Raises:
        ValueError: ``batch_sizes`` is empty or contains a value below 1.
    """
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    if any(b < 1 for b in batch_sizes):
        raise ValueError(f"every batch size must be >= 1, got {batch_sizes}")

    measurements: list[ControlMeasurement] = []
    reasons: list[str] = []
    smallest = min(batch_sizes)

    for batch_size in batch_sizes:
        x = _inputs(batch_size)
        per_variant: dict[str, ControlMeasurement] = {}
        for variant, fn in ((HOST_LOOP, host_loop), (VMAPPED_TWIN, vmapped_twin)):
            result = probe_unit(fn, (x,), n_steady=n_steady, n_wall=n_wall)
            m = ControlMeasurement(
                variant=variant,
                batch_size=batch_size,
                executions_per_unit=float(result.executions_per_unit),
                exec_per_element=result.executions_per_element(batch_size),
                steady_compile_count=int(result.steady_compile_count),
                wall_seconds_per_unit=float(result.wall_seconds_per_unit),
            )
            measurements.append(m)
            per_variant[variant] = m

        host = per_variant[HOST_LOOP]
        twin = per_variant[VMAPPED_TWIN]
        sensitivity = " (sensitivity check)" if batch_size == smallest else ""

        if host.exec_per_element < threshold:
            reasons.append(
                f"host_loop exec/element {host.exec_per_element:.3f} < threshold "
                f"{threshold} at B={batch_size}{sensitivity}: the probe does not "
                "flag a host loop"
            )
        if twin.exec_per_element >= threshold:
            reasons.append(
                f"vmapped_twin exec/element {twin.exec_per_element:.3f} >= threshold "
                f"{threshold} at B={batch_size}: the probe flags a batched unit"
            )
        if twin.steady_compile_count != 0:
            reasons.append(
                f"vmapped_twin has {twin.steady_compile_count} steady compile(s) "
                f"at B={batch_size}, expected 0"
            )

    return ControlsReport(
        threshold=float(threshold),
        batch_sizes=tuple(batch_sizes),
        measurements=tuple(measurements),
        held=not reasons,
        reasons=tuple(reasons),
    )


__all__ = [
    "HOST_LOOP",
    "VMAPPED_TWIN",
    "ControlMeasurement",
    "ControlsReport",
    "host_loop",
    "run_negative_controls",
    "vmapped_twin",
]

"""Shared chex.assert_max_traces probe helpers (port + performance gates)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import chex
import jax


@dataclass(frozen=True, slots=True)
class ProbeResult:
    qualname: str
    passed: bool
    skipped: bool = False
    reason: str = ""
    trace_probe: str | None = None
    max_traces: int = 1


def import_qualname(qualname: str) -> Any:
    module_path, _, attr = qualname.rpartition(".")
    if not module_path or not attr:
        msg = f"invalid qualname: {qualname!r}"
        raise ImportError(msg)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def import_probe(trace_probe: str) -> Callable[[Callable[..., Any]], None]:
    module_path, sep, attr = trace_probe.partition(":")
    if not sep or not module_path or not attr:
        msg = f"trace_probe must be 'module.path:callable', got {trace_probe!r}"
        raise ValueError(msg)
    module = importlib.import_module(module_path)
    probe = getattr(module, attr)
    if not callable(probe):
        msg = f"trace_probe {trace_probe!r} is not callable"
        raise TypeError(msg)
    return probe


def _guard_for_trace_count(
    target: Callable[..., Any],
    *,
    max_traces: int,
) -> Callable[..., Any]:
    """Apply @jax.jit @chex.assert_max_traces when target is not already traced."""
    try:
        return jax.jit(chex.assert_max_traces(n=max_traces)(target))
    except ValueError:
        return target


def run_trace_gate(
    target: Callable[..., Any],
    probe: Callable[[Callable[..., Any]], None],
    *,
    max_traces: int,
) -> None:
    """Run probe against target wrapped with chex.assert_max_traces."""
    chex.clear_trace_counter()
    guarded = _guard_for_trace_count(target, max_traces=max_traces)
    probe(guarded)


def run_trace_probe(
    qualname: str,
    *,
    max_traces: int,
    trace_probe: str | None = None,
) -> ProbeResult:
    """Run a configured trace probe; returns structured pass/fail/skip."""
    if not trace_probe:
        return ProbeResult(
            qualname=qualname,
            passed=False,
            skipped=True,
            reason="no trace_probe configured for kernel",
            max_traces=max_traces,
        )

    try:
        target = import_qualname(qualname)
        probe_runner = import_probe(trace_probe)

        def _invoke(guarded: Callable[..., Any]) -> None:
            probe_runner(guarded)

        run_trace_gate(target, _invoke, max_traces=max_traces)
    except Exception as exc:  # noqa: BLE001 — gate aggregates per-kernel failures
        return ProbeResult(
            qualname=qualname,
            passed=False,
            reason=str(exc),
            trace_probe=trace_probe,
            max_traces=max_traces,
        )

    return ProbeResult(
        qualname=qualname,
        passed=True,
        trace_probe=trace_probe,
        max_traces=max_traces,
    )

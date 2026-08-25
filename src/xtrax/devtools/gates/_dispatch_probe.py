"""Dispatch-count probe: n_executions/n_compilations/n_jit_traces from a traced run.

Phase C of .praxia/docs/specs/
260824_upstream-profiling-probe-tooling-from-prolix.md: profiler-backed probe
kinds for the performance gate, alongside (not replacing) the chex
trace-count probes. The counters come from xtrax.profiling.trace's
parse_dispatch_counts over a jax.profiler.trace capture of ONE guarded
invocation -- warm-up happens OUTSIDE the traced window, so steady-state
n_compilations is expected to be 0 (D9 spike: backend_compile_and_load only
appears if compilation occurs inside the window).

Event-name fragility: see xtrax.profiling.trace's docstring and the scope
doc's D9 result. Re-spike on any JAX upgrade before trusting new ceilings.
"""

import gzip
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from xtrax.devtools.gates._trace_probe import (
    _guard_for_trace_count,
    import_probe,
    import_qualname,
)
from xtrax.profiling.trace import parse_dispatch_counts


@dataclass(frozen=True, slots=True)
class DispatchResult:
    qualname: str
    counts: dict[str, int]
    passed: bool
    skipped: bool = False
    reason: str = ""


def _load_newest_events(trace_dir: Path) -> list[dict[str, Any]]:
    trace_files = sorted(trace_dir.rglob("*.trace.json.gz"))
    if not trace_files:
        msg = f"jax.profiler.trace wrote no *.trace.json.gz under {trace_dir}"
        raise RuntimeError(msg)
    with gzip.open(trace_files[-1], "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents")
    if not isinstance(events, list):
        msg = f"{trace_files[-1]} has no traceEvents list"
        raise RuntimeError(msg)
    return events


def measure_dispatch_counts(qualname: str, trace_probe: str | None) -> DispatchResult:
    """Run ONE guarded invocation inside a trace window; parse dispatch counts.

    Skipped (not failed) when the spec has no trace_probe: dispatch counting
    shares its invocation recipe with the existing trace-count probes.
    """
    if not trace_probe:
        return DispatchResult(
            qualname=qualname,
            counts={},
            passed=False,
            skipped=True,
            reason="no trace_probe configured for kernel",
        )
    target = import_qualname(qualname)
    probe_runner = import_probe(trace_probe)

    # One guarded callable, built ONCE and reused for both the warm-up call
    # and the traced call. run_trace_gate() re-wraps per invocation, which
    # would recompile INSIDE the trace window and poison n_compilations --
    # steady-state counting needs the windowed call to hit an already-
    # compiled executable.
    guarded = _guard_for_trace_count(target, max_traces=10**6)
    probe_runner(guarded)  # warm-up + compile OUTSIDE the window

    with tempfile.TemporaryDirectory(prefix="xtrax-dispatch-probe-") as tmp:
        trace_dir = Path(tmp)
        with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
            probe_runner(guarded)
        events = _load_newest_events(trace_dir)

    counts = parse_dispatch_counts(events)
    return DispatchResult(
        qualname=qualname,
        counts=counts,
        passed=True,
    )

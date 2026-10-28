"""Dispatch-count probe: n_executions/n_compilations/n_jit_traces from a traced run.

Phase C of .praxia/docs/specs/
260824_upstream-profiling-probe-tooling-from-prolix.md: profiler-backed probe
kinds for the performance gate, alongside (not replacing) the chex
trace-count probes. The measurement itself is delegated to
xtrax.profiling.unit_probe.probe_unit: one warm-up call OUTSIDE the traced
window, then one traced steady call. The counters are the raw
parse_dispatch_counts totals over that window (exposed as
UnitProbeResult.dispatch_counts), so steady-state n_compilations is expected
to be 0 (D9 spike: backend_compile_and_load only appears if compilation
occurs inside the window).

Event-name fragility: see xtrax.profiling.trace's docstring and the scope
doc's D9 result. Re-spike on any JAX upgrade before trusting new ceilings.
"""

from dataclasses import dataclass

from xtrax.devtools.gates._trace_probe import (
    _guard_for_trace_count,
    import_probe,
    import_qualname,
)
from xtrax.profiling.unit_probe import probe_unit


@dataclass(frozen=True, slots=True)
class DispatchResult:
    qualname: str
    counts: dict[str, int]
    passed: bool
    skipped: bool = False
    reason: str = ""


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

    # One guarded callable, built ONCE and reused for the warm-up call and the
    # traced call. run_trace_gate() re-wraps per invocation, which would
    # recompile INSIDE the trace window and poison n_compilations --
    # steady-state counting needs the windowed call to hit an already-
    # compiled executable.
    guarded = _guard_for_trace_count(target, max_traces=10**6)
    unit = probe_unit(lambda: probe_runner(guarded), (), n_steady=1, n_wall=1)

    return DispatchResult(
        qualname=qualname,
        counts=dict(unit.dispatch_counts),
        passed=True,
    )

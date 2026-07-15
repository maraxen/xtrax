"""Compile-time two-phase clock (T2-19, #2181, AC-27, fork 3, ORTH-4).

AC-27's claim: runtime fitness must exclude compile time (a persistent XLA cache warms the first
compile), with compile time tracked as its own metric; a compile-time blowup (> K times the rolling
median, default K=3) should be flagged loud without polluting the runtime fitness comparison.

Grounding note (verified before writing any code): no existing xtrax timing idiom separates compile
time from runtime. Two related, real precedents exist for pieces of this -- `src/xtrax/cli/
sweep_verb.py` already sets up a persistent XLA compilation cache (`jax.config.update(
"jax_compilation_cache_dir", ...)`, wired only into the sweep CLI today, not any loop gate) --
literally the mechanism AC-27's "persistent XLA cache warms first compile" text names, though this
module doesn't need to wire that path up itself (see below); `src/xtrax/tiling/estimators.py`
establishes the `.lower().compile()` AOT idiom for a different purpose (memory estimation),
confirming the pattern is known in this codebase, just not for timing yet.

Design decision (confirmed before writing code): this candidate runs **in-process** (matching
every T2-1x module except T2-14, which spawns a genuinely separate, untrusted subprocess for a
smoke test -- T2-19 needs to report two timing numbers back from a single call site, which a
one-shot subprocess doesn't naturally support). Compile time is measured by calling the same
`jax.jit`-wrapped candidate twice with the same concrete inputs: the first call pays for tracing +
XLA compilation + execution, the second call (cache-hit) pays only for execution --
`compile_time_seconds = max(first_call - second_call, 0.0)`. `jax.block_until_ready()` wraps each
timed call: JAX dispatch is asynchronous, so a bare `time.perf_counter()` around a JAX call without
blocking would measure dispatch time, not actual compilation/execution.

Rolling-median ownership (mirrors T2-18's identical stance on its own windowed history): this
module does not persist compile-time history itself -- `rolling_median` and
`assert_no_compile_time_regression` both take the history as a caller-supplied sequence; a future
loop controller owns the actual storage, matching every T2-1x module's "don't invent state you
don't own" stance.

Extension seam: this module reports; it does not decide what a future loop controller does with a
detected regression beyond raising loud, and it does not itself wire up the persistent XLA cache
directory `sweep_verb.py` sets up for the CLI path -- a caller running many candidates across
process boundaries would still want that for genuinely warm first-compiles between runs, but that
is a deployment/CLI-wiring concern orthogonal to this gate's own two-call, in-process measurement.
"""

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from xtrax.loop.schema_gate import resolve_candidate_callable


class CompileTimeClockError(Exception):
    """Base for T2-19's compile-time-clock errors."""


class CompileTimeRegressionError(CompileTimeClockError):
    """AC-27's rejection error: compile time exceeded `k_threshold * rolling_median(history)`."""


@dataclass(frozen=True, slots=True)
class TwoPhaseTiming:
    """The AC-27 compile/runtime split for a single candidate invocation.

    `result` is the candidate's actual output, taken from the second (cache-warm) call -- useful
    for a caller to use directly without a redundant third call.
    """

    compile_time_seconds: float
    runtime_seconds: float
    result: Any


def measure_two_phase_timing(
    candidate_path: Path,
    callable_name: str,
    *,
    concrete_inputs: list[Any],
) -> TwoPhaseTiming:
    """The composed AC-27 measurement: resolve, `jax.jit`, call twice, split compile from runtime.

    Args:
        candidate_path: source file to resolve (same contract as T2-11 through T2-15).
        callable_name: attribute on the resolved module to call.
        concrete_inputs: real arguments, reused identically for both timed calls -- an input-shape
            change between calls would itself trigger a recompile on the second call, defeating
            the cache-hit assumption this measurement depends on.

    Returns:
        A `TwoPhaseTiming`. `compile_time_seconds` is clamped to a minimum of 0.0 -- ordinary
        timing jitter can otherwise make the second (cache-warm) call measure slightly slower than
        the first on a noisy host, which must not be reported as negative compile time.

    Raises:
        CandidateResolutionError: `candidate_path` can't be imported, or `callable_name` isn't a
            callable attribute -- propagates unchanged, not wrapped in a T2-19 error (T2-13/T2-15
            precedent).
    """
    fn = resolve_candidate_callable(candidate_path, callable_name)
    jitted = jax.jit(fn)

    first_start = time.perf_counter()
    first_result = jitted(*concrete_inputs)
    jax.block_until_ready(first_result)
    first_call_seconds = time.perf_counter() - first_start

    second_start = time.perf_counter()
    second_result = jitted(*concrete_inputs)
    jax.block_until_ready(second_result)
    second_call_seconds = time.perf_counter() - second_start

    compile_time_seconds = max(first_call_seconds - second_call_seconds, 0.0)

    return TwoPhaseTiming(
        compile_time_seconds=compile_time_seconds,
        runtime_seconds=second_call_seconds,
        result=second_result,
    )


def rolling_median(history: Sequence[float]) -> float:
    """The median of `history`.

    Raises:
        CompileTimeClockError: `history` is empty -- a median is undefined.
    """
    if not history:
        msg = "cannot compute a rolling median of an empty history"
        raise CompileTimeClockError(msg)
    return statistics.median(history)


def assert_no_compile_time_regression(
    compile_time_seconds: float,
    *,
    compile_time_history: Sequence[float],
    k_threshold: float = 3.0,
) -> None:
    """The composed AC-27 regression gate: flag loud if `compile_time_seconds` exceeds
    `k_threshold * rolling_median(compile_time_history)`.

    Args:
        compile_time_seconds: this candidate's own `TwoPhaseTiming.compile_time_seconds`.
        compile_time_history: the caller-maintained rolling window of prior compile times --
            this module never computes or stores this history itself.
        k_threshold: default 3.0, per AC-27's own text ("K=3").

    A no-op (does not raise) if `compile_time_history` is empty -- nothing to regress against yet
    -- or if its median is `<= 0` -- a multiplicative threshold against a ~zero baseline is
    degenerate (any nonzero compile time would trivially exceed `k_threshold * 0`), so the check is
    skipped rather than silently over-triggering on a near-instant-compile candidate's first
    genuinely nonzero measurement.

    Raises:
        CompileTimeRegressionError: `compile_time_seconds > k_threshold * median`, naming both
            the actual and threshold values.
    """
    if not compile_time_history:
        return
    median = rolling_median(compile_time_history)
    if median <= 0:
        return
    threshold = k_threshold * median
    if compile_time_seconds > threshold:
        msg = (
            f"compile time {compile_time_seconds:.4f}s exceeds {k_threshold}x the rolling "
            f"median {median:.4f}s (threshold {threshold:.4f}s)"
        )
        raise CompileTimeRegressionError(msg)


__all__ = [
    "CompileTimeClockError",
    "CompileTimeRegressionError",
    "TwoPhaseTiming",
    "assert_no_compile_time_regression",
    "measure_two_phase_timing",
    "rolling_median",
]

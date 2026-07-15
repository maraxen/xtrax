"""Checkified-execution (T2-15, #2181, AC-5).

AC-5's claim: a smoke-passing candidate (T2-14) must be executed under
`SafetyManager(enabled=True)` checkify float_checks, with no NaN/Inf/overflow escaping to the
host. `SafetyManager` (`xtrax.safety.manager`, commit 71f213c) already exists and predates this
session's T2-1x work entirely -- this module is thin glue like T2-12/T2-13, not new infrastructure
like T2-14: `SafetyManager.wrap(fn)` already builds `jax.jit(checkify.checkify(fn,
errors=checkify.float_checks))` and calls `err.throw()` on the host, raising
`jax.experimental.checkify.JaxRuntimeError` (empirically confirmed: a subclass of `ValueError`,
publicly importable from `jax.experimental.checkify`, not just the private `jax._src.checkify`
module it's actually defined in) the moment a tracked violation fires.

Known, pre-existing, out-of-scope gap in `SafetyManager` itself (not this module's to fix --
`SafetyManager` predates this worktree and lives in a different package): its `check_nans`/
`check_infs` fields are declared and documented ("Include NaN checks in float_checks" / "Include
Inf checks in float_checks") but `.wrap()` never actually reads either field -- it unconditionally
uses `checkify.float_checks` regardless of their values. This module still forwards `check_nans`/
`check_infs` through to the `SafetyManager` it constructs (so the call site stays correct in
intent, and needs no update if that gap is ever fixed upstream), but do not assume passing
`check_nans=False` here actually disables NaN detection today -- it currently does not.

Real gap in `checkify.float_checks` itself, closed rather than silently under-delivered on:
`float_checks = nan_checks | div_checks` has no dedicated Inf/overflow detector -- an Inf arising
from something other than a literal division-by-zero (e.g. `jnp.exp(1000.0)`) passes through
`err.throw()` without raising (verified empirically: `checkify.float_checks` misses this case
outright, `result` comes back as `inf` with no exception). AC-5's own text says "no
NaN/Inf/overflow" -- broader than what `float_checks` alone enumerates -- so this module adds a
manual post-call `jnp.isinf` walk over the result pytree, in addition to composing `SafetyManager`
for the NaN/div-by-zero layer, to actually satisfy that claim rather than merely naming the DAG's
cited mechanism.

Extension seam (mirrors every other T2-0X module's stance): this module raises
`CheckifiedExecutionError` and stops. AC-5's "candidate marked failed; git reset" describes a
*consequence*, not something this module performs -- no #2181 loop controller exists yet, and
`xtrax.loop.ratchet_crash_atomicity` (T2-10) itself exposes only "advance best-so-far" primitives,
no "discard candidate" one, confirming the git-reset action belongs to a future controller, not to
any currently-merged T2-0X module.
"""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.tree_util
from jax.experimental import checkify

from xtrax.loop.schema_gate import resolve_candidate_callable
from xtrax.safety.manager import SafetyManager


class CheckifiedExecutionError(Exception):
    """AC-5's rejection error: a NaN/Inf/overflow violation was detected on the host.

    Candidate marked failed; do not retry. The physical git reset is a future loop controller's
    responsibility, not this module's -- see the module docstring's Extension seam note.
    """


def assert_checkified_execution(
    candidate_path: Path,
    callable_name: str,
    *,
    concrete_inputs: list[Any],
    check_nans: bool = True,
    check_infs: bool = True,
) -> Any:
    """The composed AC-5 gate: resolve, wrap under `SafetyManager(enabled=True)`, call once.

    Always constructs `SafetyManager(enabled=True, ...)` -- this gate exists specifically to
    exercise that path; the `enabled=False` "promoted best-lineage" case AC-5's parenthetical
    describes is a different, later consumer's concern, not something this function checks.

    Args:
        candidate_path: source file to resolve (same contract as T2-11/T2-12/T2-13/T2-14).
        callable_name: attribute on the resolved module to call.
        concrete_inputs: real arguments for the single execution.
        check_nans: forwarded to `SafetyManager` (see module docstring: not currently consulted
            by `SafetyManager.wrap()`, kept for call-site correctness regardless).
        check_infs: forwarded to `SafetyManager` (same caveat as `check_nans`).

    Returns:
        The candidate's result, once confirmed free of NaN/Inf/overflow -- useful for a caller to
        use directly without a redundant second call.

    Raises:
        CandidateResolutionError: `candidate_path` can't be imported, or `callable_name` isn't a
            callable attribute (see `xtrax.loop.schema_gate.resolve_candidate_callable`) --
            propagates unchanged, not wrapped in `CheckifiedExecutionError`.
        CheckifiedExecutionError: `checkify.float_checks` detected a NaN or division-by-zero, or
            the post-call `isinf` sweep found an Inf `float_checks` itself missed.
    """
    fn = resolve_candidate_callable(candidate_path, callable_name)
    manager = SafetyManager(enabled=True, check_nans=check_nans, check_infs=check_infs)
    wrapped = manager.wrap(fn)

    try:
        result = wrapped(*concrete_inputs)
    except checkify.JaxRuntimeError as exc:
        msg = f"checkify detected a NaN/division-by-zero violation: {exc}"
        raise CheckifiedExecutionError(msg) from exc

    inf_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(result)
        if hasattr(leaf, "dtype") and jnp.isinf(leaf).any()
    ]
    if inf_leaves:
        msg = (
            f"result contains Inf in {len(inf_leaves)} leaf/leaves (not caught by "
            "checkify.float_checks, which has no dedicated overflow/Inf detector -- "
            "see module docstring)"
        )
        raise CheckifiedExecutionError(msg)

    return result


__all__ = [
    "CheckifiedExecutionError",
    "assert_checkified_execution",
]

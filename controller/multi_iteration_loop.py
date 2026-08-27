"""Multi-iteration loop wiring (epic #3611 loop-controller, LC-10, AC-8b).

AC-8b's text (verbatim): "multi-iteration loop wiring: the actual 'keep going' logic across
candidates, external-stop-watchdog integration (T2-09's already-merged
`xtrax.loop.external_stop_watchdog` primitive supervises the whole run), diversity-quota
Leap-Path triggering (T2-18's already-merged `xtrax.loop.diversity_quota.assert_diversity_quota`
decision -- this item wires the RESPONSE to a `leap_path_required` signal, which T2-18 itself
never implements). AC: a real multi-iteration run (>=3 candidates) terminates correctly on both
a budget-exhaustion path and a normal-completion path." See
`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md` ("AC-8b", "finding 1",
"leap_path_required", lines 262-268) and
`.praxia/docs/roadmaps/loop-controller/260716_00-mandate.md` section 6 (phase 5), which this
item closes: "Multi-iteration loop -- the actual 'keep going' logic, budget/watchdog
integration (T2-09 already provides the primitive), diversity-quota Leap-Path triggering (T2-18
already provides the decision, not the mutation-generation response to it)." That phase was
silently dropped during the epic's initial brainstorm pass and restored as AC-8b after
adversarial review flagged that a "loop controller" that never iterates past one candidate isn't
actually a loop.

This module provides one function, `run_multi_iteration_loop`, which repeatedly calls LC-09's
`run_one_candidate_pass` (`controller.main_loop`) and adds exactly the three things AC-8b names:
a supervising watchdog, a diversity-quota rolling window with a wired Leap-Path response, and
two distinct, independently-testable termination paths.

## Watchdog wiring: real subprocess in production, NEVER in this module's own tests

`xtrax.loop.external_stop_watchdog.start_watchdog` spawns a genuinely separate, detached OS
process (`subprocess.Popen(..., start_new_session=True)`) that SIGKILLs a target PID once a
wall-clock budget elapses -- see that module's own docstring. `run_multi_iteration_loop` starts
this watchdog exactly ONCE, at the very start of the run, supervising `os.getpid()` (this
process, the whole run) -- not once per candidate. The starter is an injectable dependency
(`start_watchdog_fn`, defaulting to the real `start_watchdog`) specifically so tests can supply a
fake `WatchdogHandle`-shaped stub (`is_alive`/`join`/`stop`) instead of ever spawning the real
kill-capable subprocess against the actual test-runner process. **No test in this module's own
test suite may call the real `start_watchdog` under any circumstance** -- doing so against a
short budget would hard-kill the test runner itself, not a hypothetical failure mode (see
`external_stop_watchdog.py`'s own module docstring, PM-7). `run_multi_iteration_loop` stops the
watchdog itself in a `finally` block once the run ends (gracefully or via an uncaught exception)
-- this is ordinary resource cleanup for a subprocess this function itself started, not error/
retry policy or a `campaign_conclude`-fires-on-every-path guarantee (both explicitly out of
scope; see below). `WatchdogAlreadyStoppedError` (raised by `WatchdogHandle.stop()` if the
watchdog process already exited on its own) is caught and ignored here for exactly that reason:
by the time this function's own cleanup runs, there is nothing further to signal.

## Two distinct termination paths, per the AC's own literal text

AC-8b's AC requires "correctly on both a budget-exhaustion path and a normal-completion path" --
two *distinct* mechanisms, not one:

1. **Normal completion**: `max_candidates` (a required, caller-supplied stopping condition) is
   reached with no uncaught exception. `termination_reason="normal_completion"`.
2. **Budget exhaustion**: a SOFT, in-process, independently-testable wall-clock check
   (`wall_clock_budget_seconds` + the injectable `time_fn`, default `time.monotonic`), evaluated
   before each candidate's dispatch. This is deliberately NOT the same value as
   `watchdog_criteria.wall_clock_budget_seconds` (the watchdog's own HARD, out-of-process kill
   budget) -- the two are independent parameters, and this function enforces no relationship
   between them. The soft budget's entire purpose is to let the loop notice it is out of time and
   return a well-formed `MultiIterationLoopResult` *before* the OS-level watchdog would ever need
   to fire a real `SIGKILL`; the watchdog remains the hard backstop for a loop that is wedged and
   never reaches its own soft check at all. `termination_reason="budget_exhausted"`.

   This mirrors the LC-10 dispatch brief's own reasoning almost verbatim: "you can't/shouldn't
   test [the real watchdog hard-kill] directly, your test needs some other caller-observable
   signal proving the driver WOULD have honored a budget limit ... an injected clock/time-budget
   check inside the driver itself that's independently testable without depending on the real
   watchdog process actually firing." Tests exercise path 2 via a fake `time_fn` that crosses the
   supplied `wall_clock_budget_seconds` after several real candidate passes have already
   completed -- never by invoking the real watchdog.

A caught per-candidate failure / uncaught-exception path is explicitly NOT a third termination
reason implemented here -- that is LC-11's scope (AC-8c), described below.

## Diversity-quota Leap-Path response: what "wiring the response" concretely means, and why

`assert_diversity_quota` (T2-18, already merged) computes a `DiversityQuotaDecision` from a
caller-maintained rolling window of `is_structural_mutation()` pairwise flags -- its own
docstring is explicit that it "does NOT implement Leap-Path itself... this module returns a
`DiversityQuotaDecision` signal." AC-8b's own text is equally explicit that T2-18 "never
implements" the response to `leap_path_required=True` -- that response is this item's job.

Grounding the response in what's actually buildable today (not invented from nothing): the
mandate doc (section 6, phase 5) frames Leap-Path as "diversity-quota Leap-Path triggering (T2-18
already provides the decision, not the mutation-generation response to it)" -- and
`diversity_quota.py`'s own docstring traces "Leap-Path" to the AutoSOTA paper's "Leap Path
bifurcation" (forcing the next iteration onto a structurally-novel idea), explicitly noting "no
detail exists anywhere on *how* a forced structural mutation gets proposed/generated." Checking
what's actually available to force such a thing: `controller.dispatch.DispatchBackend.
dispatch_candidate()` (LC-03's Protocol) takes **zero parameters** -- there is no hook anywhere
in this codebase today for a caller to tell a dispatch backend "propose something structurally
different this time." Inventing that capability here would be exactly the kind of scope creep
this epic's own decomposition repeatedly warns against (see `main_loop.py`'s own scoping-decision
precedent: "wiring one of those in here would mean inventing its own data-sourcing wrapper as a
side effect of this item"). So this module wires the response as far as it can go without
inventing a mutation-generation mechanism that doesn't exist:

1. The rolling structural-mutation-flag window is genuinely maintained across iterations (see
   below) and fed to the real `assert_diversity_quota` after every candidate from the second
   onward.
2. When `leap_path_required=True` fires, an injectable hook (`on_leap_path_required`, a
   `Callable[[LeapPathEvent], None]`) is invoked exactly once for that occurrence. This is the
   concrete extension seam where a future mutation-generation strategy -- once one is designed
   and a `DispatchBackend` grows a hook to receive it -- plugs in. The default implementation
   (`_default_leap_path_handler`) does the only thing honestly available today: it surfaces the
   signal loudly via structured logging, so a human operator or downstream tooling watching the
   run's logs can see it rather than the signal being computed and silently discarded (T2-18's
   own stated gap).
3. Every firing is also aggregated onto the returned `MultiIterationLoopResult.leap_path_events`,
   so any caller can inspect (or act on, or re-raise, or feed to a future mutation-generation
   backend) the full set of occurrences after the run completes, not just react in real time via
   the hook.

This is a deliberately bounded "wire the response, don't invent the mutation" reading of AC-8b's
own text -- consistent with the mandate doc's own framing that only the *decision* (T2-18) and
now the *response-wiring* (this item) exist; the actual mutation-generation strategy remains an
explicitly out-of-scope, not-yet-designed follow-up for whichever future item grows
`DispatchBackend` a real hook to receive it.

### How the rolling window is actually built

`is_structural_mutation` (T2-18) is a pairwise check over two candidates' raw source text. This
module reads each candidate's source via `result.handoff.path.read_text()` -- `CandidateHandoff`
itself only carries `path` + `content_sha256` (LC-03), never raw content, and
`content_sha256` alone cannot feed an AST-structural diff. `CandidateHandoff.path`'s own
docstring already establishes the contract this relies on: "path to the staged candidate source
file (must be readable by controller)". One consequence worth flagging explicitly for test
authors: `MockDispatchBackend` (LC-03) computes `content_sha256` from its own
`candidate_content` field but never writes that content to `candidate_path` on disk -- a test
using `MockDispatchBackend` with this driver must independently write real, matching file content
to each `candidate_path` for the diversity-quota window to observe anything meaningful. This is
an interaction between LC-03's mock design and LC-10's read pattern, not a bug in either.

The window itself: `previous_source` starts `None` (no prior candidate to compare against on
iteration 1 -- no flag is computed or appended for the first candidate, matching
`assert_diversity_quota`'s own "over N *consecutive* iterations" framing, which needs a pair to
compare). From the second candidate onward, `is_structural_mutation(previous_source,
candidate_source)` is appended to an ever-growing `structural_flags` list (never truncated by
this module -- `assert_diversity_quota` itself slices the trailing `window_size` window, per its
own docstring), and `assert_diversity_quota` is re-evaluated. This happens for every candidate
regardless of that candidate's own gate/run outcome (`OneCandidatePassResult.accepted`) --
AC-12's own framing is about whether the *search itself* is exploring structurally, not about
gate outcomes, so a rejected candidate's source still counts toward (or against) the quota.

## What this module explicitly does NOT do (deferred to LC-11, AC-8c)

Per this item's own dispatch brief and `main_loop.py`'s identical precedent: zero error/retry
policy, and no guarantee that `campaign_conclude` (or an equivalent close-out call) fires on
every code path that exits the loop. `run_multi_iteration_loop` performs zero retries and does
not catch any exception raised by `run_one_candidate_pass` or by reading/diffing candidate
source (`FileNotFoundError`, `UnicodeDecodeError`, `DiversityQuotaInputError` on malformed
candidate source, or any of `run_one_candidate_pass`'s own documented exceptions) -- every one
propagates to the caller unmodified, after this function's own `finally` block stops the watchdog
it started (ordinary resource cleanup, not retry policy or a bathos-side close-out guarantee).
AC-8c (LC-11) owns error/retry policy and the "conclude fires on every code path" property; this
module's job is to prove the multi-iteration "keep going" sequence -- watchdog, diversity-quota
response, and both termination paths -- is wired correctly, not to also own what happens when a
step fails.
"""

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from controller.bathos_campaign_adapter import BathosCampaignAdapter
from controller.bathos_library_wrappers import call_stats_battery_gate, get_seed_trial_counts
from controller.dispatch import DispatchBackend
from controller.evaluate_adapter import BathosFrozenContext
from controller.lineage_interim import CandidateParentage
from controller.main_loop import CampaignMode, OneCandidatePassResult, run_one_candidate_pass
from xtrax.loop.candidate_smoke import assert_candidate_smoke
from xtrax.loop.candidate_static import assert_candidate_static
from xtrax.loop.checkified_execution import assert_checkified_execution
from xtrax.loop.diversity_quota import (
    DiversityQuotaDecision,
    assert_diversity_quota,
    is_structural_mutation,
)
from xtrax.loop.external_stop_watchdog import (
    WatchdogAlreadyStoppedError,
    WatchdogCriteria,
    WatchdogHandle,
    start_watchdog,
)
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict
from xtrax.loop.structure_tripwire import assert_structure_tripwire

_logger = logging.getLogger(__name__)

TerminationReason = Literal["normal_completion", "budget_exhausted"]


@dataclass(frozen=True, slots=True)
class LeapPathEvent:
    """One firing of `assert_diversity_quota`'s `leap_path_required=True` decision.

    Attributes:
        iteration_index: the 0-based index (within this run) of the candidate whose pass just
            completed when this firing was observed.
        decision: the `DiversityQuotaDecision` that triggered this event (`leap_path_required`
            is always `True` here; `window` carries the exact tail of structural-mutation flags
            that produced it, for logging/introspection).
    """

    iteration_index: int
    decision: DiversityQuotaDecision


def _default_leap_path_handler(event: LeapPathEvent) -> None:
    """Default `on_leap_path_required` hook: surface the signal via structured logging.

    See the module docstring's "Diversity-quota Leap-Path response" section for why this is the
    honest, grounded default -- no mutation-generation mechanism exists anywhere in this
    codebase yet (`DispatchBackend.dispatch_candidate()` takes zero parameters), so this default
    does the one thing actually available: make the signal loudly observable instead of letting
    it be computed and silently discarded.
    """
    _logger.warning(
        "leap-path required at iteration %d: diversity quota unmet over window=%s "
        "(no mutation-generation strategy is implemented yet -- this signal is surfaced for a "
        "future dispatch backend or human operator to act on; see AC-8b's module docstring, "
        "controller/multi_iteration_loop.py)",
        event.iteration_index,
        event.decision.window,
    )


@dataclass(frozen=True, slots=True)
class MultiIterationLoopResult:
    """The composed outcome of one `run_multi_iteration_loop` call.

    Deliberately minimal, matching `OneCandidatePassResult`'s own precedent: bundles what LC-11
    (error/retry policy, AC-8c) will need to build on, without this item anticipating that
    item's own design.

    Attributes:
        iterations: every `OneCandidatePassResult` produced, in order, one per candidate that
            actually completed a pass (a candidate skipped by an early budget-exhaustion break
            is not represented here).
        termination_reason: `"normal_completion"` (`max_candidates` reached) or
            `"budget_exhausted"` (the soft, in-process wall-clock check tripped before the next
            candidate's dispatch). See the module docstring's "Two distinct termination paths"
            section.
        leap_path_events: every `LeapPathEvent` fired during this run, in order.
        watchdog_handle: the (already-`stop()`-ped, in the normal/budget-exhaustion paths)
            handle to the watchdog this run started, for caller introspection.
    """

    iterations: tuple[OneCandidatePassResult, ...]
    termination_reason: TerminationReason
    leap_path_events: tuple[LeapPathEvent, ...]
    watchdog_handle: WatchdogHandle

    @property
    def accepted_count(self) -> int:
        """Count of iterations that were actually promotable per the ratchet (GW-02).

        `OneCandidatePassResult.accepted` gained a stricter meaning once `run_one_candidate_pass`
        wired in the real multi-metric ratchet decision (AC-10): it now additionally requires
        `ratchet_decision.improved`, not just "the bathos run succeeded and neither gate hard-
        blocked" (see `main_loop.py`'s own module docstring and `OneCandidatePassResult.accepted`
        for the full rationale -- a candidate strictly worse on every metric than the current
        best no longer reads `accepted=True` here). This property's own implementation is
        unchanged by that -- it always intended "was actually promotable," it simply inherits the
        stricter, now-correct definition of `accepted` for free.
        """
        return sum(1 for result in self.iterations if result.accepted)


def run_multi_iteration_loop(
    dispatch_backend: DispatchBackend,
    campaign_adapter: BathosCampaignAdapter,
    *,
    campaign_id: str,
    campaign_mode: CampaignMode,
    max_candidates: int,
    watchdog_criteria: WatchdogCriteria,
    parentage: CandidateParentage = CandidateParentage(),
    run_args: list[str] | None = None,
    output_paths: list[str] | None = None,
    tags: list[str] | None = None,
    agent_mode: str = "",
    no_sidecar: bool = False,
    stats_battery_kwargs: Mapping[str, Any],
    seed_trial_db: Any = None,
    hypothesis_clause_id: str = "",
    stats_battery_fn: Callable[..., BathosStatsBatteryVerdict] = call_stats_battery_gate,
    seed_trial_counts_fn: Callable[..., SeedTrialCounts] = get_seed_trial_counts,
    candidate_static_fn: Callable[..., None] = assert_candidate_static,
    candidate_static_root: Path | None = None,
    abstract_inputs: list[Any],
    structure_tripwire_fn: Callable[..., None] = assert_structure_tripwire,
    candidate_smoke_fn: Callable[..., None] = assert_candidate_smoke,
    candidate_smoke_root: Path | None = None,
    checkified_execution_fn: Callable[..., Any] = assert_checkified_execution,
    wall_clock_budget_seconds: float | None = None,
    time_fn: Callable[[], float] = time.monotonic,
    diversity_window_size: int = 5,
    on_leap_path_required: Callable[[LeapPathEvent], None] = _default_leap_path_handler,
    start_watchdog_fn: Callable[[int, WatchdogCriteria], WatchdogHandle] = start_watchdog,
    higher_is_better: Mapping[str, bool],
    frozen_context: BathosFrozenContext,
    metrics_provenance_dir: Path,
    current_config: Mapping[str, Any],
    repo: Path,
    ratchet_ref_name: str,
    callable_name: str,
    concrete_inputs: list[Any],
    commit_tree_sha: str | None = None,
    candidate_target_path: Path | None = None,
    allow_fresh_start_despite_existing_lineage: bool = False,
    bootstrap_commit_sha: str | None = None,
) -> MultiIterationLoopResult:
    """Run the actual multi-iteration "keep going" loop across candidates (AC-8b).

    Starts a supervising watchdog once, then repeatedly calls `run_one_candidate_pass` (LC-09)
    while maintaining a rolling structural-mutation-flag window for `assert_diversity_quota`
    (T2-18) and wiring its `leap_path_required` decision to `on_leap_path_required`. See the
    module docstring for the full grounding of every design choice below.

    Args:
        dispatch_backend: forwarded to `run_one_candidate_pass` on every iteration.
        campaign_adapter: forwarded to `run_one_candidate_pass` on every iteration. The caller
            is responsible for having already called `campaign_create` (same scoping as LC-09).
        campaign_id: forwarded to `run_one_candidate_pass` on every iteration.
        campaign_mode: forwarded to `run_one_candidate_pass` on every iteration.
        max_candidates: the normal-completion stopping condition -- run at most this many
            candidate passes. Must be `>= 1`.
        watchdog_criteria: the HARD, out-of-process kill budget for the real watchdog (AC-13).
            Required (no default) -- deliberately not derived from `wall_clock_budget_seconds`;
            see the module docstring's "Two distinct termination paths" section for why the two
            budgets are independent.
        parentage: forwarded to `run_one_candidate_pass` unchanged on every iteration (this
            module does not implement a per-iteration parentage/ratcheting strategy -- AC-8b's
            own text names watchdog integration and Leap-Path response, not lineage selection
            across iterations; left to a future item).
        run_args: forwarded to `run_one_candidate_pass` on every iteration.
        output_paths: forwarded to `run_one_candidate_pass` on every iteration.
        tags: forwarded to `run_one_candidate_pass` on every iteration.
        agent_mode: forwarded to `run_one_candidate_pass` on every iteration.
        no_sidecar: forwarded to `run_one_candidate_pass` on every iteration.
        stats_battery_kwargs: forwarded to `run_one_candidate_pass` unchanged on every
            iteration -- see that function's own docstring for why this module does not compute
            these values itself either.
        seed_trial_db: forwarded to `run_one_candidate_pass` on every iteration.
        hypothesis_clause_id: forwarded to `run_one_candidate_pass` on every iteration.
        stats_battery_fn: forwarded to `run_one_candidate_pass` on every iteration (injection
            seam for tests, default: LC-07's real `call_stats_battery_gate`).
        seed_trial_counts_fn: forwarded to `run_one_candidate_pass` on every iteration
            (injection seam for tests, default: LC-07's real `get_seed_trial_counts`).
        candidate_static_fn: forwarded to `run_one_candidate_pass` on every iteration
            (injection seam for tests, default: T2-11's real `assert_candidate_static`).
        candidate_static_root: forwarded to `run_one_candidate_pass` on every iteration
            (jaxlint's subprocess root; default `None` lets jaxlint fall back to `Path.cwd()`).
        wall_clock_budget_seconds: the SOFT, in-process budget-exhaustion check. `None`
            (default) disables it -- the run then only stops via `max_candidates` or the real
            watchdog's hard kill. When set, checked (via `time_fn`) before each candidate's
            dispatch; tripping it sets `termination_reason="budget_exhausted"` and stops before
            that candidate is attempted.
        time_fn: injection seam for tests -- defaults to `time.monotonic`. Never needs to touch
            the real watchdog subprocess; this is purely this function's own in-process check.
        diversity_window_size: forwarded to `assert_diversity_quota` as `window_size` (default
            5, matching T2-18's own AC-12 default).
        on_leap_path_required: injection seam -- invoked exactly once per `LeapPathEvent`
            firing. Defaults to `_default_leap_path_handler` (structured logging). See the
            module docstring's "Diversity-quota Leap-Path response" section.
        start_watchdog_fn: injection seam for tests -- defaults to the REAL
            `xtrax.loop.external_stop_watchdog.start_watchdog`. **Tests of this function must
            always override this with a fake watchdog-starter returning a fake
            `WatchdogHandle`-shaped stub** -- see the module docstring's watchdog-wiring section
            for why the real one must never be reachable from a test.
        higher_is_better: forwarded to `run_one_candidate_pass` on every iteration, but NOT
            unconditionally -- suppressed to `None` on any call where this loop's own live
            `best_fitness` state is still `None` (matching `run_one_candidate_pass`'s own
            all-or-nothing guard on `best_fitness`/`higher_is_better`, `main_loop.py:461-467`),
            and forwarded as the real, unmutated mapping once `best_fitness` becomes non-`None`.
            Required (no default) -- GW-02's ratchet decision cannot run without a per-metric
            comparison direction.
        frozen_context: forwarded to `run_one_candidate_pass` unchanged on every iteration
            (SPLIT_COMPUTE, #4133/#3657).
        metrics_provenance_dir: forwarded to `run_one_candidate_pass` unchanged on every
            iteration (backlog #3075 metrics-provenance attestation directory).
        current_config: forwarded to `run_one_candidate_pass` unchanged on every iteration.
        repo: forwarded to `run_one_candidate_pass` unchanged on every iteration.
        ratchet_ref_name: forwarded to `run_one_candidate_pass` unchanged on every iteration.
        callable_name: forwarded to `run_one_candidate_pass` unchanged on every iteration.
        concrete_inputs: forwarded to `run_one_candidate_pass` unchanged on every iteration.
        commit_tree_sha: forwarded to `run_one_candidate_pass` unchanged on every iteration.
            `None` (default) lets that function fall back to `commit_tree_sha_fn`.
        candidate_target_path: forwarded to `run_one_candidate_pass` unchanged on every
            iteration. `None` (default) is only valid there when `commit_tree_sha` is supplied
            literally -- see that function's own fail-fast validation.
        allow_fresh_start_despite_existing_lineage: forwarded to `run_one_candidate_pass`
            unchanged on every iteration. `False` (default) -- see that function's own
            lineage-conflict guard.
        bootstrap_commit_sha: forwarded to BOTH `run_one_candidate_pass`'s `commit_parent_sha`
            and `bootstrap_base_tree_sha` unconditionally on every call; inert except on a
            genuinely fresh campaign's first accepted candidate (`read_best_so_far` returns
            `None`); callers needing the two inner values to diverge must call
            `run_one_candidate_pass` directly; not validated before use (garbage-in surfaces as
            a real git-subprocess error, not a friendlier `ValueError`).

    Returns:
        A `MultiIterationLoopResult` bundling every completed iteration, the termination
        reason, every `LeapPathEvent` fired, and the (already-stopped) watchdog handle.

    Raises:
        ValueError: `max_candidates < 1`.
        Any exception `run_one_candidate_pass` itself raises (see that function's own `Raises`
            section) -- propagates unmodified after this function's own watchdog-cleanup
            `finally` block runs. Zero retry logic, matching LC-09's own precedent.
        FileNotFoundError, UnicodeDecodeError: reading a candidate's staged source file (for the
            diversity-quota window) failed.
        DiversityQuotaInputError: a candidate's staged source failed to parse as Python.
    """
    if max_candidates < 1:
        msg = f"max_candidates must be >= 1, got {max_candidates}"
        raise ValueError(msg)

    watchdog_handle = start_watchdog_fn(os.getpid(), watchdog_criteria)

    iterations: list[OneCandidatePassResult] = []
    leap_path_events: list[LeapPathEvent] = []
    structural_flags: list[bool] = []
    previous_source: str | None = None
    termination_reason: TerminationReason = "normal_completion"
    loop_start = time_fn()
    best_fitness: Mapping[str, float] | None = None

    try:
        for candidate_index in range(max_candidates):
            if (
                wall_clock_budget_seconds is not None
                and (time_fn() - loop_start) >= wall_clock_budget_seconds
            ):
                termination_reason = "budget_exhausted"
                break

            result = run_one_candidate_pass(
                dispatch_backend,
                campaign_adapter,
                campaign_id=campaign_id,
                campaign_mode=campaign_mode,
                parentage=parentage,
                run_args=run_args,
                output_paths=output_paths,
                tags=tags,
                agent_mode=agent_mode,
                no_sidecar=no_sidecar,
                stats_battery_kwargs=stats_battery_kwargs,
                seed_trial_db=seed_trial_db,
                hypothesis_clause_id=hypothesis_clause_id,
                stats_battery_fn=stats_battery_fn,
                seed_trial_counts_fn=seed_trial_counts_fn,
                candidate_static_fn=candidate_static_fn,
                candidate_static_root=candidate_static_root,
                abstract_inputs=abstract_inputs,
                structure_tripwire_fn=structure_tripwire_fn,
                candidate_smoke_fn=candidate_smoke_fn,
                candidate_smoke_root=candidate_smoke_root,
                checkified_execution_fn=checkified_execution_fn,
                frozen_context=frozen_context,
                current_config=current_config,
                metrics_provenance_dir=metrics_provenance_dir,
                best_fitness=best_fitness,
                higher_is_better=None if best_fitness is None else higher_is_better,
                repo=repo,
                ratchet_ref_name=ratchet_ref_name,
                commit_tree_sha=commit_tree_sha,
                candidate_target_path=candidate_target_path,
                allow_fresh_start_despite_existing_lineage=(
                    allow_fresh_start_despite_existing_lineage
                ),
                callable_name=callable_name,
                concrete_inputs=concrete_inputs,
                commit_parent_sha=bootstrap_commit_sha,
                bootstrap_base_tree_sha=bootstrap_commit_sha,
            )
            iterations.append(result)
            if result.accepted:
                best_fitness = result.fitness_dict

            candidate_source = result.handoff.path.read_text(encoding="utf-8")
            if previous_source is not None:
                structural_flags.append(is_structural_mutation(previous_source, candidate_source))
                decision = assert_diversity_quota(
                    structural_flags, window_size=diversity_window_size
                )
                if decision.leap_path_required:
                    event = LeapPathEvent(iteration_index=candidate_index, decision=decision)
                    leap_path_events.append(event)
                    on_leap_path_required(event)
            previous_source = candidate_source
    finally:
        try:
            watchdog_handle.stop()
        except WatchdogAlreadyStoppedError:
            pass

    return MultiIterationLoopResult(
        iterations=tuple(iterations),
        termination_reason=termination_reason,
        leap_path_events=tuple(leap_path_events),
        watchdog_handle=watchdog_handle,
    )


__all__ = [
    "LeapPathEvent",
    "MultiIterationLoopResult",
    "TerminationReason",
    "run_multi_iteration_loop",
]

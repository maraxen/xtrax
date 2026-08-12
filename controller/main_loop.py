"""Controller main-loop sequencing: one-candidate pass (epic #3611 loop-controller, LC-09, AC-8a).

AC-8a's text (verbatim): "controller main-loop sequencing -- one-candidate pass: dispatch ->
hand off source -> bathos `run` via `AC-5` -> gate checks via #2181's already-merged modules.
AC: a single iteration completes end-to-end against `MockDispatchBackend`." See
`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md` ("AC-8a", "one-
candidate pass").

This module wires together five already-merged pieces into one function, `run_one_candidate_
pass`:

1. **Dispatch** (`controller.dispatch.DispatchBackend`, LC-03): `dispatch_backend.dispatch_
   candidate()` proposes and hands off the next candidate, returning a `CandidateHandoff`.
2. **Lineage-resolved bathos run** (`controller.lineage_interim`, LC-08, wrapping `controller.
   bathos_campaign_adapter.BathosCampaignAdapter`, LC-06): `record_candidate_run` resolves the
   candidate's `parentage` to a single `derived_from` value (fail-loud on a genuine multi-parent
   merge, per AC-7) and records the run via the campaign adapter's `run` MCP call.
3. **Gate checks** (`controller.bathos_library_wrappers`, LC-07, feeding #2181's already-merged
   `xtrax.loop.stats_battery_gate`/`xtrax.loop.seed_gate`): the stats-battery verdict and the
   seed/trial counts are obtained via LC-07's direct-bathos-library wrappers, then assessed by
   the real, already-merged gate functions.
4. **Composed result**: `OneCandidatePassResult` bundles the handoff, the run result, the
   resolved lineage, and both gate decisions -- enough for LC-10 (multi-iteration wiring) and
   LC-11 (error/retry policy) to build on without this module anticipating their own designs.

## Scoping decision: campaign lifecycle (create/conclude) is NOT this function's job

AC-8a's own text says "one-candidate pass," naming exactly the inner sequence: dispatch -> hand
off -> bathos `run` -> gate checks. It does not mention `campaign_create` or `campaign_
conclude`. The architecture spec's own decomposition backs this reading explicitly: AC-8c
("error/retry policy + task_id-bearing telemetry hook", `depends_on: [AC-8a]`) is the item that
owns "`campaign_conclude` (or an equivalent close-out call) fires on every code path that exits
the loop -- success, a caught per-candidate failure, or an uncaught exception." A per-candidate
function cannot itself guarantee "fires on every code path that exits the **loop**" -- that is
a property of the multi-iteration driver (AC-8b/LC-10) wrapping *this* function in a try/finally
or equivalent, not of the single pass itself. Baking a `campaign_create`/`campaign_conclude`
call into `run_one_candidate_pass` would also be actively wrong for a multi-iteration campaign:
`campaign_create` must fire exactly once per campaign, not once per candidate, and this
function's own AC is scoped to "a single iteration."

Therefore: `run_one_candidate_pass` assumes a bathos campaign is **already open** (its
`campaign_id` is a required parameter, forwarded through to `BathosCampaignAdapter.run` via
`record_candidate_run`) and performs only the inner `run` step. Opening the campaign
(`campaign_create`) and concluding it on every exit path (`campaign_conclude`) are left to the
multi-iteration driver -- LC-10 (AC-8b) and LC-11 (AC-8c) respectively, both explicitly out of
this item's scope per its own dispatch brief.

## Scoping decision: which `xtrax.loop` gate modules are wired in

AC-8a's text says "gate checks via #2181's already-merged modules" without naming which ones.
LC-07 (`controller.bathos_library_wrappers`, AC-6) exists specifically to wrap exactly two pure
read/compute gaps: `bathos.stats_gates.run_stats_battery` (feeding `xtrax.loop.stats_battery_
gate.assess_stats_battery_verdict`) and `bathos.campaigns.count_seeds_for_script`/`count_runs_
for_script` (feeding `xtrax.loop.seed_gate.assess_seed_trial_floor`). No other `xtrax.loop` gate
module (e.g. `capability_probe_gate.py`, `multi_metric_ratchet.py`, `admission.py`) has a
corresponding LC-07-style direct-bathos wrapper merged yet -- wiring one of those in here would
mean inventing its own data-sourcing wrapper as a side effect of this item, which is exactly the
kind of scope creep LC-09's own dispatch brief warns against ("make a grounded, defensible
scoping call ... rather than guessing"). This module therefore wires in exactly the two gates
LC-07 provides: the stats-battery gate and the seed/trial-floor gate. Both are conclude-time-
shaped gates in their own module docstrings (they were designed to be called once per campaign
conclude, not once per candidate) -- applying them per-candidate here is a deliberate, bounded
interim usage matching AC-8a's own "gate checks" framing for a *single* pass, not a claim that
this is the gates' final, only call site; a future multi-iteration/conclude-time flow (LC-10/
LC-11) may call `assess_stats_battery_verdict`/`assess_seed_trial_floor` again at true campaign-
conclude time with accumulated data, independent of this per-candidate usage.

## GW-04 addendum: candidate-static gate wired in as a genuine pre-dispatch reject (T2-11, AC-1)

The "Scoping decision" above (LC-09/AC-8a) explicitly deferred `xtrax.loop` gates without an
LC-07-style bathos-data wrapper, precisely because wiring one in here would mean inventing that
wrapper as a side effect. `candidate_static` (T2-11, AC-1, F0) needs no such wrapper: `xtrax.loop.
candidate_static.assert_candidate_static` takes only the already-available `handoff.path` and an
optional `root` for its own jaxlint subprocess -- there is no bathos data-sourcing gap to invent.
Filed and wired here as [GW-04]'s first slice (backlog id 3651): a candidate that fails static
checks (an import error, or a JL-series jaxlint error) is rejected **before** `resolve_derived_
from`/the real bathos `run` call -- AC-1's own "zero GPU time spent" contract, matching `scripts/
smoke_2181_walking_skeleton.py`'s reference sequencing (`assert_candidate_static` called
immediately after a candidate's path is known, before schema/structure/smoke/checkified/prereg).
The remaining five gates in [GW-04]'s pipeline (schema_gate, structure_tripwire, candidate_smoke,
checkified_execution, prereg_match) are explicitly out of this item's scope -- each needs its own
integration surface (a StageBundle slot's declared schema, a live tiny-batch execution, the pinned
uv lockfile, a SafetyManager instance, a bathos prereg sidecar) that this narrowly-scoped change
does not invent.

Like the dispatch/lineage/bathos-run steps above, a `CandidateStaticGateError` propagates
unmodified -- this module still performs zero retry logic (LC-11/AC-8c's own scope).

## Error/retry policy is explicitly NOT this module's job

`run_one_candidate_pass` performs zero retry logic and does not catch any exception raised by
the pieces it sequences: `dispatch_backend.dispatch_candidate()` (`CandidateHandoffFailure`,
`TimeoutError`, `ValueError`), `candidate_static_fn` (`CandidateStaticGateError`, see the GW-04
addendum above), `record_candidate_run`/`resolve_derived_from`
(`MultiParentLineageUnsupportedError`), or `campaign_adapter.run` via `record_candidate_run`
(`BathosMcpToolError`, `BathosMcpTransportError`, `BathosTokenMissingError`). Every one of these
propagates to the caller unmodified. AC-8c (LC-11) owns error/retry policy and the "conclude
fires on every code path" guarantee; this module's job is to prove the happy-path sequence is
wired correctly end-to-end, not to also own what happens when a step fails.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from controller.bathos_campaign_adapter import BathosCampaignAdapter, CandidateRunResult
from controller.bathos_library_wrappers import call_stats_battery_gate, get_seed_trial_counts
from controller.dispatch import CandidateHandoff, DispatchBackend
from controller.evaluate_adapter import BathosFrozenContext, score_raw_artifacts
from controller.lineage_interim import (
    CandidateParentage,
    record_candidate_run,
    resolve_derived_from,
)
from xtrax.loop.candidate_static import assert_candidate_static
from xtrax.loop.closure_lock import guarded_evaluate
from xtrax.loop.compile_time_clock import (
    TwoPhaseTiming,
    assert_no_compile_time_regression,
    measure_two_phase_timing,
)
from xtrax.loop.multi_metric_ratchet import RatchetDecision, compute_ratchet_decision
from xtrax.loop.ratchet_crash_atomicity import (
    advance_best_so_far,
    create_pending_commit,
    read_best_so_far,
    reset_worktree_to_best_so_far,
)
from xtrax.loop.seed_gate import (
    SeedTrialCounts,
    SeedTrialFloorDecision,
    assess_seed_trial_floor,
)
from xtrax.loop.stats_battery_gate import (
    BathosStatsBatteryVerdict,
    ConcludeStatsDecision,
    assess_stats_battery_verdict,
)

# Redefined locally rather than imported from either sibling gate module, matching every
# xtrax.loop gate module's own stated independence-from-siblings convention (see e.g.
# stats_battery_gate.py's docstring) -- controller/ sits outside that package but mirrors the
# same convention rather than picking one sibling's alias as authoritative over the other.
CampaignMode = Literal["exploration", "confirmation", "sequential"]


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Both gate decisions computed for one candidate pass.

    Attributes:
        stats_battery: `xtrax.loop.stats_battery_gate.assess_stats_battery_verdict`'s decision.
        seed_trial: `xtrax.loop.seed_gate.assess_seed_trial_floor`'s decision.
    """

    stats_battery: ConcludeStatsDecision
    seed_trial: SeedTrialFloorDecision

    @property
    def hard_blocked(self) -> bool:
        """True iff either gate hard-blocked this candidate.

        This is the load-bearing signal a failing gate check must actually produce: a
        `confirmation`/`sequential` campaign whose stats-battery verdict downgraded, or whose
        seed/trial floor isn't cleared, hard-blocks here -- it is not silently plumbed through
        and ignored by `OneCandidatePassResult.accepted` below.
        """
        return self.stats_battery.hard_blocked or self.seed_trial.hard_blocked


@dataclass(frozen=True, slots=True)
class OneCandidatePassResult:
    """The composed outcome of one `run_one_candidate_pass` call.

    Deliberately minimal: bundles what LC-10 (multi-iteration wiring) and LC-11 (error/retry
    policy) will need to build on, without this item anticipating either one's own design.

    Attributes:
        handoff: the `CandidateHandoff` returned by the dispatch backend.
        derived_from: the single parent run ID resolved from `parentage` (`""` for a root
            candidate) -- the value actually threaded through to `campaign_adapter.run`.
        run_result: the `CandidateRunResult` from the bathos `run` call.
        gate_outcome: both gate decisions (stats-battery, seed/trial-floor).
        ratchet_decision: the AC-10 multi-metric ratchet verdict (GW-02) computed from this
            candidate's scored fitness dict against `best_fitness`. Populated only when scoring
            succeeds -- an exception raised anywhere upstream (dispatch, the candidate-static
            gate, the bathos run, closure-lock scoring) aborts the whole `run_one_candidate_pass`
            call before any `OneCandidatePassResult` -- partial or otherwise -- is constructed.
            For the first candidate in a campaign (`best_fitness=None`), this is the automatic-
            accept sentinel described in `run_one_candidate_pass`'s own docstring, not a real
            `compute_ratchet_decision` verdict.
    """

    handoff: CandidateHandoff
    derived_from: str
    run_result: CandidateRunResult
    gate_outcome: GateOutcome
    ratchet_decision: RatchetDecision

    @property
    def accepted(self) -> bool:
        """True iff the bathos run succeeded, neither gate hard-blocked, AND the candidate is a
        genuine multi-metric ratchet improvement over the current best (AC-10, GW-02).

        An `advisory`-only downgrade (an `exploration`-mode campaign) does not flip this to
        `False` -- per both gate modules' own docstrings, an advisory downgrade is a normal
        campaign state the loop should proceed past, just surfaced loudly in whatever report a
        downstream caller composes. That composition is out of this function's scope.

        `ratchet_decision.improved` is the load-bearing addition GW-02 makes to this property: a
        candidate strictly worse on every metric than the current best used to still read
        `accepted=True` here as long as the run succeeded and neither gate hard-blocked -- that
        was exactly the bug this item fixes (see the module docstring). For the FIRST candidate
        in a campaign (no prior best-so-far exists yet -- `run_one_candidate_pass` was called
        with `best_fitness=None`/`higher_is_better=None`), `ratchet_decision` is the automatic-
        accept sentinel constructed directly rather than computed via `compute_ratchet_decision`
        -- there is nothing to ratchet against yet, so a first candidate is never rejected on
        ratchet grounds alone.
        """
        return (
            self.run_result.success
            and not self.gate_outcome.hard_blocked
            and self.ratchet_decision.improved
        )


def run_one_candidate_pass(
    dispatch_backend: DispatchBackend,
    campaign_adapter: BathosCampaignAdapter,
    *,
    campaign_id: str,
    campaign_mode: CampaignMode,
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
    frozen_context: BathosFrozenContext,
    current_config: Mapping[str, Any],
    candidate_touched_paths: frozenset[Path] = frozenset(),
    guarded_evaluate_fn: Callable[..., dict[str, float]] = guarded_evaluate,
    best_fitness: Mapping[str, float] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
    repo: Path,
    ratchet_ref_name: str,
    commit_tree_sha: str,
    commit_parent_sha: str | None = None,
    commit_message: str | None = None,
    callable_name: str,
    concrete_inputs: list[Any],
    compile_time_history: Sequence[float] = (),
    k_threshold: float = 3.0,
    measure_two_phase_timing_fn: Callable[..., TwoPhaseTiming] = measure_two_phase_timing,
) -> OneCandidatePassResult:
    """Run one candidate through the full dispatch -> bathos-run -> gate-check sequence.

    See the module docstring for the campaign-lifecycle and gate-module scoping decisions, and
    for why error/retry handling is deliberately absent here.

    Args:
        dispatch_backend: the `DispatchBackend` (LC-03) to propose this candidate. Its own
            `dispatch_candidate()` exceptions (`CandidateHandoffFailure`, `TimeoutError`,
            `ValueError`) propagate unmodified.
        campaign_adapter: the `BathosCampaignAdapter` (LC-06) to record the run through. The
            caller is responsible for having already called `campaign_create` -- see the module
            docstring's scoping decision.
        campaign_id: the already-open campaign's ID, forwarded to `campaign_adapter.run`.
        campaign_mode: the campaign's mode, forwarded to both gate-assessment calls.
        parentage: this candidate's proposed parentage (LC-08). Defaults to a root candidate
            (no parents). Resolved to a single `derived_from` value via `resolve_derived_from`
            *before* any bathos call -- a genuine multi-parent `parentage` raises
            `MultiParentLineageUnsupportedError` here, before `record_candidate_run` (and so
            before `campaign_adapter.run`) is ever reached.
        run_args: forwarded to `record_candidate_run`/`campaign_adapter.run`.
        output_paths: forwarded to `record_candidate_run`/`campaign_adapter.run`.
        tags: forwarded to `record_candidate_run`/`campaign_adapter.run`.
        agent_mode: forwarded to `record_candidate_run`/`campaign_adapter.run`.
        no_sidecar: forwarded to `record_candidate_run`/`campaign_adapter.run`.
        stats_battery_kwargs: keyword arguments forwarded verbatim to `stats_battery_fn`
            (default: `call_stats_battery_gate`) -- e.g. `candidate_values`/`baseline_values`
            and the optional HPO-budget fields. This function does not compute or source these
            values itself (neither does LC-07's own wrapper, nor `xtrax.loop.stats_battery_
            gate` -- see both modules' docstrings): obtaining the real statistics from a
            candidate's run output is explicitly outside this item's scope, deferred to
            whatever caller (LC-10/LC-12) has real run data to supply.
        seed_trial_db: forwarded as `seed_trial_counts_fn`'s `db` argument (default:
            `get_seed_trial_counts`) -- a bathos database connection, caller-supplied (see
            `get_seed_trial_counts`'s own docstring: "caller's responsibility to provide"). May
            be `None` when `seed_trial_counts_fn` is replaced by a test double that ignores it.
        hypothesis_clause_id: forwarded to `seed_trial_counts_fn`, then threaded through to the
            resulting `SeedTrialFloorDecision` for logging/traceability only.
        stats_battery_fn: injection seam for tests -- defaults to LC-07's real
            `call_stats_battery_gate` (which lazily imports bathos, no MCP call).
        seed_trial_counts_fn: injection seam for tests -- defaults to LC-07's real
            `get_seed_trial_counts` (which lazily imports bathos, no MCP call).
        candidate_static_fn: injection seam for tests -- defaults to T2-11's real
            `assert_candidate_static` (clean import + zero jaxlint JL-series errors). Called
            immediately after dispatch, before lineage resolution or the real bathos run -- see
            the module docstring's GW-04 addendum.
        candidate_static_root: forwarded to `candidate_static_fn` as its `root` kwarg (jaxlint's
            subprocess root; default `None` lets jaxlint fall back to `Path.cwd()`).
        frozen_context: the fixed `BathosFrozenContext` (SPLIT_COMPUTE, #4133/#3657) this
            candidate's raw artifacts are scored against -- `frozen_context.locked` is the
            closure-lock's `ClosureManifest`, forwarded to `guarded_evaluate_fn` unchanged.
        current_config: this iteration's actual in-effect config, forwarded to
            `guarded_evaluate_fn` -- deliberately distinct from `frozen_context.locked.config`
            (see `verify_closure`'s own docstring: recomputing against the locked manifest's own
            stored config would make config drift undetectable by construction).
        candidate_touched_paths: forwarded to `guarded_evaluate_fn` unchanged (default: empty,
            matching `guarded_evaluate`'s own default) -- paths this candidate is known to have
            mutated, checked against the locked closure's protected-path set.
        guarded_evaluate_fn: injection seam for tests -- defaults to the real
            `xtrax.loop.closure_lock.guarded_evaluate`. Called with `frozen_context.locked`,
            `score_raw_artifacts`, `frozen_context`, `tuple(output_paths)`,
            `current_config=current_config`, `candidate_touched_paths=candidate_touched_paths` --
            never `frozen_context.score_fn` or `score_raw_artifacts` called directly outside this
            seam. `ClosureHashMismatchError`/`EvaluatorClosureDriftError`/`UnlistedReadError`
            propagate unmodified (see the module docstring: this is a non-recoverable HALT
            signal, not a candidate-rejection result).
        best_fitness: the current best-so-far candidate's fitness dict, or `None` for the first
            candidate in a campaign (no prior best to ratchet against). Must be supplied
            together with `higher_is_better` (both `None` or both set) -- a `ValueError` is
            raised immediately if only one is supplied. When both are `None`,
            `compute_ratchet_decision` is never called; `OneCandidatePassResult.ratchet_decision`
            is instead the automatic-accept sentinel described on that field.
        higher_is_better: per-metric direction for `compute_ratchet_decision` -- see that
            function's own docstring. Must be supplied together with `best_fitness`.
        repo: the git repository whose worktree/ref this campaign ratchets best-so-far lineage
            against, forwarded to `read_best_so_far`/`create_pending_commit`/
            `advance_best_so_far`/`reset_worktree_to_best_so_far`.
        ratchet_ref_name: the git ref name tracking best-so-far (T2-10, AC-14) -- the SAME
            `repo`/`ratchet_ref_name` pair is reused for both the accept path (`create_pending_
            commit`/`advance_best_so_far`) and the reject path (`reset_worktree_to_best_so_far`),
            one source of truth for which repo/ref this campaign ratchets against.
        commit_tree_sha: real git tree data the caller supplies for the accept-path pending
            commit -- this function does not construct or verify it (mirrors the
            `stats_battery_kwargs` trust boundary: caller-sourced, not computed here).
        commit_parent_sha: bootstrap fallback parent SHA, used only when `read_best_so_far`
            returns `None` (no prior best-so-far ref exists yet). A `ValueError` is raised if
            both are `None` on an accept.
        commit_message: optional commit message override for the accept-path pending commit;
            defaults to an auto-generated message naming the candidate's content SHA and
            resolved lineage.
        callable_name: the attribute name on the candidate module `measure_two_phase_timing_fn`
            resolves and times, forwarded unchanged.
        concrete_inputs: real arguments reused identically for both of `measure_two_phase_timing_
            fn`'s timed calls, forwarded unchanged.
        compile_time_history: the caller-maintained rolling window of prior compile times,
            forwarded to `assert_no_compile_time_regression` (default: empty tuple, that
            function's own documented no-op case -- this function does not compute or persist
            this history itself).
        k_threshold: forwarded to `assert_no_compile_time_regression` (default 3.0, matching that
            module's own default).
        measure_two_phase_timing_fn: injection seam for tests -- defaults to T2-19's real
            `xtrax.loop.compile_time_clock.measure_two_phase_timing`. Called with
            `handoff.path` (never a separately constructed path) and `callable_name`, with
            `concrete_inputs=concrete_inputs`.

    Returns:
        A `OneCandidatePassResult` bundling the handoff, resolved lineage, run result, both gate
        decisions, and the AC-10 multi-metric ratchet decision.

    Raises:
        CandidateHandoffFailure: dispatch's own staging-write failure.
        TimeoutError: dispatch timed out.
        ValueError: dispatch's completion could not be parsed; `output_paths` is empty/`None`
            when scoring is attempted; `best_fitness`/`higher_is_better` were not both supplied
            or both omitted; or `commit_parent_sha` was not supplied on an accept when no prior
            best-so-far ref exists yet.
        CandidateStaticGateError: the candidate fails clean-import or has a JL-series jaxlint
            error -- raised right after dispatch, before lineage resolution or any bathos call
            (T2-11, AC-1).
        MultiParentLineageUnsupportedError: `parentage` names more than one distinct, real
            parent run ID -- raised before any bathos call.
        BathosMcpToolError: `campaign_adapter.run` itself failed (bathos-side validation or the
            script run reported failure).
        BathosMcpTransportError: the MCP round-trip to bathos failed.
        BathosTokenMissingError: no local bathos MCP write-token is available.
        ClosureHashMismatchError, ProtectedPathMutatedError, UnlistedReadError: closure-lock
            drift detected by `guarded_evaluate_fn` (AC-7) -- non-recoverable, propagates
            unmodified (see the module docstring's HALT-not-retry stance).
        RatchetInputError: `compute_ratchet_decision`'s own structural validation failure
            (mismatched metric keys, or fewer than 2 metrics).
        CommitCreationError, RefUpdateConflictError: the accept-path crash-atomicity commit/ref
            update failed (T2-10, AC-14) -- propagates unmodified.
        RatchetCrashAtomicityError: the reject-path worktree reset failed, or (unreachable in
            normal operation) no prior best-so-far ref existed to reset to.
        CompileTimeRegressionError: this candidate's compile time exceeded `k_threshold *
            rolling_median(compile_time_history)` (AC-27) -- raised independently of, and never
            altering, the ratchet accept/reject outcome above.
        StatsBatteryGateInputError: `campaign_mode` is not a recognized `CampaignMode` (raised
            by `assess_stats_battery_verdict`).
        SeedGateInputError: `campaign_mode` is not recognized, or the seed/trial counts are
            malformed (raised by `assess_seed_trial_floor`).
    """
    # 1. Dispatch -> hand off source (LC-03).
    handoff = dispatch_backend.dispatch_candidate()

    # 1.5. Candidate-static gate (T2-11, AC-1, F0; [GW-04] first slice) -- reject a candidate
    # that fails clean-import or jaxlint JL-series checks before lineage resolution or any real
    # bathos run is attempted. See the module docstring's GW-04 addendum.
    candidate_static_fn(handoff.path, root=candidate_static_root)

    # Resolved directly here (not only inside record_candidate_run) so a genuine multi-parent
    # parentage fails loud before the bathos-run step is even attempted, and so the resolved
    # value is available on the composed result without record_candidate_run needing to expose
    # it. resolve_derived_from is a pure, side-effect-free function of `parentage` -- calling it
    # twice (once here, once again inside record_candidate_run below) recomputes the same
    # answer, not a behavior change.
    derived_from = resolve_derived_from(parentage)

    # 2. Lineage-resolved bathos run (LC-08 wrapping LC-06). Assumes campaign_id names an
    # already-open campaign -- see the module docstring's campaign-lifecycle scoping decision.
    run_result: CandidateRunResult = record_candidate_run(
        campaign_adapter,
        str(handoff.path),
        parentage,
        args=run_args,
        campaign_id=campaign_id,
        output_paths=output_paths,
        tags=tags,
        agent_mode=agent_mode,
        no_sidecar=no_sidecar,
    )

    # 2.5. Score raw artifacts through the AC-7 closure-lock gate (S2.1, GW-02). output_paths is
    # this function's OWN existing parameter -- CandidateRunResult carries no output_paths field
    # to read this off of instead. guarded_evaluate_fn is the sole call site through which
    # frozen_context.score_fn/score_raw_artifacts may be reached; ClosureHashMismatchError/
    # EvaluatorClosureDriftError/UnlistedReadError propagate uncaught (non-recoverable HALT, per
    # closure_lock's own module docstring -- never wrapped in try/except here).
    if not output_paths:
        msg = (
            "output_paths must be non-empty to score a candidate's raw artifacts "
            "(guarded_evaluate_fn has nothing to read otherwise)"
        )
        raise ValueError(msg)

    fitness_dict = guarded_evaluate_fn(
        frozen_context.locked,
        score_raw_artifacts,
        frozen_context,
        tuple(output_paths),
        current_config=current_config,
        candidate_touched_paths=candidate_touched_paths,
    )

    # 2.6. Multi-metric ratchet decision (S2.1b, AC-10). best_fitness/higher_is_better must both
    # be supplied or both omitted -- validated before either the ratchet or the crash-atomicity
    # step below is ever reached.
    if (best_fitness is None) != (higher_is_better is None):
        msg = (
            f"best_fitness and higher_is_better must both be supplied or both omitted "
            f"(got best_fitness={'set' if best_fitness is not None else None}, "
            f"higher_is_better={'set' if higher_is_better is not None else None})"
        )
        raise ValueError(msg)

    if best_fitness is None:
        # First candidate in a campaign: nothing to ratchet against yet -- automatic accept
        # sentinel, never a real compute_ratchet_decision verdict. See OneCandidatePassResult.
        # ratchet_decision's own docstring.
        ratchet_decision = RatchetDecision(
            improved=True,
            win_rate=1.0,
            breakdown_point=1.0,
            cohens_d=math.inf,
            per_metric_delta={},
        )
    else:
        ratchet_decision = compute_ratchet_decision(
            fitness_dict, best_fitness, higher_is_better=higher_is_better
        )

    # 2.7. Crash-safe best-so-far lineage (S2.1c/S2.4, AC-14) -- accept vs reject branch, driven
    # solely by ratchet_decision.improved (whether from a real win or the sentinel above).
    if ratchet_decision.improved:
        prior_best_sha = read_best_so_far(repo, ratchet_ref_name)
        parent_sha = prior_best_sha if prior_best_sha is not None else commit_parent_sha
        if parent_sha is None:
            msg = (
                f"commit_parent_sha must be supplied when no prior best-so-far ref exists yet "
                f"(read_best_so_far({ratchet_ref_name!r}) returned None)"
            )
            raise ValueError(msg)
        message = commit_message or (
            f"ratchet best-so-far: candidate {handoff.content_sha256}"
            + (f" derived_from={derived_from}" if derived_from else " (root candidate)")
        )
        # prior_best_sha serves as BOTH the parent-SHA source AND the CAS expected_old_sha --
        # one read_best_so_far call above, not two. CommitCreationError/RefUpdateConflictError
        # propagate uncaught.
        pending_sha = create_pending_commit(repo, commit_tree_sha, parent_sha, message)
        advance_best_so_far(repo, ratchet_ref_name, pending_sha, prior_best_sha)
    else:
        # RatchetCrashAtomicityError propagates uncaught. reset_worktree_to_best_so_far reuses
        # the same repo/ratchet_ref_name as the accept branch above -- one source of truth for
        # which repo/ref this campaign ratchets against.
        reset_worktree_to_best_so_far(repo, ratchet_ref_name)

    # 3. Gate checks (LC-07 wrappers feeding #2181's already-merged xtrax.loop gates).
    stats_verdict = stats_battery_fn(**dict(stats_battery_kwargs))
    stats_decision = assess_stats_battery_verdict(stats_verdict, campaign_mode=campaign_mode)

    seed_counts = seed_trial_counts_fn(seed_trial_db, handoff.content_sha256, hypothesis_clause_id)
    seed_decision = assess_seed_trial_floor(seed_counts, campaign_mode=campaign_mode)

    # 4. Compile-time two-phase clock (S2.2, AC-27) -- strictly AFTER the accept/reject decision
    # above, never an input to it. compile_time_seconds is intentionally never merged into
    # fitness_dict / never passed to compute_ratchet_decision -- only runtime_seconds is
    # fitness-comparable. CompileTimeRegressionError is raised/escalated independently, without
    # altering the ratchet accept/reject outcome computed above.
    timing = measure_two_phase_timing_fn(
        handoff.path, callable_name, concrete_inputs=concrete_inputs
    )
    assert_no_compile_time_regression(
        timing.compile_time_seconds,
        compile_time_history=compile_time_history,
        k_threshold=k_threshold,
    )

    # 5. Composed result.
    return OneCandidatePassResult(
        handoff=handoff,
        derived_from=derived_from,
        run_result=run_result,
        gate_outcome=GateOutcome(stats_battery=stats_decision, seed_trial=seed_decision),
        ratchet_decision=ratchet_decision,
    )


__all__ = [
    "CampaignMode",
    "GateOutcome",
    "OneCandidatePassResult",
    "run_one_candidate_pass",
]

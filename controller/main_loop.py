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

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from controller.bathos_campaign_adapter import BathosCampaignAdapter, CandidateRunResult
from controller.bathos_library_wrappers import call_stats_battery_gate, get_seed_trial_counts
from controller.dispatch import CandidateHandoff, DispatchBackend
from controller.lineage_interim import (
    CandidateParentage,
    record_candidate_run,
    resolve_derived_from,
)
from xtrax.loop.candidate_static import assert_candidate_static
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
    """

    handoff: CandidateHandoff
    derived_from: str
    run_result: CandidateRunResult
    gate_outcome: GateOutcome

    @property
    def accepted(self) -> bool:
        """True iff the bathos run itself succeeded AND neither gate hard-blocked.

        An `advisory`-only downgrade (an `exploration`-mode campaign) does not flip this to
        `False` -- per both gate modules' own docstrings, an advisory downgrade is a normal
        campaign state the loop should proceed past, just surfaced loudly in whatever report a
        downstream caller composes. That composition is out of this function's scope.
        """
        return self.run_result.success and not self.gate_outcome.hard_blocked


def _emit_candidate_pass_probe_record(
    *,
    out_dir: Path,
    campaign_id: str,
    derived_from: str,
    handoff_sha: str,
    wall_seconds: float,
    accepted: bool,
    hard_blocked: bool,
) -> Path:
    """Emit a Stage-0 provenance ProbeRecord for one candidate pass (Phase C).

    Opt-in via ``probe_record_dir``; the only bathos-adjacent site allowed to
    attach probe records to campaign context (scope doc D7). Written once ALL
    gates have resolved -- every COMPLETED pass gets one (including
    hard-blocked ones; the outcome is in the config). STRUCTURAL-only:
    wall_seconds is a host-side dispatch+run+gates duration, not a device
    measurement, so the record deliberately carries no scopes/attribution and
    cannot back DISPATCH_COUNT or ranking claims.
    """
    from xtrax.profiling.emitters import emit_probe_record

    slug = f"{campaign_id}" if campaign_id else "candidate"
    if derived_from:
        slug = f"{slug}__{derived_from}"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)[-80:]
    path = out_dir / f"pass_{safe}.json"
    emit_probe_record(
        path=path,
        probe_id=f"candidate_pass_{safe}",
        stage=0,
        n_atoms=1,
        platform="cpu",
        metrics={"wall_seconds": wall_seconds},
        config={
            "campaign_id": campaign_id,
            "derived_from": derived_from,
            "handoff_content_sha256": handoff_sha,
            "accepted": str(accepted).lower(),
            "hard_blocked": str(hard_blocked).lower(),
            "source": "controller.run_one_candidate_pass",
            "axis_note": (
                "provenance artifact; n_atoms placeholder by contract"
            ),
        },
    )
    return path


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
    probe_record_dir: Path | None = None,
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
        probe_record_dir: opt-in Phase C seam -- when set, one Stage-0 provenance ProbeRecord
            (wall-clock pass duration + campaign/run identity) is written under this directory
            after the gates resolve. `None` (default) emits nothing.

    Returns:
        A `OneCandidatePassResult` bundling the handoff, resolved lineage, run result, and both
        gate decisions.

    Raises:
        CandidateHandoffFailure: dispatch's own staging-write failure.
        TimeoutError: dispatch timed out.
        ValueError: dispatch's completion could not be parsed.
        CandidateStaticGateError: the candidate fails clean-import or has a JL-series jaxlint
            error -- raised right after dispatch, before lineage resolution or any bathos call
            (T2-11, AC-1).
        MultiParentLineageUnsupportedError: `parentage` names more than one distinct, real
            parent run ID -- raised before any bathos call.
        BathosMcpToolError: `campaign_adapter.run` itself failed (bathos-side validation or the
            script run reported failure).
        BathosMcpTransportError: the MCP round-trip to bathos failed.
        BathosTokenMissingError: no local bathos MCP write-token is available.
        StatsBatteryGateInputError: `campaign_mode` is not a recognized `CampaignMode` (raised
            by `assess_stats_battery_verdict`).
        SeedGateInputError: `campaign_mode` is not recognized, or the seed/trial counts are
            malformed (raised by `assess_seed_trial_floor`).
    """
    # 1. Dispatch -> hand off source (LC-03).
    pass_started_at = perf_counter()
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

    # 3. Gate checks (LC-07 wrappers feeding #2181's already-merged xtrax.loop gates).
    stats_verdict = stats_battery_fn(**dict(stats_battery_kwargs))
    stats_decision = assess_stats_battery_verdict(stats_verdict, campaign_mode=campaign_mode)

    seed_counts = seed_trial_counts_fn(seed_trial_db, handoff.content_sha256, hypothesis_clause_id)
    seed_decision = assess_seed_trial_floor(seed_counts, campaign_mode=campaign_mode)

    # 3.5. Optional provenance record (Phase C, scope doc D7): written only when
    # the caller opted in, AFTER all gates resolve so the artifact always
    # carries their verdicts. Emission failures are CONTAINED: a completed
    # pass (dispatch + run + gates all done) must never be discarded because
    # provenance bookkeeping failed -- the pass result stands, the missing
    # record is reported loudly instead.
    if probe_record_dir is not None:
        try:
            probe_record_dir.mkdir(parents=True, exist_ok=True)
            _emit_candidate_pass_probe_record(
                out_dir=probe_record_dir,
                campaign_id=campaign_id,
                derived_from=derived_from,
                handoff_sha=handoff.content_sha256,
                wall_seconds=perf_counter() - pass_started_at,
                accepted=(
                    run_result.success
                    and stats_decision.honored
                    and seed_decision.held
                ),
                hard_blocked=(
                    stats_decision.hard_blocked or seed_decision.hard_blocked
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- contained by design; see comment above
            print(
                f"WARNING: candidate-pass ProbeRecord emission failed for "
                f"campaign {campaign_id!r}; pass result unaffected: {exc}",
                file=sys.stderr,
            )

    # 4. Composed result.
    return OneCandidatePassResult(
        handoff=handoff,
        derived_from=derived_from,
        run_result=run_result,
        gate_outcome=GateOutcome(stats_battery=stats_decision, seed_trial=seed_decision),
    )


__all__ = [
    "CampaignMode",
    "GateOutcome",
    "OneCandidatePassResult",
    "run_one_candidate_pass",
]

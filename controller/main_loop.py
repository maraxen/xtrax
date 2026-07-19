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

## GW-01 addendum: post-run integrity/provenance gates (attestation_evidence_gate, sidecar_drift)

Backlog id 3648 ([GW-01]) named 5 gates to wire in post-run: closure_lock, metrics_provenance,
info_barrier_lint, attestation_evidence_gate, sidecar_drift_gate. Verified during implementation
(2026-07-19) that the first 3 assume the loop controller invokes an evaluator IN-PROCESS
(`xtrax.loop.closure_lock.guarded_evaluate(locked, evaluator: EvaluateFn, ...)` calls `evaluator`
directly) -- an execution model this controller does not have: candidates are dispatched to bathos
as OUT-OF-PROCESS script runs (`record_candidate_run`/`campaign_adapter.run` above), never through
`xtrax.stages.evaluate.EvaluateFn`/`seal_evaluator` anywhere in `controller/`. Wiring those 3 as
literally shown in `scripts/smoke_2181_walking_skeleton.py` (a synthetic, no-bathos, in-process-only
demo per its own docstring) would not actually provenance this controller's real bathos-run data --
a gate that doesn't check the real thing is worse than no gate (false confidence). Split out as a
new backlog item (id 3657) pending an architecture decision; NOT wired here.

The remaining 2 gates genuinely fit this controller's existing bathos-dispatch architecture --
both are pure decision functions over caller-supplied, already-computed bathos data, the same shape
LC-07's wrappers already use for stats-battery/seed-floor:

- `attestation_evidence_gate.admit_evidence` (T2-26, AC-19): filters a collection of candidate
  runs (here, always a 1-element collection: this candidate's own just-completed run) down to
  those whose manifest verifies. Fed by the new `get_evidence_candidate_for_run` wrapper
  (`controller.bathos_library_wrappers`), which calls `bathos.query.get_run` +
  `bathos.prereg.verify_run_manifest` directly (no MCP), mirroring LC-07's exact convention.
- `sidecar_drift_gate.assert_sidecar_drift_reaction` (T2-25, AC-18): reacts to a sidecar-hash
  drift signal for this candidate's script, fed by the new `get_sidecar_drift_signal` wrapper
  (same module, same direct-library-import convention). Denies (raises `SidecarHashMismatchError`,
  propagating unmodified like every other raise in this function) only under
  `sidecar_drift_agent_mode="autonomous"`; otherwise returns a decision, `should_warn` surfaced but
  non-blocking. `sidecar_drift_agent_mode` is a **distinct parameter** from the existing
  `agent_mode: str` (forwarded to bathos's `run` call) -- the gate module's own docstring is
  explicit that `AgentMode` (collaborative/autonomous) and `CampaignMode` are orthogonal axes, and
  by the same reasoning this is orthogonal to whatever free-form `agent_mode` string bathos's `run`
  tool receives; conflating the two into one parameter would silently couple unrelated concerns.
  Defaults to `"collaborative"` (warn-only) -- the safer rollout choice for a gate with no
  existing production caller yet setting any particular mode; a caller wanting the harder
  `"autonomous"` denial opts in explicitly.

`admit_evidence` itself has no `campaign_mode` concept (by design -- it is a pure set-filtering
concern, see its own docstring). This controller therefore composes the mode-awareness itself, in
`EvidenceIntegrityOutcome`, mirroring `stats_battery_gate`/`seed_gate`'s own established convention:
a run whose manifest fails verification is `advisory`-only for an `exploration` campaign,
`hard_blocked` for `confirmation`/`sequential`. Deliberately keyed off `EvidenceCandidate.
manifest_verified` directly, NOT off `admit_evidence`'s own `excluded` collection: with today's
real bathos, nothing anywhere independently captures and re-hashes a run's stdout, so
`stdout_verified` is always `None` for every real call -- and `admit_evidence` itself always
treats that as an exclusion reason (`stdout_hash_not_recorded`), even when the manifest is
genuinely valid. Keying `hard_blocked` off `excluded` (as this addendum's first draft did, caught
during implementation -- see the `EvidenceIntegrityOutcome` docstring) would have permanently
hard-blocked every non-exploration candidate forever, for a systemic capability gap that no caller
can currently close, not a real per-candidate integrity failure. `manifest_verified` is the one
sub-check bathos can genuinely answer today; the full `EvidenceAdmissionResult` is still surfaced
on the outcome for observability, just not used to drive this decision.

Both new gates run immediately after the bathos run (`record_candidate_run`) and *before* the
stats-battery/seed-floor checks -- checking whether this run's own evidence is trustworthy at all
is a logical precondition to checking whether its statistics look good, matching [GW-01]'s own
"post-run integrity/provenance gates" framing as a prerequisite tier.

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from controller.bathos_campaign_adapter import BathosCampaignAdapter, CandidateRunResult
from controller.bathos_library_wrappers import (
    call_stats_battery_gate,
    get_evidence_candidate_for_run,
    get_seed_trial_counts,
    get_sidecar_drift_signal,
)
from controller.dispatch import CandidateHandoff, DispatchBackend
from controller.lineage_interim import (
    CandidateParentage,
    record_candidate_run,
    resolve_derived_from,
)
from xtrax.loop.attestation_evidence_gate import (
    EvidenceAdmissionResult,
    EvidenceCandidate,
    admit_evidence,
)
from xtrax.loop.candidate_static import assert_candidate_static
from xtrax.loop.seed_gate import (
    SeedTrialCounts,
    SeedTrialFloorDecision,
    assess_seed_trial_floor,
)
from xtrax.loop.sidecar_drift_gate import (
    AgentMode as SidecarAgentMode,
)
from xtrax.loop.sidecar_drift_gate import (
    SidecarDriftDecision,
    SidecarDriftSignal,
    assert_sidecar_drift_reaction,
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
class EvidenceIntegrityOutcome:
    """This candidate's post-run integrity/provenance decision ([GW-01], T2-26/T2-25).

    See the module docstring's GW-01 addendum for why `hard_blocked`/`advisory` are computed
    here (a controller-level composition) rather than by `admit_evidence` itself (which has no
    `campaign_mode` concept -- pure set-filtering, by design).

    Attributes:
        evidence: the real `xtrax.loop.attestation_evidence_gate.admit_evidence` result for
            this candidate's single run (always a 1-element input collection) -- surfaced for
            observability (e.g. a later cross-run evidence audit); does NOT itself drive
            `hard_blocked`/`advisory` below, see their own docstrings for why.
        sidecar_drift: the real `xtrax.loop.sidecar_drift_gate.assert_sidecar_drift_reaction`
            decision. Only ever populated for a non-denying outcome -- a denying outcome
            (`sidecar_drift_agent_mode="autonomous"` and drift detected) raises
            `SidecarHashMismatchError` instead, before this outcome is ever constructed.
        hard_blocked: True iff `manifest_verified` was False on the input `EvidenceCandidate`
            AND `campaign_mode` is not `"exploration"` -- mirrors `stats_battery_gate`/
            `seed_gate`'s own advisory-for-exploration, hard-blocking-otherwise convention.
            Deliberately NOT keyed off `evidence.excluded` (see the module docstring's GW-01
            addendum): with today's real bathos, `stdout_verified` is always `None` for every
            real run (nothing anywhere independently captures+hashes stdout), which
            `admit_evidence` itself always treats as an exclusion reason even when the
            manifest is genuinely valid -- that would permanently hard-block every
            non-exploration candidate forever, for a systemic capability gap, not a real
            integrity failure.
        advisory: True iff `manifest_verified` was False but `campaign_mode` IS
            `"exploration"` (surfaced, not blocking).
    """

    evidence: EvidenceAdmissionResult
    sidecar_drift: SidecarDriftDecision
    hard_blocked: bool
    advisory: bool


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """All gate decisions computed for one candidate pass.

    Attributes:
        stats_battery: `xtrax.loop.stats_battery_gate.assess_stats_battery_verdict`'s decision.
        seed_trial: `xtrax.loop.seed_gate.assess_seed_trial_floor`'s decision.
        evidence_integrity: the post-run integrity/provenance decision ([GW-01]) -- see
            `EvidenceIntegrityOutcome`.
    """

    stats_battery: ConcludeStatsDecision
    seed_trial: SeedTrialFloorDecision
    evidence_integrity: EvidenceIntegrityOutcome

    @property
    def hard_blocked(self) -> bool:
        """True iff any gate hard-blocked this candidate.

        This is the load-bearing signal a failing gate check must actually produce: a
        `confirmation`/`sequential` campaign whose stats-battery verdict downgraded, whose
        seed/trial floor isn't cleared, or whose run's manifest fails verification, hard-blocks
        here -- it is not silently plumbed through and ignored by
        `OneCandidatePassResult.accepted` below.
        """
        return (
            self.stats_battery.hard_blocked
            or self.seed_trial.hard_blocked
            or self.evidence_integrity.hard_blocked
        )


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
    catalog_dir: str = "",
    evidence_candidate_fn: Callable[..., EvidenceCandidate] = get_evidence_candidate_for_run,
    stdout_verified: bool | None = None,
    sidecar_drift_signal_fn: Callable[..., SidecarDriftSignal] = get_sidecar_drift_signal,
    sidecar_drift_agent_mode: SidecarAgentMode = "collaborative",
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
        catalog_dir: bathos catalog directory, forwarded to `evidence_candidate_fn`/
            `sidecar_drift_signal_fn` (default `""` resolves the same way bathos's own MCP
            layer does -- see `controller.bathos_library_wrappers._resolve_catalog_dir`).
        evidence_candidate_fn: injection seam -- defaults to [GW-01]'s real
            `get_evidence_candidate_for_run` (fetches this run's bathos `Run` + calls
            `verify_run_manifest`, no MCP). Called with this run's `run_id` right after the
            bathos run, before the stats-battery/seed-floor checks -- see the module
            docstring's GW-01 addendum.
        stdout_verified: forwarded to `evidence_candidate_fn` -- caller-supplied result of
            independently re-hashing a stored stdout artifact against `Run.stdout_sha256`
            (bathos does not persist raw stdout to re-hash against itself). Defaults to
            `None` ("not recorded/not checked").
        sidecar_drift_signal_fn: injection seam -- defaults to [GW-01]'s real
            `get_sidecar_drift_signal` (fetches this run's sidecar hash + calls
            `check_sidecar_drift`, no MCP).
        sidecar_drift_agent_mode: `"autonomous"` denies (raises) on a detected sidecar-hash
            drift; `"collaborative"` (default) only warns. A distinct axis from `agent_mode`
            above and from `campaign_mode` -- see the module docstring's GW-01 addendum for
            why this is not derived from either.

    Returns:
        A `OneCandidatePassResult` bundling the handoff, resolved lineage, run result, and all
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
        SidecarHashMismatchError: a sidecar-hash drift was detected for this candidate's
            script AND `sidecar_drift_agent_mode == "autonomous"` (T2-25, AC-18) -- a
            structural anomaly, not a normal candidate rejection; propagates unmodified.
        SidecarDriftGateInputError: `sidecar_drift_agent_mode` is not a recognized `AgentMode`.
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

    # 3. Post-run integrity/provenance gates ([GW-01] first slice: attestation_evidence_gate +
    # sidecar_drift_gate). Run before the stats-battery/seed-floor checks -- see the module
    # docstring's GW-01 addendum for why (evidence trustworthiness is a precondition to
    # judging the statistics). sidecar_drift's deny path raises SidecarHashMismatchError here,
    # propagating unmodified like every other raise in this function.
    evidence_candidate = evidence_candidate_fn(
        run_result.run_id, catalog_dir=catalog_dir, stdout_verified=stdout_verified
    )
    evidence_result = admit_evidence([evidence_candidate])
    # Deliberately NOT `len(evidence_result.excluded) > 0` -- see the module docstring's GW-01
    # addendum: with today's real bathos, nothing anywhere independently captures+hashes stdout,
    # so `stdout_verified` is always `None` for every real run, which `admit_evidence` itself
    # always treats as an exclusion reason (`stdout_hash_not_recorded`) even when the manifest is
    # genuinely valid -- keying hard_blocked off `excluded` would permanently hard-block every
    # non-exploration candidate forever, for a systemic capability gap, not a real integrity
    # failure. `manifest_verified` is the one sub-check bathos can genuinely answer today.
    manifest_unverified = not evidence_candidate.manifest_verified

    sidecar_signal = sidecar_drift_signal_fn(
        handoff.path, run_result.run_id, catalog_dir=catalog_dir
    )
    sidecar_decision = assert_sidecar_drift_reaction(
        sidecar_signal, agent_mode=sidecar_drift_agent_mode
    )

    evidence_integrity = EvidenceIntegrityOutcome(
        evidence=evidence_result,
        sidecar_drift=sidecar_decision,
        hard_blocked=manifest_unverified and campaign_mode != "exploration",
        advisory=manifest_unverified and campaign_mode == "exploration",
    )

    # 4. Gate checks (LC-07 wrappers feeding #2181's already-merged xtrax.loop gates).
    stats_verdict = stats_battery_fn(**dict(stats_battery_kwargs))
    stats_decision = assess_stats_battery_verdict(stats_verdict, campaign_mode=campaign_mode)

    seed_counts = seed_trial_counts_fn(seed_trial_db, handoff.content_sha256, hypothesis_clause_id)
    seed_decision = assess_seed_trial_floor(seed_counts, campaign_mode=campaign_mode)

    # 5. Composed result.
    return OneCandidatePassResult(
        handoff=handoff,
        derived_from=derived_from,
        run_result=run_result,
        gate_outcome=GateOutcome(
            stats_battery=stats_decision,
            seed_trial=seed_decision,
            evidence_integrity=evidence_integrity,
        ),
    )


__all__ = [
    "CampaignMode",
    "GateOutcome",
    "OneCandidatePassResult",
    "run_one_candidate_pass",
]

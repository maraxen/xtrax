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

## GW-04 addendum: 4-of-6 candidate-validation gates wired pre-bathos (T2-11, T2-13, T2-14, T2-15)

The "Scoping decision" above (LC-09/AC-8a) explicitly deferred `xtrax.loop` gates without an
LC-07-style bathos-data wrapper. Four of the six gates in [GW-04]'s pipeline need NO such wrapper:

1. `candidate_static` (T2-11, AC-1, F0): takes `handoff.path` + optional jaxlint `root`.
2. `structure_tripwire` (T2-13, AC-3): takes `handoff.path`, `callable_name`, `abstract_inputs`
   (caller-sourced), `concrete_inputs` (caller-sourced). No bathos dependency.
3. `candidate_smoke` (T2-14, AC-4): takes `handoff.path`, `callable_name`, `concrete_inputs`
   (caller-sourced), optional `root`. No bathos dependency.
4. `checkified_execution` (T2-15, AC-5): takes `handoff.path`, `callable_name`, `concrete_inputs`
   (caller-sourced). No bathos dependency.

All four are wired here (backlog #3651) as [GW-04]'s composition: candidates rejected by any of
these gates fail **before** `resolve_derived_from`/the real bathos `run` call, matching AC-1's
own "zero GPU time spent" contract and `smoke_2181_walking_skeleton.py`'s reference sequencing.

The remaining two gates in [GW-04]'s pipeline (schema_gate, prereg_match) are explicitly
DEFERRED to a follow-up backlog item -- schema_gate needs a `declared_schema` data type in
controller/'s data model (CandidateHandoff or a new StageBundle field, zero existing producers);
prereg_match needs a bathos-sidecar sourcing gap closed (see backlog #3652 for the narrower,
deferred `capability_probe_gate` wiring and its own documented deferrals). These two data-sourcing
gaps are structural, not implementation-easy, so this narrowly-scoped item does not attempt them.

Like the dispatch/lineage/bathos-run steps above, exceptions from any of these four gate functions
propagate unmodified -- this module still performs zero retry logic (LC-11/AC-8c's own scope).

## Error/retry policy is explicitly NOT this module's job

`run_one_candidate_pass` performs zero retry logic and does not catch any exception raised by
the pieces it sequences: `dispatch_backend.dispatch_candidate()` (`CandidateHandoffFailure`,
`TimeoutError`, `ValueError`), `candidate_static_fn` (`CandidateStaticGateError`), the three
[GW-04] gates (`StructureMismatchError`, `CandidateResolutionError`, `CandidateSmokeTimeoutError`,
`CandidateSmokeFailedError`, `CheckifiedExecutionError`, see the GW-04 addendum above),
`record_candidate_run`/`resolve_derived_from` (`MultiParentLineageUnsupportedError`), or
`campaign_adapter.run` via `record_candidate_run` (`BathosMcpToolError`, `BathosMcpTransportError`,
`BathosTokenMissingError`). Every one of these propagates to the caller unmodified. AC-8c (LC-11)
owns error/retry policy and the "conclude
fires on every code path" guarantee; this module's job is to prove the happy-path sequence is
wired correctly end-to-end, not to also own what happens when a step fails.
"""

import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from controller.bathos_campaign_adapter import BathosCampaignAdapter, CandidateRunResult
from controller.bathos_library_wrappers import (
    call_stats_battery_gate,
    get_evidence_candidate_for_run,
    get_seed_trial_counts,
    get_sidecar_drift_signal,
)
from controller.dispatch import CandidateHandoff, DispatchBackend
from controller.evaluate_adapter import BathosFrozenContext, score_raw_artifacts
from controller.lineage_interim import (
    CandidateParentage,
    record_candidate_run,
    resolve_derived_from,
)
from xtrax.loop.attestation_evidence_gate import EvidenceAdmissionResult, EvidenceCandidate
from xtrax.loop.candidate_smoke import assert_candidate_smoke
from xtrax.loop.candidate_static import assert_candidate_static
from xtrax.loop.checkified_execution import assert_checkified_execution
from xtrax.loop.closure_lock import guarded_evaluate
from xtrax.loop.compile_time_clock import (
    TwoPhaseTiming,
    assert_no_compile_time_regression,
    measure_two_phase_timing,
)
from xtrax.loop.info_barrier_lint import (
    DEFAULT_DELTA_POLICY,
    DeltaExposurePolicy,
    SanctionedEnvelope,
    build_sanctioned_envelope,
)
from xtrax.loop.metrics_provenance import (
    MetricsProvenanceRecord,
    verify_metrics_provenance,
    write_metrics_provenance_attestation,
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
from xtrax.loop.structure_tripwire import assert_structure_tripwire

# Redefined locally rather than imported from either sibling gate module, matching every
# xtrax.loop gate module's own stated independence-from-siblings convention (see e.g.
# stats_battery_gate.py's docstring) -- controller/ sits outside that package but mirrors the
# same convention rather than picking one sibling's alias as authoritative over the other.
CampaignMode = Literal["exploration", "confirmation", "sequential"]


class PriorBestSoFarLineageConflictError(Exception):
    """Raised when a fresh-start call (`best_fitness=None`) finds a prior best-so-far commit
    already present at `ratchet_ref_name` (C6, backlog #4203).

    A fresh-start call implicitly assumes no prior lineage exists yet for `ratchet_ref_name` --
    when `read_best_so_far` finds one anyway, that's either a genuine crash-resume (the caller
    intends to continue an existing campaign) or an accidental ref-name collision with a stale,
    unrelated campaign. This exception makes that ambiguity loud by default; the caller must
    either supply `allow_fresh_start_despite_existing_lineage=True` (an explicit, opt-in
    acknowledgment -- never a silent default) or, more likely correct, use a fresh, unique
    `ratchet_ref_name` for a genuinely new campaign instead of reaching for the override flag.
    """


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """All gate decisions computed for one candidate pass (GW-01, GW-02, GW-04, etc.).

    Attributes:
        stats_battery: `xtrax.loop.stats_battery_gate.assess_stats_battery_verdict`'s decision.
        seed_trial: `xtrax.loop.seed_gate.assess_seed_trial_floor`'s decision.
        evidence_admission: `xtrax.loop.attestation_evidence_gate.admit_evidence`'s decision
            (GW-01, advisory-only -- not folded into hard_blocked). None when evidence_candidate_fn
            is not called or returns empty result.
        sidecar_drift: `xtrax.loop.sidecar_drift_gate.assert_sidecar_drift_reaction`'s decision
            (GW-01, AC-18). Carries should_warn flag for collaborative mode, or is absent when
            agent_mode='' or no drift detected. None when sidecar_drift_signal_fn is skipped.
    """

    stats_battery: ConcludeStatsDecision
    seed_trial: SeedTrialFloorDecision
    evidence_admission: EvidenceAdmissionResult | None = None
    sidecar_drift: SidecarDriftDecision | None = None

    @property
    def hard_blocked(self) -> bool:
        """True iff either gate hard-blocked this candidate.

        This is the load-bearing signal a failing gate check must actually produce: a
        `confirmation`/`sequential` campaign whose stats-battery verdict downgraded, or whose
        seed/trial floor isn't cleared, hard-blocks here -- it is not silently plumbed through
        and ignored by `OneCandidatePassResult.accepted` below. Sidecar drift does NOT fold
        into hard_blocked; SidecarHashMismatchError is raised directly (autonomous mode) or
        sidecar_drift.should_warn is set (collaborative mode).
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
        fitness_dict: the raw scored fitness dict returned by `guarded_evaluate_fn` for this
            candidate (GW-02 addendum, backlog #4203) -- byte-identical to that return value.
            Populated at the same point as `ratchet_decision`: an exception raised anywhere
            upstream of `guarded_evaluate_fn`'s return (dispatch, the candidate-static gate, the
            bathos run, closure-lock scoring itself) aborts the whole `run_one_candidate_pass`
            call before any `OneCandidatePassResult` -- partial or otherwise -- is constructed,
            so `fitness_dict` is never observed in a half-populated state (mirrors
            `ratchet_decision`'s own no-partial-construction guarantee above).
        provenance_record: the `MetricsProvenanceRecord` wrapping `fitness_dict` after
            `verify_metrics_provenance` + attestation write succeed (backlog #3075). Never
            constructed as a partial result -- empty `run_id` and provenance-verify failures
            abort before this dataclass is built (and before any ratchet commit).
        sanctioned_envelope: the agent-facing `SanctionedEnvelope` built after the #3075
            provenance wrap and before ratchet commit (backlog #3077). `InfoBarrierViolationError`
            from the builder aborts before this dataclass is constructed (and before any
            ratchet commit).
    """

    handoff: CandidateHandoff
    derived_from: str
    run_result: CandidateRunResult
    gate_outcome: GateOutcome
    ratchet_decision: RatchetDecision
    fitness_dict: dict[str, float]
    provenance_record: MetricsProvenanceRecord
    sanctioned_envelope: SanctionedEnvelope

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


def _run_git(
    repo: Path, *args: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand in `repo`, raising `ValueError` with `stderr` on failure.

    `env`, when supplied, REPLACES the subprocess's environment entirely (matching
    `subprocess.run`'s own `env` semantics) -- callers that need `GIT_INDEX_FILE` set while
    still inheriting the rest of the process environment must build that dict themselves (see
    `compute_candidate_tree_sha`).
    """
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=dict(env) if env is not None else None,
    )
    if result.returncode != 0:
        msg = f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}"
        raise ValueError(msg)
    return result


def compute_candidate_tree_sha(
    handoff: CandidateHandoff,
    repo: Path,
    candidate_target_path: Path,
    base_sha: str | None,
) -> str:
    """Real default for `run_one_candidate_pass`'s `commit_tree_sha_fn` injection seam (C3-C5,
    backlog #4203).

    Produces a whole-repo-tree substitution: `base_sha`'s tree with every path unchanged except
    `candidate_target_path`, which is written (newly added, or replaced if already present) with
    `handoff.path`'s content -- never a synthetic single-file tree. This matters because
    `reset_worktree_to_best_so_far` does a real `git reset --hard` onto whatever lineage this
    tree ends up in; a single-file tree would corrupt the worktree the next time a candidate is
    rejected.

    `base_sha` is the resolved base COMMIT to substitute against (the caller's own
    `read_best_so_far`/`bootstrap_base_tree_sha` resolution) -- explicitly and unambiguously NOT
    `commit_tree_sha` itself, which is this function's own return value, never one of its inputs.

    Git plumbing (via a scratch `GIT_INDEX_FILE` -- `repo`'s real index is never read or
    mutated):
        1. Resolve `base_sha^{tree}` (`base_sha` is a COMMIT sha, not a tree sha).
        2. `git read-tree` that resolved tree into the scratch index.
        3. `git hash-object -w handoff.path` to write the candidate's blob FIRST.
        4. `git update-index --add --cacheinfo 100644 <blob-sha> <candidate_target_path>` (file
           mode `100644` unconditionally -- an executable candidate is an explicitly out-of-scope
           future extension). `--add` stages the path whether it is new or already present in the
           base tree -- git's own plumbing does not distinguish the two cases, so no
           special-casing is needed here for either.
        5. `git write-tree` from the scratch index, returning the resulting tree sha.

    Raises:
        ValueError: `base_sha` is `None` (no prior best-so-far commit exists and no
            `bootstrap_base_tree_sha` fallback was supplied) -- raised immediately, before any
            git write. Also raised (via the internal git-subcommand runner) if any git command
            above fails.
    """
    if base_sha is None:
        msg = (
            "compute_candidate_tree_sha: base_sha is None -- no prior best-so-far commit "
            "exists (read_best_so_far returned None) and no bootstrap_base_tree_sha fallback "
            "was supplied"
        )
        raise ValueError(msg)

    repo = repo.resolve()
    with tempfile.TemporaryDirectory() as scratch_dir:
        index_env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch_dir) / "scratch-index")}

        # 1. base_sha is a COMMIT sha -- resolve its tree explicitly before read-tree (C5).
        base_tree_sha = _run_git(repo, "rev-parse", f"{base_sha}^{{tree}}").stdout.strip()
        # 2. Load that tree into the scratch index -- repo's real index untouched.
        _run_git(repo, "read-tree", base_tree_sha, env=index_env)

        # 3. Write the candidate's blob FIRST, before staging it (C5).
        blob_sha = _run_git(repo, "hash-object", "-w", str(handoff.path)).stdout.strip()
        # 4. Stage it at candidate_target_path, mode 100644 unconditionally (C5/C9).
        _run_git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            "100644",
            blob_sha,
            str(candidate_target_path),
            env=index_env,
        )

        # 5. Materialize the resulting tree object.
        return _run_git(repo, "write-tree", env=index_env).stdout.strip()


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
    # Per-pass uniqueness (review finding): repeated passes of one campaign
    # are distinct forensic events -- a second pass must never overwrite the
    # first's record, and truncated long slugs must not collide either.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    nonce = uuid.uuid4().hex[:6]
    path = out_dir / f"pass_{safe}_{stamp}_{nonce}.json"
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
            "axis_note": ("provenance artifact; n_atoms placeholder by contract"),
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
    evidence_candidate_fn: Callable[..., EvidenceCandidate] = get_evidence_candidate_for_run,
    stdout_verified: bool | None = None,
    sidecar_drift_signal_fn: Callable[..., SidecarDriftSignal] = get_sidecar_drift_signal,
    candidate_static_fn: Callable[..., None] = assert_candidate_static,
    candidate_static_root: Path | None = None,
    probe_record_dir: Path | None = None,
    abstract_inputs: list[Any],
    structure_tripwire_fn: Callable[..., None] = assert_structure_tripwire,
    candidate_smoke_fn: Callable[..., None] = assert_candidate_smoke,
    candidate_smoke_root: Path | None = None,
    checkified_execution_fn: Callable[..., Any] = assert_checkified_execution,
    frozen_context: BathosFrozenContext,
    current_config: Mapping[str, Any],
    candidate_touched_paths: frozenset[Path] = frozenset(),
    guarded_evaluate_fn: Callable[..., dict[str, float]] = guarded_evaluate,
    metrics_provenance_dir: Path,
    verify_metrics_provenance_fn: Callable[..., None] = verify_metrics_provenance,
    iteration: int,
    build_sanctioned_envelope_fn: Callable[..., SanctionedEnvelope] = build_sanctioned_envelope,
    info_barrier_score_key: str = "score",
    delta_policy: DeltaExposurePolicy = DEFAULT_DELTA_POLICY,
    best_fitness: Mapping[str, float] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
    repo: Path,
    ratchet_ref_name: str,
    commit_tree_sha: str | None = None,
    commit_tree_sha_fn: Callable[
        [CandidateHandoff, Path, Path, str | None], str
    ] = compute_candidate_tree_sha,
    candidate_target_path: Path | None = None,
    bootstrap_base_tree_sha: str | None = None,
    allow_fresh_start_despite_existing_lineage: bool = False,
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
        evidence_candidate_fn: injection seam for tests -- defaults to GW-01's real
            `get_evidence_candidate_for_run` (which lazily imports bathos, no MCP call). Called
            after the bathos run to verify evidence attestation. Raises ValueError if run_id is
            empty (fallback strategy in bathos_campaign_adapter didn't find the run).
        stdout_verified: Optional caller-supplied stdout verification result (True iff
            `Run.stdout_sha256` was recorded AND re-hash matched; False if recorded but
            mismatched; None if never recorded). Passed through to evidence_candidate_fn and
            then threaded into EvidenceCandidate for consumption by admit_evidence.
        sidecar_drift_signal_fn: injection seam for tests -- defaults to GW-01's real
            `get_sidecar_drift_signal` (which lazily imports bathos, no MCP call). Called
            immediately after the bathos run, BEFORE ratchet/commit (audit-required ordering).
            Forwards ``run_id`` so the default wrapper can look up sidecar_sha256; this
            function itself does not import bathos. Skipped entirely when agent_mode is not
            'collaborative'/'autonomous' (agent_mode='' is an explicit opt-out for existing
            callers). On drift + autonomous mode, raises SidecarHashMismatchError (HALT before
            any commit lands). On drift + collaborative mode, returns a decision with
            should_warn=True for non-blocking warning.
        candidate_static_fn: injection seam for tests -- defaults to T2-11's real
            `assert_candidate_static` (clean import + zero jaxlint JL-series errors). Called
            immediately after dispatch, before lineage resolution or the real bathos run -- see
            the module docstring's GW-04 addendum.
        candidate_static_root: forwarded to `candidate_static_fn` as its `root` kwarg (jaxlint's
            subprocess root; default `None` lets jaxlint fall back to `Path.cwd()`).
        probe_record_dir: opt-in Phase C seam -- when set, one Stage-0 provenance ProbeRecord
            (wall-clock pass duration + campaign/run identity) is written under this directory
            after the gates resolve. `None` (default) emits nothing.
        abstract_inputs: required list of abstract input shapes (JAX ShapeDtypeStruct instances)
            for structure-tripwire and checkified-execution gates -- supplied by the caller,
            caller-sourced, not computed here.
        structure_tripwire_fn: injection seam for tests -- defaults to T2-13's real
            `assert_structure_tripwire` (abstract vs concrete pytree/shape/dtype comparison).
            Called immediately after candidate_static_fn, in the same pre-bathos window, to reject
            candidates with data-dependent control flow before any real execution. Raises
            `StructureMismatchError` or `CandidateResolutionError` unchanged.
        candidate_smoke_fn: injection seam for tests -- defaults to T2-14's real
            `assert_candidate_smoke` (L1 dry-run + L2 CPU smoke test). Called after
            structure_tripwire_fn, before checkified_execution_fn, in the same pre-bathos window.
            Raises `CandidateSmokeTimeoutError` or `CandidateSmokeFailedError` unchanged.
        candidate_smoke_root: forwarded to `candidate_smoke_fn` as its `root` kwarg (working
            directory for `uv run --locked`; default `None` falls back to the repo root).
        checkified_execution_fn: injection seam for tests -- defaults to T2-15's real
            `assert_checkified_execution` (SafetyManager NaN/Inf/overflow checks). Called after
            candidate_smoke_fn, in the same pre-bathos window, to reject candidates with numeric
            violations before bathos submission. Raises `CheckifiedExecutionError` or
            `CandidateResolutionError` unchanged.
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
        metrics_provenance_dir: directory that receives `write_metrics_provenance_attestation`
            output as `{run_id}.toml` (backlog #3075). Required -- callers (including tests)
            supply an isolated path; this function does not default to cwd.
        verify_metrics_provenance_fn: injection seam for tests -- defaults to the real
            `xtrax.loop.metrics_provenance.verify_metrics_provenance`. Called after wrapping
            the already-scored `fitness_dict`; `UnprovenancedMetricsError` propagates uncaught
            (HALT before ratchet commit and before `OneCandidatePassResult` construction).
        iteration: 1-based candidate index for the agent-facing envelope (backlog #3077).
            Required -- `run_multi_iteration_loop` passes `candidate_index + 1`.
        build_sanctioned_envelope_fn: injection seam for tests -- defaults to the real
            `xtrax.loop.info_barrier_lint.build_sanctioned_envelope`. Called after the #3075
            provenance wrap and before ratchet commit. `InfoBarrierViolationError` propagates
            uncaught (HALT before ratchet commit and before `OneCandidatePassResult`
            construction).
        info_barrier_score_key: fitness-dict key used to read the current/previous score for
            the envelope (default `"score"`). If missing from `record.fitness`,
            `previous_score` is `None`.
        delta_policy: forwarded to `build_sanctioned_envelope_fn` (default
            `DEFAULT_DELTA_POLICY`).
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
        commit_tree_sha: real git tree data the caller supplies directly for the accept-path
            pending commit, or `None` (the default) to have `commit_tree_sha_fn` compute one
            instead. When supplied literally, this function does not construct or verify it
            (mirrors the `stats_battery_kwargs` trust boundary: caller-sourced, not computed
            here), and `commit_tree_sha_fn` is never invoked -- `candidate_target_path` may be
            omitted entirely on this path.
        commit_tree_sha_fn: injection seam for tests -- defaults to the real
            `compute_candidate_tree_sha` (C3-C5, backlog #4203). Invoked lazily, only inside the
            accept branch and only when `commit_tree_sha is None` (never eagerly right after
            dispatch -- nothing between dispatch and the accept branch reads `commit_tree_sha`,
            so a candidate that is rejected, gate-blocked, or aborts via an upstream exception
            before ever reaching the accept branch never triggers this call). Called as
            `commit_tree_sha_fn(handoff, repo, candidate_target_path, base_sha)`, where
            `base_sha` is resolved via a second, explicit `read_best_so_far` call (falling back
            to `bootstrap_base_tree_sha`) -- see that param's own docstring.
        candidate_target_path: the repo-relative destination path for the candidate's content
            inside the tree substitution `commit_tree_sha_fn` computes (C7, backlog #4203).
            `None` by default; required (raises `ValueError` immediately, before dispatch) when
            `commit_tree_sha is None`, since `commit_tree_sha_fn` needs an explicit destination
            and this function never infers one from `handoff.path`'s basename. May be omitted
            entirely by literal-`commit_tree_sha` callers, since it is never used on that path.
        bootstrap_base_tree_sha: fallback base-commit SHA for `commit_tree_sha_fn`'s tree
            substitution (C3, backlog #4203), used only when
            `read_best_so_far(repo, ratchet_ref_name)` returns `None` (no prior best-so-far ref
            exists yet) AND `commit_tree_sha is None`. A `ValueError` is raised (by
            `compute_candidate_tree_sha`, or an equivalent custom `commit_tree_sha_fn`) if both
            are `None` on an accept -- mirrors `commit_parent_sha`'s own identical
            bootstrap-required precedent below.
        allow_fresh_start_despite_existing_lineage: when `best_fitness is None` (a fresh-start
            call) and `read_best_so_far(repo, ratchet_ref_name)` finds a prior best-so-far commit
            already exists, this function raises `PriorBestSoFarLineageConflictError` by default
            (C6, backlog #4203) -- the caller likely intended a genuinely new campaign but reused
            a `ratchet_ref_name` that already has unrelated (or stale) lineage on it. Set `True`
            only when this really is an intentional resume/overwrite of that existing lineage;
            prefer a fresh, unique `ratchet_ref_name` for a genuinely new campaign instead of
            reaching for this flag. Default `False` -- silent overwrite is never the default.
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
            or both omitted; `candidate_target_path` was omitted while `commit_tree_sha is None`
            (raised immediately, before dispatch); `commit_parent_sha` was not supplied on an
            accept when no prior best-so-far ref exists yet; or (via `commit_tree_sha_fn`)
            `bootstrap_base_tree_sha` was not supplied on an accept when `commit_tree_sha is
            None` and no prior best-so-far ref exists yet.
        PriorBestSoFarLineageConflictError: `best_fitness is None` (a fresh-start call) and
            `read_best_so_far(repo, ratchet_ref_name)` found a prior best-so-far commit already
            exists, and `allow_fresh_start_despite_existing_lineage` was not set.
        CandidateStaticGateError: the candidate fails clean-import or has a JL-series jaxlint
            error -- raised right after dispatch, before lineage resolution or any bathos call
            (T2-11, AC-1).
        StructureMismatchError, CandidateResolutionError: the structure-tripwire gate (T2-13, AC-3)
            detected an abstract vs concrete pytree/shape/dtype mismatch, or candidate resolution
            failed -- raised before the real bathos run.
        CandidateSmokeTimeoutError, CandidateSmokeFailedError: the candidate-smoke gate (T2-14,
            AC-4) L1/L2 budget was exhausted or a phase exited nonzero -- raised before the real
            bathos run.
        CheckifiedExecutionError, CandidateResolutionError: the checkified-execution gate (T2-15,
            AC-5) detected a NaN/Inf/overflow violation, or candidate resolution failed -- raised
            before the real bathos run.
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
        InfoBarrierViolationError: the sanctioned envelope failed lint (AC-9, backlog #3077) --
            non-recoverable HALT, propagates unmodified, before ratchet commit.
        StatsBatteryGateInputError: `campaign_mode` is not a recognized `CampaignMode` (raised
            by `assess_stats_battery_verdict`).
        SeedGateInputError: `campaign_mode` is not recognized, or the seed/trial counts are
            malformed (raised by `assess_seed_trial_floor`).
    """
    # 0. Fail-fast validation (C7, backlog #4203): commit_tree_sha_fn needs an explicit
    # candidate_target_path to write to -- literal-commit_tree_sha callers may omit it entirely
    # (never used on that path). Checked before dispatch so this never burns a real dispatch
    # call on a call that was going to fail anyway.
    if commit_tree_sha is None and candidate_target_path is None:
        msg = "candidate_target_path is required when commit_tree_sha is not supplied directly"
        raise ValueError(msg)

    # 1. Dispatch -> hand off source (LC-03).
    pass_started_at = perf_counter()
    handoff = dispatch_backend.dispatch_candidate()

    # 1.5. Candidate-static gate (T2-11, AC-1, F0; [GW-04] first slice) -- reject a candidate
    # that fails clean-import or jaxlint JL-series checks before lineage resolution or any real
    # bathos run is attempted. See the module docstring's GW-04 addendum.
    candidate_static_fn(handoff.path, root=candidate_static_root)

    # 1.6. Structure-tripwire gate (T2-13, AC-3; [GW-04]) -- verify abstract vs concrete pytree
    # structure before any real bathos run. Same pre-bathos window as candidate_static (T2-11).
    structure_tripwire_fn(
        handoff.path,
        callable_name,
        abstract_inputs=abstract_inputs,
        concrete_inputs=concrete_inputs,
    )

    # 1.7. Candidate-smoke gate (T2-14, AC-4; [GW-04]) -- L1 dry-run + L2 CPU smoke test before
    # any GPU/cluster submission. Same pre-bathos window as candidate_static/structure_tripwire.
    candidate_smoke_fn(
        handoff.path, callable_name, concrete_inputs=concrete_inputs, root=candidate_smoke_root
    )

    # 1.8. Checkified-execution gate (T2-15, AC-5; [GW-04]) -- SafetyManager NaN/Inf/overflow
    # checks before bathos submission. Same pre-bathos window as prior gates.
    checkified_execution_fn(handoff.path, callable_name, concrete_inputs=concrete_inputs)

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

    # 2.4. Sidecar-drift check (GW-01, AC-18) -- MUST fire BEFORE any best-so-far commit
    # lands, to prevent a drift-tainted candidate from becoming the new best-so-far. Skip
    # entirely when agent_mode is not 'collaborative'/'autonomous' (explicit opt-out for
    # existing callers). On drift + autonomous: raise SidecarHashMismatchError (HALT). On
    # drift + collaborative: return decision with should_warn=True (non-blocking warning).
    sidecar_drift_decision: SidecarDriftDecision | None = None
    if agent_mode in ("collaborative", "autonomous") and run_result.run_id:
        # Sidecar hash lookup lives in the default `get_sidecar_drift_signal` wrapper
        # (lazy bathos import), not here -- this function must stay importable without
        # the controller extra so CI/--extra dev tests can inject a stub.
        sidecar_signal = sidecar_drift_signal_fn(
            script_path=str(handoff.path),
            catalog_dir="",
            current_sidecar_sha256="",
            script_id=handoff.path.stem,
            run_id=run_result.run_id,
        )
        sidecar_drift_decision = assert_sidecar_drift_reaction(
            sidecar_signal,
            agent_mode=agent_mode,
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

    # 2.55. Metrics provenance (#3075) -- wrap already-scored fitness_dict; do not re-evaluate.
    # Empty run_id and UnprovenancedMetricsError HALT before ratchet commit and before
    # OneCandidatePassResult is constructed.
    if not run_result.run_id:
        msg = "run_id must be non-empty to attest metrics provenance"
        raise ValueError(msg)
    record = MetricsProvenanceRecord(
        fitness=fitness_dict,
        evaluator_closure_hash=frozen_context.locked.closure_hash,
        run_id=run_result.run_id,
    )
    verify_metrics_provenance_fn(record, locked=frozen_context.locked)
    write_metrics_provenance_attestation(
        record, metrics_provenance_dir / f"{run_result.run_id}.toml"
    )

    # 2.56. Info barrier (#3077) -- lint the agent-facing envelope after the scored fitness
    # attestation and before ratchet commit. InfoBarrierViolationError HALTs uncaught.
    previous_score = None
    if best_fitness is not None and info_barrier_score_key in record.fitness:
        previous_score = best_fitness.get(info_barrier_score_key)
    envelope = build_sanctioned_envelope_fn(
        record,
        iteration=iteration,
        previous_score=previous_score,
        score_key=info_barrier_score_key,
        delta_policy=delta_policy,
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

    if best_fitness is None or higher_is_better is None:
        # The XOR check above already means both are None by the time we get here; the
        # second disjunct adds no runtime behaviour and exists so that pairing is
        # *provable* rather than merely true. Without it a type checker cannot correlate
        # the two names across the XOR, and reads the else branch below as passing
        # `Mapping[str, bool] | None` into a parameter declared `Mapping[str, bool]`.
        #
        # C6 (backlog #4203): a fresh-start call (best_fitness=None) implicitly assumes no
        # prior lineage exists yet for ratchet_ref_name -- if one does, that's either a genuine
        # crash-resume or an accidental ref-name collision with stale/unrelated data. Fail loud
        # by default, before the automatic-accept sentinel below is ever constructed;
        # allow_fresh_start_despite_existing_lineage=True is the caller's explicit opt-in
        # acknowledgment of that conflict (and skips this read entirely when set).
        if not allow_fresh_start_despite_existing_lineage:
            conflicting_sha = read_best_so_far(repo, ratchet_ref_name)
            if conflicting_sha is not None:
                msg = (
                    f"a prior best-so-far commit already exists at ratchet_ref_name "
                    f"{ratchet_ref_name!r} (sha={conflicting_sha!r}), but this call supplied "
                    "best_fitness=None (a fresh-start call). If this is genuinely a new "
                    "campaign, use a fresh, unique ratchet_ref_name -- do not reach for "
                    "allow_fresh_start_despite_existing_lineage=True unless you intend to "
                    "proceed despite the existing lineage (e.g. an intentional crash-resume)."
                )
                raise PriorBestSoFarLineageConflictError(msg)
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
        if commit_tree_sha is None:
            # C3/C4/P1.6 (backlog #4203): a second, explicit, side-effect-free
            # read_best_so_far call -- deliberately not reusing prior_best_sha above, mirroring
            # this same accept branch's own precedent of resolving commit_parent_sha
            # independently. commit_tree_sha_fn is invoked lazily, HERE, only on the accept
            # path -- see its own docstring for why (nothing upstream of this branch reads
            # commit_tree_sha, so a rejected/gate-blocked/exception-aborted candidate never
            # reaches this call).
            # AC-h (#4215, adversarial review finding C3, warning, pre-existing/orthogonal --
            # not blocking): this is the second of two independent read_best_so_far calls in
            # this accept branch (the first, above, feeds parent_sha). A theoretical TOCTOU
            # window exists where a concurrent writer could advance ratchet_ref_name between the
            # two reads, making commit_parent_sha and bootstrap_base_tree_sha resolve to
            # different commits even when a caller (e.g. #4215's bootstrap_commit_sha fan-out)
            # supplies them identically. This race pre-dates #4215 and is not introduced or
            # worsened by it -- documented here as a known, low-likelihood limitation
            # (single-process sequential campaign loop, no evidence of concurrent multi-process
            # writers to the same ref anywhere in this codebase), not a bug or new test
            # requirement.
            base_sha = read_best_so_far(repo, ratchet_ref_name)
            if base_sha is None:
                base_sha = bootstrap_base_tree_sha
            assert candidate_target_path is not None, (
                "guaranteed non-None here by the fail-fast validation at the top of this "
                "function (commit_tree_sha is None implies candidate_target_path is not None)"
            )
            resolved_tree_sha = commit_tree_sha_fn(handoff, repo, candidate_target_path, base_sha)
        else:
            resolved_tree_sha = commit_tree_sha
        # prior_best_sha serves as BOTH the parent-SHA source AND the CAS expected_old_sha --
        # one read_best_so_far call above, not two. CommitCreationError/RefUpdateConflictError
        # propagate uncaught.
        pending_sha = create_pending_commit(repo, resolved_tree_sha, parent_sha, message)
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

    # 3.5. Evidence attestation check (GW-01, AC-19, advisory-only) -- verify run provenance.
    # May raise ValueError if run_id lookup failed (indicating a broken fallback strategy).
    evidence_admission: EvidenceAdmissionResult | None = None
    if run_result.run_id:
        try:
            evidence_candidate = evidence_candidate_fn(
                run_id=run_result.run_id,
                catalog_dir="",
                stdout_verified=stdout_verified,
            )
            # Admit evidence (advisory only, not hard-blocking) -- returns partitioned
            # (admitted, excluded) sets of candidates.
            from xtrax.loop.attestation_evidence_gate import admit_evidence

            evidence_admission = admit_evidence([evidence_candidate])
        except (ValueError, ImportError):
            # run_id lookup failed, or bathos is not installed (CI / no controller extra);
            # evidence is advisory-only -- proceed without a result.
            pass

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
    # 4.5. Optional provenance record (Phase C, scope doc D7): written only when
    # the caller opted in, AFTER everything that decides the pass outcome so
    # the artifact carries those verdicts. Emission failures are CONTAINED:
    # a completed pass must never be discarded because provenance
    # bookkeeping failed -- the pass result stands, the missing record is
    # reported loudly instead.
    if probe_record_dir is not None:
        try:
            probe_record_dir.mkdir(parents=True, exist_ok=True)
            _emit_candidate_pass_probe_record(
                out_dir=probe_record_dir,
                campaign_id=campaign_id,
                derived_from=derived_from,
                handoff_sha=handoff.content_sha256,
                wall_seconds=perf_counter() - pass_started_at,
                accepted=(run_result.success and stats_decision.honored and seed_decision.held),
                hard_blocked=(stats_decision.hard_blocked or seed_decision.hard_blocked),
            )
        except Exception as exc:  # noqa: BLE001 -- contained by design; see comment above
            print(
                f"WARNING: candidate-pass ProbeRecord emission failed for "
                f"campaign {campaign_id!r}; pass result unaffected: {exc}",
                file=sys.stderr,
            )

    return OneCandidatePassResult(
        handoff=handoff,
        derived_from=derived_from,
        run_result=run_result,
        gate_outcome=GateOutcome(
            stats_battery=stats_decision,
            seed_trial=seed_decision,
            evidence_admission=evidence_admission,
            sidecar_drift=sidecar_drift_decision,
        ),
        ratchet_decision=ratchet_decision,
        fitness_dict=fitness_dict,
        provenance_record=record,
        sanctioned_envelope=envelope,
    )


__all__ = [
    "CampaignMode",
    "GateOutcome",
    "OneCandidatePassResult",
    "PriorBestSoFarLineageConflictError",
    "compute_candidate_tree_sha",
    "run_one_candidate_pass",
]

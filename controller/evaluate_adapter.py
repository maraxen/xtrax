"""SPLIT_COMPUTE evaluator integration for #3611's controller (resolves #3657, backlog #4133).

Wires xtrax.stages.evaluate.EvaluateFn to controller/'s bathos dispatch, per the SPLIT_COMPUTE
decision (.praxia/docs/specs/260807_evaluator-integration-decision.md): the bathos-dispatched
candidate subprocess must produce only RAW artifacts (predictions, checkpoints, logs) into its
declared output_paths -- it must never be trusted to compute or report a fitness number itself.
This module's EvaluateFn implementation performs the out-of-process bathos dispatch
(campaign_adapter.run) to obtain those raw artifacts, then scores them in-process against
xtrax.loop.closure_lock.ClosureManifest's locked split_paths using a caller-supplied, injectable
scoring function -- mirroring xtrax's existing stats_battery_fn/seed_trial_counts_fn
injection-seam convention (see controller/main_loop.py).

This module deliberately does NOT call controller.main_loop.run_one_candidate_pass: that
function's own stats_battery gate requires a fitness dict as INPUT (stats_battery_kwargs),
which is exactly what this module produces -- calling it here would be circular. A future
multi-iteration driver (out of this item's scope) is expected to call this module's
BathosSplitComputeEvaluator FIRST (via xtrax.loop.metrics_provenance.evaluate_with_provenance)
to get a fitness dict, then feed it into run_one_candidate_pass's stats_battery_kwargs
downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from controller.bathos_campaign_adapter import BathosCampaignAdapter, CandidateRunResult
from xtrax.loop.closure_lock import ClosureManifest


class RawArtifactsUnavailableError(Exception):
    """Raised when the bathos dispatch did not succeed, so no raw artifacts exist to score.

    EvaluateFn must not proceed to score a candidate whose dispatch failed (success=False /
    non-zero exit_code) -- there is nothing legitimate to read, and scoring a partial or absent
    output_paths set would silently manufacture a fitness number from noise.
    """


@dataclass(frozen=True, slots=True)
class BathosFrozenContext:
    """The FrozenContext type parameter for controller/'s EvaluateFn: fixed across every
    candidate dispatched against one locked evaluator closure.

    Attributes:
        locked: the locked ClosureManifest whose split_paths are the trusted ground truth
            raw artifacts get scored against.
        campaign_adapter: the BathosCampaignAdapter this EvaluateFn dispatches candidates
            through.
        campaign_id: the already-open bathos campaign's ID, forwarded to campaign_adapter.run.
        score_fn: scores a candidate's raw artifact paths (produced by the bathos dispatch)
            against locked.split_paths, returning the fitness dict. Injectable -- xtrax does
            not know in advance what metric a given #2181 research loop cares about (mirrors
            run_one_candidate_pass's stats_battery_fn/seed_trial_counts_fn convention).
    """

    locked: ClosureManifest
    campaign_adapter: BathosCampaignAdapter
    campaign_id: str
    score_fn: Callable[[tuple[Path, ...], tuple[Path, ...]], dict[str, float]]


@dataclass(frozen=True, slots=True)
class BathosCandidate:
    """The Candidate type parameter for controller/'s EvaluateFn: one dispatched candidate.

    Attributes:
        script_path: path to the candidate's staged source script, forwarded to
            campaign_adapter.run as the script to execute.
        output_paths: paths the candidate's bathos-dispatched subprocess is declared to write
            RAW artifacts to -- never a pre-computed summary fitness value (SPLIT_COMPUTE's
            core invariant; enforced by convention/documentation, not mechanically checkable
            from this module -- what a candidate script actually writes is opaque to xtrax).
            Read back by this module after dispatch to compute the fitness dict.
        derived_from: forwarded to campaign_adapter.run's derived_from parameter.
        run_args: forwarded to campaign_adapter.run's args parameter.
        tags: forwarded to campaign_adapter.run.
        agent_mode: forwarded to campaign_adapter.run.
        no_sidecar: forwarded to campaign_adapter.run.
    """

    script_path: str
    output_paths: tuple[str, ...]
    derived_from: str = ""
    run_args: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    agent_mode: str = ""
    no_sidecar: bool = False


class BathosSplitComputeEvaluator:
    """EvaluateFn[BathosFrozenContext, BathosCandidate]: dispatches via bathos, scores raw
    artifacts in-process (SPLIT_COMPUTE). The sole call site in controller/ through which a
    candidate's fitness is obtained -- resolves #3657.
    """

    def __call__(
        self, frozen_context: BathosFrozenContext, candidate: BathosCandidate
    ) -> dict[str, float]:
        run_result: CandidateRunResult = frozen_context.campaign_adapter.run(
            candidate.script_path,
            campaign_id=frozen_context.campaign_id,
            derived_from=candidate.derived_from,
            args=list(candidate.run_args) or None,
            output_paths=list(candidate.output_paths) or None,
            tags=list(candidate.tags) or None,
            agent_mode=candidate.agent_mode,
            no_sidecar=candidate.no_sidecar,
        )

        if not run_result.success:
            msg = (
                f"bathos dispatch did not succeed for {candidate.script_path!r} "
                f"(exit_code={run_result.exit_code}) -- refusing to score raw artifacts that "
                "may be partial or absent"
            )
            raise RawArtifactsUnavailableError(msg)

        raw_artifact_paths = tuple(Path(p) for p in candidate.output_paths)
        return frozen_context.score_fn(raw_artifact_paths, frozen_context.locked.split_paths)


__all__ = [
    "BathosCandidate",
    "BathosFrozenContext",
    "BathosSplitComputeEvaluator",
    "RawArtifactsUnavailableError",
]

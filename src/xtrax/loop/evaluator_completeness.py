"""Evaluator-completeness gate (T2-08, #2181, AC-11, PM-2).

PM-2's failure mode (`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md`): a
candidate improved a tracked fitness scalar while degrading an UNtracked property, because the
evaluator's returned `dict[str, float]` was incomplete and nothing checked for that incompleteness
-- "what you don't measure, the agent games." This module is the fix: before a loop may start, the
evaluator must (1) hold a *reviewed* invariant-manifest (correctness/stability floors beyond the
optimization target) and (2) pass a synthetic-ground-truth sanity check (the one-hot fitness case
`tests.stages.test_evaluate.TestOneHotSanityCheck` already worked through for T1-07, generalized
here into a reusable primitive rather than left as a test-only illustration).

"Reviewed" reuses `xtrax.devtools.freshness.Attestation`/`evaluate_freshness` rather than inventing
a second review/TTL mechanism -- a manifest is "reviewed" iff its attestation is still fresh.

Registration-site scope: `SealedEvaluatorRegistry.seal()` (T1-07, `xtrax.stages.evaluate`) is left
untouched, matching the pattern T2-04 and T2-05 already set of composing new checks around the
frozen seam rather than widening it. `assert_evaluator_completeness` is a separate call a future
loop controller invokes once before its first iteration -- not a hook inside `seal()` itself.

Extension seam (mirrors `xtrax.loop.admission`'s and `xtrax.loop.closure_lock`'s stance): this
module raises on failure and stops there. It does not decide what a human-in-the-loop response
looks like, and it does not own *authoring or reviewing* the invariant-manifest's content -- no
DAG item is responsible for that yet (checked: it is not among the roadmap's wired human-gate
nodes), so that process is out of scope here, flagged rather than invented. This module only
checks a manifest that's handed to it.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from xtrax.devtools.freshness import Attestation, evaluate_freshness
from xtrax.stages.evaluate import Candidate, EvaluateFn, FrozenContext


class EvaluatorCompletenessError(Exception):
    """Base for AC-11's registration-refusal errors. The loop must not start; do not retry."""


class InvariantManifestStaleError(EvaluatorCompletenessError):
    """The invariant-manifest's attestation is not fresh (unreviewed, or review has expired)."""


class SyntheticSanityCheckFailedError(EvaluatorCompletenessError):
    """The evaluator did not score the sanity case's known-best candidate strictly top."""


class InvariantViolationError(EvaluatorCompletenessError):
    """A named invariant failed against the sanity case's evaluator output."""


@dataclass(frozen=True, slots=True)
class Invariant:
    """A correctness/stability floor the evaluator's fitness dict must satisfy.

    `check` sees only the evaluator's returned `dict[str, float]` -- PM-2's own framing is
    specifically that the *returned dict* was incomplete, not that the evaluator's inputs were
    mishandled, so a narrower `dict[str, float] -> bool` signature is sufficient and simpler than
    threading `frozen_context`/`candidate` through as well.
    """

    name: str
    description: str
    check: Callable[[dict[str, float]], bool]


@dataclass(frozen=True, slots=True)
class InvariantManifest:
    """A reviewed set of invariants. "Reviewed" = `attestation` is still fresh."""

    invariants: tuple[Invariant, ...]
    attestation: Attestation


@dataclass(frozen=True, slots=True)
class SyntheticGroundTruthCase[FrozenContext, Candidate]:
    """A one-hot fitness case: one candidate is the known-best, the rest are not.

    Generalizes `TestOneHotSanityCheck`'s inline pattern (`tests/stages/test_evaluate.py`) into a
    reusable primitive instead of leaving it as a test-only illustration.
    """

    candidates: Mapping[Candidate, FrozenContext]
    known_best: Candidate


def run_synthetic_sanity_check(
    evaluator: EvaluateFn[FrozenContext, Candidate],
    sanity_case: SyntheticGroundTruthCase[FrozenContext, Candidate],
) -> dict[Candidate, dict[str, float]]:
    """Run `evaluator` over every candidate in `sanity_case`; return each fitness dict.

    Raises:
        SyntheticSanityCheckFailedError: `known_best` did not score strictly higher (by the
            `"score"` key) than every other candidate.
    """
    if sanity_case.known_best not in sanity_case.candidates:
        msg = "sanity_case.known_best is not one of sanity_case.candidates"
        raise SyntheticSanityCheckFailedError(msg)

    results = {
        candidate: evaluator(frozen_context, candidate)
        for candidate, frozen_context in sanity_case.candidates.items()
    }
    known_best_score = results[sanity_case.known_best]["score"]
    worse_or_tied = {
        candidate: fitness["score"]
        for candidate, fitness in results.items()
        if candidate != sanity_case.known_best and fitness["score"] >= known_best_score
    }
    if worse_or_tied:
        msg = (
            f"known_best={sanity_case.known_best!r} (score={known_best_score}) did not score "
            f"strictly top: {worse_or_tied}"
        )
        raise SyntheticSanityCheckFailedError(msg)

    return results


def assert_evaluator_completeness(
    evaluator: EvaluateFn[FrozenContext, Candidate],
    manifest: InvariantManifest,
    sanity_case: SyntheticGroundTruthCase[FrozenContext, Candidate],
) -> None:
    """The composed AC-11 gate: reviewed manifest + sanity check + every invariant, in order.

    Raises:
        InvariantManifestStaleError: `manifest.attestation` is not fresh.
        SyntheticSanityCheckFailedError: see `run_synthetic_sanity_check`.
        InvariantViolationError: the first invariant (in `manifest.invariants` order) that fails
            against any sanity-case fitness dict.
    """
    verdict = evaluate_freshness(manifest.attestation)
    if not verdict.fresh:
        msg = f"invariant-manifest is not reviewed/fresh: {'; '.join(verdict.reasons)}"
        raise InvariantManifestStaleError(msg)

    results = run_synthetic_sanity_check(evaluator, sanity_case)

    for invariant in manifest.invariants:
        for candidate, fitness in results.items():
            if not invariant.check(fitness):
                msg = (
                    f"invariant {invariant.name!r} ({invariant.description}) failed for "
                    f"candidate={candidate!r}: fitness={fitness}"
                )
                raise InvariantViolationError(msg)


__all__ = [
    "EvaluatorCompletenessError",
    "Invariant",
    "InvariantManifest",
    "InvariantManifestStaleError",
    "InvariantViolationError",
    "SyntheticGroundTruthCase",
    "SyntheticSanityCheckFailedError",
    "assert_evaluator_completeness",
    "run_synthetic_sanity_check",
]

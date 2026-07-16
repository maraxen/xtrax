"""Attestation-as-evidence gate (T2-26, #2181, AC-19).

AC-19's claim (spec `260702_design-the-2181-agentic-algorithm-evolut.md:129`, DAG doc
`260702_02-dag-2181-autoresearch.md:554-564`): a run admitted as evidence must verify its signed
``manifest_sha256`` + stdout hash. Success: attested runs only in the evidence set. Fast/loud: an
unverifiable run is excluded from evidence, loud (full K-Veritas RSA-PSS nonrepudiation deferred to
a late milestone) -- not silently dropped.

Grounding, and a wording correction already made upstream (verified before writing any code, not
guessed): the local `bathos` checkout at the start of this task was stale against
`origin/main` (missing merged PR #16/B2-07 and others) -- re-verified directly against
`origin/main` via `git show origin/main:src/bathos/prereg.py` rather than trusting the stale
working tree. `bathos.prereg.verify_run_manifest(run) -> bool` (B2-07, schema v10 -> v11, merged)
is the function this gate's ``manifest_verified`` input traces to; its own docstring already states
the correction this module's docstring restates for its own callers: "'Signed' here means the
existing self-signed-manifest scheme (content-hash + git-commit-bound ...) -- there is no
cryptographic signature key anywhere in this codebase", citing bathos's own ADR
`.praxia/docs/decisions/260526_nonrepudiation-v06.md` (accepted 2026-05-26), which explicitly
rejects RSA-PSS/K-Veritas-style signing for this milestone and defers it to v0.7+. So: "signed
manifest_sha256" in this module's own docstring and the DAG text it implements means
**hash-verified against a self-signed (content-hash + git-commit-bound) manifest**, not literal
cryptographic signing -- the DAG text's own "K-Veritas RSA-PSS deferred" clause already says as
much; this is stated here explicitly so no future reader re-guesses "signed" as real crypto.

Also grounded: `bathos.prereg.verify_run_manifest`'s own docstring states it does *not* verify
`Run.stdout_sha256` against anything -- "bathos does not persist raw stdout for a general run to
re-hash against; that field exists for a caller ... to populate and independently verify against
its own stored artifact." So the stdout-hash half of AC-19's admission criterion has no bathos
function to call at all; it is entirely the caller's responsibility, one layer further up than even
`manifest_verified` is. Both fields on :class:`EvidenceCandidate` are therefore plain,
caller-supplied results of checks this module does not perform itself -- matching
`stdout_sha256`'s own nullability on the `Run` schema (schema v11; the field defaults to `None`
because bathos's general `bth run` invocation does not yet capture stdout to a hashable sink), this
module models "no stdout hash was ever recorded to check" (``stdout_verified=None``) as a state
distinct from "recorded, but the hash didn't match" (``stdout_verified=False``) -- both exclude the
run, but a caller reading `ExcludedRun.reasons` can tell a genuine tampering signal apart from an
older run that simply predates stdout-hash capture. `manifest_verified` has no analogous `None`
state: `verify_run_manifest` itself already collapses "no manifest recorded" into a definite
``False`` (per its own docstring), so there is nothing further to distinguish on this module's side.

Cross-repo integration seam (matches `xtrax.run.repro_floor`, `xtrax.loop.metrics_provenance`,
`xtrax.run.component_binding`, `xtrax.loop.capability_probe_gate`, and
`xtrax.loop.stats_battery_gate` exactly -- all already-merged #2181 items crossing into bathos):
xtrax has no dependency on bathos, by design. This module never imports `bathos.schema.Run`, never
calls `bathos.prereg.verify_run_manifest`, and never re-hashes anything itself -- both
`manifest_verified` and `stdout_verified` are plain, caller-supplied values, assembled from whatever
the caller obtained by calling bathos's `verify_run_manifest` (for the manifest) and by
independently re-hashing its own stored stdout artifact against `Run.stdout_sha256` (for stdout).
The actual bathos call, and the actual stdout re-hash, both happen one layer up, outside this
module.

Shape choice: this is a SET-filtering concern -- "admit only runs whose ... verify; unverifiable ->
excluded" describes filtering a *collection* of candidate evidence runs down to an admitted subset,
with the excluded ones surfaced loudly (with reasons), not silently dropped -- a different shape
from most single-decision T2-1x gates. `xtrax.loop.diversity_quota` is the closer precedent here
(a pure decision function, not a raising gate, for the expected/common outcome) more than the
single-value gates (`capability_probe_gate`, `stats_battery_gate`): an unverifiable run is a normal,
expected occurrence in this evidence-admission process (a run predating stdout-hash capture, e.g.),
not a structural anomaly -- so `admit_evidence` never raises for that. It returns an
`EvidenceAdmissionResult` partitioning the input collection into `admitted` and `excluded` (each
excluded entry tagged with every reason it failed, per AC-19's "loud"). Only a genuinely malformed
candidate collection (a duplicate `run_id`) raises `EvidenceAdmissionInputError` -- matching the
`*InputError` naming convention used by `RatchetInputError`, `DiversityQuotaInputError`,
`StatsBatteryGateInputError`, and `CapabilityProbeInputError`.
"""

from collections.abc import Iterable
from dataclasses import dataclass

#: Reasons `admit_evidence` can attach to an `ExcludedRun` -- see module docstring.
_REASON_MANIFEST_UNVERIFIED = "manifest_unverified"
_REASON_STDOUT_NOT_RECORDED = "stdout_hash_not_recorded"
_REASON_STDOUT_MISMATCH = "stdout_hash_mismatch"


class EvidenceAdmissionInputError(Exception):
    """A candidate collection is structurally malformed: more than one candidate shares the same
    `run_id`.

    Distinct from a normal per-run exclusion (`ExcludedRun`, not an exception) -- an unverifiable
    run is this gate's expected, common outcome, not an error. This is raised only when the
    collection itself cannot be sensibly partitioned (ambiguous which candidate a `run_id` refers
    to).
    """


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """One candidate run's caller-supplied verification inputs.

    Both fields are the *result* of a check this module does not perform itself -- see the module
    docstring's cross-repo integration seam.

    Attributes:
        run_id: identifier for the candidate run (mirrors bathos `Run.id`).
        manifest_verified: the caller-obtained result of `bathos.prereg.verify_run_manifest(run)`
            (B2-07) -- `True` iff the manifest file recorded at run time still hashes to the
            `manifest_sha256` recorded at run time. Always a definite bool (see module docstring:
            `verify_run_manifest` itself already collapses "no manifest recorded" into `False`).
        stdout_verified: `True` iff `Run.stdout_sha256` (schema v11, nullable) was recorded AND the
            caller's own re-hash of the actual stored stdout artifact matched it; `False` if
            recorded but mismatched (a tampering signal); `None` if `Run.stdout_sha256` was never
            recorded at all -- there is genuinely nothing to check, a distinct condition from a
            verified mismatch (see module docstring).
    """

    run_id: str
    manifest_verified: bool
    stdout_verified: bool | None


@dataclass(frozen=True, slots=True)
class ExcludedRun:
    """A candidate excluded from the evidence set, plus every reason it failed.

    `reasons` may hold more than one entry (e.g. a run whose manifest AND stdout hash both fail to
    verify) -- AC-19's "excluded, loud" means surfacing everything wrong, not just the first check
    that failed.
    """

    candidate: EvidenceCandidate
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceAdmissionResult:
    """The composed AC-19 decision: `candidates` partitioned into `admitted` and `excluded`.

    `admitted` holds the full `EvidenceCandidate` (not just its `run_id`) so a caller has everything
    it needs without cross-referencing back into the original collection.
    """

    admitted: tuple[EvidenceCandidate, ...]
    excluded: tuple[ExcludedRun, ...]


def admit_evidence(candidates: Iterable[EvidenceCandidate]) -> EvidenceAdmissionResult:
    """The composed AC-19 gate: filter a collection of candidate runs down to only those whose
    attestation verifies.

    Args:
        candidates: candidate runs to evaluate for evidence-set admission. May be empty -- not
            every campaign step has candidate runs pending evidence admission; that is a normal
            input, not an error.

    Returns:
        An `EvidenceAdmissionResult`. A run failing either check is a normal, expected outcome
        (e.g. an older run that predates stdout-hash capture) -- this function never raises for
        it; every excluded run is tagged with every reason it failed, per AC-19's "excluded, loud".

    Raises:
        EvidenceAdmissionInputError: more than one candidate in `candidates` shares the same
            `run_id` -- a structurally malformed collection, not a normal exclusion.
    """
    candidate_list = list(candidates)

    run_ids = [candidate.run_id for candidate in candidate_list]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
        msg = f"duplicate run_id(s) in evidence-candidate collection: {duplicates}"
        raise EvidenceAdmissionInputError(msg)

    admitted: list[EvidenceCandidate] = []
    excluded: list[ExcludedRun] = []
    for candidate in candidate_list:
        reasons: list[str] = []
        if not candidate.manifest_verified:
            reasons.append(_REASON_MANIFEST_UNVERIFIED)
        if candidate.stdout_verified is None:
            reasons.append(_REASON_STDOUT_NOT_RECORDED)
        elif not candidate.stdout_verified:
            reasons.append(_REASON_STDOUT_MISMATCH)

        if reasons:
            excluded.append(ExcludedRun(candidate=candidate, reasons=tuple(reasons)))
        else:
            admitted.append(candidate)

    return EvidenceAdmissionResult(admitted=tuple(admitted), excluded=tuple(excluded))


__all__ = [
    "EvidenceAdmissionInputError",
    "EvidenceAdmissionResult",
    "EvidenceCandidate",
    "ExcludedRun",
    "admit_evidence",
]

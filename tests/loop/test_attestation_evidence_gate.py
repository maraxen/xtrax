"""Tests for the attestation-as-evidence gate (T2-26, #2181, AC-19)."""

import pytest

from xtrax.loop.attestation_evidence_gate import (
    EvidenceAdmissionInputError,
    EvidenceCandidate,
    admit_evidence,
)


class TestEmptyCollection:
    def test_empty_input_admits_nothing_and_excludes_nothing(self) -> None:
        result = admit_evidence([])
        assert result.admitted == ()
        assert result.excluded == ()


class TestAllVerified:
    def test_single_fully_verified_run_is_admitted(self) -> None:
        candidate = EvidenceCandidate(run_id="run-1", manifest_verified=True, stdout_verified=True)
        result = admit_evidence([candidate])
        assert result.admitted == (candidate,)
        assert result.excluded == ()

    def test_multiple_fully_verified_runs_are_all_admitted(self) -> None:
        candidates = [
            EvidenceCandidate(run_id=f"run-{i}", manifest_verified=True, stdout_verified=True)
            for i in range(5)
        ]
        result = admit_evidence(candidates)
        assert result.admitted == tuple(candidates)
        assert result.excluded == ()


class TestAllUnverifiable:
    def test_manifest_unverified_excludes_with_reason(self) -> None:
        candidate = EvidenceCandidate(run_id="run-1", manifest_verified=False, stdout_verified=True)
        result = admit_evidence([candidate])
        assert result.admitted == ()
        assert len(result.excluded) == 1
        assert result.excluded[0].candidate == candidate
        assert result.excluded[0].reasons == ("manifest_unverified",)

    def test_stdout_mismatch_excludes_with_reason(self) -> None:
        candidate = EvidenceCandidate(run_id="run-1", manifest_verified=True, stdout_verified=False)
        result = admit_evidence([candidate])
        assert result.admitted == ()
        assert result.excluded[0].reasons == ("stdout_hash_mismatch",)

    def test_stdout_not_recorded_excludes_with_distinct_reason(self) -> None:
        """A run with no stdout_sha256 ever recorded (schema-nullable, B2-07) is a distinct
        exclusion condition from a verified mismatch -- an older run predating stdout-hash
        capture, not necessarily a tampering signal."""
        candidate = EvidenceCandidate(run_id="run-1", manifest_verified=True, stdout_verified=None)
        result = admit_evidence([candidate])
        assert result.admitted == ()
        assert result.excluded[0].reasons == ("stdout_hash_not_recorded",)

    def test_both_checks_failing_reports_both_reasons(self) -> None:
        candidate = EvidenceCandidate(
            run_id="run-1", manifest_verified=False, stdout_verified=False
        )
        result = admit_evidence([candidate])
        assert result.excluded[0].reasons == (
            "manifest_unverified",
            "stdout_hash_mismatch",
        )

    def test_manifest_unverified_and_stdout_not_recorded_reports_both_reasons(self) -> None:
        candidate = EvidenceCandidate(run_id="run-1", manifest_verified=False, stdout_verified=None)
        result = admit_evidence([candidate])
        assert result.excluded[0].reasons == (
            "manifest_unverified",
            "stdout_hash_not_recorded",
        )

    def test_all_candidates_unverifiable_none_admitted(self) -> None:
        candidates = [
            EvidenceCandidate(run_id="run-1", manifest_verified=False, stdout_verified=True),
            EvidenceCandidate(run_id="run-2", manifest_verified=True, stdout_verified=False),
            EvidenceCandidate(run_id="run-3", manifest_verified=True, stdout_verified=None),
        ]
        result = admit_evidence(candidates)
        assert result.admitted == ()
        assert len(result.excluded) == 3


class TestMixedSet:
    def test_partitions_admitted_and_excluded_correctly(self) -> None:
        verified = EvidenceCandidate(
            run_id="run-verified", manifest_verified=True, stdout_verified=True
        )
        bad_manifest = EvidenceCandidate(
            run_id="run-bad-manifest", manifest_verified=False, stdout_verified=True
        )
        bad_stdout = EvidenceCandidate(
            run_id="run-bad-stdout", manifest_verified=True, stdout_verified=False
        )
        no_stdout_hash = EvidenceCandidate(
            run_id="run-no-stdout-hash", manifest_verified=True, stdout_verified=None
        )
        result = admit_evidence([verified, bad_manifest, bad_stdout, no_stdout_hash])

        assert result.admitted == (verified,)
        excluded_run_ids = {excluded.candidate.run_id for excluded in result.excluded}
        assert excluded_run_ids == {"run-bad-manifest", "run-bad-stdout", "run-no-stdout-hash"}

    def test_preserves_input_order_within_each_bucket(self) -> None:
        a = EvidenceCandidate(run_id="a", manifest_verified=True, stdout_verified=True)
        b = EvidenceCandidate(run_id="b", manifest_verified=False, stdout_verified=True)
        c = EvidenceCandidate(run_id="c", manifest_verified=True, stdout_verified=True)
        d = EvidenceCandidate(run_id="d", manifest_verified=False, stdout_verified=True)

        result = admit_evidence([a, b, c, d])

        assert result.admitted == (a, c)
        assert [excluded.candidate.run_id for excluded in result.excluded] == ["b", "d"]


class TestDuplicateRunIdRaises:
    def test_duplicate_run_id_raises(self) -> None:
        candidates = [
            EvidenceCandidate(run_id="dup", manifest_verified=True, stdout_verified=True),
            EvidenceCandidate(run_id="dup", manifest_verified=False, stdout_verified=None),
        ]
        with pytest.raises(EvidenceAdmissionInputError, match="dup"):
            admit_evidence(candidates)

    def test_duplicate_among_otherwise_distinct_ids_raises(self) -> None:
        candidates = [
            EvidenceCandidate(run_id="run-1", manifest_verified=True, stdout_verified=True),
            EvidenceCandidate(run_id="run-2", manifest_verified=True, stdout_verified=True),
            EvidenceCandidate(run_id="run-1", manifest_verified=True, stdout_verified=True),
        ]
        with pytest.raises(EvidenceAdmissionInputError, match="run-1"):
            admit_evidence(candidates)


class TestEvidenceCandidateIsFrozen:
    def test_is_frozen(self) -> None:
        candidate = EvidenceCandidate(run_id="run-1", manifest_verified=True, stdout_verified=True)
        with pytest.raises(AttributeError):
            candidate.manifest_verified = False

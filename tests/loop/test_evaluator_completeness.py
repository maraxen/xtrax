"""Tests for the evaluator-completeness gate (T2-08, #2181, AC-11, PM-2)."""

from datetime import UTC, datetime

import pytest

from xtrax.devtools.freshness import Attestation
from xtrax.loop.evaluator_completeness import (
    Invariant,
    InvariantManifest,
    InvariantManifestStaleError,
    InvariantViolationError,
    SyntheticGroundTruthCase,
    SyntheticSanityCheckFailedError,
    assert_evaluator_completeness,
    run_synthetic_sanity_check,
)

FITNESS_BY_CANDIDATE = {"a": 0.0, "b": 1.0, "c": 0.2}


def _one_hot_evaluator(frozen_context: object, candidate: str) -> dict[str, float]:
    del frozen_context
    return {"score": FITNESS_BY_CANDIDATE[candidate]}


def _sanity_case() -> SyntheticGroundTruthCase:
    return SyntheticGroundTruthCase(
        candidates={c: None for c in FITNESS_BY_CANDIDATE}, known_best="b"
    )


def _fresh_attestation(now: datetime | None = None) -> Attestation:
    return Attestation(
        attested_at=(now or datetime.now(UTC)).isoformat(), ttl_days=30.0, attested_by="test"
    )


def _stale_attestation() -> Attestation:
    return Attestation(attested_at="2000-01-01T00:00:00+00:00", ttl_days=1.0, attested_by="test")


def _manifest(invariants: tuple[Invariant, ...] = ()) -> InvariantManifest:
    return InvariantManifest(invariants=invariants, attestation=_fresh_attestation())


class TestRunSyntheticSanityCheck:
    def test_returns_every_candidates_fitness(self) -> None:
        results = run_synthetic_sanity_check(_one_hot_evaluator, _sanity_case())
        assert results == {
            "a": {"score": 0.0},
            "b": {"score": 1.0},
            "c": {"score": 0.2},
        }

    def test_raises_when_known_best_does_not_score_top(self) -> None:
        case = SyntheticGroundTruthCase(
            candidates={c: None for c in FITNESS_BY_CANDIDATE}, known_best="a"
        )
        with pytest.raises(SyntheticSanityCheckFailedError, match="did not score strictly top"):
            run_synthetic_sanity_check(_one_hot_evaluator, case)

    def test_raises_on_tie(self) -> None:
        def tied_evaluator(frozen_context: object, candidate: str) -> dict[str, float]:
            del frozen_context
            return {"score": 1.0}

        with pytest.raises(SyntheticSanityCheckFailedError):
            run_synthetic_sanity_check(tied_evaluator, _sanity_case())

    def test_raises_when_known_best_not_in_candidates(self) -> None:
        case = SyntheticGroundTruthCase(candidates={"a": None}, known_best="z")
        with pytest.raises(SyntheticSanityCheckFailedError, match="not one of"):
            run_synthetic_sanity_check(_one_hot_evaluator, case)


class TestAssertEvaluatorCompleteness:
    def test_passes_silently_when_manifest_fresh_and_sanity_check_passes(self) -> None:
        assert_evaluator_completeness(_one_hot_evaluator, _manifest(), _sanity_case())

    def test_raises_invariant_manifest_stale_error(self) -> None:
        stale_manifest = InvariantManifest(invariants=(), attestation=_stale_attestation())
        with pytest.raises(InvariantManifestStaleError, match="not reviewed/fresh"):
            assert_evaluator_completeness(_one_hot_evaluator, stale_manifest, _sanity_case())

    def test_stale_manifest_short_circuits_before_running_sanity_check(self) -> None:
        stale_manifest = InvariantManifest(invariants=(), attestation=_stale_attestation())
        calls = []

        def counting_evaluator(frozen_context: object, candidate: str) -> dict[str, float]:
            calls.append(candidate)
            return {"score": 0.0}

        with pytest.raises(InvariantManifestStaleError):
            assert_evaluator_completeness(counting_evaluator, stale_manifest, _sanity_case())
        assert calls == []

    def test_raises_synthetic_sanity_check_failed_error(self) -> None:
        case = SyntheticGroundTruthCase(
            candidates={c: None for c in FITNESS_BY_CANDIDATE}, known_best="a"
        )
        with pytest.raises(SyntheticSanityCheckFailedError):
            assert_evaluator_completeness(_one_hot_evaluator, _manifest(), case)

    def test_passing_invariant_does_not_raise(self) -> None:
        always_positive = Invariant(
            name="score_non_negative",
            description="fitness score must never be negative",
            check=lambda fitness: fitness["score"] >= 0.0,
        )
        assert_evaluator_completeness(
            _one_hot_evaluator, _manifest((always_positive,)), _sanity_case()
        )

    def test_raises_invariant_violation_error_naming_the_failing_invariant(self) -> None:
        impossible = Invariant(
            name="score_above_ten",
            description="fitness score must exceed 10 (deliberately impossible, for this test)",
            check=lambda fitness: fitness["score"] > 10.0,
        )
        with pytest.raises(InvariantViolationError, match="score_above_ten"):
            assert_evaluator_completeness(
                _one_hot_evaluator, _manifest((impossible,)), _sanity_case()
            )

    def test_first_failing_invariant_short_circuits_remaining_invariants(self) -> None:
        checked_names = []

        def make_invariant(name: str, *, passes: bool) -> Invariant:
            def check(fitness: dict[str, float]) -> bool:
                checked_names.append(name)
                return passes

            return Invariant(name=name, description=name, check=check)

        manifest = _manifest(
            (
                make_invariant("first_fails", passes=False),
                make_invariant("second_never_reached", passes=True),
            )
        )
        with pytest.raises(InvariantViolationError, match="first_fails"):
            assert_evaluator_completeness(_one_hot_evaluator, manifest, _sanity_case())
        assert "second_never_reached" not in checked_names

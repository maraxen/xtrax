"""Tests for the info-barrier-lint gate (T2-07, #2181, AC-9, F3, PM-5, ORTH-2)."""

import pytest

from xtrax.loop.info_barrier_lint import (
    DeltaExposurePolicy,
    FitnessDeltaExposureError,
    NonWhitelistedFieldError,
    PerSplitGranularityError,
    RawLogPathExposedError,
    SanctionedEnvelope,
    build_sanctioned_envelope,
    lint_agent_envelope,
)
from xtrax.loop.metrics_provenance import MetricsProvenanceRecord


def _record(score: float, run_id: str = "run-1") -> MetricsProvenanceRecord:
    return MetricsProvenanceRecord(
        fitness={"score": score},
        evaluator_closure_hash="deadbeef",
        run_id=run_id,
        created_at="2026-07-15T00:00:00+00:00",
    )


class TestLintAgentEnvelope:
    def test_accepts_a_fully_sanctioned_envelope(self) -> None:
        lint_agent_envelope(
            {"status": "ok", "run_id": "run-1", "iteration": 1, "fitness_trend": "improved"}
        )

    def test_raises_on_raw_log_field_name(self) -> None:
        with pytest.raises(RawLogPathExposedError, match="stdout"):
            lint_agent_envelope({"status": "ok", "stdout": "..."})

    def test_raises_on_raw_log_marker_in_value(self) -> None:
        with pytest.raises(RawLogPathExposedError, match="error_taxonomy"):
            lint_agent_envelope(
                {
                    "status": "error",
                    "run_id": "run-1",
                    "iteration": 1,
                    "error_taxonomy": "see /var/log/train.log for details",
                }
            )

    def test_raises_on_per_split_field(self) -> None:
        with pytest.raises(PerSplitGranularityError, match="train_accuracy"):
            lint_agent_envelope({"status": "ok", "train_accuracy": 0.9})

    def test_raises_on_per_split_field_alias(self) -> None:
        with pytest.raises(PerSplitGranularityError, match="splits"):
            lint_agent_envelope({"status": "ok", "splits": {"train": 0.9}})

    def test_raises_on_non_whitelisted_field(self) -> None:
        with pytest.raises(NonWhitelistedFieldError, match="internal_debug_id"):
            lint_agent_envelope({"status": "ok", "internal_debug_id": 42})

    def test_raw_log_check_takes_priority_over_whitelist_check(self) -> None:
        with pytest.raises(RawLogPathExposedError):
            lint_agent_envelope({"stdout": "..."})

    def test_raises_on_over_precision_fitness_delta(self) -> None:
        envelope = {
            "status": "ok",
            "run_id": "run-1",
            "iteration": 3,
            "fitness_delta": 0.123456,
        }
        with pytest.raises(FitnessDeltaExposureError, match="0.123456"):
            lint_agent_envelope(envelope, delta_policy=DeltaExposurePolicy(precision=2))

    def test_accepts_correctly_rounded_fitness_delta(self) -> None:
        envelope = {
            "status": "ok",
            "run_id": "run-1",
            "iteration": 3,
            "fitness_delta": 0.12,
        }
        lint_agent_envelope(envelope, delta_policy=DeltaExposurePolicy(precision=2))

    def test_skips_delta_precision_check_when_no_policy_given(self) -> None:
        envelope = {
            "status": "ok",
            "run_id": "run-1",
            "iteration": 3,
            "fitness_delta": 0.123456789,
        }
        lint_agent_envelope(envelope)

    def test_raises_fitness_delta_exposure_error_on_non_numeric_delta(self) -> None:
        envelope = {
            "status": "ok",
            "run_id": "run-1",
            "iteration": 3,
            "fitness_delta": "not-a-number",
        }
        with pytest.raises(FitnessDeltaExposureError, match="not numeric"):
            lint_agent_envelope(envelope, delta_policy=DeltaExposurePolicy())


class TestDeltaExposurePolicy:
    def test_reveal_every_zero_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="reveal_every must be >= 1"):
            DeltaExposurePolicy(reveal_every=0)

    def test_negative_reveal_every_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="reveal_every must be >= 1"):
            DeltaExposurePolicy(reveal_every=-1)

    def test_negative_precision_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="precision must be >= 0"):
            DeltaExposurePolicy(precision=-1)


class TestSanctionedEnvelope:
    def test_to_json_dict_drops_none_fields(self) -> None:
        envelope = SanctionedEnvelope(status="ok", run_id="run-1", iteration=0)
        assert envelope.to_json_dict() == {"status": "ok", "run_id": "run-1", "iteration": 0}

    def test_to_json_dict_keeps_set_fields(self) -> None:
        envelope = SanctionedEnvelope(
            status="ok",
            run_id="run-1",
            iteration=3,
            fitness_trend="improved",
            fitness_delta=0.1,
        )
        assert envelope.to_json_dict() == {
            "status": "ok",
            "run_id": "run-1",
            "iteration": 3,
            "fitness_trend": "improved",
            "fitness_delta": 0.1,
        }


class TestBuildSanctionedEnvelope:
    def test_first_iteration_omits_trend_and_delta(self) -> None:
        envelope = build_sanctioned_envelope(_record(1.0), iteration=0)
        assert envelope.fitness_trend is None
        assert envelope.fitness_delta is None

    def test_improved_score_yields_improved_trend(self) -> None:
        reveal_always = DeltaExposurePolicy(reveal_every=1)
        envelope = build_sanctioned_envelope(
            _record(1.0), iteration=3, previous_score=0.5, delta_policy=reveal_always
        )
        assert envelope.fitness_trend == "improved"

    def test_regressed_score_yields_regressed_trend(self) -> None:
        reveal_always = DeltaExposurePolicy(reveal_every=1)
        envelope = build_sanctioned_envelope(
            _record(0.2), iteration=3, previous_score=0.5, delta_policy=reveal_always
        )
        assert envelope.fitness_trend == "regressed"

    def test_unchanged_score_yields_unchanged_trend(self) -> None:
        reveal_always = DeltaExposurePolicy(reveal_every=1)
        envelope = build_sanctioned_envelope(
            _record(0.5), iteration=3, previous_score=0.5, delta_policy=reveal_always
        )
        assert envelope.fitness_trend == "unchanged"

    def test_delta_only_revealed_on_reveal_every_boundary(self) -> None:
        policy = DeltaExposurePolicy(precision=2, reveal_every=3)

        withheld = build_sanctioned_envelope(
            _record(1.0), iteration=1, previous_score=0.5, delta_policy=policy
        )
        revealed = build_sanctioned_envelope(
            _record(1.0), iteration=3, previous_score=0.5, delta_policy=policy
        )

        assert withheld.fitness_delta is None
        assert withheld.fitness_trend == "improved"
        assert revealed.fitness_delta == pytest.approx(0.5)

    def test_revealed_delta_is_rounded_to_policy_precision(self) -> None:
        policy = DeltaExposurePolicy(precision=2, reveal_every=1)
        envelope = build_sanctioned_envelope(
            _record(0.123456), iteration=1, previous_score=0.0, delta_policy=policy
        )
        assert envelope.fitness_delta == pytest.approx(0.12)

    def test_result_always_passes_its_own_lint(self) -> None:
        envelope = build_sanctioned_envelope(_record(1.0), iteration=3, previous_score=0.5)
        lint_agent_envelope(envelope.to_json_dict())

    def test_error_taxonomy_is_carried_through(self) -> None:
        envelope = build_sanctioned_envelope(
            _record(1.0), iteration=0, status="error", error_taxonomy="shape_mismatch"
        )
        assert envelope.status == "error"
        assert envelope.error_taxonomy == "shape_mismatch"

    def test_uses_custom_score_key(self) -> None:
        record = MetricsProvenanceRecord(
            fitness={"custom_metric": 2.0},
            evaluator_closure_hash="deadbeef",
            run_id="run-1",
        )
        envelope = build_sanctioned_envelope(
            record,
            iteration=1,
            previous_score=1.0,
            score_key="custom_metric",
            delta_policy=DeltaExposurePolicy(reveal_every=1),
        )
        assert envelope.fitness_trend == "improved"

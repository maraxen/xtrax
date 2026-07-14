"""Tests for T1-13's `graph_author_bench` harness (#3068)."""

from pathlib import Path

import pytest

from scripts.author_bench_fixtures import TASKS
from scripts.graph_author_bench import (
    SPOT_CHECK_PASS_BAR,
    ModeResult,
    TaskOutcome,
    _spot_check_gate_passes,
    build_report,
    decide_fork12,
    main,
    run_generate_then_validate_mode,
    run_spot_check,
)


class TestGenerateThenValidateMode:
    def test_is_deterministic(self) -> None:
        result_a = run_generate_then_validate_mode()
        result_b = run_generate_then_validate_mode()
        assert result_a == result_b

    def test_rates_are_in_bounds(self) -> None:
        result = run_generate_then_validate_mode()
        assert 0.0 <= result.structural_validity_yield <= 1.0
        assert 0.0 <= result.faithfulness_rate <= 1.0

    def test_covers_every_task(self) -> None:
        result = run_generate_then_validate_mode()
        assert len(result.task_outcomes) == len(TASKS)

    def test_structural_validity_is_high(self) -> None:
        """TemplateGenerator always builds resolvable, structurally-valid graphs from the
        example-callable pool -- this must not silently regress.
        """
        result = run_generate_then_validate_mode()
        assert result.structural_validity_yield == 1.0


class TestSpotCheck:
    def test_passes_fully_against_the_real_fixture(self) -> None:
        result = run_spot_check()
        assert result.pass_rate == 1.0

    def test_gate_boundary(self) -> None:
        assert _spot_check_gate_passes(SPOT_CHECK_PASS_BAR) is True
        assert _spot_check_gate_passes(SPOT_CHECK_PASS_BAR - 0.01) is False
        assert _spot_check_gate_passes(1.0) is True
        assert _spot_check_gate_passes(0.0) is False


def _mode_result(mode: str, validity: float, faithfulness: float) -> ModeResult:
    """Build a ModeResult with the given aggregate rates via synthetic pass/fail outcomes.

    n=100 so two-decimal-place rates (0.59, 0.45, ...) are exactly representable -- n=10
    rounds them to the nearest tenth, silently shifting a test's intended boundary value.
    """
    n = 100
    n_valid = round(validity * n)
    n_faithful = round(faithfulness * n)
    outcomes = tuple(TaskOutcome(f"t{i:02d}", i < n_valid, i < n_faithful) for i in range(n))
    return ModeResult(mode, outcomes)


class TestDecideFork12:
    def test_undecided_when_constrained_decoding_not_run(self) -> None:
        gtv = _mode_result("generate-then-validate", 1.0, 0.1)
        assert decide_fork12(gtv, None) == "UNDECIDED (constrained-decoding mode not run)"

    def test_adopts_constrained_decoding_when_validity_gain_and_faithfulness_hold(self) -> None:
        gtv = _mode_result("generate-then-validate", 0.5, 0.5)
        cd = _mode_result("constrained-decoding", 0.6, 0.5)  # +10pp validity, 0pp faithfulness
        assert decide_fork12(gtv, cd) == "ADOPT_CONSTRAINED_DECODING"

    def test_keeps_generate_then_validate_when_validity_gain_insufficient(self) -> None:
        gtv = _mode_result("generate-then-validate", 0.5, 0.5)
        cd = _mode_result("constrained-decoding", 0.59, 0.5)  # +9pp, below the 10pp margin
        assert decide_fork12(gtv, cd) == "KEEP_GENERATE_THEN_VALIDATE"

    def test_keeps_generate_then_validate_when_faithfulness_regresses_beyond_margin(self) -> None:
        gtv = _mode_result("generate-then-validate", 0.5, 0.5)
        cd = _mode_result("constrained-decoding", 0.7, 0.4)  # +20pp validity, -10pp faithfulness
        assert decide_fork12(gtv, cd) == "KEEP_GENERATE_THEN_VALIDATE"

    def test_faithfulness_regression_within_margin_still_adopts(self) -> None:
        gtv = _mode_result("generate-then-validate", 0.5, 0.5)
        cd = _mode_result("constrained-decoding", 0.7, 0.45)  # +20pp validity, -5pp (at margin)
        assert decide_fork12(gtv, cd) == "ADOPT_CONSTRAINED_DECODING"


class TestBuildReport:
    def test_report_always_carries_both_absolute_rates(self) -> None:
        gtv = run_generate_then_validate_mode()
        spot_check = run_spot_check()
        report = build_report(gtv, None, spot_check)
        assert report["generate_then_validate"]["structural_validity_yield"] is not None
        assert report["generate_then_validate"]["faithfulness_rate"] is not None
        assert report["constrained_decoding"] is None
        assert report["human_labeled_spot_check"]["gate_passed"] is True


class TestMain:
    def test_exits_zero_without_constrained_decoding_flag(self, tmp_path: Path) -> None:
        out_path = tmp_path / "report.json"
        exit_code = main(["--out", str(out_path)])
        assert exit_code == 0
        assert out_path.exists()

    def test_hard_fails_when_constrained_decoding_requested_but_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.graph_author_bench as bench

        def _raise_import_error(**kwargs):
            raise ImportError("outlines not installed")

        monkeypatch.setattr(bench, "run_constrained_decoding_mode", _raise_import_error)
        out_path = tmp_path / "report.json"
        exit_code = main(["--include-constrained-decoding", "--out", str(out_path)])
        assert exit_code == 1
        assert not out_path.exists()

    def test_exits_nonzero_when_spot_check_gate_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.graph_author_bench as bench

        def _failing_spot_check():
            return bench.SpotCheckResult(n=30, n_passed=10)  # 33% < 80% bar

        monkeypatch.setattr(bench, "run_spot_check", _failing_spot_check)
        out_path = tmp_path / "report.json"
        exit_code = main(["--out", str(out_path)])
        assert exit_code == 1
        # The report is still written -- the gate fails loud, it doesn't hide the measurement.
        assert out_path.exists()


class TestRunConstrainedDecodingModeDegenerateDecode:
    """T1-12's own confirmed live run produced schema_version=0, empty nodes -- a document
    that fails HostPrepGraph deserialization outright. This must be scored as structurally
    invalid / unfaithful, not crash the harness. Fully mocked -- exercises real harness logic
    without requiring outlines/transformers/torch to be installed.
    """

    def test_degenerate_decode_scores_as_invalid_and_unfaithful(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.graph_author_bench as bench
        import scripts.smoke_outlines_constrained_decode as smoke

        monkeypatch.setattr(smoke, "run_smoke", lambda **kwargs: {"schema_version": 0, "nodes": []})
        result = bench.run_constrained_decoding_mode(tasks=TASKS[:2])
        assert len(result.task_outcomes) == 2
        assert result.structural_validity_yield == 0.0
        assert result.faithfulness_rate == 0.0
        for outcome in result.task_outcomes:
            assert outcome.error is not None

    def test_malformed_callable_ref_scores_as_invalid_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.graph_author_bench as bench
        import scripts.smoke_outlines_constrained_decode as smoke

        def _fake_decode(**kwargs):
            return {
                "schema_version": 1,
                "nodes": [
                    {
                        "id": "n0",
                        "callable_ref": "not.a.real:module",
                        "metadata": {"nl_description": "fake"},
                    }
                ],
                "edges": [],
            }

        monkeypatch.setattr(smoke, "run_smoke", _fake_decode)
        result = bench.run_constrained_decoding_mode(tasks=TASKS[:1])
        assert result.task_outcomes[0].structurally_valid is False
        assert result.task_outcomes[0].faithful is False

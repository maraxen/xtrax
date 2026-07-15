"""Tests for the compile-time two-phase clock (T2-19, #2181, AC-27, fork 3, ORTH-4)."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from xtrax.loop.compile_time_clock import (
    CompileTimeClockError,
    CompileTimeRegressionError,
    assert_no_compile_time_regression,
    measure_two_phase_timing,
    rolling_median,
)
from xtrax.loop.schema_gate import CandidateResolutionError

_CANDIDATE_SOURCE = """
def add(a, b):
    return a + b
"""


@pytest.fixture
def candidate_path(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.py"
    path.write_text(_CANDIDATE_SOURCE)
    return path


class TestMeasureTwoPhaseTiming:
    def test_returns_nonnegative_compile_time_and_correct_result(
        self, candidate_path: Path
    ) -> None:
        timing = measure_two_phase_timing(
            candidate_path,
            "add",
            concrete_inputs=[jnp.array(2.0), jnp.array(3.0)],
        )
        assert timing.compile_time_seconds >= 0.0
        assert timing.runtime_seconds >= 0.0
        assert float(timing.result) == pytest.approx(5.0)

    def test_candidate_resolution_error_propagates_unwrapped(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.py"
        with pytest.raises(CandidateResolutionError):
            measure_two_phase_timing(
                missing_path,
                "add",
                concrete_inputs=[jnp.array(1.0), jnp.array(1.0)],
            )

    def test_missing_callable_raises_candidate_resolution_error(self, candidate_path: Path) -> None:
        with pytest.raises(CandidateResolutionError, match="no attribute"):
            measure_two_phase_timing(
                candidate_path,
                "does_not_exist",
                concrete_inputs=[jnp.array(1.0), jnp.array(1.0)],
            )


class TestRollingMedian:
    def test_odd_length_history(self) -> None:
        assert rolling_median([1.0, 3.0, 2.0]) == pytest.approx(2.0)

    def test_even_length_history(self) -> None:
        assert rolling_median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_empty_history_raises(self) -> None:
        with pytest.raises(CompileTimeClockError, match="empty history"):
            rolling_median([])


class TestRegressionGate:
    def test_compile_time_far_exceeding_threshold_raises(self) -> None:
        with pytest.raises(CompileTimeRegressionError, match="exceeds"):
            assert_no_compile_time_regression(
                10.0, compile_time_history=[1.0, 1.0, 1.0], k_threshold=3.0
            )

    def test_compile_time_within_budget_does_not_raise(self) -> None:
        assert_no_compile_time_regression(
            2.0, compile_time_history=[1.0, 1.0, 1.0], k_threshold=3.0
        )

    def test_compile_time_exactly_at_threshold_does_not_raise(self) -> None:
        assert_no_compile_time_regression(
            3.0, compile_time_history=[1.0, 1.0, 1.0], k_threshold=3.0
        )

    def test_empty_history_never_raises(self) -> None:
        assert_no_compile_time_regression(1000.0, compile_time_history=[], k_threshold=3.0)

    def test_all_zero_history_never_raises(self) -> None:
        assert_no_compile_time_regression(
            1000.0, compile_time_history=[0.0, 0.0, 0.0], k_threshold=3.0
        )

    def test_custom_k_threshold(self) -> None:
        with pytest.raises(CompileTimeRegressionError):
            assert_no_compile_time_regression(
                1.5, compile_time_history=[1.0, 1.0, 1.0], k_threshold=1.0
            )

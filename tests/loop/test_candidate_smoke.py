"""Tests for candidate-smoke (T2-14, #2181, AC-4, F4)."""

import time
from pathlib import Path

import numpy as np
import pytest

from xtrax.loop.candidate_smoke import (
    _OUTPUT_TRUNCATE_CHARS,
    CandidateSmokeFailedError,
    CandidateSmokeTimeoutError,
    assert_candidate_smoke,
)

_CLEAN_CANDIDATE_SRC = """
def transform(x):
    return x * 2
"""

_SYNTAX_ERROR_SRC = """
def transform(:
    pass
"""

_HANGING_IGNORES_SIGTERM_SRC = """
import signal
import time

def transform(x):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
    return x
"""

_RAISES_DURING_CALL_SRC = """
def transform(x):
    raise ValueError("deliberate failure for test")
"""

_CHECKS_CPU_ONLY_ENV_SRC = """
import os

def transform(x):
    platforms = os.environ.get("JAX_PLATFORMS")
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert platforms == "cpu", f"JAX_PLATFORMS={platforms!r}"
    assert cuda_devices == "", f"CUDA_VISIBLE_DEVICES={cuda_devices!r}"
    return x * 2
"""

_RAISES_ONLY_IF_CALLED_SRC = """
def transform(x):
    raise RuntimeError("should never run during L1")
"""

_FLOODS_STDOUT_THEN_RAISES_SRC = """
def transform(x):
    print("x" * 100_000)
    raise ValueError("failure after a flood of stdout")
"""

_DEFAULT_INPUTS = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]


def _write_candidate(tmp_path: Path, source: str, filename: str = "candidate.py") -> Path:
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


class TestAssertCandidateSmoke:
    def test_clean_candidate_passes_silently(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _CLEAN_CANDIDATE_SRC)
        result = assert_candidate_smoke(
            path,
            "transform",
            concrete_inputs=_DEFAULT_INPUTS,
            wall_clock_budget_seconds=15.0,
        )
        assert result is None

    def test_syntax_error_fails_at_l1_before_l2_ever_runs(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SYNTAX_ERROR_SRC)
        with pytest.raises(CandidateSmokeFailedError) as exc_info:
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=15.0,
            )
        message = str(exc_info.value)
        assert "L1 dry-run" in message
        assert "L2 CPU smoke" not in message

    def test_hanging_candidate_ignoring_sigterm_is_hard_killed(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _HANGING_IGNORES_SIGTERM_SRC)
        started = time.monotonic()
        with pytest.raises(CandidateSmokeTimeoutError):
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=2.0,
                poll_interval_seconds=0.1,
            )
        elapsed = time.monotonic() - started
        # Bounded well below the candidate's own 30s sleep -- proves the watchdog's SIGKILL (not
        # a softer termination the candidate's own SIGTERM-ignore handler could survive) is what
        # actually stopped it, and assert_candidate_smoke's communicate() call didn't hang.
        assert elapsed < 10.0

    def test_zero_budget_rejects_pre_budget_with_no_subprocess_spawned(
        self, tmp_path: Path
    ) -> None:
        path = _write_candidate(tmp_path, _CLEAN_CANDIDATE_SRC)
        started = time.monotonic()
        with pytest.raises(CandidateSmokeTimeoutError, match="pre-budget"):
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=0.0,
            )
        elapsed = time.monotonic() - started
        # A real `uv run --locked` subprocess costs tens of ms minimum (confirmed ~50ms); a
        # near-instant return proves no subprocess was spawned at all.
        assert elapsed < 0.05

    def test_negative_budget_also_rejects_pre_budget(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _CLEAN_CANDIDATE_SRC)
        with pytest.raises(CandidateSmokeTimeoutError, match="pre-budget"):
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=-1.0,
            )

    def test_candidate_exception_during_l2_reported_with_truncated_traceback(
        self, tmp_path: Path
    ) -> None:
        path = _write_candidate(tmp_path, _RAISES_DURING_CALL_SRC)
        with pytest.raises(CandidateSmokeFailedError) as exc_info:
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=15.0,
            )
        message = str(exc_info.value)
        assert "L2 CPU smoke" in message
        assert "ValueError" in message
        assert "deliberate failure for test" in message

    def test_oversized_output_is_actually_truncated_not_just_documented(
        self, tmp_path: Path
    ) -> None:
        path = _write_candidate(tmp_path, _FLOODS_STDOUT_THEN_RAISES_SRC)
        with pytest.raises(CandidateSmokeFailedError) as exc_info:
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=15.0,
            )
        message = str(exc_info.value)
        # The candidate printed 100_000 chars; a length regression (e.g. someone widening
        # _OUTPUT_TRUNCATE_CHARS or dropping the slice) would silently balloon this message --
        # bound it explicitly, not just check substring presence.
        assert len(message) < _OUTPUT_TRUNCATE_CHARS + 100
        assert "ValueError" in message

    def test_cpu_only_environment_reaches_the_subprocess(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _CHECKS_CPU_ONLY_ENV_SRC)
        result = assert_candidate_smoke(
            path,
            "transform",
            concrete_inputs=_DEFAULT_INPUTS,
            wall_clock_budget_seconds=15.0,
        )
        assert result is None

    def test_l1_does_not_execute_the_candidates_function_body(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _RAISES_ONLY_IF_CALLED_SRC)
        with pytest.raises(CandidateSmokeFailedError) as exc_info:
            assert_candidate_smoke(
                path,
                "transform",
                concrete_inputs=_DEFAULT_INPUTS,
                wall_clock_budget_seconds=15.0,
            )
        message = str(exc_info.value)
        # If L1 had executed the function body, this would fail at "L1 dry-run" instead.
        assert "L2 CPU smoke" in message
        assert "RuntimeError" in message
        assert "should never run during L1" in message

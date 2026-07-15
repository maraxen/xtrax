"""Tests for checkified-execution (T2-15, #2181, AC-5)."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from xtrax.loop.checkified_execution import (
    CheckifiedExecutionError,
    assert_checkified_execution,
)
from xtrax.loop.schema_gate import CandidateResolutionError

_CLEAN_CANDIDATE_SRC = """
def transform(x):
    return x * 2.0
"""

_NAN_CANDIDATE_SRC = """
import jax.numpy as jnp

def transform(x):
    return jnp.sqrt(x - x - 1.0)
"""

_DIV_BY_ZERO_CANDIDATE_SRC = """
import jax.numpy as jnp

def transform(x):
    return x / jnp.array(0.0)
"""

_OVERFLOW_INF_CANDIDATE_SRC = """
import jax.numpy as jnp

def transform(x):
    return jnp.exp(x * 0.0 + 1000.0)
"""

_WRONG_ARITY_CANDIDATE_SRC = """
def transform(x, y):
    return x + y
"""


def _write_candidate(tmp_path: Path, source: str, filename: str = "candidate.py") -> Path:
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


class TestAssertCheckifiedExecution:
    def test_clean_candidate_returns_correct_result(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _CLEAN_CANDIDATE_SRC)
        result = assert_checkified_execution(path, "transform", concrete_inputs=[jnp.array(3.0)])
        assert float(result) == pytest.approx(6.0)

    def test_nan_is_detected_by_checkify(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _NAN_CANDIDATE_SRC)
        with pytest.raises(CheckifiedExecutionError, match="NaN/division-by-zero"):
            assert_checkified_execution(path, "transform", concrete_inputs=[jnp.array(1.0)])

    def test_division_by_zero_is_detected_by_checkify(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _DIV_BY_ZERO_CANDIDATE_SRC)
        with pytest.raises(CheckifiedExecutionError, match="NaN/division-by-zero"):
            assert_checkified_execution(path, "transform", concrete_inputs=[jnp.array(5.0)])

    def test_overflow_inf_is_detected_by_manual_isinf_sweep_not_checkify(
        self, tmp_path: Path
    ) -> None:
        """checkify.float_checks has no dedicated overflow/Inf detector -- verified empirically
        (see module docstring) that jnp.exp(1000.0) returns Inf without checkify's err.throw()
        ever firing. This test's error message must confirm it was OUR sweep, not checkify's
        exception, that caught it -- proving the manual isinf check is load-bearing, not
        redundant with the checkify-detected cases above.
        """
        path = _write_candidate(tmp_path, _OVERFLOW_INF_CANDIDATE_SRC)
        with pytest.raises(CheckifiedExecutionError, match="not caught by checkify.float_checks"):
            assert_checkified_execution(path, "transform", concrete_inputs=[jnp.array(1.0)])

    def test_candidate_resolution_error_propagates_unwrapped(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, "def transform(:\n    pass\n")
        with pytest.raises(CandidateResolutionError):
            assert_checkified_execution(path, "transform", concrete_inputs=[jnp.array(1.0)])

    def test_unrelated_candidate_bug_is_not_mislabeled_as_checkified_execution_error(
        self, tmp_path: Path
    ) -> None:
        path = _write_candidate(tmp_path, _WRONG_ARITY_CANDIDATE_SRC)
        with pytest.raises(TypeError):
            assert_checkified_execution(path, "transform", concrete_inputs=[jnp.array(1.0)])

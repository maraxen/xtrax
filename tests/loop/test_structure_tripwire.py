"""Tests for the structure-tripwire (T2-13, #2181, AC-3)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.loop.schema_gate import CandidateResolutionError
from xtrax.loop.structure_tripwire import StructureMismatchError, assert_structure_tripwire

_SCALE_CANDIDATE_SRC = """
def transform(x):
    return x * 2
"""

_SYNTAX_ERROR_SRC = """
def transform(:
    pass
"""


def _write_candidate(tmp_path: Path, source: str, filename: str = "candidate.py") -> Path:
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


def _counting_candidate_src(counter_path: Path) -> str:
    return f"""
import jax

def transform(x):
    if not isinstance(x, jax.core.Tracer):
        with open({str(counter_path)!r}, "a") as f:
            f.write("1")
    return x * 2
"""


class TestAssertStructureTripwire:
    def test_matching_structure_passes_silently(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SCALE_CANDIDATE_SRC)
        result = assert_structure_tripwire(
            path,
            "transform",
            abstract_inputs=[ShapeDtypeStruct((3,), np.float32)],
            concrete_inputs=[jnp.ones((3,), dtype=jnp.float32)],
        )
        assert result is None

    def test_raises_on_shape_divergence(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SCALE_CANDIDATE_SRC)
        with pytest.raises(StructureMismatchError, match="shape"):
            assert_structure_tripwire(
                path,
                "transform",
                abstract_inputs=[ShapeDtypeStruct((3,), np.float32)],
                concrete_inputs=[jnp.ones((4,), dtype=jnp.float32)],
            )

    def test_raises_on_dtype_divergence(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SCALE_CANDIDATE_SRC)
        with pytest.raises(StructureMismatchError, match="dtype"):
            assert_structure_tripwire(
                path,
                "transform",
                abstract_inputs=[ShapeDtypeStruct((3,), np.float32)],
                concrete_inputs=[jnp.ones((3,), dtype=jnp.int32)],
            )

    def test_candidate_resolution_error_propagates_unwrapped(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SYNTAX_ERROR_SRC)
        with pytest.raises(CandidateResolutionError):
            assert_structure_tripwire(
                path,
                "transform",
                abstract_inputs=[ShapeDtypeStruct((3,), np.float32)],
                concrete_inputs=[jnp.ones((3,), dtype=jnp.float32)],
            )

    def test_candidate_resolution_error_short_circuits_before_any_execution(
        self, tmp_path: Path
    ) -> None:
        path = _write_candidate(tmp_path, _SYNTAX_ERROR_SRC)
        with pytest.raises(CandidateResolutionError):
            assert_structure_tripwire(
                path,
                "does_not_matter",
                abstract_inputs=[ShapeDtypeStruct((3,), np.float32)],
                concrete_inputs=[jnp.ones((3,), dtype=jnp.float32)],
            )

    def test_candidate_invoked_exactly_once_concretely(self, tmp_path: Path) -> None:
        """AC-3's explicit claim: not zero-cost, exactly one concrete (real-data) execution.

        The candidate's abstract trace (inside jax.eval_shape) also invokes the Python body once,
        with a Tracer -- that's the "zero FLOPs" symbolic pass, not a real execution. The
        candidate distinguishes the two by checking `isinstance(x, jax.core.Tracer)` and only
        records a hit (via an on-disk side-channel, since a fresh module import each call means
        in-memory globals can't be inspected from outside) on the real, concrete call.
        """
        counter_path = tmp_path / "counter.txt"
        path = _write_candidate(tmp_path, _counting_candidate_src(counter_path))

        assert_structure_tripwire(
            path,
            "transform",
            abstract_inputs=[ShapeDtypeStruct((3,), np.float32)],
            concrete_inputs=[jnp.ones((3,), dtype=jnp.float32)],
        )

        assert counter_path.read_text() == "1"

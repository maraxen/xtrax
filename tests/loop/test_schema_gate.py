"""Tests for the schema-gate (T2-12, #2181, AC-2)."""

from pathlib import Path

import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.inference.schema import BundleSchema, extract_schema
from xtrax.loop.schema_gate import (
    CandidateResolutionError,
    SchemaMismatchError,
    assert_schema_gate,
    resolve_candidate_callable,
)

_MATCHING_CANDIDATE_SRC = """
def transform(x):
    return {"output": x * 2}
"""

_SYNTAX_ERROR_SRC = """
def transform(:
    pass
"""

_WRONG_SHAPE_CANDIDATE_SRC = """
def transform(x):
    return {"output": x[:2] * 2}
"""

_EXTRA_FIELD_CANDIDATE_SRC = """
def transform(x):
    return {"output": x * 2, "extra": x * 3}
"""

_MISSING_FIELD_CANDIDATE_SRC = """
def transform(x):
    return {"other_name": x * 2}
"""


def _write_candidate(tmp_path: Path, source: str, filename: str = "candidate.py") -> Path:
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


def _abstract_inputs() -> list[ShapeDtypeStruct]:
    return [ShapeDtypeStruct((3,), np.float32)]


def _declared_schema() -> BundleSchema:
    return BundleSchema(fields={"output": ShapeDtypeStruct((3,), np.float32)})


class TestResolveCandidateCallable:
    def test_resolves_a_clean_candidate(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _MATCHING_CANDIDATE_SRC)
        fn = resolve_candidate_callable(path, "transform")
        assert callable(fn)

    def test_raises_on_syntax_error(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SYNTAX_ERROR_SRC)
        with pytest.raises(CandidateResolutionError, match="failed to import candidate"):
            resolve_candidate_callable(path, "transform")

    def test_raises_on_missing_attribute(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _MATCHING_CANDIDATE_SRC)
        with pytest.raises(CandidateResolutionError, match="no attribute 'does_not_exist'"):
            resolve_candidate_callable(path, "does_not_exist")

    def test_raises_when_attribute_is_not_callable(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, "transform = 42\n")
        with pytest.raises(CandidateResolutionError, match="is not callable"):
            resolve_candidate_callable(path, "transform")


class TestAssertSchemaGate:
    def test_matching_schema_returns_derived_schema(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _MATCHING_CANDIDATE_SRC)
        result = assert_schema_gate(
            path,
            "transform",
            abstract_inputs=_abstract_inputs(),
            declared_schema=_declared_schema(),
        )
        assert result.fields["output"].shape == (3,)
        assert result.fields["output"].dtype == np.float32

    def test_result_equals_extract_schema_directly(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _MATCHING_CANDIDATE_SRC)
        fn = resolve_candidate_callable(path, "transform")
        expected = extract_schema(fn, _abstract_inputs())

        result = assert_schema_gate(
            path,
            "transform",
            abstract_inputs=_abstract_inputs(),
            declared_schema=_declared_schema(),
        )
        assert result.fields.keys() == expected.fields.keys()

    def test_propagates_candidate_resolution_error(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _SYNTAX_ERROR_SRC)
        with pytest.raises(CandidateResolutionError):
            assert_schema_gate(
                path,
                "transform",
                abstract_inputs=_abstract_inputs(),
                declared_schema=_declared_schema(),
            )

    def test_raises_on_wrong_shape(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _WRONG_SHAPE_CANDIDATE_SRC)
        expected = r"declared shape=\(3,\).*derived shape=\(2,\)"
        with pytest.raises(SchemaMismatchError, match=expected):
            assert_schema_gate(
                path,
                "transform",
                abstract_inputs=_abstract_inputs(),
                declared_schema=_declared_schema(),
            )

    def test_raises_naming_extra_field(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _EXTRA_FIELD_CANDIDATE_SRC)
        expected = "'extra'.*produced by the candidate but not declared"
        with pytest.raises(SchemaMismatchError, match=expected):
            assert_schema_gate(
                path,
                "transform",
                abstract_inputs=_abstract_inputs(),
                declared_schema=_declared_schema(),
            )

    def test_raises_naming_missing_field(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _MISSING_FIELD_CANDIDATE_SRC)
        with pytest.raises(SchemaMismatchError, match="'output'.*declared but not produced"):
            assert_schema_gate(
                path,
                "transform",
                abstract_inputs=_abstract_inputs(),
                declared_schema=_declared_schema(),
            )

    def test_raises_on_dtype_mismatch(self, tmp_path: Path) -> None:
        path = _write_candidate(tmp_path, _MATCHING_CANDIDATE_SRC)
        wrong_dtype_schema = BundleSchema(fields={"output": ShapeDtypeStruct((3,), np.int32)})
        with pytest.raises(SchemaMismatchError, match="dtype"):
            assert_schema_gate(
                path,
                "transform",
                abstract_inputs=_abstract_inputs(),
                declared_schema=wrong_dtype_schema,
            )

    def test_no_flops_executed_on_mismatch(self, tmp_path: Path) -> None:
        """A candidate that would raise if given concrete data must still be gate-checkable.

        `jax.eval_shape` traces with abstract ShapeDtypeStructs only, never dispatching real ops
        -- so a candidate whose *shape logic* is wrong is still safely traceable and reported as a
        SchemaMismatchError, not a crash from actually running the (wrong) computation.
        """
        path = _write_candidate(tmp_path, _WRONG_SHAPE_CANDIDATE_SRC)
        # Should raise SchemaMismatchError (a controlled rejection), not some other execution error.
        with pytest.raises(SchemaMismatchError):
            assert_schema_gate(
                path,
                "transform",
                abstract_inputs=_abstract_inputs(),
                declared_schema=_declared_schema(),
            )

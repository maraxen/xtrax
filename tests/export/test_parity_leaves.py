"""#5688: one parity rule for every backend -- per output leaf, integers and bools exact.

Before #5688 only the onnx path compared integer outputs exactly. Native/IREE parity
went through ``np.allclose(rtol=1e-5)``, so an index of ``1_000_009`` passed for
``1_000_000`` and ``ExportResult.verified`` was True for a wrong artifact. The same
program FAILED on onnx: one oracle, two verdicts.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.export.conftest import FAKE_VMFB
from xtrax.export.parity import LeafParityResult, compare, compare_leaves, verify_native_parity
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

BIG = np.full((4,), 1_000_000, dtype=np.int32)
OFF_BY_9 = BIG + 9  # within rtol=1e-5 of BIG: allclose says equal


class TestCompareIsExactForIntegersAndBools:
    def test_an_integer_off_by_nine_fails(self):
        """Red on the pre-#5688 code: np.allclose(BIG, BIG + 9, rtol=1e-5) is True."""
        assert np.allclose(BIG, OFF_BY_9, rtol=1e-5, atol=1e-5)  # the trap, demonstrated
        result = compare(BIG, OFF_BY_9)
        assert result.passed is False
        assert result.max_abs_diff == 9.0
        assert "exact comparison" in result.summary()

    def test_identical_integers_pass(self):
        assert compare(BIG, BIG.copy()).passed is True

    def test_a_flipped_bool_fails(self):
        mask = np.array([True, False, True])
        assert compare(mask, ~mask).passed is False
        assert compare(mask, mask.copy()).passed is True

    def test_floats_keep_their_tolerance(self):
        """Control: the float path is unchanged -- noise inside tolerance passes."""
        a = np.ones((3,), np.float32)
        assert compare(a, a + 1e-7).passed is True
        assert compare(a, a + 1.0).passed is False


class TestNativeParityIsPerLeafAndExact:
    def _run(self, tmp_path: Path, expected, actual) -> LeafParityResult:
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        return verify_native_parity(expected, artifact, [np.zeros((4,), np.float32)])

    def test_integer_output_off_by_nine_fails_on_native(self, fake_runtime, tmp_path):
        fake_runtime["result"] = OFF_BY_9
        result = self._run(tmp_path, BIG, OFF_BY_9)
        assert isinstance(result, LeafParityResult)
        assert result.passed is False
        assert "exact comparison" in result.summary()

    def test_exact_integer_output_passes_on_native(self, fake_runtime, tmp_path):
        fake_runtime["result"] = BIG.copy()
        assert self._run(tmp_path, BIG, BIG).passed is True

    def test_multi_output_is_compared_leaf_by_leaf(self, fake_runtime, tmp_path):
        """IREE returns a multi-output entry point as a flat tuple of leaves."""
        floats = np.ones((2, 3), np.float32)
        idx = np.array([2, 2], np.int32)
        fake_runtime["result"] = (floats + 1e-7, idx + 1)
        result = self._run(tmp_path, (floats, idx), None)
        assert result.passed is False
        assert result.leaf_results[0].passed is True
        assert result.leaf_results[1].passed is False
        assert result.summary().startswith("FAIL: leaf 1 of 2")

    def test_a_dtype_change_fails_on_native(self, fake_runtime, tmp_path):
        fake_runtime["result"] = BIG.astype(np.float32)
        result = self._run(tmp_path, BIG, None)
        assert result.passed is False
        assert result.dtype_mismatches == ("leaf 0: expected int32, got float32",)


class TestOneOracleFlattensTheSameWayEverywhere:
    def test_a_list_oracle_is_one_array_for_a_single_output(self):
        """Finding 9: tree_leaves split [[1, 2], [3, 4]] into four scalars, so onnx
        reported 'expected 4 output leaves, got 1' where native passed."""
        actual = [np.array([[1, 2], [3, 4]], np.int32)]
        assert compare_leaves([[1, 2], [3, 4]], actual).passed is True
        assert compare_leaves([[1, 2], [3, 5]], actual).passed is False

    def test_a_tuple_oracle_still_maps_to_several_outputs(self):
        """Control: a genuine multi-output pytree is not collapsed into one array."""
        actual = [np.zeros((2,), np.float32), np.ones((3,), np.int32)]
        oracle = (np.zeros((2,), np.float32), np.ones((3,), np.int32))
        result = compare_leaves(oracle, actual)
        assert result.passed is True
        assert len(result.leaf_results) == 2

    def test_a_leaf_count_mismatch_still_fails(self):
        result = compare_leaves(np.zeros(2, np.float32), [np.zeros(2), np.zeros(2)])
        assert result.passed is False
        assert "expected 1 output leaves, got 2" in result.summary()


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _vmap_plan(rows: int) -> _Plan:
    spec = AxisSpec(name="batch", cardinality=rows, default_batch_size=0)
    return _Plan([AxisDecision(spec=spec, batch_size=0, reasoning="t", strategy=Vmap())])


@pytest.mark.parametrize(("offset", "verified"), [(0, True), (9, False)])
def test_native_export_of_an_index_pipeline_verifies_exactly(offset: int, verified: bool):
    """End to end on a real IREE native artifact: an oracle off by 9 on a ~1e6 index
    must NOT verify (it did before #5688), and the exact oracle must."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    from xtrax.export import NATIVE, export_pipeline

    xs = (np.arange(8, dtype=np.int32) + 1_000_000).reshape(4, 2)

    def fn(row):
        return jnp.max(row)

    result = export_pipeline(
        fn,
        _vmap_plan(4),
        (jax.ShapeDtypeStruct(xs.shape, xs.dtype),),
        (xs,),
        targets=(NATIVE,),
        reference_fn=lambda inp: np.max(inp[0], axis=1).astype(np.int32) + offset,
    )["native"]
    assert result.verified is verified, result.parity.summary() if result.parity else None


# --- review of #180 ------------------------------------------------------------------


@pytest.fixture
def x64():
    prior = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


class TestExactDiffAndDtypes:
    def test_a_float_output_for_an_integer_reference_is_a_kind_change(self):
        """Was: int64 casts reported 'FAIL: max|diff| = 0.000e+00' for [1, 2] vs
        [1.4, 2.0] -- a FAIL whose number said the values matched. A change of kind is
        now inf, never a value diff that can read as agreement."""
        result = compare(np.array([1, 2], np.int32), np.array([1.4, 2.0], np.float32))
        assert (result.passed, result.max_abs_diff) == (False, float("inf"))

    def test_a_nan_output_reads_as_an_infinite_diff(self):
        result = compare(np.array([1], np.int32), np.array([np.nan], np.float32))
        assert (result.passed, result.max_abs_diff) == (False, float("inf"))

    def test_a_uint64_diff_of_2_64_minus_1_is_not_reported_as_1(self, x64):
        """int64 differencing is exact only modulo 2**64: 0 vs 2**64 - 1 wraps to a
        reported diff of 1. Under x64: uint64 only exists there (32-bit jnp.asarray
        narrows it)."""
        result = compare(np.array([0], np.uint64), np.array([2**64 - 1], np.uint64))
        assert result.passed is False
        assert result.max_abs_diff == float(2**64 - 1)

    def test_a_bool_reference_against_a_float_output_fails(self):
        """Was: compare([True], [1.0]) passed -- equal values, different kind."""
        assert compare(np.array([True]), np.array([1.0], np.float32)).passed is False


class TestOracleShapes:
    def test_a_dict_oracle_against_one_output_is_a_count_mismatch_not_a_crash(self):
        oracle = {"a": np.zeros(2, np.float32), "b": np.zeros(2, np.float32)}
        result = compare_leaves(oracle, [np.zeros(2, np.float32)])
        assert result.passed is False
        assert "expected 2 output leaves, got 1" in result.summary()

    def test_a_none_oracle_is_a_count_mismatch_not_a_crash(self):
        result = compare_leaves(None, [np.zeros(2, np.float32)])
        assert result.passed is False
        assert "expected 0 output leaves, got 1" in result.summary()


class TestOracleWidthUnderX64:
    def test_a_numpy_float64_oracle_verifies_an_f32_output(self, x64):
        """Under x64 jnp.asarray keeps NumPy's f64, while an f32 program returns f32."""
        result = compare_leaves(np.ones(3), [np.ones(3, np.float32)])
        assert result.passed, result.summary()

    def test_a_numpy_int64_oracle_verifies_an_i32_output_exactly(self, x64):
        assert compare_leaves(np.array([1, 2]), [np.array([1, 2], np.int32)]).passed
        assert not compare_leaves(np.array([1, 2]), [np.array([1, 3], np.int32)]).passed

    def test_an_artifact_wider_than_the_oracle_is_still_a_dtype_change(self, x64):
        result = compare_leaves(np.array([1, 2], np.int32), [np.array([1, 2], np.int64)])
        assert result.dtype_mismatches == ("leaf 0: expected int32, got int64",)

    def test_a_jax_array_oracle_dtype_is_explicit(self, x64):
        result = compare_leaves(jnp.ones(3, jnp.float64), [np.ones(3, np.float32)])
        assert result.passed is False


def test_in_32_bit_a_narrower_artifact_is_still_a_dtype_change():
    """Control: the width allowance is x64-only. In 32-bit, jnp.asarray already
    narrowed the oracle, so f32 -> f16 across the export is a real change."""
    result = compare_leaves(np.ones(3), [np.ones(3, np.float16)])
    assert result.dtype_mismatches == ("leaf 0: expected float32, got float16",)


# --- second review of #180 (xhigh) ---------------------------------------------------


def test_a_two_output_tuple_oracle_never_verifies_one_stacked_output():
    """Was: any tuple oracle was stacked when the artifact had one output, so an export
    that collapsed two outputs into one (2, 2) array verified."""
    oracle = (np.ones(2, np.float32), np.ones(2, np.float32))
    result = compare_leaves(oracle, [np.ones((2, 2), np.float32)])
    assert result.passed is False
    assert "expected 2 output leaves, got 1" in result.summary()


def test_a_two_output_tuple_oracle_never_verifies_under_x64_either(x64):
    oracle = (jnp.array([1, 2], jnp.int32), jnp.array([3, 4], jnp.int32))
    assert compare_leaves(oracle, [np.array([[1, 2], [3, 4]], np.int16)]).passed is False


@pytest.mark.parametrize(
    ("oracle_dtype", "artifact_dtype"),
    [(np.int32, np.int16), (np.float32, np.float16), (np.int64, np.int8), (np.float64, np.float16)],
)
def test_under_x64_only_the_64_to_32_bit_default_pair_is_tolerated(
    x64, oracle_dtype, artifact_dtype
):
    """Was: any same-kind narrowing passed under x64."""
    result = compare_leaves(np.ones(3, oracle_dtype), [np.ones(3, artifact_dtype)])
    assert result.dtype_mismatches, result.summary()


def test_a_bool_int_swap_fails_with_an_infinite_diff():
    """Was: compare([True, False], int8 [1, 0]) passed."""
    for exp, act in [
        (np.array([True, False]), np.array([1, 0], np.int8)),
        (np.array([1, 0], np.int32), np.array([True, False])),
    ]:
        result = compare(exp, act)
        assert (result.passed, result.max_abs_diff) == (False, float("inf"))


def test_int32_diff_is_exact_on_the_vectorised_path():
    result = compare(np.array([-(2**31), 0], np.int32), np.array([2**31 - 1, 0], np.int32))
    assert result.max_abs_diff == float(2**32 - 1)


def test_iree_returns_flat_typed_leaves(tmp_path):
    """Pins what verify_native_parity relies on: the IREE runtime returns a multi-output
    (and dict) entry point as a flat tuple of leaves, keeping int32/bool dtypes."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    from xtrax.export.compile import compile_for_target, run_native_vmfb
    from xtrax.export.targets import NATIVE

    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    def run(fn):
        exported = jax.export.export(jax.jit(fn))(jax.ShapeDtypeStruct(x.shape, x.dtype))
        return run_native_vmfb(compile_for_target(exported.mlir_module(), NATIVE).path, x)

    out = run(lambda v: (v * 2, jnp.argmax(v, axis=1), v > 2))
    assert [np.asarray(o).dtype for o in out] == [np.float32, np.int32, np.bool_]
    out = run(lambda v: {"a": v, "b": jnp.argmax(v, axis=1)})
    assert isinstance(out, tuple)
    assert [np.asarray(o).dtype for o in out] == [np.float32, np.int32]


def test_a_float64_numpy_input_verifies_on_native():
    """Was: IREE raised 'input0 element type mismatch; expected f32' -- only run_onnx
    narrowed inputs. Both backends now share parity.narrow_inputs."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    from xtrax.export import NATIVE, export_pipeline

    xs = np.ones((4, 3))  # float64
    result = export_pipeline(
        lambda x: x * 2,
        _vmap_plan(4),
        (jax.ShapeDtypeStruct((4, 3), jnp.float32),),
        (xs,),
        targets=(NATIVE,),
        reference_fn=lambda inp: np.asarray(inp[0], np.float32) * 2,
    )["native"]
    assert result.verified, result.parity.summary()

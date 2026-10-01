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
        assert "exact integer comparison" in result.summary()

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
        assert "exact integer comparison" in result.summary()

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

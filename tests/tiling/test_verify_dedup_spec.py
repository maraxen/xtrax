"""Tests for xtrax.tiling.dedup_synthesis.verify_dedup_spec (spec 260922 §4, AC-18..AC-26).

verify_dedup_spec does not exist prior to this change, so every test in this
file fails at collection (ImportError) against the unmodified module — that is
the correct "red" state for a brand-new public symbol, not a masked/vacuous
test. Once implemented, each test exercises real row-identity logic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.dedup_synthesis import (
    DedupSpecVerificationError,
    DedupSynthesisUnsupportedError,
    DedupVerificationResult,
    synthesize_dedup_spec,
    verify_dedup_spec,
)


def _tiled_batch(n_unique: int, N: int, dtype=np.float32) -> np.ndarray:
    unique_rows = np.arange(n_unique, dtype=dtype).reshape(-1, 1)
    return np.tile(unique_rows, (N // n_unique + 1, 1))[:N]


class TestAC18SynthesizedSpecVerifies:
    def test_ac18_ac7_fixture_verifies_with_expected_result(self):
        """AC-18: spec from synthesize_dedup_spec on the AC7 fixture verifies clean."""
        N = 10000
        n_unique = 30
        batch = _tiled_batch(n_unique, N)

        result = synthesize_dedup_spec([batch], threshold=0.5)
        assert result.stage == "synthesized"
        spec = result.spec
        assert spec is not None

        verification = verify_dedup_spec(spec, [batch])

        assert verification == DedupVerificationResult(n_rows=N, k=spec.k, transfer_bytes_spent=N)


class TestAC19HandBuiltSoundSpec:
    def test_ac19_last_occurrence_unsorted_spec_passes(self):
        """AC-19: a sound spec need not be F3's ascending-first-occurrence form."""
        # Rows: [9, 3, 9, 3]. A LAST-occurrence, unsorted canonical assignment
        # is still sound: every row is byte-identical to its claimed canonical.
        leaf = np.array([[9.0], [3.0], [9.0], [3.0]], dtype=np.float32)
        # unique_indices = [2, 3] (last occurrences, in this order -- not F3's
        # ascending-first-occurrence [0, 1]).
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([2, 3], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
        )

        result = verify_dedup_spec(spec, [leaf])
        assert result.n_rows == 4
        assert result.k == 2


class TestAC20RowMismatchUnion:
    def _two_leaf_fixture(self):
        N = 10
        leaf0 = np.arange(N, dtype=np.float32).reshape(-1, 1)
        leaf1 = np.arange(N, dtype=np.float32).reshape(-1, 1) * 10.0
        # canonical for every row is row 0.
        unique_indices = np.array([0], dtype=np.int32)
        index_map = np.zeros(N, dtype=np.int32)
        spec = DedupSpec(axis_name="batch", unique_indices=unique_indices, index_map=index_map, k=1)
        return leaf0, leaf1, spec

    def test_ac20_union_semantics_report_first_bad_row_and_leaf(self):
        leaf0, leaf1, spec = self._two_leaf_fixture()
        # Row 5 differs from canonical (row 0) in leaf1 only; leaf0 happens to
        # differ at every non-canonical row too (arange), so make leaf0 match
        # canonical everywhere except row 9.
        leaf0 = leaf0.copy()
        leaf0[:] = leaf0[0]
        leaf0[9] = 999.0  # leaf0 mismatch at row 9 only

        leaf1 = leaf1.copy()
        leaf1[:] = leaf1[0]
        leaf1[5] = 999.0  # leaf1 mismatch at row 5 only

        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, [leaf0, leaf1])

        err = exc_info.value
        assert err.check == "row_mismatch"
        assert err.first_bad_row == 5
        assert err.n_bad == 2
        assert err.leaf_index == 1

    def test_ac20_repaired_rows_pass(self):
        leaf0, leaf1, spec = self._two_leaf_fixture()
        leaf0 = leaf0.copy()
        leaf0[:] = leaf0[0]
        leaf1 = leaf1.copy()
        leaf1[:] = leaf1[0]
        # All rows now byte-identical to canonical row 0.
        result = verify_dedup_spec(spec, [leaf0, leaf1])
        assert result.n_rows == 10
        assert result.k == 1


class TestAC21BitwiseNotNumericTolerance:
    def test_ac21_signed_zero_mapped_together_raises(self):
        leaf = np.array([[0.0], [-0.0]], dtype=np.float32)
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0], dtype=np.int32),
            index_map=np.array([0, 0], dtype=np.int32),
            k=1,
        )
        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, [leaf])
        assert exc_info.value.check == "row_mismatch"

    def test_ac21_nextafter_mapped_together_raises(self):
        leaf = np.array([[1.0], [np.nextafter(np.float32(1.0), np.float32(2.0))]], dtype=np.float32)
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0], dtype=np.int32),
            index_map=np.array([0, 0], dtype=np.int32),
            k=1,
        )
        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, [leaf])
        assert exc_info.value.check == "row_mismatch"

    def test_ac21_identical_nan_payload_mapped_together_passes(self):
        nan_bits = np.float32(np.nan)
        leaf = np.array([[nan_bits], [nan_bits]], dtype=np.float32)
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0], dtype=np.int32),
            index_map=np.array([0, 0], dtype=np.int32),
            k=1,
        )
        result = verify_dedup_spec(spec, [leaf])
        assert result.n_rows == 2
        assert result.k == 1


def _pair_leaf(a_row, b_row, n_each, dtype):
    a = np.tile(np.asarray(a_row, dtype=dtype), (n_each, 1))
    b = np.tile(np.asarray(b_row, dtype=dtype), (n_each, 1))
    return np.concatenate([a, b], axis=0)


class TestAC22DtypeMatrix:
    """AC-22: dtype matrix. Each case builds a distinct pair (k==2) and a merged
    pair (k==1), verified both via synthesize_dedup_spec and verify_dedup_spec."""

    def _assert_distinct_then_merged(self, distinct_leaf, merged_leaf, *, axis=0, n_each=50):
        threshold = 0.1
        result = synthesize_dedup_spec([distinct_leaf], axis=axis, threshold=threshold)
        assert result.stage == "synthesized"
        assert result.spec is not None
        assert result.spec.k == 2
        verify_dedup_spec(result.spec, [distinct_leaf], axis=axis)

        merged_result = synthesize_dedup_spec([merged_leaf], axis=axis, threshold=threshold)
        assert merged_result.stage == "synthesized"
        assert merged_result.spec is not None
        assert merged_result.spec.k == 1
        verify_dedup_spec(merged_result.spec, [merged_leaf], axis=axis)

    def test_bfloat16_jax_array(self):
        distinct = jnp.asarray(_pair_leaf([1.0], [2.0], 50, np.float32), dtype=jnp.bfloat16)
        merged = jnp.asarray(_pair_leaf([1.0], [1.0], 50, np.float32), dtype=jnp.bfloat16)
        self._assert_distinct_then_merged(distinct, merged)

    def test_complex64_jax_array_imag_only(self):
        distinct = jnp.asarray(
            _pair_leaf([1 + 2j], [1 + 3j], 50, np.complex64), dtype=jnp.complex64
        )
        merged = jnp.asarray(_pair_leaf([1 + 2j], [1 + 2j], 50, np.complex64), dtype=jnp.complex64)
        self._assert_distinct_then_merged(distinct, merged)

    def test_complex64_numpy_imag_only(self):
        distinct = _pair_leaf([1 + 2j], [1 + 3j], 50, np.complex64)
        merged = _pair_leaf([1 + 2j], [1 + 2j], 50, np.complex64)
        self._assert_distinct_then_merged(distinct, merged)

    def test_bool_jax_array(self):
        distinct = jnp.asarray(_pair_leaf([True], [False], 50, np.bool_))
        merged = jnp.asarray(_pair_leaf([True], [True], 50, np.bool_))
        self._assert_distinct_then_merged(distinct, merged)

    def test_numpy_int64_x64_off_wide_values(self):
        distinct = _pair_leaf([5], [2**32 + 5], 50, np.int64)
        merged = _pair_leaf([5], [5], 50, np.int64)
        self._assert_distinct_then_merged(distinct, merged)

    def test_numpy_float64_near_epsilon(self):
        distinct = _pair_leaf([1.0], [1.0 + 2**-40], 50, np.float64)
        merged = _pair_leaf([1.0], [1.0], 50, np.float64)
        self._assert_distinct_then_merged(distinct, merged)

    def test_numpy_transposed_strided_leaf_axis1(self):
        # Build (2, 100) so axis=1 is batch; make it non-contiguous via a
        # transpose from a contiguous (100, 2) array.
        base_distinct = _pair_leaf([1.0], [2.0], 50, np.float64)  # (100, 1)
        base_distinct2 = _pair_leaf([10.0], [20.0], 50, np.float64)  # (100, 1)
        contig = np.concatenate([base_distinct, base_distinct2], axis=1)  # (100, 2)
        distinct = contig.T  # (2, 100), strided
        assert not distinct.flags["C_CONTIGUOUS"]

        base_merged = _pair_leaf([1.0], [1.0], 50, np.float64)
        base_merged2 = _pair_leaf([10.0], [10.0], 50, np.float64)
        merged = np.concatenate([base_merged, base_merged2], axis=1).T

        self._assert_distinct_then_merged(distinct, merged, axis=1)

    def test_int4_jax_array_n3(self):
        distinct = jnp.asarray(_pair_leaf([1, 2, 3], [1, 2, -3], 50, np.int8), dtype=jnp.int4)
        merged = jnp.asarray(_pair_leaf([1, 2, 3], [1, 2, 3], 50, np.int8), dtype=jnp.int4)
        self._assert_distinct_then_merged(distinct, merged)

    def test_uint4_jax_array_n3(self):
        distinct = jnp.asarray(_pair_leaf([1, 2, 3], [1, 2, 4], 50, np.uint8), dtype=jnp.uint4)
        merged = jnp.asarray(_pair_leaf([1, 2, 3], [1, 2, 3], 50, np.uint8), dtype=jnp.uint4)
        self._assert_distinct_then_merged(distinct, merged)

    def test_uint4_jax_array_n2_consistency_case(self):
        """Not a red control (P11): (N,2) packs losslessly even without widening."""
        distinct = jnp.asarray(_pair_leaf([1, 2], [1, 3], 50, np.uint8), dtype=jnp.uint4)
        merged = jnp.asarray(_pair_leaf([1, 2], [1, 2], 50, np.uint8), dtype=jnp.uint4)
        self._assert_distinct_then_merged(distinct, merged)


class TestAC23UnsupportedLeaves:
    def test_typed_key_leaf_refused_naming_key_data(self):
        keys = jax.random.split(jax.random.key(0), 4)
        with pytest.raises(DedupSynthesisUnsupportedError, match="key_data"):
            verify_dedup_spec(
                DedupSpec(
                    axis_name="batch",
                    unique_indices=np.array([0], dtype=np.int32),
                    index_map=np.array([0, 0, 0, 0], dtype=np.int32),
                    k=1,
                ),
                [keys],
            )
        with pytest.raises(DedupSynthesisUnsupportedError, match="key_data"):
            synthesize_dedup_spec([keys])

    def test_key_data_leaf_accepted(self):
        keys = jax.random.split(jax.random.key(0), 4)
        data = jax.random.key_data(keys)
        result = synthesize_dedup_spec([data], threshold=0.1)
        # Should not raise DedupSynthesisUnsupportedError; stage may vary.
        assert result.stage in {
            "no_duplication",
            "below_threshold",
            "synthesized",
            "k_over_limit",
        }

    def test_float4_leaf_refused_naming_dtype(self):
        from jax._src import dtypes as jax_internal_dtypes

        float4_dtype = jax_internal_dtypes.float4_e2m1fn
        leaf = jnp.asarray(np.array([[1.0], [1.0]], dtype=np.float32), dtype=float4_dtype)
        with pytest.raises(DedupSynthesisUnsupportedError, match="float4"):
            synthesize_dedup_spec([leaf], threshold=0.1)

    def test_numpy_unicode_leaf_refused(self):
        leaf = np.array([["aa"], ["bb"]], dtype="U4")
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0], dtype=np.int32),
            index_map=np.array([0, 0], dtype=np.int32),
            k=1,
        )
        with pytest.raises(DedupSynthesisUnsupportedError):
            verify_dedup_spec(spec, [leaf])


class TestAC24StructuralChecksNoTransfer:
    def _valid_spec_and_leaves(self, N=10):
        leaf0 = jnp.asarray(np.arange(N, dtype=np.float32).reshape(-1, 1))
        leaf1 = jnp.asarray(np.arange(N, dtype=np.float32).reshape(-1, 1) * 2.0)
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.arange(N, dtype=np.int32),
            index_map=np.arange(N, dtype=np.int32),
            k=N,
        )
        return spec, [leaf0, leaf1]

    def _spy(self, monkeypatch, calls):
        import xtrax.tiling.dedup_synthesis as ds

        original = ds._to_host

        def wrapper(x):
            calls.append((type(x), getattr(x, "shape", None)))
            return original(x)

        monkeypatch.setattr(ds, "_to_host", wrapper)

    # NB: every mutation here uses object.__setattr__ (or in-place array
    # mutation) on an already-VALID spec, bypassing DedupSpec.__post_init__ --
    # matching spec 260922 §4.2's framing ("spec fields mutated after
    # construction"). Rebuilding via the DedupSpec constructor is not viable:
    # DedupSpec's own __post_init__ (and beartype's constructor typecheck)
    # would reject most of these inputs before verify_dedup_spec ever runs.

    @pytest.mark.parametrize(
        "mutate,expected_check",
        [
            (
                lambda spec: object.__setattr__(
                    spec, "unique_indices", jnp.asarray(spec.unique_indices)
                ),
                "index_type",
            ),
            (
                lambda spec: object.__setattr__(
                    spec, "index_map", spec.index_map.astype(np.float64)
                ),
                "index_dtype",
            ),
        ],
    )
    def test_ac24_type_and_dtype_checks_zero_transfer(self, monkeypatch, mutate, expected_check):
        spec, leaves = self._valid_spec_and_leaves()
        mutate(spec)
        calls: list = []
        self._spy(monkeypatch, calls)

        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, leaves)

        assert exc_info.value.check == expected_check
        assert calls == []

    def test_ac24_index_map_length_zero_transfer(self, monkeypatch):
        spec, leaves = self._valid_spec_and_leaves()
        object.__setattr__(spec, "index_map", spec.index_map[:-1].copy())
        calls: list = []
        self._spy(monkeypatch, calls)
        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, leaves)
        assert exc_info.value.check == "index_map_length"
        assert calls == []

    def test_ac24_unique_indices_out_of_bounds_high_zero_transfer(self, monkeypatch):
        spec, leaves = self._valid_spec_and_leaves(N=10)
        spec.unique_indices[-1] = 10  # N; out of [0, N)
        calls: list = []
        self._spy(monkeypatch, calls)
        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, leaves)
        assert exc_info.value.check == "unique_indices_bounds"
        assert calls == []

    def test_ac24_unique_indices_out_of_bounds_negative_zero_transfer(self, monkeypatch):
        spec, leaves = self._valid_spec_and_leaves(N=10)
        spec.unique_indices[0] = -1
        calls: list = []
        self._spy(monkeypatch, calls)
        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, leaves)
        assert exc_info.value.check == "unique_indices_bounds"
        assert calls == []

    def test_ac24_index_map_out_of_bounds_zero_transfer(self, monkeypatch):
        spec, leaves = self._valid_spec_and_leaves(N=10)
        spec.index_map[0] = spec.k  # out of [0, k)
        calls: list = []
        self._spy(monkeypatch, calls)
        with pytest.raises(DedupSpecVerificationError) as exc_info:
            verify_dedup_spec(spec, leaves)
        assert exc_info.value.check == "index_map_bounds"
        assert calls == []


class TestAC25TransferSpyControl:
    def test_ac25_exactly_one_to_host_call_shape_n_by_l(self, monkeypatch):
        import xtrax.tiling.dedup_synthesis as ds

        N = 10
        leaf0 = jnp.asarray(np.arange(N, dtype=np.float32).reshape(-1, 1))
        leaf1 = jnp.asarray(np.arange(N, dtype=np.float32).reshape(-1, 1) * 2.0)
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.arange(N, dtype=np.int32),
            index_map=np.arange(N, dtype=np.int32),
            k=N,
        )

        original = ds._to_host
        calls: list = []

        def wrapper(x):
            calls.append((type(x), getattr(x, "shape", None)))
            return original(x)

        monkeypatch.setattr(ds, "_to_host", wrapper)

        verify_dedup_spec(spec, [leaf0, leaf1])

        assert len(calls) == 1
        arg_type, arg_shape = calls[0]
        assert arg_type is jax.Array or issubclass(arg_type, jax.Array)
        assert arg_shape == (N, 2)


class TestAC26PublicSurfaceAndDocs:
    def test_public_names_importable(self):
        from xtrax.tiling.dedup_synthesis import (
            DedupSpecVerificationError as _E,
        )
        from xtrax.tiling.dedup_synthesis import (
            DedupVerificationResult as _R,
        )
        from xtrax.tiling.dedup_synthesis import (
            verify_dedup_spec as _V,
        )

        assert issubclass(_E, ValueError)
        assert _R is not None
        assert callable(_V)

    def test_docs_mention_dedup_synthesis_automodule(self):
        import pathlib

        docs = pathlib.Path("docs/api/tiling.md").read_text()
        assert "dedup_synthesis" in docs

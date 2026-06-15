"""Tests for xtrax.tiling.dedup module."""

import numpy as np
import pytest

from xtrax.tiling.dedup import DedupSpec, get_k_bucket


class TestGetKBucket:
    """Test get_k_bucket function."""

    def test_get_k_bucket_1(self) -> None:
        """get_k_bucket(1) should return 1."""
        assert get_k_bucket(1) == 1

    def test_get_k_bucket_7(self) -> None:
        """get_k_bucket(7) should return 8 (next power of 2)."""
        assert get_k_bucket(7) == 8

    def test_get_k_bucket_8(self) -> None:
        """get_k_bucket(8) should return 8 (already power of 2)."""
        assert get_k_bucket(8) == 8

    def test_get_k_bucket_9(self) -> None:
        """get_k_bucket(9) should return 16 (next power of 2)."""
        assert get_k_bucket(9) == 16

    def test_get_k_bucket_zero_raises(self) -> None:
        """get_k_bucket(0) should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            get_k_bucket(0)

    def test_get_k_bucket_negative_raises(self) -> None:
        """get_k_bucket(-1) should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            get_k_bucket(-1)


class TestDedupSpec:
    """Test DedupSpec dataclass."""

    def test_dedup_spec_valid(self) -> None:
        """DedupSpec with valid inputs should instantiate."""
        unique_indices = np.array([0, 2, 5])
        index_map = np.array([0, 1, 2, 0, 1, 2])
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=unique_indices,
            index_map=index_map,
            k=3,
        )
        assert spec.axis_name == "batch"
        assert spec.k == 3
        np.testing.assert_array_equal(spec.unique_indices, unique_indices)
        np.testing.assert_array_equal(spec.index_map, index_map)

    def test_dedup_spec_k_mismatch_raises(self) -> None:
        """DedupSpec with k != len(unique_indices) should raise ValueError."""
        unique_indices = np.array([0, 2, 5])
        index_map = np.array([0, 1, 2, 0, 1, 2])
        with pytest.raises(ValueError, match=r"len\(unique_indices\)"):
            DedupSpec(
                axis_name="batch",
                unique_indices=unique_indices,
                index_map=index_map,
                k=4,  # Wrong: should be 3
            )

    def test_dedup_spec_index_map_mismatch_raises(self) -> None:
        """DedupSpec where index_map doesn't reference all k slots should raise ValueError."""
        unique_indices = np.array([0, 2, 5])
        index_map = np.array([0, 0, 0, 0, 0, 0])  # Only references slot 0
        with pytest.raises(ValueError, match="distinct slots"):
            DedupSpec(
                axis_name="batch",
                unique_indices=unique_indices,
                index_map=index_map,
                k=3,  # Claims 3 but only 1 is used
            )

    def test_dedup_spec_to_dedup_gather(self) -> None:
        """to_dedup_gather() should return DedupGather with padded indices."""
        unique_indices = np.array([0, 2], dtype=np.int32)
        index_map = np.array([0, 1, 0, 1], dtype=np.int32)
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=unique_indices,
            index_map=index_map,
            k=2,
        )
        dg = spec.to_dedup_gather()
        assert dg.k == 2
        assert dg.k_bucket == 2  # 2 is already a power of 2
        np.testing.assert_array_equal(dg.index_map, index_map)
        # unique_indices should be padded to k_bucket
        assert len(dg.unique_indices) == dg.k_bucket

    def test_dedup_spec_frozen(self) -> None:
        """DedupSpec should be frozen (immutable)."""
        unique_indices = np.array([0, 2, 5])
        index_map = np.array([0, 1, 2, 0, 1, 2])
        spec = DedupSpec(
            axis_name="batch",
            unique_indices=unique_indices,
            index_map=index_map,
            k=3,
        )
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            spec.k = 4  # type: ignore

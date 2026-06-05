"""Tests for xtrax.tiling.dedup module."""

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
        with pytest.raises(ValueError, match="must be > 0"):
            get_k_bucket(0)

    def test_get_k_bucket_negative_raises(self) -> None:
        """get_k_bucket(-1) should raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            get_k_bucket(-1)


class TestDedupSpec:
    """Test DedupSpec dataclass."""

    def test_dedup_spec_valid(self) -> None:
        """DedupSpec with valid k_bucket and max_unique should instantiate."""
        spec = DedupSpec(k_bucket=8, max_unique=4)
        assert spec.k_bucket == 8
        assert spec.max_unique == 4

    def test_dedup_spec_k_bucket_equals_max_unique(self) -> None:
        """DedupSpec with k_bucket == max_unique should be valid."""
        spec = DedupSpec(k_bucket=8, max_unique=8)
        assert spec.k_bucket == 8
        assert spec.max_unique == 8

    def test_dedup_spec_k_bucket_not_power_of_2(self) -> None:
        """DedupSpec with non-power-of-2 k_bucket should raise ValueError."""
        with pytest.raises(ValueError, match="power of 2"):
            DedupSpec(k_bucket=3, max_unique=2)

    def test_dedup_spec_k_bucket_less_than_max_unique(self) -> None:
        """DedupSpec with k_bucket < max_unique should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= max_unique"):
            DedupSpec(k_bucket=4, max_unique=8)

    def test_dedup_spec_k_bucket_zero(self) -> None:
        """DedupSpec with k_bucket=0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DedupSpec(k_bucket=0, max_unique=0)

    def test_dedup_spec_k_bucket_negative(self) -> None:
        """DedupSpec with negative k_bucket should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DedupSpec(k_bucket=-1, max_unique=0)

    def test_dedup_spec_frozen(self) -> None:
        """DedupSpec should be frozen (immutable)."""
        spec = DedupSpec(k_bucket=8, max_unique=4)
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            spec.k_bucket = 16  # type: ignore

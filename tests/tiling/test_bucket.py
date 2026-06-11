"""Tests for xtrax.tiling.bucket — host-side select_bucket and bucketize."""

import numpy as np
import pytest

from xtrax.tiling.bucket import bucketize, select_bucket


class TestSelectBucket:
    """select_bucket: smallest boundary >= length, with overflow contract."""

    def test_exact_match_returns_boundary(self):
        """A length equal to a boundary selects that boundary (pad_amount=0)."""
        assert select_bucket(8, (8, 16, 32)) == 8

    def test_between_boundaries_rounds_up(self):
        """A length between boundaries rounds up to the next boundary."""
        assert select_bucket(9, (8, 16, 32)) == 16
        assert select_bucket(1, (8, 16, 32)) == 8

    def test_zero_length_selects_first_boundary(self):
        """A zero length selects the smallest boundary."""
        assert select_bucket(0, (8, 16)) == 8

    def test_exceeds_largest_boundary_raises(self):
        """A length above the largest boundary is an error (recompilation contract)."""
        with pytest.raises(ValueError, match="exceeds the largest"):
            select_bucket(33, (8, 16, 32))

    def test_negative_length_raises(self):
        """Negative length is rejected."""
        with pytest.raises(ValueError, match=">= 0"):
            select_bucket(-1, (8, 16))

    def test_empty_boundaries_raises(self):
        """Empty boundaries is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            select_bucket(5, ())


class TestBucketize:
    """bucketize: host-side NumPy padding of the leading axis + mask."""

    def test_pads_leading_axis_to_bucket_size(self):
        """A 1-D array is padded up to bucket_size along the leading axis."""
        xs = np.arange(5)
        padded, mask = bucketize(xs, 8)
        assert padded.shape == (8,)
        # Original values preserved, tail zero-padded.
        np.testing.assert_array_equal(padded[:5], np.arange(5))
        np.testing.assert_array_equal(padded[5:], np.zeros(3))

    def test_mask_marks_original_positions(self):
        """The mask is True over the original length, False over padding."""
        xs = np.arange(5)
        _, mask = bucketize(xs, 8)
        assert mask.dtype == np.bool_
        np.testing.assert_array_equal(mask, np.array([True] * 5 + [False] * 3))

    def test_pads_multidim_leaf_only_on_leading_axis(self):
        """A 2-D leaf is padded only on the leading axis."""
        xs = np.ones((3, 4))
        padded, mask = bucketize(xs, 8)
        assert padded.shape == (8, 4)
        np.testing.assert_array_equal(padded[:3], np.ones((3, 4)))
        np.testing.assert_array_equal(padded[3:], np.zeros((5, 4)))

    def test_pytree_inputs_padded_consistently(self):
        """All leaves of a pytree are padded to the same bucket size."""
        xs = {"a": np.arange(3), "b": np.ones((3, 2))}
        padded, mask = bucketize(xs, 4)
        assert padded["a"].shape == (4,)
        assert padded["b"].shape == (4, 2)
        np.testing.assert_array_equal(mask, np.array([True, True, True, False]))

    def test_exact_fit_no_padding(self):
        """bucket_size == length pads nothing and masks everything True."""
        xs = np.arange(4)
        padded, mask = bucketize(xs, 4)
        np.testing.assert_array_equal(padded, np.arange(4))
        assert mask.all()

    def test_bucket_size_smaller_than_input_raises(self):
        """A bucket_size below the input length is an error."""
        with pytest.raises(ValueError, match="smaller than"):
            bucketize(np.arange(10), 8)

    def test_mismatched_leaf_lengths_raise(self):
        """Leaves with different leading-axis lengths are rejected."""
        xs = {"a": np.arange(3), "b": np.arange(4)}
        with pytest.raises(ValueError, match="same leading-axis length"):
            bucketize(xs, 8)

    def test_no_leaves_raises(self):
        """An empty pytree has nothing to pad."""
        with pytest.raises(ValueError, match="no array leaves"):
            bucketize({}, 8)


class TestBucketizeRoundTrip:
    """Composition: select_bucket → bucketize → drop padding via mask."""

    def test_select_then_bucketize_then_unmask(self):
        """select_bucket + bucketize round-trips through a device-like step."""
        seq = np.arange(5) + 1  # [1, 2, 3, 4, 5]
        bucket = select_bucket(len(seq), (4, 8, 16))
        assert bucket == 8

        padded, mask = bucketize(seq, bucket)

        # Stand-in for a jitted step: same executable for any length in this bucket.
        out = padded * 10
        recovered = out[mask]
        np.testing.assert_array_equal(recovered, (np.arange(5) + 1) * 10)

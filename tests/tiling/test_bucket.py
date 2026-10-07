"""Tests for xtrax.tiling.bucket — host-side select_bucket and bucketize."""

import subprocess
import sys

import numpy as np
import pytest

from xtrax.tiling.bucket import (
    BUCKET_LADDER,
    bucketize,
    pad_axis,
    select_bucket,
    select_rung,
    trim_axis,
    valid_span,
)


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


class TestBucketLadder:
    """BUCKET_LADDER lives in tiling and is the rings object."""

    def test_documented_rungs(self):
        assert BUCKET_LADDER == (64, 128, 256, 512, 1024, 1536, 2048)

    def test_same_object_as_export_rings(self):
        from xtrax.export.rings import BUCKET_LADDER as rings_ladder
        from xtrax.tiling import BUCKET_LADDER as tiling_ladder

        assert tiling_ladder is rings_ladder
        assert tiling_ladder is BUCKET_LADDER

    def test_importing_tiling_does_not_import_export(self):
        """Runtime tiling must not pull in the export stack."""
        code = (
            "import sys; "
            "import xtrax.tiling; "
            "bad = [m for m in sys.modules if m == 'xtrax.export' "
            "or m.startswith('xtrax.export.')]; "
            "assert not bad, bad"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestValidSpan:
    """valid_span: positions from the leading edge through the last valid one."""

    def test_trailing_padding_is_excluded(self):
        assert valid_span([1, 1, 1, 0, 0]) == 3

    def test_hole_counts_through_the_last_valid(self):
        """A gap is inside the span; the rung has to cover the last valid index."""
        assert valid_span([1, 0, 1, 0]) == 3

    def test_batch_uses_the_longest_row(self):
        mask = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
        assert valid_span(mask) == 2

    def test_all_masked_or_empty_is_zero(self):
        assert valid_span([0, 0, 0]) == 0
        assert valid_span([]) == 0
        assert valid_span(np.zeros((2, 0), dtype=bool)) == 0


class TestSelectRung:
    """select_rung: smallest rung >= span, then capped."""

    def test_rounds_up_when_cap_is_above_the_rung(self):
        assert select_rung(80, 200, (64, 128, 256)) == 128

    def test_cap_below_the_rung_wins(self):
        """cap need not sit on the ladder."""
        assert select_rung(80, 100, (64, 128, 256)) == 100

    def test_cap_equal_to_the_rung(self):
        assert select_rung(64, 64, (64, 128, 256)) == 64

    def test_span_past_the_ladder_raises_even_if_cap_is_larger(self):
        with pytest.raises(ValueError, match="exceeds the largest"):
            select_rung(3000, 4000, BUCKET_LADDER)

    def test_negative_cap_raises(self):
        with pytest.raises(ValueError, match="cap"):
            select_rung(1, -1, (8, 16))


class TestTrimPadAxis:
    """trim_axis / pad_axis round-trip on a non-leading axis."""

    def test_round_trip_on_axis_1(self):
        arr = np.arange(24).reshape(2, 3, 4)
        trimmed = trim_axis(arr, 2, axis=1)
        assert trimmed.shape == (2, 2, 4)
        np.testing.assert_array_equal(trimmed, arr[:, :2, :])

        padded = pad_axis(trimmed, 3, axis=1, fill=-1)
        assert padded.shape == (2, 3, 4)
        np.testing.assert_array_equal(padded[:, :2, :], trimmed)
        assert np.all(padded[:, 2, :] == -1)

    def test_round_trip_on_negative_axis(self):
        arr = np.ones((2, 3, 4))
        trimmed = trim_axis(arr, 3, axis=-1)
        assert trimmed.shape == (2, 3, 3)
        padded = pad_axis(trimmed, 4, axis=-1, fill=7)
        assert padded.shape == arr.shape
        assert np.all(padded[..., :3] == 1)
        assert np.all(padded[..., 3] == 7)

    def test_pad_is_identity_when_length_matches(self):
        arr = np.arange(6).reshape(2, 3)
        assert pad_axis(arr, 3, axis=1) is arr

    def test_pad_rejects_a_shorter_target(self):
        with pytest.raises(ValueError, match="cannot pad"):
            pad_axis(np.ones((2, 5)), 3, axis=1)

    def test_none_passes_through(self):
        assert trim_axis(None, 2, axis=0) is None
        assert pad_axis(None, 4, axis=0, fill=1) is None

    def test_jax_pad_round_trip_on_axis_1(self):
        import jax.numpy as jnp

        arr = jnp.arange(6).reshape(2, 3)
        trimmed = trim_axis(arr, 2, axis=1)
        padded = pad_axis(trimmed, 3, axis=1, fill=0)
        assert padded.shape == (2, 3)
        np.testing.assert_array_equal(np.asarray(padded[:, :2]), np.asarray(trimmed))
        assert int(np.asarray(padded[0, 2])) == 0

"""Tests for xtrax.tiling.dedup_synthesis — spec 260825 §4.3 (AC7/AC8/AC10/AC11)."""

from __future__ import annotations

import numpy as np
import pytest

from xtrax.inference.errors import (
    DedupSpecCollisionError,
    DedupSynthesisCollisionError,
    DedupSynthesisUnsupportedError,
)
from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.dedup_synthesis import (
    DedupSynthesisResult,
    merge_dedup_specs,
    synthesize_dedup_spec,
)


class TestSampleStage:
    """Tests for the sample-stage behavior."""

    def test_ac8_all_unique_batch(self):
        """AC8: all-unique batch → stage='no_duplication', spec=None, minimal transfer."""
        # N=100, all unique rows
        batch = np.arange(100, dtype=np.float32).reshape(-1, 1)
        result = synthesize_dedup_spec([batch])

        assert result.spec is None
        assert result.stage == "no_duplication"
        assert result.sampled_ratio == 0.0
        # Transfer bytes = sample stage only (100 rows * 4 bytes each)
        assert result.transfer_bytes_spent == 100 * 4
        assert result.k_bucket_bytes == 0

    def test_below_threshold(self):
        """Sample-stage under threshold → stage='below_threshold', zero O(N)."""
        # N=1000, duplication below threshold (10%)
        N = 1000
        # Create rows where each row appears 1-2 times (sparse duplication)
        batch = np.concatenate([
            np.arange(900, dtype=np.float32).reshape(-1, 1),  # 900 unique rows
            np.arange(100, dtype=np.float32).reshape(-1, 1),  # 100 repeats of first rows
        ], axis=0)[:N]
        result = synthesize_dedup_spec([batch], threshold=0.5, max_unique_k=200)

        assert result.spec is None
        assert result.stage == "below_threshold"
        assert 0.0 < result.sampled_ratio < 0.5
        # Sample-only transfer
        assert result.transfer_bytes_spent > 0
        assert result.k_bucket_bytes == 0

    def test_high_duplication_fires_exact_stage(self):
        """High duplication ratio (>threshold) fires exact stage."""
        # N=1000, ~30 unique rows (97% duplication)
        N = 1000
        unique_rows = np.arange(30, dtype=np.float32).reshape(-1, 1)
        batch = np.tile(unique_rows, (N // 30 + 1, 1))[:N]
        result = synthesize_dedup_spec([batch], threshold=0.5)

        # Exact stage should fire, producing a spec.
        assert result.spec is not None
        assert result.stage == "synthesized"
        # Sampled ratio should be high
        assert result.sampled_ratio > 0.5


class TestAC7SynthesizedSpec:
    """AC7: batch with >50% duplicated rows → synthesized spec, round-trips."""

    def test_ac7_high_duplication_synthesized(self):
        """AC7: N=10000, ~30 uniques → stage='synthesized', valid spec."""
        N = 10000
        n_unique = 30
        unique_rows = np.arange(n_unique, dtype=np.float32).reshape(-1, 1)
        # Repeat pattern to create duplicates
        batch = np.tile(unique_rows, (N // n_unique + 1, 1))[:N]

        result = synthesize_dedup_spec([batch], threshold=0.5)

        assert result.stage == "synthesized"
        assert result.spec is not None

        spec = result.spec
        assert spec.axis_name == "batch"
        assert len(spec.unique_indices) == n_unique
        assert len(spec.index_map) == N

        # Self-assertion: len(index_map) == N (checked by DedupSpec.__post_init__)
        assert len(spec.index_map) == N

        # Round-trip: to_dedup_gather() should succeed
        gather = spec.to_dedup_gather()
        assert gather.k == n_unique
        assert gather.k_bucket >= n_unique
        assert len(gather.unique_indices) == gather.k_bucket
        assert len(gather.index_map) == N

    def test_ac7_multiple_leaves(self):
        """AC7 with multiple batch leaves (concatenated for analysis)."""
        N = 5000
        n_unique = 25
        leaf1 = np.tile(np.arange(n_unique, dtype=np.float32).reshape(-1, 1),
                        (N // n_unique + 1, 1))[:N]
        leaf2 = leaf1 * 2.0  # Correlated but not identical

        result = synthesize_dedup_spec([leaf1, leaf2], threshold=0.5)

        assert result.stage == "synthesized"
        assert result.spec is not None
        # k should reflect actual unique rows when stacked
        assert result.spec.k > 0
        assert len(result.spec.index_map) == N


class TestAC8NoDuplication:
    """AC8: all-unique batch → no_duplication stage, zero O(N) transfer."""

    def test_ac8_zero_transfer_in_sample_only(self):
        """AC8: verify transfer_bytes_spent reflects sample-only cost."""
        N = 5000
        batch = np.arange(N, dtype=np.float32).reshape(-1, 1)

        result = synthesize_dedup_spec([batch], threshold=0.5, max_sample_rows=100)

        assert result.stage == "no_duplication"
        assert result.spec is None
        assert result.sampled_ratio == 0.0
        # Transfer should be approximately sample rows only.
        # max_sample_rows = 100, so roughly 100 * 4 bytes (float32) = 400 bytes
        assert result.transfer_bytes_spent < 1000  # Conservative upper bound
        assert result.k_bucket_bytes == 0


class TestAC10KOverLimit:
    """AC10: exact k > max_unique_k → k_over_limit stage, full O(N) transfer."""

    def test_ac10_k_exceeds_limit(self):
        """AC10: k > max_unique_k after exact stage → k_over_limit."""
        # Create data with exactly 270 unique rows (exceeds max_unique_k=256)
        n_unique = 270
        batch = np.arange(n_unique, dtype=np.float32).reshape(-1, 1)
        # Pad to N=500 rows using first row repeated
        batch = np.vstack([batch, np.tile(batch[0:1], (500 - n_unique, 1))])

        result = synthesize_dedup_spec(
            [batch], threshold=0.01, max_unique_k=256
        )

        assert result.stage == "k_over_limit"
        assert result.spec is None
        # transfer_bytes_spent should reflect full O(N) pass (sample + exact)
        assert result.transfer_bytes_spent > 500  # At least the full batch
        assert result.k_bucket_bytes == 0

    def test_ac10_transfer_bytes_include_sample_and_exact(self):
        """AC10: transfer_bytes_spent includes both sample and exact stages."""
        # Create data with 270 unique rows (exceeds max_unique_k=256)
        n_unique = 270
        batch = np.arange(n_unique, dtype=np.float32).reshape(-1, 1)
        # Pad to N=1000 rows using repeats
        batch = np.vstack([batch, np.tile(batch[0:1], (1000 - n_unique, 1))])

        result = synthesize_dedup_spec(
            [batch], threshold=0.01, max_sample_rows=100, max_unique_k=256
        )

        assert result.stage == "k_over_limit"
        # Expect: ~100 sample rows + 1000 exact rows transferred
        element_width = 4  # float32
        min_expected_bytes = 100 * element_width + 1000 * element_width
        assert result.transfer_bytes_spent >= min_expected_bytes


class TestAC11CollisionPolicy:
    """AC11: existing_specs collision → DedupSynthesisCollisionError."""

    def test_ac11_collision_with_existing_spec(self):
        """AC11: existing_specs declares 'batch' → collision error."""
        N = 100
        batch = np.tile(np.array([1.0, 2.0], dtype=np.float32).reshape(-1, 1),
                        (N // 2 + 1, 1))[:N]

        # Pre-declare a spec for 'batch' axis
        existing_spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0, 1]),
            index_map=np.array([0, 1] * (N // 2) + [0] * (N % 2)),
            k=2,
        )
        existing_specs = {"batch": existing_spec}

        # Should raise DedupSynthesisCollisionError
        with pytest.raises(DedupSynthesisCollisionError, match="already declares"):
            synthesize_dedup_spec([batch], existing_specs=existing_specs)

    def test_ac11_caller_spec_untouched(self):
        """AC11: collision does not modify caller's spec."""
        N = 100
        batch = np.tile(np.array([1.0, 2.0], dtype=np.float32).reshape(-1, 1),
                        (N // 2 + 1, 1))[:N]

        existing_spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0, 1]),
            index_map=np.array([0, 1] * (N // 2) + [0] * (N % 2)),
            k=2,
        )
        existing_specs = {"batch": existing_spec}

        with pytest.raises(DedupSynthesisCollisionError):
            synthesize_dedup_spec([batch], existing_specs=existing_specs)

        # Verify caller's spec is unchanged
        assert existing_specs["batch"] is existing_spec

    def test_ac11_non_colliding_existing_specs(self):
        """AC11: non-colliding existing_specs (different axis_name) must NOT raise."""
        N = 100
        batch = np.tile(np.array([1.0, 2.0], dtype=np.float32).reshape(-1, 1),
                        (N // 2 + 1, 1))[:N]

        # Pre-declare a spec for a different axis (not "batch")
        other_spec = DedupSpec(
            axis_name="other_axis",
            unique_indices=np.array([0]),
            index_map=np.array([0] * N),
            k=1,
        )
        existing_specs = {"other_axis": other_spec}

        # Should NOT raise; synthesis should proceed normally
        result = synthesize_dedup_spec([batch], existing_specs=existing_specs,
                                       threshold=0.5)

        assert result.spec is not None
        assert result.stage == "synthesized"
        assert result.spec.axis_name == "batch"
        # other_axis spec should remain unchanged
        assert existing_specs["other_axis"] is other_spec


class TestMergeDedupSpecs:
    """Tests for merge_dedup_specs helper."""

    def test_merge_no_collision(self):
        """merge_dedup_specs: no collision → merged dict."""
        spec1 = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0, 1]),
            index_map=np.array([0, 1, 0]),
            k=2,
        )
        spec2 = DedupSpec(
            axis_name="feat",
            unique_indices=np.array([0]),
            index_map=np.array([0]),
            k=1,
        )

        result = merge_dedup_specs({"batch": spec1}, {"feat": spec2})

        assert len(result) == 2
        assert result["batch"] is spec1
        assert result["feat"] is spec2

    def test_merge_collision_same_axis(self):
        """merge_dedup_specs: duplicate axis_name → DedupSpecCollisionError."""
        spec1 = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0, 1]),
            index_map=np.array([0, 1, 0]),
            k=2,
        )
        spec2 = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0]),
            index_map=np.array([0]),
            k=1,
        )

        with pytest.raises(DedupSpecCollisionError, match="appears in multiple"):
            merge_dedup_specs({"batch": spec1}, {"batch": spec2})

    def test_merge_multiple_mappings(self):
        """merge_dedup_specs: three+ mappings without collision."""
        specs = [
            {"batch": DedupSpec(
                axis_name="batch",
                unique_indices=np.array([0, 1]),
                index_map=np.array([0, 1, 0]),
                k=2,
            )},
            {"feat": DedupSpec(
                axis_name="feat",
                unique_indices=np.array([0]),
                index_map=np.array([0]),
                k=1,
            )},
            {"seq": DedupSpec(
                axis_name="seq",
                unique_indices=np.array([0, 1, 2]),
                index_map=np.array([0, 1, 2]),
                k=3,
            )},
        ]

        result = merge_dedup_specs(*specs)

        assert len(result) == 3
        assert "batch" in result
        assert "feat" in result
        assert "seq" in result

    def test_merge_empty_mappings(self):
        """merge_dedup_specs: empty or empty mappings."""
        spec1 = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0]),
            index_map=np.array([0]),
            k=1,
        )

        result = merge_dedup_specs({"batch": spec1}, {})

        assert len(result) == 1
        assert result["batch"] is spec1


class TestF3FirstOccurrenceOrder:
    """F3 discriminator: first-occurrence-position order vs sorted-value order."""

    def test_f3_discriminator_first_occurrence_order(self):
        """F3: unique_indices must be in first-occurrence order, not sorted-value order.

        Spec's own example: rows [9, 3, 9, 3] must produce unique_indices=[0, 1]
        (first occurrence of 9 at position 0, first occurrence of 3 at position 1),
        NOT [1, 0] or any value-sorted ordering.
        """
        # Create a batch with values [9, 3, 9, 3] as 1-column array
        batch = np.array([[9.0], [3.0], [9.0], [3.0]], dtype=np.float32)

        result = synthesize_dedup_spec([batch], threshold=0.01)

        assert result.spec is not None
        assert result.stage == "synthesized"

        spec = result.spec
        # unique_indices should be [0, 1] (first occurrences in order)
        np.testing.assert_array_equal(spec.unique_indices, np.array([0, 1]))

        # index_map should map each position to its canonical index:
        # row 0 (value 9) → canonical 0, row 1 (value 3) → canonical 1,
        # row 2 (value 9, dup of row 0) → canonical 0, row 3 (value 3, dup of row 1) → canonical 1
        np.testing.assert_array_equal(spec.index_map, np.array([0, 1, 0, 1]))


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_batch_leaves(self):
        """Empty batch_leaves → ValueError."""
        with pytest.raises(ValueError, match="empty"):
            synthesize_dedup_spec([])

    def test_zero_length_axis(self):
        """Zero-length batch axis → ValueError."""
        batch = np.array([], dtype=np.float32).reshape(0, 1)
        with pytest.raises(ValueError, match="length 0"):
            synthesize_dedup_spec([batch])

    def test_mismatched_batch_dimensions(self):
        """Mismatched batch dimensions across leaves → ValueError."""
        batch1 = np.arange(100, dtype=np.float32).reshape(-1, 1)
        batch2 = np.arange(50, dtype=np.float32).reshape(-1, 1)

        with pytest.raises(ValueError, match="batch dimension"):
            synthesize_dedup_spec([batch1, batch2])

    def test_invalid_axis(self):
        """Invalid axis parameter → ValueError."""
        batch = np.arange(100, dtype=np.float32).reshape(-1, 1)
        with pytest.raises(ValueError, match="out of range"):
            synthesize_dedup_spec([batch], axis=5)

    def test_heterogeneous_ragged_batch_axis(self):
        """Heterogeneous/ragged batch axis → DedupSynthesisUnsupportedError (spec §4.3).

        A Python list with sub-arrays/tuples of inconsistent shapes (ragged) converts to
        dtype=object under np.asarray; v1 does not support such heterogeneous axes.
        """
        # Create a batch_leaf that is a Python list of lists with inconsistent shapes.
        # When converted with np.asarray, this produces dtype=object.
        batch_leaf_ragged = [
            [1.0, 2.0],  # length 2
            [3.0, 4.0, 5.0],  # length 3 — ragged!
            [6.0, 7.0],  # length 2
        ]

        with pytest.raises(DedupSynthesisUnsupportedError, match="heterogeneous"):
            synthesize_dedup_spec([batch_leaf_ragged])

    def test_result_frozen(self):
        """DedupSynthesisResult should be frozen (immutable)."""
        result = DedupSynthesisResult(
            spec=None,
            stage="no_duplication",
            sampled_ratio=0.0,
            transfer_bytes_spent=0,
            k_bucket_bytes=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            # Frozen dataclass prevents mutation.
            result.stage = "synthesized"  # type: ignore


class TestResultMetadata:
    """Verify result metadata is populated correctly."""

    def test_synthesized_result_metadata(self):
        """Synthesized result has all metadata fields."""
        N = 1000
        n_unique = 50
        unique_rows = np.arange(n_unique, dtype=np.float32).reshape(-1, 1)
        batch = np.tile(unique_rows, (N // n_unique + 1, 1))[:N]

        result = synthesize_dedup_spec([batch], threshold=0.5)

        assert result.spec is not None
        assert result.stage == "synthesized"
        assert result.sampled_ratio > 0.5
        assert result.transfer_bytes_spent > 0
        assert result.k_bucket_bytes > 0  # Should be k_bucket * element_width

    def test_no_duplication_result_metadata(self):
        """no_duplication result has correct metadata."""
        N = 100
        batch = np.arange(N, dtype=np.float32).reshape(-1, 1)

        result = synthesize_dedup_spec([batch], threshold=0.5)

        assert result.spec is None
        assert result.stage == "no_duplication"
        assert result.sampled_ratio == 0.0
        assert result.transfer_bytes_spent > 0
        assert result.k_bucket_bytes == 0


class TestSpecValidity:
    """Ensure synthesized specs are valid DedupSpec instances."""

    def test_synthesized_spec_self_assertion(self):
        """Synthesized spec passes DedupSpec.__post_init__ self-assertions."""
        N = 500
        n_unique = 50
        unique_rows = np.arange(n_unique, dtype=np.float32).reshape(-1, 1)
        batch = np.tile(unique_rows, (N // n_unique + 1, 1))[:N]

        result = synthesize_dedup_spec([batch], threshold=0.5)

        if result.spec is not None:
            spec = result.spec
            # These assertions should pass without raising ValueError
            assert len(spec.unique_indices) == spec.k
            assert len(spec.index_map) == N
            # Verify index_map values are in valid range [0, k)
            assert spec.index_map.min() >= 0
            assert spec.index_map.max() < spec.k

    def test_round_trip_to_dedup_gather(self):
        """Synthesized spec round-trips through to_dedup_gather()."""
        N = 2000
        n_unique = 100
        unique_rows = np.arange(n_unique, dtype=np.float32).reshape(-1, 1)
        batch = np.tile(unique_rows, (N // n_unique + 1, 1))[:N]

        result = synthesize_dedup_spec([batch], threshold=0.5)

        if result.spec is not None:
            gather = result.spec.to_dedup_gather()
            assert gather.k == result.spec.k
            assert len(gather.unique_indices) == gather.k_bucket
            assert len(gather.index_map) == N
            # index_map values should still be in [0, k)
            assert gather.index_map.max() < result.spec.k

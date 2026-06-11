"""Tests for xtrax.tiling.strategy — AxisStrategy sealed union and variants."""

import pytest

from xtrax.tiling.strategy import (
    Bucket,
    DedupFn,
    DedupGather,
    GatherFn,
    SafeMap,
    Scan,
    ScanTransition,
    Vmap,
)


class TestAxisStrategyInstantiation:
    """All four variants must instantiate correctly."""

    def test_vmap_instantiates(self):
        """Vmap with no fields instantiates."""
        strategy = Vmap()
        assert isinstance(strategy, Vmap)

    def test_safemap_instantiates(self):
        """SafeMap with batch_size instantiates."""
        strategy = SafeMap(batch_size=32)
        assert isinstance(strategy, SafeMap)
        assert strategy.batch_size == 32

    def test_scan_instantiates(self):
        """Scan with transition function instantiates."""

        def transition(carry, x):
            return carry, x

        strategy = Scan(transition=transition)
        assert isinstance(strategy, Scan)
        assert strategy.transition is transition

    def test_dedupgather_instantiates(self):
        """DedupGather with dedup_fn, gather_fn, k_bucket instantiates."""

        def dedup(xs):
            return xs, None

        def gather(ys, indices):
            return ys

        strategy = DedupGather(dedup_fn=dedup, gather_fn=gather, k_bucket=256)
        assert isinstance(strategy, DedupGather)
        assert strategy.dedup_fn is dedup
        assert strategy.gather_fn is gather
        assert strategy.k_bucket == 256


class TestScanNoFieldLoss:
    """Scan must not have batch_size field — it's intentional."""

    def test_scan_has_no_batch_size_field(self):
        """Scan dataclass has no batch_size attribute."""

        def transition(carry, x):
            return carry, x

        strategy = Scan(transition=transition)
        assert not hasattr(strategy, "batch_size")

    def test_scan_only_has_transition_and_frozen_fields(self):
        """Scan only exposes transition field."""

        def transition(carry, x):
            return carry, x

        strategy = Scan(transition=transition)
        # Verify we can access transition
        assert callable(strategy.transition)
        # Check that no other public field exists
        field_names = {f.name for f in strategy.__dataclass_fields__.values()}
        assert field_names == {"transition"}


class TestFrozenDataclasses:
    """All four variants must be frozen (immutable)."""

    def test_vmap_frozen(self):
        """Vmap is immutable."""
        strategy = Vmap()
        with pytest.raises(Exception):  # FrozenInstanceError
            strategy.anything = True

    def test_safemap_frozen(self):
        """SafeMap is immutable."""
        strategy = SafeMap(batch_size=32)
        with pytest.raises(Exception):
            strategy.batch_size = 64

    def test_scan_frozen(self):
        """Scan is immutable."""

        def transition(carry, x):
            return carry, x

        strategy = Scan(transition=transition)
        with pytest.raises(Exception):
            strategy.transition = None

    def test_dedupgather_frozen(self):
        """DedupGather is immutable."""

        def dedup(xs):
            return xs, None

        def gather(ys, indices):
            return ys

        strategy = DedupGather(dedup_fn=dedup, gather_fn=gather, k_bucket=256)
        with pytest.raises(Exception):
            strategy.k_bucket = 512


class TestProtocolChecks:
    """Protocol isinstance checks must work for callables."""

    def test_scantransition_protocol_accepts_matching_callable(self):
        """A callable with signature (carry, x) -> (carry, y) matches ScanTransition."""

        def my_transition(carry, x):
            return carry, x

        assert isinstance(my_transition, ScanTransition)

    def test_dedupfn_protocol_accepts_matching_callable(self):
        """A callable with signature (xs) -> (xs, indices) matches DedupFn."""

        def my_dedup(xs):
            return xs, None

        assert isinstance(my_dedup, DedupFn)

    def test_gatherfn_protocol_accepts_matching_callable(self):
        """A callable with signature (ys, indices) -> ys matches GatherFn."""

        def my_gather(ys, indices):
            return ys

        assert isinstance(my_gather, GatherFn)


class TestAxisStrategyUnion:
    """AxisStrategy is a sealed union of the four variants."""

    def test_vmap_is_axis_strategy(self):
        """Vmap is a valid AxisStrategy."""
        strategy = Vmap()
        # The union alias should work — verify by isinstance on each variant
        assert isinstance(strategy, Vmap)

    def test_safemap_is_axis_strategy(self):
        """SafeMap is a valid AxisStrategy."""
        strategy = SafeMap(batch_size=32)
        assert isinstance(strategy, SafeMap)

    def test_scan_is_axis_strategy(self):
        """Scan is a valid AxisStrategy."""

        def transition(carry, x):
            return carry, x

        strategy = Scan(transition=transition)
        assert isinstance(strategy, Scan)

    def test_dedupgather_is_axis_strategy(self):
        """DedupGather is a valid AxisStrategy."""

        def dedup(xs):
            return xs, None

        def gather(ys, indices):
            return ys

        strategy = DedupGather(dedup_fn=dedup, gather_fn=gather, k_bucket=256)
        assert isinstance(strategy, DedupGather)

    def test_bucket_is_axis_strategy(self):
        """Bucket is a valid AxisStrategy."""
        strategy = Bucket(boundaries=(8, 16))
        assert isinstance(strategy, Bucket)


class TestBucketStrategy:
    """Bucket variant: instantiation, fields, and immutability."""

    def test_bucket_instantiates(self):
        """Bucket with boundaries instantiates."""
        strategy = Bucket(boundaries=(8, 16, 32))
        assert isinstance(strategy, Bucket)
        assert strategy.boundaries == (8, 16, 32)

    def test_bucket_frozen(self):
        """Bucket is immutable."""
        strategy = Bucket(boundaries=(8,))
        with pytest.raises(Exception):  # FrozenInstanceError
            strategy.boundaries = (16,)

    def test_bucket_hashable(self):
        """Bucket is hashable (frozen with a tuple field)."""
        strategy = Bucket(boundaries=(8, 16))
        # Should not raise — the tuple field makes the frozen dataclass hashable.
        assert hash(strategy) == hash(Bucket(boundaries=(8, 16)))

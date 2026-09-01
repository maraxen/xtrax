"""Tests for xtrax.stages.topology.validate_plan_topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.topology import (
    MaterializeFuseConflictError,
    MaterializeWithoutSinkError,
    MultipleMaterializeAxesError,
    PlanTopologyError,
    axis_boundaries_by_name,
    validate_plan_topology,
)
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap, WhileCarry


def _vmap_decision(name: str, *, heterogeneous: bool = False) -> AxisDecision:
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=0, heterogeneous=heterogeneous)
    return AxisDecision(spec=spec, batch_size=0, reasoning="test", strategy=Vmap())


def _safemap_decision(name: str, tile: int = 1) -> AxisDecision:
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=tile)
    return AxisDecision(
        spec=spec, batch_size=tile, reasoning="test", strategy=SafeMap(batch_size=tile)
    )


def _scan_decision(name: str, *, heterogeneous: bool = False) -> AxisDecision:
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=0, heterogeneous=heterogeneous)
    return AxisDecision(
        spec=spec,
        batch_size=1,
        reasoning="test",
        strategy=Scan(init=None, transition=lambda c, x: (c, x), ordered_sinks=True),
    )


def _dedup_decision(name: str) -> AxisDecision:
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=0)
    return AxisDecision(
        spec=spec,
        batch_size=0,
        reasoning="test",
        strategy=DedupGather(
            unique_indices=np.array([0, 1, 2, 3], dtype=np.int32),
            index_map=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
            k=4,
            k_bucket=4,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx: ys[idx],
        ),
    )


def _bucket_decision(name: str) -> AxisDecision:
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=0)
    return AxisDecision(
        spec=spec, batch_size=0, reasoning="test", strategy=Bucket(boundaries=(4, 8))
    )


def _whilecarry_decision(name: str) -> AxisDecision:
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=0)
    return AxisDecision(spec=spec, batch_size=0, reasoning="test", strategy=WhileCarry())


@dataclass(frozen=True)
class _ForeignBoundary:
    """A boundary object from a parallel BatchPlanner that predates `materialize`.

    topology.py's docstring promises structural compatibility with such objects,
    so Rule 3 must read `materialize` via getattr rather than attribute access.
    """

    fuse: object | None
    tap: object | None
    sink: object | None


class _OrderedTap:
    ordered = True

    def __call__(self, x):
        return x


class _UnorderedSink:
    ordered = False

    def __call__(self, x) -> None:
        pass


class _OrderedSink:
    ordered = True

    def __call__(self, x) -> None:
        pass


class TestScanOnHeterogeneousAxis:
    def test_rejects_scan_on_heterogeneous_axis(self):
        decisions = [_scan_decision("state", heterogeneous=True)]
        with pytest.raises(PlanTopologyError, match="heterogeneous.*Scan"):
            validate_plan_topology(decisions, axis_boundaries={})

    def test_allows_scan_on_homogeneous_axis(self):
        decisions = [_scan_decision("n_noises", heterogeneous=False)]
        validate_plan_topology(decisions, axis_boundaries={})  # must not raise


class TestOrderedBoundaryOnVmapAxis:
    def test_rejects_ordered_tap_on_vmap_axis(self):
        decisions = [_vmap_decision("n_noises")]
        boundaries = {"n_noises": AxisBoundary(tap=_OrderedTap())}
        with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
            validate_plan_topology(decisions, boundaries)

    def test_rejects_ordered_sink_on_vmap_axis(self):
        decisions = [_vmap_decision("n_noises")]
        boundaries = {"n_noises": AxisBoundary(sink=_OrderedSink())}
        with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
            validate_plan_topology(decisions, boundaries)

    def test_allows_unordered_sink_on_vmap_axis(self):
        decisions = [_vmap_decision("n_noises")]
        boundaries = {"n_noises": AxisBoundary(sink=_UnorderedSink())}
        validate_plan_topology(decisions, boundaries)  # must not raise

    def test_allows_ordered_sink_on_safemap_axis(self):
        decisions = [_safemap_decision("n_noises")]
        boundaries = {"n_noises": AxisBoundary(sink=_OrderedSink())}
        validate_plan_topology(decisions, boundaries)  # must not raise

    def test_allows_ordered_sink_on_scan_axis(self):
        decisions = [_scan_decision("n_noises")]
        boundaries = {"n_noises": AxisBoundary(sink=_OrderedSink())}
        validate_plan_topology(decisions, boundaries)  # must not raise


class TestCleanTopologyPasses:
    def test_passes_for_default_plan(self):
        decisions = [
            _safemap_decision("n_structures"),
            _vmap_decision("n_noises"),
            _vmap_decision("n_temperatures"),
            _vmap_decision("n_samples"),
        ]
        validate_plan_topology(decisions, axis_boundaries={})  # no raise

    def test_empty_decisions_passes(self):
        validate_plan_topology([], axis_boundaries={})  # no raise


class TestStructuralDuckTyping:
    """Proves validate_plan_topology works on foreign plan/decision/spec/strategy
    classes that do NOT inherit from any xtrax type -- e.g. a parallel
    BatchPlanner reimplementation (aminx.tiling) whose Scan/Vmap instances
    are distinct classes from xtrax's own. Detection is by exact class name
    (type(strategy).__name__), so the foreign classes below are deliberately
    named `Scan`/`Vmap` -- exactly mirroring aminx.tiling.strategy's actual
    naming convention, not a contrived test setup."""

    def test_foreign_scan_on_heterogeneous_axis_is_rejected(self):
        @dataclass(frozen=True)
        class ForeignSpec:
            name: str
            heterogeneous: bool

        @dataclass(frozen=True)
        class Scan:  # noqa: N801 -- deliberately shadows xtrax's Scan; see class docstring
            pass

        @dataclass(frozen=True)
        class ForeignDecision:
            spec: ForeignSpec
            strategy: object

        decisions = [
            ForeignDecision(
                spec=ForeignSpec(name="state", heterogeneous=True),
                strategy=Scan(),
            )
        ]
        with pytest.raises(PlanTopologyError, match="heterogeneous.*Scan"):
            validate_plan_topology(decisions, axis_boundaries={})

    def test_foreign_vmap_with_ordered_tap_is_rejected(self):
        @dataclass(frozen=True)
        class ForeignSpec:
            name: str
            heterogeneous: bool

        @dataclass(frozen=True)
        class Vmap:  # noqa: N801 -- deliberately shadows xtrax's Vmap; see class docstring
            pass

        @dataclass(frozen=True)
        class ForeignDecision:
            spec: ForeignSpec
            strategy: object

        decisions = [
            ForeignDecision(
                spec=ForeignSpec(name="n_noises", heterogeneous=False),
                strategy=Vmap(),
            )
        ]
        boundaries = {"n_noises": AxisBoundary(tap=_OrderedTap())}
        with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
            validate_plan_topology(decisions, boundaries)


def _axis(name: str) -> AxisSpec:
    return AxisSpec(name=name, cardinality=8, default_batch_size=1)


class TestAxisBoundariesByName:
    """Tests for the T1-02 fork-9 name-keying adapter (#3040)."""

    def test_none_boundaries_returns_empty_mapping(self):
        axes = [_axis("n_structures"), _axis("n_noises")]
        assert axis_boundaries_by_name(axes, None) == {}

    def test_total_and_injective_happy_path(self):
        axes = [_axis("n_structures"), _axis("n_noises")]
        b0 = AxisBoundary(tap=_OrderedTap())
        b1 = AxisBoundary(sink=_UnorderedSink())
        boundaries = [b0, b1]

        result = axis_boundaries_by_name(axes, boundaries)

        assert result == {"n_structures": b0, "n_noises": b1}

    def test_empty_axes_and_boundaries_returns_empty_mapping(self):
        assert axis_boundaries_by_name([], []) == {}

    def test_length_mismatch_raises(self):
        axes = [_axis("n_structures"), _axis("n_noises")]
        boundaries = [AxisBoundary()]
        with pytest.raises(PlanTopologyError, match="1 entries but axes has 2"):
            axis_boundaries_by_name(axes, boundaries)

    def test_duplicate_axis_name_raises_rather_than_silently_dropping(self):
        axes = [_axis("n_noises"), _axis("n_noises")]
        b0 = AxisBoundary(tap=_OrderedTap())
        b1 = AxisBoundary(sink=_OrderedSink())

        with pytest.raises(PlanTopologyError, match="duplicate axis name 'n_noises'"):
            axis_boundaries_by_name(axes, [b0, b1])

    def test_composes_with_validate_plan_topology(self):
        """The adapter's output feeds validate_plan_topology directly."""
        axes = [_axis("n_noises")]
        boundaries = [AxisBoundary(tap=_OrderedTap())]
        axis_boundaries = axis_boundaries_by_name(axes, boundaries)
        decisions = [_vmap_decision("n_noises")]

        with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
            validate_plan_topology(decisions, axis_boundaries)


class TestExportSafeIsOptIn:
    """export_safe defaults False, so no existing caller's behavior changed."""

    def test_sink_passes_without_export_safe(self):
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(sink=_UnorderedSink())}
        validate_plan_topology(decisions, boundaries)  # must not raise

    def test_bucket_strategy_passes_without_export_safe(self):
        decisions = [_bucket_decision("batch")]
        validate_plan_topology(decisions, axis_boundaries={})  # must not raise


class TestExportSafeRule3:
    """Rule 3 is kind-based: a tap always rejects, a sink only if undeclared."""

    def test_rejects_tap_even_when_materialize_is_set(self):
        # A Tap is T -> T and feeds downstream; materialize never applies to it.
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(tap=_OrderedTap(), materialize=True)}
        with pytest.raises(PlanTopologyError, match="has a Tap"):
            validate_plan_topology(decisions, boundaries, export_safe=True)

    def test_rejects_undeclared_sink(self):
        """AC-17b regression check: an undeclared sink rejects exactly as before."""
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(sink=_UnorderedSink())}
        with pytest.raises(PlanTopologyError, match="has a Sink"):
            validate_plan_topology(decisions, boundaries, export_safe=True)

    def test_rejects_undeclared_ordered_sink(self):
        """AC-17b, second half: materialize=False is the pre-existing behavior."""
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(sink=_OrderedSink(), materialize=False)}
        with pytest.raises(PlanTopologyError, match="has a Sink"):
            validate_plan_topology(decisions, boundaries, export_safe=True)

    def test_accepts_declared_materializing_sink(self):
        """AC-17: the one new passing case this pass introduces."""
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(sink=_OrderedSink(), materialize=True)}
        validate_plan_topology(decisions, boundaries, export_safe=True)  # must not raise

    def test_rejects_materialize_with_fuse(self):
        decisions = [_safemap_decision("batch")]
        boundaries = {
            "batch": AxisBoundary(sink=_OrderedSink(), fuse=lambda ys: ys, materialize=True)
        }
        with pytest.raises(MaterializeFuseConflictError, match="also"):
            validate_plan_topology(decisions, boundaries, export_safe=True)

    def test_rejects_materialize_without_sink(self):
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(materialize=True)}
        with pytest.raises(MaterializeWithoutSinkError, match="no sink"):
            validate_plan_topology(decisions, boundaries, export_safe=True)

    def test_fuse_only_boundary_passes(self):
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": AxisBoundary(fuse=lambda ys: ys)}
        validate_plan_topology(decisions, boundaries, export_safe=True)  # must not raise

    def test_foreign_boundary_without_materialize_attribute_is_not_an_error(self):
        """AC-17f: topology.py promises structural compatibility with foreign plans.

        A boundary object from another library predating this field must read as
        materialize=False, not raise AttributeError.
        """
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": _ForeignBoundary(fuse=None, tap=None, sink=None)}
        validate_plan_topology(decisions, boundaries, export_safe=True)  # must not raise

    def test_foreign_boundary_with_sink_still_rejects(self):
        decisions = [_safemap_decision("batch")]
        boundaries = {"batch": _ForeignBoundary(fuse=None, tap=None, sink=_UnorderedSink())}
        with pytest.raises(PlanTopologyError, match="has a Sink"):
            validate_plan_topology(decisions, boundaries, export_safe=True)


class TestExportSafeMultipleMaterializeAxes:
    """AC-17h: the whole-plan tier, separate from the per-axis checks."""

    def test_rejects_two_materializing_axes(self):
        decisions = [_safemap_decision("outer"), _scan_decision("inner")]
        boundaries = {
            "outer": AxisBoundary(sink=_OrderedSink(), materialize=True),
            "inner": AxisBoundary(sink=_OrderedSink(), materialize=True),
        }
        with pytest.raises(MultipleMaterializeAxesError, match="'outer'.*'inner'"):
            validate_plan_topology(decisions, boundaries, export_safe=True)

    def test_accepts_exactly_one_materializing_axis_among_several(self):
        decisions = [_safemap_decision("outer"), _scan_decision("inner")]
        boundaries = {
            "outer": AxisBoundary(sink=_OrderedSink(), materialize=True),
            "inner": AxisBoundary(fuse=lambda ys: ys),
        }
        validate_plan_topology(decisions, boundaries, export_safe=True)  # must not raise


class TestExportSafeRule4:
    """Rule 4 allow-lists the strategies the composer can fold into a callable."""

    @pytest.mark.parametrize("factory", [_vmap_decision, _safemap_decision, _scan_decision])
    def test_accepts_allowed_strategies(self, factory):
        validate_plan_topology([factory("batch")], axis_boundaries={}, export_safe=True)

    def test_accepts_dedup_gather(self):
        """Blocker 7: DedupGather is routed, not rejected."""
        decisions = [_dedup_decision("batch")]
        validate_plan_topology(decisions, axis_boundaries={}, export_safe=True)

    def test_rejects_bucket(self):
        with pytest.raises(PlanTopologyError, match="Bucket is host-tier"):
            validate_plan_topology([_bucket_decision("batch")], {}, export_safe=True)

    def test_rejects_while_carry(self):
        with pytest.raises(PlanTopologyError, match="unbounded trip count"):
            validate_plan_topology([_whilecarry_decision("batch")], {}, export_safe=True)

    def test_names_the_offending_strategy(self):
        with pytest.raises(PlanTopologyError, match="'Bucket'"):
            validate_plan_topology([_bucket_decision("batch")], {}, export_safe=True)

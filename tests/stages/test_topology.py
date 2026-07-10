"""Tests for xtrax.stages.topology.validate_plan_topology."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.topology import (
    PlanTopologyError,
    axis_boundaries_by_name,
    validate_plan_topology,
)
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import SafeMap, Scan, Vmap


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

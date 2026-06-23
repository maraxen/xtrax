"""E1.3b — KEYSTONE: BatchPlanner.plan() fails loud on UNKNOWN-role axes (AC3).

Tests:
- AC3: UNKNOWN-role spec raises AmbiguousAxisError with axis name in message.
- residues-first guard: 'residues' UNKNOWN spec raises, never silently planned.
- Non-regression: KNOWN-role spec (default) plans successfully.
"""

from __future__ import annotations

import dataclasses

import pytest

from xtrax.tiling.roles import AmbiguousAxisError, AxisRole
from xtrax.tiling.plan import AxisSpec, BatchPlan, BatchPlanner

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def known_spec() -> AxisSpec:
    """A well-formed KNOWN-role spec (the default)."""
    return AxisSpec(name="batch", cardinality=64, default_batch_size=128)


@pytest.fixture()
def unknown_spec(known_spec: AxisSpec) -> AxisSpec:
    """Same spec but with role=AxisRole.UNKNOWN (the sentinel that must fail loud)."""
    return dataclasses.replace(known_spec, role=AxisRole.UNKNOWN)


# ---------------------------------------------------------------------------
# AC3: fail-loud on UNKNOWN-role axis
# ---------------------------------------------------------------------------

def test_unknown_role_raises_ambiguous_axis_error(unknown_spec: AxisSpec) -> None:
    """AC3: BatchPlanner.plan([unknown_spec]) must raise AmbiguousAxisError."""
    planner = BatchPlanner()
    with pytest.raises(AmbiguousAxisError):
        planner.plan([unknown_spec])


def test_unknown_role_error_contains_axis_name(unknown_spec: AxisSpec) -> None:
    """AC3: The raised AmbiguousAxisError must name the axis in its message."""
    planner = BatchPlanner()
    with pytest.raises(AmbiguousAxisError, match=unknown_spec.name):
        planner.plan([unknown_spec])


# ---------------------------------------------------------------------------
# residues-first guard: pre-mortem scenario
# ---------------------------------------------------------------------------

def test_residues_unknown_raises_not_silently_planned() -> None:
    """A 'residues' axis with UNKNOWN role must NEVER be silently auto-batched."""
    residues_spec = AxisSpec(
        name="residues",
        cardinality=512,
        default_batch_size=128,
        role=AxisRole.UNKNOWN,
    )
    planner = BatchPlanner()
    with pytest.raises(AmbiguousAxisError, match="residues"):
        planner.plan([residues_spec])


# ---------------------------------------------------------------------------
# Non-regression: KNOWN-role specs plan normally
# ---------------------------------------------------------------------------

def test_known_role_spec_plans_successfully(known_spec: AxisSpec) -> None:
    """KNOWN-role spec (default) must return a BatchPlan without raising."""
    planner = BatchPlanner()
    result = planner.plan([known_spec])
    assert isinstance(result, BatchPlan)
    assert len(result.decisions) == 1
    assert result.decisions[0].spec == known_spec


def test_mixed_specs_with_unknown_raises_on_unknown() -> None:
    """Mixed list: KNOWN + UNKNOWN — must raise on the UNKNOWN axis."""
    good = AxisSpec(name="batch", cardinality=64, default_batch_size=128)
    bad = AxisSpec(
        name="inferred_axis", cardinality=32, default_batch_size=32, role=AxisRole.UNKNOWN
    )
    planner = BatchPlanner()
    with pytest.raises(AmbiguousAxisError, match="inferred_axis"):
        planner.plan([good, bad])


def test_all_known_specs_plan_successfully() -> None:
    """Multiple KNOWN-role specs all plan without error."""
    specs = [
        AxisSpec(name="batch", cardinality=64, default_batch_size=128),
        AxisSpec(name="seq", cardinality=512, default_batch_size=128),
    ]
    planner = BatchPlanner()
    result = planner.plan(specs)
    assert isinstance(result, BatchPlan)
    assert len(result.decisions) == 2

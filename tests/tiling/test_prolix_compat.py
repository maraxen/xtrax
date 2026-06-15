"""Falsifiable compatibility test: can xtrax.tiling CORE back prolix's 6-axis planner?

D7 criterion (spec 260611_aminx-xtrax-refactor.md:125-126):
  1. Build prolix's 6-axis plan using xtrax.AxisSpec + xtrax.BatchPlanner.
  2. Assert decision for n_mols and n_conformers has batch_size > 0.
  3. Assert xtrax.tiling CORE exports no DedupGather or io_callback_sink.

Both PASS and FAIL are valid outcomes: FAIL documents the gap.

API translation from prolix.AxisSpec to xtrax.AxisSpec:
  prolix default_batch_size=0 (vmap)    -> xtrax batch_size=cardinality
  prolix default_batch_size=1 (safe_map) -> xtrax batch_size=1
  prolix tile_granularity               -> xtrax granularity
"""
from __future__ import annotations

import xtrax.tiling as m
from xtrax.tiling import AxisSpec, BatchPlanner

N_ATOMS = AxisSpec(name="n_atoms", cardinality=60_000, batch_size=60_000, granularity=128, heterogeneous=False)
N_BONDS = AxisSpec(name="n_bonds", cardinality=512, batch_size=512, granularity=64, heterogeneous=False)
N_ANGLES = AxisSpec(name="n_angles", cardinality=512, batch_size=512, granularity=64, heterogeneous=False)
N_TORSIONS = AxisSpec(name="n_torsions", cardinality=512, batch_size=512, granularity=64, heterogeneous=False)
N_CONFORMERS = AxisSpec(name="n_conformers", cardinality=2048, batch_size=1, granularity=1, heterogeneous=True)
N_MOLS = AxisSpec(name="n_mols", cardinality=64, batch_size=1, granularity=1, heterogeneous=True)
PROLIX_6_AXES = [N_ATOMS, N_BONDS, N_ANGLES, N_TORSIONS, N_CONFORMERS, N_MOLS]


def _decision_for(plan, name: str):
    for d in plan.decisions:
        if d.spec.name == name:
            return d
    raise KeyError(name)


class TestProlixPlannerCompat:
    def test_planner_accepts_6_axes(self):
        plan = BatchPlanner().plan(PROLIX_6_AXES)
        assert len(plan.decisions) == 6

    def test_n_mols_batch_size_positive(self):
        plan = BatchPlanner().plan(PROLIX_6_AXES)
        d = _decision_for(plan, "n_mols")
        assert d.batch_size > 0, f"n_mols batch_size={d.batch_size!r}; strategy={d.strategy!r}"

    def test_n_conformers_batch_size_positive(self):
        plan = BatchPlanner().plan(PROLIX_6_AXES)
        d = _decision_for(plan, "n_conformers")
        assert d.batch_size > 0, f"n_conformers batch_size={d.batch_size!r}; strategy={d.strategy!r}"

    def test_homogeneous_axes_use_vmap_strategy(self):
        from xtrax.tiling.strategy import Vmap
        plan = BatchPlanner().plan(PROLIX_6_AXES)
        for name in ("n_atoms", "n_bonds", "n_angles", "n_torsions"):
            d = _decision_for(plan, name)
            assert isinstance(d.strategy, Vmap), f"{name}: got {type(d.strategy).__name__!r}"


class TestCoreNonLeakyAssertions:
    """D7-part2: CORE must not export OPTIONAL symbols.
    Expected: test_no_io_callback_sink PASS, test_no_dedup_gather FAIL.
    Gap: DedupGather in xtrax.tiling.__all__ violates D4/D5 CORE/OPTIONAL split.
    """

    def test_no_io_callback_sink(self):
        assert not hasattr(m, "io_callback_sink")

    def test_no_dedup_gather(self):
        """EXPECTED FAIL until D4/D5 ships: DedupGather IS in xtrax.tiling.__all__.
        Gap: move to xtrax.tiling.ext or xtrax.tiling.dedup before R1 DoD.
        """
        assert not hasattr(m, "DedupGather"), (
            "DedupGather exported in xtrax.tiling CORE — OPTIONAL extension; "
            "gap: move to opt-in path before R1 DoD (D4/D5)"
        )

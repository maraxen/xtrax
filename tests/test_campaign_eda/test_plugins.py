"""Unit tests for Campaign EDA domain plugin protocol, quarantine, and immutability."""

import numpy as np
import pytest

from xtrax.campaign.manifest import ManifestRow, ResourceRequirements
from xtrax.eda.campaign.domains.aminx import AminxDomainPlugin
from xtrax.eda.campaign.domains.demistify import DemistifyDomainPlugin
from xtrax.eda.campaign.plugins import (
    clear_quarantined_units,
    enforce_cell_immutability,
    execute_domain_plugin,
    get_quarantined_units,
)
from xtrax.eda.campaign.types import (
    DomainMetricResult,
    LibraryCell,
    ResidueTable,
    SinkSchema,
)


def _make_dummy_cell() -> LibraryCell:
    L = 10
    N = 4
    tokens = np.zeros((N, L), dtype=np.int64)
    mask = np.ones(L, dtype=bool)
    logits = np.zeros((N, L, 21), dtype=np.float64)
    scores = np.ones(N, dtype=np.float64)
    real_mask = np.ones(L, dtype=bool)

    residues = ResidueTable(
        native_tokens="A" * L,
        chain_ids=tuple(["A"] * L),
        res_numbers=tuple(range(1, L + 1)),
        insertion_codes=tuple([""] * L),
        real_mask=real_mask,
        parser_name="test",
        parser_version="1.0",
    )

    return LibraryCell(
        knobs={"temp": 0.3},
        seed=1001,
        target="dummy",
        tokens=tokens,
        sequences=("A" * L,) * N,
        logits=logits,
        scores=scores,
        is_score_derived=False,
        residues=residues,
        mask=mask,
    )


def test_zero_copy_array_immutability():
    """Verify that enforce_cell_immutability sets writeable=False (MAJ-01)."""
    cell = _make_dummy_cell()
    enforce_cell_immutability(cell)

    assert not cell.tokens.flags.writeable
    assert not cell.mask.flags.writeable
    assert cell.logits is not None and not cell.logits.flags.writeable
    assert cell.scores is not None and not cell.scores.flags.writeable
    assert not cell.residues.real_mask.flags.writeable

    # Attempting to mutate in-place must raise ValueError
    with pytest.raises(ValueError, match="read-only"):
        cell.tokens[0, 0] = 99

    with pytest.raises(ValueError, match="read-only"):
        cell.mask[0] = False


def test_plugin_quarantine_on_exception_and_system_exit(tmp_path):
    """Verify that plugin failures and SystemExit trigger quarantine without crashing (BLK-06)."""
    clear_quarantined_units()

    class RoguePlugin:
        @property
        def domain_name(self) -> str:
            return "rogue"

        @property
        def version(self) -> str:
            return "0.1"

        def extract_cell(self, store_path, manifest_row, sink_schema):
            # Attempt to terminate runner process
            raise SystemExit(1)

        def compute_metrics(self, cell):
            return DomainMetricResult(metrics={}, flags={}, tracks={})

    row = ManifestRow(
        manifest_row_hash="deadbeef" * 8,
        input_hash="feedface" * 8,
        campaign_id="test_camp",
        job_id="job_01",
        chunk_id=0,
        sample_start=0,
        sample_count=16,
        target_input="dummy",
        arm_id="rogue_arm",
        arm_backend="local",
        checkpoint_id="chk_0",
        fixed_policy="none",
        state_weight_profile="uniform",
        seed=1001,
        parameters={},
        output_locator=str(tmp_path),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )

    plugin = RoguePlugin()
    results = execute_domain_plugin(plugin, tmp_path, row, SinkSchema(arrays={}))

    # Must return empty list, not crash
    assert results == []

    # Must be recorded in quarantined units
    quarantined = get_quarantined_units()
    assert len(quarantined) >= 1
    assert quarantined[-1].unit_key == row.manifest_row_hash
    assert quarantined[-1].domain_name == "rogue"
    assert "SystemExit" in quarantined[-1].reason


def test_builtin_aminx_plugin(tmp_path):
    """Verify builtin AminxDomainPlugin extraction and metric computation."""
    plugin = AminxDomainPlugin()
    assert plugin.domain_name == "aminx"

    cell = _make_dummy_cell()
    metric_res = plugin.compute_metrics(cell)
    metric_res.validate_schema()

    assert "diversity" in metric_res.metrics
    assert "recovery" in metric_res.metrics
    assert "nll" in metric_res.metrics


def test_builtin_demistify_plugin():
    """Verify builtin DemistifyDomainPlugin extraction and metric computation."""
    plugin = DemistifyDomainPlugin()
    assert plugin.domain_name == "demistify"

    cell = _make_dummy_cell()
    metric_res = plugin.compute_metrics(cell)
    metric_res.validate_schema()

    assert "s_total" in metric_res.metrics
    assert "entropy_per_residue" in metric_res.metrics

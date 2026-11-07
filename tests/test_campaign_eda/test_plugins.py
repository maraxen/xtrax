"""Unit tests for Campaign EDA domain plugin protocol, quarantine, and immutability."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
)
from xtrax.eda.campaign import run_campaign_eda
from xtrax.eda.campaign.domains import (
    DomainPlugin,
    PluginRegistry,
    validate_domain_plugin,
)
from xtrax.eda.campaign.domains.aminx import AminxDomainPlugin
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


class SyntheticTestDomainPlugin:
    """Synthetic domain plugin adhering strictly to the DomainPlugin protocol."""

    @property
    def domain_name(self) -> str:
        return "synthetic_test"

    @property
    def version(self) -> str:
        return "1.0.0"

    def extract_cell(
        self,
        store_path: Path,
        manifest_row: ManifestRow,
        sink_schema: SinkSchema,
    ) -> Sequence[LibraryCell]:
        return [_make_dummy_cell()]

    def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
        mean_score = float(np.mean(cell.scores)) if cell.scores is not None else 0.0
        return DomainMetricResult(
            metrics={
                "synthetic_mean_score": mean_score,
                "synthetic_score_doubled": mean_score * 2.0,
            },
            flags={"has_scores": cell.scores is not None},
            tracks={"synthetic_track": np.ones(cell.tokens.shape[1], dtype=np.float64)},
        )


def test_synthetic_domain_plugin_protocol_and_registration(tmp_path):
    """Verify that domain plugins adhering to DomainPlugin can be registered,
    validated, and executed.
    """
    clear_quarantined_units()
    plugin = SyntheticTestDomainPlugin()

    # 1. Runtime protocol compliance check
    assert isinstance(plugin, DomainPlugin)

    # 2. Schema validation
    val_result = validate_domain_plugin(plugin)
    assert val_result.is_valid
    assert not val_result.has_errors
    assert len(val_result.issues) == 0

    # 3. Registration and discovery via PluginRegistry
    PluginRegistry.register(plugin)
    discovered = PluginRegistry.discover()
    assert "synthetic_test" in discovered
    assert PluginRegistry.get("synthetic_test") is plugin

    # 4. Defensive execution via execute_domain_plugin
    row = ManifestRow(
        manifest_row_hash="aabbccdd" * 8,
        input_hash="11223344" * 8,
        campaign_id="test_camp",
        job_id="job_02",
        chunk_id=0,
        sample_start=0,
        sample_count=4,
        target_input="dummy",
        arm_id="test_arm",
        arm_backend="local",
        checkpoint_id="chk_1",
        fixed_policy="none",
        state_weight_profile="uniform",
        seed=1002,
        parameters={},
        output_locator=str(tmp_path),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )

    results = execute_domain_plugin(plugin, tmp_path, row, SinkSchema(arrays={}))
    assert len(results) == 1
    cell, metric_result = results[0]

    # Immutability verification
    assert not cell.tokens.flags.writeable
    assert not cell.mask.flags.writeable

    # Genuine metric assertions
    assert metric_result.metrics["synthetic_mean_score"] == 1.0
    assert metric_result.metrics["synthetic_score_doubled"] == 2.0
    assert metric_result.flags["has_scores"] is True
    assert metric_result.tracks["synthetic_track"].shape == (10,)

    # Confirm no quarantine for valid execution
    assert len(get_quarantined_units()) == 0


def test_synthetic_domain_plugin_quarantine_on_invalid_metric_schema(tmp_path):
    """Verify that domain plugins emitting non-finite or non-numeric metrics
    are quarantined (MAJ-02).
    """
    clear_quarantined_units()

    class BadMetricPlugin:
        @property
        def domain_name(self) -> str:
            return "bad_metric"

        @property
        def version(self) -> str:
            return "1.0.0"

        def extract_cell(self, store_path, manifest_row, sink_schema):
            return [_make_dummy_cell()]

        def compute_metrics(self, cell):
            # Non-finite NaN metric violates schema
            return DomainMetricResult(
                metrics={"invalid_metric": float("nan")},
                flags={},
                tracks={},
            )

    plugin = BadMetricPlugin()
    row = ManifestRow(
        manifest_row_hash="eeff0011" * 8,
        input_hash="99887766" * 8,
        campaign_id="test_camp",
        job_id="job_03",
        chunk_id=0,
        sample_start=0,
        sample_count=4,
        target_input="dummy",
        arm_id="bad_arm",
        arm_backend="local",
        checkpoint_id="chk_2",
        fixed_policy="none",
        state_weight_profile="uniform",
        seed=1003,
        parameters={},
        output_locator=str(tmp_path),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )

    results = execute_domain_plugin(plugin, tmp_path, row, SinkSchema(arrays={}))
    assert results == []

    quarantined = get_quarantined_units()
    assert len(quarantined) == 1
    assert quarantined[0].unit_key == row.manifest_row_hash
    assert quarantined[0].domain_name == "bad_metric"
    assert "TypeError" in quarantined[0].reason


def test_synthetic_domain_plugin_quarantine_on_non_string_metric_keys(tmp_path):
    """Verify that domain plugins with non-string metric keys trigger TypeError and quarantine."""
    clear_quarantined_units()

    class NonStringKeyPlugin:
        @property
        def domain_name(self) -> str:
            return "non_string_key"

        @property
        def version(self) -> str:
            return "1.0.0"

        def extract_cell(self, store_path, manifest_row, sink_schema):
            return [_make_dummy_cell()]

        def compute_metrics(self, cell):
            # Non-string metric key violates schema;
            # instantiate via __new__ to exercise validate_schema
            obj = object.__new__(DomainMetricResult)
            object.__setattr__(obj, "metrics", {123: 1.0})
            object.__setattr__(obj, "flags", {})
            object.__setattr__(obj, "tracks", {})
            return obj

    plugin = NonStringKeyPlugin()
    row = ManifestRow(
        manifest_row_hash="nonstr0011" * 8,
        input_hash="99887766" * 8,
        campaign_id="test_camp",
        job_id="job_non_str",
        chunk_id=0,
        sample_start=0,
        sample_count=4,
        target_input="dummy",
        arm_id="test_arm",
        arm_backend="local",
        checkpoint_id="chk_nonstr",
        fixed_policy="none",
        state_weight_profile="uniform",
        seed=1004,
        parameters={},
        output_locator=str(tmp_path),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )

    results = execute_domain_plugin(plugin, tmp_path, row, SinkSchema(arrays={}))
    assert results == []

    quarantined = get_quarantined_units()
    assert len(quarantined) == 1
    assert quarantined[0].unit_key == row.manifest_row_hash
    assert quarantined[0].domain_name == "non_string_key"
    assert "TypeError" in quarantined[0].reason
    assert "Metric key 123 must be str" in quarantined[0].reason


def test_domain_metric_result_validate_schema_flag_types():
    """Verify validate_schema rejects non-string flag keys and non-boolean flag values."""
    # Valid schema
    res_ok = DomainMetricResult(metrics={"m": 1.0}, flags={"f": True}, tracks={})
    res_ok.validate_schema()

    # Non-string flag key
    obj_bad_key = object.__new__(DomainMetricResult)
    object.__setattr__(obj_bad_key, "metrics", {"m": 1.0})
    object.__setattr__(obj_bad_key, "flags", {123: True})
    object.__setattr__(obj_bad_key, "tracks", {})
    with pytest.raises(TypeError, match="Flag.*invalid"):
        obj_bad_key.validate_schema()

    # Non-bool flag value
    obj_bad_val = object.__new__(DomainMetricResult)
    object.__setattr__(obj_bad_val, "metrics", {"m": 1.0})
    object.__setattr__(obj_bad_val, "flags", {"f": "not_bool"})
    object.__setattr__(obj_bad_val, "tracks", {})
    with pytest.raises(TypeError, match="Flag.*invalid"):
        obj_bad_val.validate_schema()


def test_runtime_quarantined_units_append_coverage_banners(tmp_path):
    """Verify that runtime quarantined units append banners to q4_coverage["banners"]."""
    clear_quarantined_units()

    campaign_dir = tmp_path / "flaky_campaign"
    units_dir = campaign_dir / "units"
    units_dir.mkdir(parents=True)

    # Unit 1: Valid committed unit
    u1_hash = "u1_good_hash"
    u1_dir = units_dir / u1_hash
    u1_dir.mkdir()
    np.save(u1_dir / "sequences.npy", np.array(["ACDEFGHIKL"]))
    np.save(u1_dir / "scores.npy", np.array([1.0]))
    with open(u1_dir / "done.json", "w", encoding="utf-8") as f:
        json.dump({"manifest_row_hash": u1_hash, "status": "committed", "duration_s": 1.0}, f)

    # Unit 2: Committed on disk but fails during domain plugin execution
    u2_hash = "u2_flaky_hash"
    u2_dir = units_dir / u2_hash
    u2_dir.mkdir()
    np.save(u2_dir / "sequences.npy", np.array(["ACDEFGHIKL"]))
    np.save(u2_dir / "scores.npy", np.array([2.0]))
    with open(u2_dir / "done.json", "w", encoding="utf-8") as f:
        json.dump({"manifest_row_hash": u2_hash, "status": "committed", "duration_s": 1.0}, f)

    row1 = ManifestRow(
        manifest_row_hash=u1_hash,
        input_hash=u1_hash,
        campaign_id="flaky_camp",
        job_id="j1",
        chunk_id=0,
        sample_start=0,
        sample_count=1,
        target_input="dummy",
        arm_id="arm1",
        arm_backend="cpu",
        checkpoint_id="chk1",
        fixed_policy="none",
        state_weight_profile="uniform",
        seed=1,
        parameters={"temperature": 0.1},
        output_locator=str(u1_dir),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )
    row2 = ManifestRow(
        manifest_row_hash=u2_hash,
        input_hash=u2_hash,
        campaign_id="flaky_camp",
        job_id="j2",
        chunk_id=0,
        sample_start=0,
        sample_count=1,
        target_input="dummy",
        arm_id="arm1",
        arm_backend="cpu",
        checkpoint_id="chk1",
        fixed_policy="none",
        state_weight_profile="uniform",
        seed=2,
        parameters={"temperature": 0.1},
        output_locator=str(u2_dir),
        resource_requirements=ResourceRequirements(gpu_memory_bytes=0),
    )

    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=ManifestMetadata(
            campaign_id="flaky_camp",
            planner_version="1.0",
            spec_sha256="abc",
            total_rows=2,
            created_at_utc="2026-11-06T20:30:00Z",
        ),
        rows=[row1, row2],
    )
    with open(campaign_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    class FlakyDomainPlugin:
        @property
        def domain_name(self) -> str:
            return "flaky"

        @property
        def version(self) -> str:
            return "1.0.0"

        def extract_cell(self, store_path, manifest_row, sink_schema):
            if manifest_row.manifest_row_hash == u2_hash:
                raise RuntimeError("Corrupt weights in unit")
            return [_make_dummy_cell()]

        def compute_metrics(self, cell):
            return DomainMetricResult(
                metrics={"diversity": 0.5, "nll": 1.0},
                flags={},
                tracks={},
            )

    plugin = FlakyDomainPlugin()
    PluginRegistry.register(plugin)
    results = run_campaign_eda(
        campaign_dir=campaign_dir,
        manifest=manifest,
        domain_name="flaky",
        export_artifacts=False,
    )

    # 1. Verify that u2 was quarantined
    quarantined = get_quarantined_units()
    assert any(q.unit_key == u2_hash for q in quarantined)

    # 2. Verify that only 1 unit made it to findings
    assert len(results["q1_composition"]["rows"]) == 1
    assert results["q1_composition"]["rows"][0]["manifest_row_hash"] == u1_hash

    # 3. Verify that q4_coverage["banners"] contains the quarantine banner
    banners = results["q4_coverage"]["banners"]
    assert len(banners) >= 1
    q_banners = [b for b in banners if isinstance(b, dict) and b.get("unit_id") == u2_hash]
    assert len(q_banners) == 1
    assert q_banners[0]["type"] == "quarantined"
    assert "Corrupt weights in unit" in q_banners[0]["reason"]

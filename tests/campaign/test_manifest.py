"""Tests for xtrax.campaign.manifest schemas, pure host planner, and hash generation."""

import dataclasses
import json
from pathlib import Path

import pytest

from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestIntegrityError,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
    build_manifest_row,
    canonical_json_bytes,
    compute_manifest_row_hash,
    compute_unit_input_hash,
    plan_campaign,
)
from xtrax.campaign.spec import (
    ArmSpec,
    CampaignSpec,
    ParamGrid,
    SpecPartitionError,
)


def test_resource_requirements_dataclass() -> None:
    res = ResourceRequirements(
        gpu_memory_bytes=8000000000,
        cpu_cores=2,
        timeout_seconds=1200,
        padding_bucket_id=256,
    )
    assert res.gpu_memory_bytes == 8000000000
    assert res.cpu_cores == 2
    assert res.timeout_seconds == 1200
    assert res.padding_bucket_id == 256
    assert hasattr(res, "__slots__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.cpu_cores = 4  # type: ignore[misc]

    data = res.to_dict()
    rebuilt = ResourceRequirements.from_dict(data)
    assert rebuilt == res


def test_manifest_row_dataclass_and_separation_of_hashes() -> None:
    row = build_manifest_row(
        campaign_id="test_camp",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=100,
        target_input="target.pdb",
        arm_id="arm_1",
        arm_backend="boltz2",
        checkpoint_id="ckpt_1",
        parameters={"temperature": 0.2},
    )
    assert row.manifest_row_hash != row.input_hash
    assert len(row.manifest_row_hash) == 64
    assert len(row.input_hash) == 64
    assert hasattr(row, "__slots__")

    d = row.to_dict()
    rebuilt = ManifestRow.from_dict(d)
    assert rebuilt == row


def test_campaign_manifest_schema_and_roundtrip(tmp_path: Path) -> None:
    meta = ManifestMetadata(
        campaign_id="test_camp",
        created_at_utc="2026-11-05T18:00:00Z",
        planner_version="xtrax@0.5.0",
        spec_sha256="c" * 64,
        total_rows=1,
    )
    row = build_manifest_row(
        campaign_id="test_camp",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=100,
        target_input="target.pdb",
        arm_id="arm_1",
        arm_backend="boltz2",
        checkpoint_id="ckpt_1",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=42,
        parameters={"temp": 0.1},
    )
    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=meta,
        rows=[row],
    )
    assert manifest.schema_version == "campaign_manifest_v1"

    out_file = tmp_path / "manifest.json"
    out_file.write_bytes(canonical_json_bytes(manifest.to_dict()))

    loaded_dict = json.loads(out_file.read_text(encoding="utf-8"))
    rebuilt = CampaignManifest.from_dict(loaded_dict)
    assert rebuilt == manifest


def test_compute_manifest_row_hash_deterministic() -> None:
    payload1 = {
        "campaign_id": "c1",
        "job_id": "j1",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 100,
        "target_input": "t.pdb",
        "arm_id": "a1",
        "arm_backend": "b1",
        "checkpoint_id": "k1",
        "fixed_policy": "none",
        "state_weight_profile": "equal",
        "seed": 42,
        "parameters": {"temperature": 0.2, "step": 10},
    }
    payload2 = {
        "parameters": {"step": 10, "temperature": 0.2},
        "seed": 42,
        "state_weight_profile": "equal",
        "fixed_policy": "none",
        "checkpoint_id": "k1",
        "arm_backend": "b1",
        "arm_id": "a1",
        "target_input": "t.pdb",
        "sample_count": 100,
        "sample_start": 0,
        "chunk_id": 0,
        "job_id": "j1",
        "campaign_id": "c1",
    }
    h1 = compute_manifest_row_hash(payload1)
    h2 = compute_manifest_row_hash(payload2)
    assert len(h1) == 64
    assert h1 == h2


def test_compute_unit_input_hash_filters_campaign_owned_keys() -> None:
    spec_params = {
        "temperature": 0.2,
        "step": 50,
        "output_path": "/ignore/me",
        "job_id": "ignored_job",
        "chunk_id": 999,
    }
    h = compute_unit_input_hash(
        spec_dict=spec_params,
        campaign_id="c1",
        job_id="real_job",
        chunk_id=0,
        sample_start=0,
        sample_count=100,
        checkpoint_id="ckpt_1",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=42,
    )
    assert len(h) == 64


def test_plan_campaign_cartesian_expansion_and_chunking() -> None:
    arm1 = ArmSpec(arm_id="arm_1", backend="b1", checkpoint_id="c1")
    arm2 = ArmSpec(arm_id="arm_2", backend="b2", checkpoint_id="c2")
    grid = ParamGrid(axes={"temperature": [0.1, 0.2], "mode": ["fast"]})
    # 2 targets * 2 arms * 2 temp values = 8 parameter combinations
    # samples_per_target = 250, chunk_size = 100 -> 3 chunks per combination (100, 100, 50)
    # Total rows = 8 * 3 = 24 rows
    spec = CampaignSpec(
        campaign_id="plan_test",
        target_inputs=["t1.pdb", "t2.pdb"],
        arms=[arm1, arm2],
        grid=grid,
        samples_per_target=250,
        chunk_size=100,
        seed=42,
    )

    manifest = plan_campaign(spec)
    assert len(manifest.rows) == 24
    assert manifest.metadata.total_rows == 24
    assert manifest.metadata.campaign_id == "plan_test"

    # Verify chunk partitioning
    chunk_0 = manifest.rows[0]
    chunk_1 = manifest.rows[1]
    chunk_2 = manifest.rows[2]
    assert chunk_0.chunk_id == 0 and chunk_0.sample_start == 0 and chunk_0.sample_count == 100
    assert chunk_1.chunk_id == 1 and chunk_1.sample_start == 100 and chunk_1.sample_count == 100
    assert chunk_2.chunk_id == 2 and chunk_2.sample_start == 200 and chunk_2.sample_count == 50


def test_gate_01_positive_control_byte_for_byte_identical_manifest(tmp_path: Path) -> None:
    """GATE-01: Byte-for-byte identical manifest.json on successive planning runs."""
    arm1 = ArmSpec(arm_id="arm_1", backend="b1", checkpoint_id="c1", parameters={"a": 1})
    arm2 = ArmSpec(arm_id="arm_2", backend="b2", checkpoint_id="c2", parameters={"b": 2})
    grid = ParamGrid(axes={"temp": [0.1, 0.5, 1.0]})
    # 2 targets * 2 arms * 3 grid combinations = 12 units total
    spec = CampaignSpec(
        campaign_id="gate01_campaign",
        target_inputs=["target_alpha.pdb", "target_beta.pdb"],
        arms=[arm1, arm2],
        grid=grid,
        samples_per_target=100,
        chunk_size=100,
        seed=12345,
    )

    path1 = tmp_path / "run1" / "manifest.json"
    path2 = tmp_path / "run2" / "manifest.json"

    m1 = plan_campaign(spec, output_manifest_path=path1)
    m2 = plan_campaign(spec, output_manifest_path=path2)

    assert len(m1.rows) == 12
    assert len(m2.rows) == 12

    bytes1 = path1.read_bytes()
    bytes2 = path2.read_bytes()

    assert bytes1 == bytes2, "GATE-01 FAILED: Successive planning runs produced differing manifests"


def test_gate_02_negative_control_spec_partition_error_on_callable() -> None:
    """GATE-02: SpecPartitionError raised when callable or lambda is injected into parameters."""
    arm = ArmSpec(
        arm_id="arm_1",
        backend="b1",
        checkpoint_id="c1",
        parameters={"alpha": 1.0},
    )

    # Negative control 1: Injecting callable into CampaignSpec.parameters raises SpecPartitionError
    with pytest.raises(SpecPartitionError, match="Callable"):
        CampaignSpec(
            campaign_id="gate02_campaign",
            target_inputs=["target.pdb"],
            arms=[arm],
            parameters={"scoring_fn": lambda x: x * 2},
        )

    # Negative control 2: Injecting callable into ArmSpec.parameters raises SpecPartitionError
    with pytest.raises(SpecPartitionError, match="Callable"):
        ArmSpec(
            arm_id="arm_bad",
            backend="b",
            checkpoint_id="c",
            parameters={"loss_fn": lambda y: y},
        )

    # Negative control 3: Injecting callable into ParamGrid.axes raises SpecPartitionError
    with pytest.raises(SpecPartitionError, match="Callable"):
        ParamGrid(axes={"scoring_fn": [lambda x: x * 2]})

    # Negative control 4: If a spec somehow has a callable, plan_campaign catches and fails
    bypassed_spec = CampaignSpec.__new__(CampaignSpec)
    object.__setattr__(bypassed_spec, "campaign_id", "gate02_bypassed")
    object.__setattr__(bypassed_spec, "target_inputs", ["target.pdb"])
    object.__setattr__(bypassed_spec, "arms", [arm])
    object.__setattr__(bypassed_spec, "grid", ParamGrid())
    object.__setattr__(bypassed_spec, "samples_per_target", 100)
    object.__setattr__(bypassed_spec, "chunk_size", 100)
    object.__setattr__(bypassed_spec, "seed", 42)
    object.__setattr__(bypassed_spec, "output_dir", "outputs")
    object.__setattr__(bypassed_spec, "metadata", {})
    object.__setattr__(bypassed_spec, "parameters", {"fn": lambda z: z})

    with pytest.raises(SpecPartitionError, match="Callable"):
        plan_campaign(bypassed_spec)


def _planned_manifest_dict(tmp_path: Path) -> dict:
    """Plan a small campaign with awkward parameter values, write it, and reload the JSON."""
    arm = ArmSpec(
        arm_id="arm_1",
        backend="b1",
        checkpoint_id="c1",
        parameters={"alpha": 0.1, "neg_zero": -0.0, "tags": ["x", "y"]},
    )
    spec = CampaignSpec(
        campaign_id="integrity_campaign",
        target_inputs=["target.pdb"],
        arms=[arm],
        grid=ParamGrid(axes={"temp": [0.1, 0.5]}),
        samples_per_target=100,
        chunk_size=100,
        seed=7,
        parameters={"scale": 2.5},
    )
    path = tmp_path / "manifest.json"
    plan_campaign(spec, output_manifest_path=path)
    return json.loads(path.read_text(encoding="utf-8"))


def test_planned_manifest_round_trip_verifies(tmp_path: Path) -> None:
    data = _planned_manifest_dict(tmp_path)
    assert len(data["rows"]) == 2
    loaded = CampaignManifest.from_dict(data)
    assert [row.manifest_row_hash for row in loaded.rows] == [
        r["manifest_row_hash"] for r in data["rows"]
    ]
    assert [row.input_hash for row in loaded.rows] == [r["input_hash"] for r in data["rows"]]


def test_edited_parameters_raise_manifest_integrity_error(tmp_path: Path) -> None:
    data = _planned_manifest_dict(tmp_path)
    data["rows"][0]["parameters"]["temp"] = 0.9999
    with pytest.raises(ManifestIntegrityError, match="job_id") as excinfo:
        CampaignManifest.from_dict(data)
    assert data["rows"][0]["job_id"] in str(excinfo.value)


def test_edited_input_hash_raises_manifest_integrity_error(tmp_path: Path) -> None:
    data = _planned_manifest_dict(tmp_path)
    data["rows"][1]["input_hash"] = "0" * 64
    with pytest.raises(ManifestIntegrityError, match="input_hash"):
        CampaignManifest.from_dict(data)


def test_edited_manifest_row_hash_raises_manifest_integrity_error(tmp_path: Path) -> None:
    data = _planned_manifest_dict(tmp_path)
    data["rows"][0]["manifest_row_hash"] = "f" * 64
    with pytest.raises(ManifestIntegrityError, match="manifest_row_hash"):
        CampaignManifest.from_dict(data)


def _identity_spec(temps: list[float], targets: list[str]) -> CampaignSpec:
    """One-arm spec with a single-chunk grid over ``temp``; used by the identity tests."""
    return CampaignSpec(
        campaign_id="identity_campaign",
        target_inputs=targets,
        arms=[ArmSpec(arm_id="arm_1", backend="b1", checkpoint_id="c1")],
        grid=ParamGrid(axes={"temp": temps}),
        samples_per_target=100,
        chunk_size=100,
        seed=7,
    )


def test_inserting_grid_value_mid_axis_keeps_existing_row_hashes() -> None:
    """Inserting 0.5 between 0.1 and 1.0 must not change identity of existing units."""
    targets = ["t1.pdb", "t2.pdb"]
    before = plan_campaign(_identity_spec([0.1, 1.0], targets)).rows
    after = plan_campaign(_identity_spec([0.1, 0.5, 1.0], targets)).rows
    after_hashes = {row.manifest_row_hash for row in after}
    assert len(before) == 4
    for row in before:
        assert row.manifest_row_hash in after_hashes, f"row {row.job_id} changed identity"


def test_prepending_target_keeps_existing_row_hashes() -> None:
    """Prepending a target must not shift the identity of rows for the existing targets."""
    before = plan_campaign(_identity_spec([0.1, 1.0], ["t1.pdb"])).rows
    after = plan_campaign(_identity_spec([0.1, 1.0], ["t_new.pdb", "t1.pdb"])).rows
    after_hashes = {row.manifest_row_hash for row in after}
    assert len(before) == 2
    for row in before:
        assert row.manifest_row_hash in after_hashes, f"row {row.job_id} changed identity"


def test_same_stem_in_different_directories_gets_distinct_identity() -> None:
    """Targets that share a file stem but differ in directory must not collide."""
    rows = plan_campaign(_identity_spec([0.1], ["dir_a/target.pdb", "dir_b/target.pdb"])).rows
    assert len(rows) == 2
    assert len({row.job_id for row in rows}) == 2
    assert len({row.manifest_row_hash for row in rows}) == 2


def test_planning_is_deterministic_with_unique_job_ids(tmp_path: Path) -> None:
    """Same spec planned twice yields identical manifest bytes and unique job_ids."""
    spec = _identity_spec([0.1, 0.5, 1.0], ["dir_a/t.pdb", "dir_b/t.pdb"])
    path1 = tmp_path / "run1" / "manifest.json"
    path2 = tmp_path / "run2" / "manifest.json"
    m1 = plan_campaign(spec, output_manifest_path=path1)
    plan_campaign(spec, output_manifest_path=path2)
    assert path1.read_bytes() == path2.read_bytes()
    assert len({row.job_id for row in m1.rows}) == len(m1.rows)

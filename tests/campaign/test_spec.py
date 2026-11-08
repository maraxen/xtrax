"""Tests for xtrax.campaign.spec declarative schemas and serialization."""

import dataclasses
from pathlib import Path

import pytest

from xtrax.campaign.spec import (
    ArmSpec,
    CampaignSpec,
    ParamGrid,
    SpecPartitionError,
)


def test_param_grid_defaults_and_slots() -> None:
    grid = ParamGrid()
    assert grid.axes == {}
    assert hasattr(grid, "__slots__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.axes = {"a": [1]}  # type: ignore[misc]


def test_param_grid_dict_roundtrip() -> None:
    axes = {"temperature": [0.1, 0.2], "steps": [20, 50]}
    grid = ParamGrid(axes=axes)
    data = grid.to_dict()
    assert data == {"axes": axes}
    rebuilt = ParamGrid.from_dict(data)
    assert rebuilt == grid


def test_arm_spec_defaults_and_immutability() -> None:
    arm = ArmSpec(arm_id="arm_01", backend="boltz2", checkpoint_id="ckpt_v1")
    assert arm.arm_id == "arm_01"
    assert arm.backend == "boltz2"
    assert arm.checkpoint_id == "ckpt_v1"
    assert arm.fixed_policy == "none"
    assert arm.state_weight_profile == "equal"
    assert arm.parameters == {}
    assert arm.resource_overrides == {}
    assert hasattr(arm, "__slots__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        arm.arm_id = "other"  # type: ignore[misc]


def test_arm_spec_validation() -> None:
    with pytest.raises(ValueError, match="arm_id"):
        ArmSpec(arm_id="", backend="boltz2", checkpoint_id="ckpt_v1")
    with pytest.raises(ValueError, match="backend"):
        ArmSpec(arm_id="arm_01", backend="", checkpoint_id="ckpt_v1")
    with pytest.raises(ValueError, match="checkpoint_id"):
        ArmSpec(arm_id="arm_01", backend="boltz2", checkpoint_id="")


def test_arm_spec_dict_roundtrip() -> None:
    arm = ArmSpec(
        arm_id="arm_01",
        backend="boltz2",
        checkpoint_id="ckpt_v1",
        fixed_policy="strict",
        state_weight_profile="weighted",
        parameters={"temp": 0.5},
        resource_overrides={"gpu_memory_bytes": 16000000000},
    )
    data = arm.to_dict()
    rebuilt = ArmSpec.from_dict(data)
    assert rebuilt == arm


def test_campaign_spec_validation() -> None:
    arm = ArmSpec(arm_id="a1", backend="b1", checkpoint_id="c1")
    # Empty campaign_id
    with pytest.raises(ValueError, match="campaign_id"):
        CampaignSpec(campaign_id="", target_inputs=["t1.pdb"], arms=[arm])

    # Empty target_inputs
    with pytest.raises(ValueError, match="target_inputs"):
        CampaignSpec(campaign_id="c1", target_inputs=[], arms=[arm])

    # Empty arms
    with pytest.raises(ValueError, match="arms"):
        CampaignSpec(campaign_id="c1", target_inputs=["t1.pdb"], arms=[])

    # chunk_size < 1
    with pytest.raises(ValueError, match="chunk_size"):
        CampaignSpec(campaign_id="c1", target_inputs=["t1.pdb"], arms=[arm], chunk_size=0)

    # samples_per_target < 1
    with pytest.raises(ValueError, match="samples_per_target"):
        CampaignSpec(campaign_id="c1", target_inputs=["t1.pdb"], arms=[arm], samples_per_target=0)


def test_campaign_spec_dict_roundtrip() -> None:
    arm1 = ArmSpec(arm_id="a1", backend="b1", checkpoint_id="c1", parameters={"p1": 1})
    arm2 = ArmSpec(arm_id="a2", backend="b2", checkpoint_id="c2", parameters={"p2": 2})
    grid = ParamGrid(axes={"temp": [0.1, 0.2]})
    spec = CampaignSpec(
        campaign_id="camp_test",
        target_inputs=["t1.pdb", "t2.pdb"],
        arms=[arm1, arm2],
        grid=grid,
        samples_per_target=200,
        chunk_size=50,
        seed=123,
        output_dir="custom_outputs",
        metadata={"author": "teamwork", "version": "1.0"},
    )
    d = spec.to_dict()
    rebuilt = CampaignSpec.from_dict(d)
    assert rebuilt == spec


def test_campaign_spec_toml_roundtrip(tmp_path: Path) -> None:
    arm1 = ArmSpec(
        arm_id="arm_alpha",
        backend="backend_alpha",
        checkpoint_id="ckpt_alpha",
        parameters={"temperature": 0.2, "recycle_count": 3},
        resource_overrides={"gpu_memory_bytes": 17179869184},
    )
    grid = ParamGrid(axes={"sampling_steps": [50, 100]})
    spec = CampaignSpec(
        campaign_id="glp1r_optimization",
        target_inputs=["inputs/glp1r.pdb", "inputs/glp1r_alt.pdb"],
        arms=[arm1],
        grid=grid,
        samples_per_target=100,
        chunk_size=50,
        seed=42,
        output_dir="outputs/glp1r",
        metadata={"created_at_utc": "2026-11-05T18:00:00Z"},
    )

    toml_text = spec.to_toml()
    assert "glp1r_optimization" in toml_text
    assert "arm_alpha" in toml_text

    # Parse from string
    from_str = CampaignSpec.from_toml(toml_text)
    assert from_str == spec

    # Save to file and parse from path
    toml_file = tmp_path / "campaign.toml"
    spec.to_toml(toml_file)
    assert toml_file.exists()
    from_file = CampaignSpec.from_toml(toml_file)
    assert from_file == spec


def test_spec_partition_error_on_callables() -> None:
    # Lambda in arm parameters
    with pytest.raises(SpecPartitionError, match="Callable"):
        ArmSpec(
            arm_id="arm_bad",
            backend="b",
            checkpoint_id="c",
            parameters={"fn": lambda x: x},
        )

    # Callable in CampaignSpec parameters
    with pytest.raises(SpecPartitionError, match="Callable"):
        CampaignSpec(
            campaign_id="c_bad",
            target_inputs=["t.pdb"],
            arms=[ArmSpec(arm_id="a", backend="b", checkpoint_id="c")],
            parameters={"fn": lambda x: x},  # type: ignore[call-arg]
        )

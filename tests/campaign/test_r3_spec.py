"""Review round 3: prevent content-derived unit identity collisions in spec and planner."""

from __future__ import annotations

import pytest

from xtrax.campaign.manifest import plan_campaign
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid


def test_param_grid_rejects_negative_and_positive_zero() -> None:
    with pytest.raises(ValueError, match="lr"):
        ParamGrid(axes={"lr": [-0.0, 0.0]})


def test_campaign_spec_rejects_duplicate_target_inputs() -> None:
    arm = ArmSpec(arm_id="arm_1", backend="b1", checkpoint_id="c1")
    with pytest.raises(ValueError, match=r"a\.pdb"):
        CampaignSpec(campaign_id="c1", target_inputs=["a.pdb", "a.pdb"], arms=[arm])


def test_campaign_spec_rejects_duplicate_arm_ids() -> None:
    arm1 = ArmSpec(arm_id="dup_arm", backend="b1", checkpoint_id="c1")
    arm2 = ArmSpec(arm_id="dup_arm", backend="b2", checkpoint_id="c2")
    with pytest.raises(ValueError, match="dup_arm"):
        CampaignSpec(campaign_id="c1", target_inputs=["a.pdb"], arms=[arm1, arm2])


def test_plan_campaign_rejects_duplicate_manifest_row_hash() -> None:
    arm = ArmSpec(arm_id="arm_1", backend="b1", checkpoint_id="c1")
    spec = CampaignSpec(campaign_id="c1", target_inputs=["a.pdb"], arms=[arm])
    object.__setattr__(spec, "target_inputs", ["a.pdb", "a.pdb"])
    with pytest.raises(ValueError, match="manifest_row_hash"):
        plan_campaign(spec)


def test_valid_spec_plans_with_all_unique_hashes() -> None:
    arm1 = ArmSpec(arm_id="arm_1", backend="b1", checkpoint_id="c1")
    arm2 = ArmSpec(arm_id="arm_2", backend="b1", checkpoint_id="c2")
    spec = CampaignSpec(
        campaign_id="valid_c",
        target_inputs=["a.pdb", "b.pdb"],
        arms=[arm1, arm2],
        grid=ParamGrid(axes={"lr": [0.0, 0.1]}),
        samples_per_target=100,
        chunk_size=50,
    )
    manifest = plan_campaign(spec)
    hashes = [row.manifest_row_hash for row in manifest.rows]
    assert len(hashes) == 16
    assert len(set(hashes)) == len(hashes)

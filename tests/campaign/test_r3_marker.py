"""Regression tests for R3 review findings in xtrax.campaign.marker."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from xtrax.campaign.manifest import ManifestRow, ResourceRequirements
from xtrax.campaign.marker import (
    compute_content_digest,
    create_done_marker,
    quarantine_output,
    validate_unit_done,
)


def _make_manifest_row(output_locator: str) -> ManifestRow:
    return ManifestRow(
        manifest_row_hash="r3_row_hash_001",
        input_hash="r3_input_hash_001",
        campaign_id="r3_campaign",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=1,
        target_input="target.pdb",
        arm_id="arm_test",
        arm_backend="test_backend",
        checkpoint_id="chk_01",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=1,
        parameters={"temp": 0.5},
        output_locator=output_locator,
        resource_requirements=ResourceRequirements(gpu_memory_bytes=1000),
    )


def test_compute_content_digest_and_deep_verify_include_root_seed_attr(tmp_path: Path) -> None:
    """Finding 1: compute_content_digest must include root provenance attrs such as seed."""
    store1 = tmp_path / "store1.zarr"
    store2 = tmp_path / "store2.zarr"

    for store, seed_val in ((store1, 1), (store2, 2)):
        grp = zarr.open_group(str(store), mode="w")
        grp.create_array("values", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        grp.attrs["seed"] = seed_val

    digest1 = compute_content_digest(store1)
    digest2 = compute_content_digest(store2)
    assert digest1 != digest2

    # Commit store1 with its done marker
    row = _make_manifest_row(str(store1))
    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(store1),
        content_digest_sha256=digest1,
        duration_s=0.5,
    )
    marker.write(store1.with_name(f"{store1.name}.done.json"))

    assert validate_unit_done(store1, expected_row=row, deep_verify=True).is_valid is True

    # Corrupt root attr 'seed' of committed store1
    grp1 = zarr.open_group(str(store1), mode="r+")
    grp1.attrs["seed"] = 2

    res = validate_unit_done(store1, expected_row=row, deep_verify=True)
    assert res.is_valid is False
    assert res.error_type == "ContentDigestMismatchError"


def test_quarantine_output_leaves_sibling_and_parent_decoy_files_untouched(
    tmp_path: Path,
) -> None:
    """Finding 2: quarantine_output only touches output_path and <output_path>.done.json."""
    campaign_dir = tmp_path / "campaign_dir"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    unit_store = campaign_dir / "unit1.zarr"
    unit_store.mkdir()
    (unit_store / "zarr.json").write_text('{"zarr_format": 3}', encoding="utf-8")
    unit_done = campaign_dir / "unit1.zarr.done.json"
    unit_done.write_text('{"schema_version": "campaign_done_marker_v3"}', encoding="utf-8")

    # Decoy files in campaign_dir and parent dir
    decoy_stem_same_dir = campaign_dir / "unit1.done.json"
    decoy_stem_parent_dir = tmp_path / "unit1.done.json"
    decoy_zarr_done_parent_dir = tmp_path / "unit1.zarr.done.json"
    decoy_other_unit_done = campaign_dir / "unit2.zarr.done.json"
    decoy_files = [
        decoy_stem_same_dir,
        decoy_stem_parent_dir,
        decoy_zarr_done_parent_dir,
        decoy_other_unit_done,
    ]
    for d in decoy_files:
        d.write_text("decoy", encoding="utf-8")

    q_path = quarantine_output(unit_store, reason="corrupt")
    assert q_path.exists()
    assert not unit_store.exists()
    assert not unit_done.exists()

    for d in decoy_files:
        assert d.exists(), f"Decoy file {d} was touched by quarantine_output"
        assert d.read_text(encoding="utf-8") == "decoy"


def test_quarantine_output_enospc_and_exdev_leave_decoys_untouched(tmp_path: Path) -> None:
    """Finding 2: ENOSPC and EXDEV fallback branches also leave decoy files untouched."""
    campaign_dir = tmp_path / "campaign_dir"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    decoy_stem_same_dir = campaign_dir / "unit1.done.json"
    decoy_stem_parent_dir = tmp_path / "unit1.done.json"
    decoy_stem_same_dir.write_text("decoy_same", encoding="utf-8")
    decoy_stem_parent_dir.write_text("decoy_parent", encoding="utf-8")

    # ENOSPC branch
    unit_store = campaign_dir / "unit1.zarr"
    unit_store.mkdir()
    (unit_store / "zarr.json").write_text('{"zarr_format": 3}', encoding="utf-8")
    unit_done = campaign_dir / "unit1.zarr.done.json"
    unit_done.write_text("{}", encoding="utf-8")

    with patch.object(Path, "mkdir", side_effect=OSError(errno.ENOSPC, "No space left")):
        quarantine_output(unit_store, reason="enospc_check")

    assert decoy_stem_same_dir.exists()
    assert decoy_stem_parent_dir.exists()

    # EXDEV branch
    unit_store.mkdir()
    (unit_store / "zarr.json").write_text('{"zarr_format": 3}', encoding="utf-8")
    unit_done.write_text("{}", encoding="utf-8")
    orig_replace = os.replace

    def simulated_replace(src: Path | str, dst: Path | str) -> None:
        if ".quarantine" in str(dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        orig_replace(src, dst)

    with patch("os.replace", side_effect=simulated_replace):
        quarantine_output(unit_store, reason="exdev_check")

    assert decoy_stem_same_dir.exists()
    assert decoy_stem_parent_dir.exists()

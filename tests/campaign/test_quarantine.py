"""Tests for two-tier verified resume and defensive quarantine.

Covering [T2-05], GATE-04, GATE-05, GATE-06, and AC-06 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from xtrax.campaign.manifest import ManifestRow, ResourceRequirements
from xtrax.campaign.marker import (
    CampaignDoneMarker,
    ProducerInfo,
    quarantine_output,
    validate_unit_done,
)


def _create_test_zarr_store(
    store_path: Path, array_data: bytes = b"initial_chunk_bytes_0123456789"
) -> None:
    """Create a minimal valid Zarr directory store with root group and an array chunk."""
    store_path.mkdir(parents=True, exist_ok=True)
    # .zgroup
    (store_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    # array subdirectory
    arr_dir = store_path / "data"
    arr_dir.mkdir(parents=True, exist_ok=True)
    (arr_dir / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [len(array_data)],
                "chunks": [len(array_data)],
                "dtype": "|u1",
                "order": "C",
                "filters": None,
                "fill_value": 0,
                "compressor": None,
            }
        ),
        encoding="utf-8",
    )
    # chunk 0
    (arr_dir / "0").write_bytes(array_data)


def _make_manifest_row(
    row_hash: str = "row_hash_q001",
    input_hash: str = "input_hash_q001",
    output_locator: str = "unit_q001.zarr",
) -> ManifestRow:
    return ManifestRow(
        manifest_row_hash=row_hash,
        input_hash=input_hash,
        campaign_id="test_campaign",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=100,
        target_input="target.pdb",
        arm_id="arm_test",
        arm_backend="test_backend",
        checkpoint_id="chk_01",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=42,
        parameters={"temp": 0.5},
        output_locator=output_locator,
        resource_requirements=ResourceRequirements(gpu_memory_bytes=1000),
    )


def test_gate_04_tier1_fast_resume_under_5ms(tmp_path: Path):
    """GATE-04 Positive Control: Fast Resume checks done marker & Zarr metadata in < 5 ms."""
    store_path = tmp_path / "unit_gate04.zarr"
    _create_test_zarr_store(store_path)

    row = _make_manifest_row(output_locator=str(store_path))
    content_digest = "dummy_fast_resume_digest_04"

    marker = CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        producer=ProducerInfo(version="xtrax@0.5.0", git_sha="clean", hostname="localhost"),
        output_locator=str(store_path),
        content_digest_sha256=content_digest,
        completed_at_utc="2026-11-05T19:00:00Z",
        completed_at_unix_s=time.time(),
        duration_s=1.0,
    )
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    done_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")

    # Time the Tier-1 validation
    t0 = time.perf_counter()
    res = validate_unit_done(store_path, expected_row=row, deep_verify=False)
    elapsed_s = time.perf_counter() - t0

    assert res.is_valid is True
    assert res.tier == 1
    assert res.manifest_row_hash == row.manifest_row_hash
    # GATE-04: Must execute strictly under 5 ms (0.005 s)
    msg = f"Tier-1 Fast Resume took {elapsed_s * 1000:.3f} ms, exceeding 5 ms ceiling!"
    assert elapsed_s < 0.005, msg


def test_gate_05_tampered_chunk_relocated_to_quarantine_under_tier2(tmp_path: Path):
    """GATE-05 Negative Control: Tampered chunk triggers relocation to .quarantine/."""
    from xtrax.campaign.marker import compute_content_digest

    store_path = tmp_path / "unit_gate05.zarr"
    _create_test_zarr_store(store_path, array_data=b"original_tensor_bytes")

    row = _make_manifest_row(output_locator=str(store_path))
    original_digest = compute_content_digest(store_path)

    marker = CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        producer=ProducerInfo(version="xtrax@0.5.0", git_sha="clean", hostname="localhost"),
        output_locator=str(store_path),
        content_digest_sha256=original_digest,
        completed_at_utc="2026-11-05T19:00:00Z",
        completed_at_unix_s=time.time(),
        duration_s=1.0,
    )
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    done_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")

    # Simulate bit rot / tampering by modifying 1 byte in the chunk file
    chunk_path = store_path / "data" / "0"
    tampered_bytes = b"Xriginal_tensor_bytes"
    chunk_path.write_bytes(tampered_bytes)

    # Under Tier-1 (default), chunk tampering is NOT checked
    tier1_res = validate_unit_done(
        store_path, expected_row=row, deep_verify=False, quarantine_on_failure=False
    )
    assert tier1_res.is_valid is True

    # Under Tier-2 (deep_verify=True), content digest mismatch is detected
    tier2_res = validate_unit_done(
        store_path, expected_row=row, deep_verify=True, quarantine_on_failure=True
    )

    assert tier2_res.is_valid is False
    assert tier2_res.tier == 2
    assert "digest mismatch" in (tier2_res.error_message or "").lower()
    assert tier2_res.quarantine_path is not None
    assert tier2_res.quarantine_path.exists()
    assert ".quarantine" in str(tier2_res.quarantine_path)

    # Verify output store and done marker were moved out of their original location
    assert not store_path.exists()
    assert not done_path.exists()

    # Verify quarantine_manifest.json was written with forensic details
    q_manifest_file = tier2_res.quarantine_path / "quarantine_manifest.json"
    assert q_manifest_file.exists()
    q_manifest = json.loads(q_manifest_file.read_text(encoding="utf-8"))
    assert q_manifest["original_path"] == str(store_path)
    assert "digest" in q_manifest["reason"].lower()


def test_gate_06_quarantine_enospc_inplace_fallback(tmp_path: Path):
    """GATE-06 Edge Case Control: in-place rename to .corrupt.<uuid> on simulated ENOSPC."""
    store_path = tmp_path / "unit_gate06.zarr"
    _create_test_zarr_store(store_path)
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    done_path.write_text('{"schema_version": "campaign_done_marker_v3"}', encoding="utf-8")

    # Simulate ENOSPC on Path.mkdir
    with patch.object(Path, "mkdir", side_effect=OSError(errno.ENOSPC, "No space left on device")):
        quarantine_dir = quarantine_output(
            output_path=store_path,
            reason="simulated_enospc_test",
            exc=OSError(errno.ENOSPC, "Disk full"),
        )

    # In-place fallback was triggered
    assert ".corrupt." in str(quarantine_dir)
    assert quarantine_dir.exists()
    # The original store path should no longer exist under original name
    assert not store_path.exists()
    # The done marker should also have been renamed in-place
    assert not done_path.exists()
    # Look for corrupt done marker
    corrupt_markers = list(tmp_path.glob("unit_gate06.zarr.done.json.corrupt.*"))
    assert len(corrupt_markers) == 1
    assert corrupt_markers[0].exists()


def test_quarantine_exdev_cross_device_fallback(tmp_path: Path):
    """Verify quarantine_output falls back to shutil.move on EXDEV (cross-device link)."""
    store_path = tmp_path / "unit_exdev.zarr"
    _create_test_zarr_store(store_path)
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    done_path.write_text('{"schema_version": "campaign_done_marker_v3"}', encoding="utf-8")

    orig_replace = os.replace

    def simulated_replace(src, dst):
        if ".quarantine" in str(dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return orig_replace(src, dst)

    with patch("os.replace", side_effect=simulated_replace):
        quarantined = quarantine_output(
            output_path=store_path,
            reason="simulated_exdev_test",
        )

    assert quarantined.exists()
    assert ".quarantine" in str(quarantined)
    assert not store_path.exists()
    assert not done_path.exists()


def test_tier1_failure_modes(tmp_path: Path):
    """Verify Tier-1 correctly flags and quarantines various corruptions."""
    store_path = tmp_path / "unit_failures.zarr"
    _create_test_zarr_store(store_path)
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    row = _make_manifest_row(output_locator=str(store_path))

    # Case 1: Corrupt JSON in .done.json
    done_path.write_text("NOT_VALID_JSON{", encoding="utf-8")
    res1 = validate_unit_done(store_path, expected_row=row, quarantine_on_failure=True)
    assert res1.is_valid is False
    assert res1.error_type == "DoneMarkerCorruptError"

    # Reset store
    _create_test_zarr_store(store_path)

    # Case 2: Outdated schema version v2
    v2_marker = {
        "schema_version": "campaign_done_marker_v2",
        "manifest_row_hash": row.manifest_row_hash,
        "input_hash": row.input_hash,
    }
    done_path.write_text(json.dumps(v2_marker), encoding="utf-8")
    res2 = validate_unit_done(store_path, expected_row=row, quarantine_on_failure=True)
    assert res2.is_valid is False
    assert res2.error_type == "DoneMarkerVersionError"

    # Reset store
    _create_test_zarr_store(store_path)

    # Case 3: Input hash mismatch (parameter drift F-09)
    v3_marker_mismatch = {
        "schema_version": "campaign_done_marker_v3",
        "manifest_row_hash": row.manifest_row_hash,
        "input_hash": "different_hash_from_old_run",
        "producer": {"version": "xtrax@0.5.0", "git_sha": "clean", "hostname": "localhost"},
        "output_locator": str(store_path),
        "content_digest_sha256": "abc",
        "completed_at_utc": "2026-11-05T19:00:00Z",
        "completed_at_unix_s": time.time(),
        "duration_s": 1.0,
    }
    done_path.write_text(json.dumps(v3_marker_mismatch), encoding="utf-8")
    res3 = validate_unit_done(store_path, expected_row=row, quarantine_on_failure=True)
    assert res3.is_valid is False
    assert res3.error_type == "InputHashMismatchError"

    # Reset store
    _create_test_zarr_store(store_path)

    # Case 4: Missing store on disk
    shutil.rmtree(store_path)
    v3_marker_valid = {
        "schema_version": "campaign_done_marker_v3",
        "manifest_row_hash": row.manifest_row_hash,
        "input_hash": row.input_hash,
        "producer": {"version": "xtrax@0.5.0", "git_sha": "clean", "hostname": "localhost"},
        "output_locator": str(store_path),
        "content_digest_sha256": "abc",
        "completed_at_utc": "2026-11-05T19:00:00Z",
        "completed_at_unix_s": time.time(),
        "duration_s": 1.0,
    }
    done_path.write_text(json.dumps(v3_marker_valid), encoding="utf-8")
    res4 = validate_unit_done(store_path, expected_row=row, quarantine_on_failure=True)
    assert res4.is_valid is False
    assert res4.error_type == "StoreMissingError"

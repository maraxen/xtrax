"""Tests for packed unit support in xtrax.campaign.marker.

Verifies that compute_content_digest and validate_unit_done seamlessly support
packed units (members.tar.zst + unit.json) without GroupNotFoundError or spurious
quarantine, resolving Track 2 Debt #3257 / #3142 seam collision.
"""

from __future__ import annotations

import json
from pathlib import Path

from xtrax.campaign.manifest import ManifestRow, ResourceRequirements
from xtrax.campaign.marker import (
    compute_content_digest,
    create_done_marker,
    validate_unit_done,
)
from xtrax.pack import (
    UnitPackRefused,
    pack_unit,
    unit_content_digest,
    write_unit_record,
)


def _make_row(output_locator: str) -> ManifestRow:
    return ManifestRow(
        manifest_row_hash="row_hash_packed_001",
        input_hash="input_hash_packed_001",
        campaign_id="camp_packed",
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
        seed=42,
        parameters={"alpha": 1.0},
        output_locator=output_locator,
        resource_requirements=ResourceRequirements(gpu_memory_bytes=1000),
    )


def _populate_sample_unit(unit_dir: Path) -> None:
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "summary.json").write_text(json.dumps({"score": 42.0}), encoding="utf-8")
    (unit_dir / "data.bin").write_bytes(b"\x00\x01\x02\x03" * 256)


def test_compute_content_digest_delegates_to_pack_unit_content_digest(tmp_path: Path):
    """compute_content_digest should delegate to unit_content_digest when packed."""
    unit_dir = tmp_path / "unit_01"
    _populate_sample_unit(unit_dir)

    write_unit_record(unit_dir, input_digest="in_01", run_id="run_01")
    expected_digest = unit_content_digest(unit_dir)

    # Now pack the unit
    pack_unit(unit_dir)
    assert (unit_dir / "members.tar.zst").exists()

    # compute_content_digest must succeed and return the uncompressed member digest
    computed = compute_content_digest(unit_dir)
    assert computed == expected_digest


def test_validate_unit_done_packed_unit_tier1_success(tmp_path: Path):
    """Tier-1 validation of packed unit must succeed without demanding loose zarr metadata."""
    unit_dir = tmp_path / "unit_02"
    _populate_sample_unit(unit_dir)
    row = _make_row(str(unit_dir))

    write_unit_record(unit_dir, input_digest=row.input_hash, run_id="run_02")
    pack_unit(unit_dir)

    digest = compute_content_digest(unit_dir)
    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(unit_dir),
        content_digest_sha256=digest,
        duration_s=2.5,
    )
    marker_path = unit_dir.with_name(f"{unit_dir.name}.done.json")
    marker.write(marker_path)

    res = validate_unit_done(unit_dir, row, deep_verify=False)
    assert res.is_valid is True
    assert res.tier == 1


def test_validate_unit_done_packed_unit_deep_verify_success(tmp_path: Path):
    """Tier-2 deep validation of packed unit must verify member digest."""
    unit_dir = tmp_path / "unit_03"
    _populate_sample_unit(unit_dir)
    row = _make_row(str(unit_dir))

    write_unit_record(unit_dir, input_digest=row.input_hash, run_id="run_03")
    pack_unit(unit_dir)

    digest = compute_content_digest(unit_dir)
    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(unit_dir),
        content_digest_sha256=digest,
        duration_s=3.0,
    )
    marker.write(unit_dir.with_name(f"{unit_dir.name}.done.json"))

    res = validate_unit_done(unit_dir, row, deep_verify=True)
    assert res.is_valid is True
    assert res.tier == 2


def test_packed_unit_deep_verify_detects_bitflip(tmp_path: Path):
    """C-NEG-3: Injected bitflip into members.tar.zst must fail deep verification."""
    unit_dir = tmp_path / "unit_04"
    _populate_sample_unit(unit_dir)
    row = _make_row(str(unit_dir))

    write_unit_record(unit_dir, input_digest=row.input_hash, run_id="run_04")
    pack_unit(unit_dir)

    digest = compute_content_digest(unit_dir)
    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(unit_dir),
        content_digest_sha256=digest,
        duration_s=1.0,
    )
    marker.write(unit_dir.with_name(f"{unit_dir.name}.done.json"))

    # Mutate a byte in members.tar.zst inside the first frame
    tar_path = unit_dir / "members.tar.zst"
    data = bytearray(tar_path.read_bytes())
    # Flip a byte within the first member's frame payload
    data[10] ^= 0xFF
    tar_path.write_bytes(bytes(data))

    # deep_verify=True should catch corruption (either via UnitPackRefused or ContentDigestMismatch)
    try:
        res = validate_unit_done(unit_dir, row, deep_verify=True)
        assert res.is_valid is False
    except UnitPackRefused:
        # Raising UnitPackRefused is also a valid refusal of corrupted archive
        pass


def test_validate_unit_done_packed_unit_missing_pack_file(tmp_path: Path):
    """Unit specifying pack storage without members.tar.zst fails Tier-1."""
    unit_dir = tmp_path / "unit_05"
    _populate_sample_unit(unit_dir)
    row = _make_row(str(unit_dir))

    write_unit_record(unit_dir, input_digest=row.input_hash, run_id="run_05")
    pack_unit(unit_dir)

    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(unit_dir),
        content_digest_sha256="dummy",
        duration_s=1.0,
    )
    marker.write(unit_dir.with_name(f"{unit_dir.name}.done.json"))

    # Delete members.tar.zst
    (unit_dir / "members.tar.zst").unlink()

    res = validate_unit_done(unit_dir, row, deep_verify=False)
    assert res.is_valid is False
    assert res.error_type == "StoreMissingError"


def test_validate_unit_done_packed_unit_corrupt_unit_record(tmp_path: Path):
    """Tampered unit.json record_digest fails Tier-1."""
    unit_dir = tmp_path / "unit_06"
    _populate_sample_unit(unit_dir)
    row = _make_row(str(unit_dir))

    write_unit_record(unit_dir, input_digest=row.input_hash, run_id="run_06")
    pack_unit(unit_dir)

    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(unit_dir),
        content_digest_sha256="dummy",
        duration_s=1.0,
    )
    marker.write(unit_dir.with_name(f"{unit_dir.name}.done.json"))

    # Tamper with unit.json
    unit_record_path = unit_dir / "unit.json"
    rec = json.loads(unit_record_path.read_text(encoding="utf-8"))
    rec["record_digest"] = "bad" * 16
    unit_record_path.write_text(json.dumps(rec), encoding="utf-8")

    res = validate_unit_done(unit_dir, row, deep_verify=False)
    assert res.is_valid is False
    assert res.error_type == "DoneMarkerCorruptError"

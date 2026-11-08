"""Tests for done marker schema, validation, and adapter consolidation.

Covering [T2-04] and Section 3.4 of SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xtrax.campaign.marker import (
    AlynxrFragmentAdapter,
    AminxDoneMarkerAdapter,
    CampaignDoneMarker,
    DoneMarkerAdapter,
    ProducerInfo,
    create_done_marker,
)


def test_producer_info_and_campaign_done_marker_schema():
    """Verify ProducerInfo and CampaignDoneMarker fields, immutability, and serialization."""
    producer = ProducerInfo(
        version="xtrax@0.5.0",
        git_sha="28a87130",
        hostname="node104.engaging.mit.edu",
    )
    assert producer.version == "xtrax@0.5.0"
    assert producer.git_sha == "28a87130"
    assert producer.hostname == "node104.engaging.mit.edu"

    marker = CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        input_hash="a17f93b2c8e4d019f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3",
        producer=producer,
        output_locator="outputs/glp1r/unit_001.zarr",
        content_digest_sha256="8c4d2e1a9f0b3c5e7d1a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c",
        completed_at_utc="2026-11-05T18:15:32Z",
        completed_at_unix_s=1730830532.15,
        duration_s=532.15,
    )
    expected_row_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert marker.schema_version == "campaign_done_marker_v3"
    assert marker.manifest_row_hash == expected_row_hash
    assert marker.duration_s == 532.15

    # Frozen dataclass check
    with pytest.raises((AttributeError, TypeError)):
        marker.duration_s = 600.0  # type: ignore

    # Serialization roundtrip
    d = marker.to_dict()
    assert d["schema_version"] == "campaign_done_marker_v3"
    assert d["producer"]["hostname"] == "node104.engaging.mit.edu"
    restored = CampaignDoneMarker.from_dict(d)
    assert restored == marker


def test_create_done_marker_helper(tmp_path: Path):
    """Verify create_done_marker generates a complete marker with host and duration."""
    out = tmp_path / "unit_helper.zarr"
    marker = create_done_marker(
        manifest_row_hash="row_hash_h01",
        input_hash="input_hash_h01",
        output_locator=str(out),
        content_digest_sha256="digest_h01",
        duration_s=12.5,
    )
    assert marker.schema_version == "campaign_done_marker_v3"
    assert marker.manifest_row_hash == "row_hash_h01"
    assert marker.input_hash == "input_hash_h01"
    assert marker.content_digest_sha256 == "digest_h01"
    assert marker.duration_s == 12.5
    assert len(marker.producer.hostname) > 0
    assert len(marker.completed_at_utc) > 0


def test_aminx_done_marker_adapter_roundtrip():
    """Verify AminxDoneMarkerAdapter translates aminx v3 markers to and from marker."""
    adapter = AminxDoneMarkerAdapter()
    assert isinstance(adapter, DoneMarkerAdapter)

    aminx_raw = {
        "schema_version": "campaign_done_marker_v3",
        "manifest_row_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "input_hash": "a17f93b2c8e4d019f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3",
        "producer": {
            "version": "aminx@0.4.0",
            "git_sha": "a1b2c3d4",
            "hostname": "node42.engaging.mit.edu",
        },
        "output_locator": "campaign_run/e3b0c442.zarr",
        "content_digest_sha256": "8c4d2e1a9f0b3c5e7d1a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c",
        "completed_at_utc": "2026-11-05T18:15:32Z",
        "completed_at_unix_s": 1730830532.15,
        "duration_s": 532.15,
    }

    marker = adapter.to_done_marker(aminx_raw)
    assert marker.schema_version == "campaign_done_marker_v3"
    assert marker.producer.version == "aminx@0.4.0"
    assert marker.duration_s == 532.15

    exported = adapter.from_done_marker(marker)
    assert exported == aminx_raw


def test_alynxr_fragment_adapter_translation():
    """Verify AlynxrFragmentAdapter translates alynxr records to and from marker."""
    adapter = AlynxrFragmentAdapter()
    assert isinstance(adapter, DoneMarkerAdapter)

    # Simulated record from alynxr fold_stats / fold campaign store
    alynxr_record = {
        "prediction_id": "pred_001_boltz2",
        "arm_key": "boltz2_standard",
        "manifest_row_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "input_hash": "a17f93b2c8e4d019f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3",
        "output_locator": "fold_campaigns/arms/boltz2/pred_001.zarr",
        "content_digest_sha256": "8c4d2e1a9f0b3c5e7d1a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c",
        "completed_at_utc": "2026-11-05T18:15:32Z",
        "completed_at_unix_s": 1730830532.15,
        "duration_s": 120.0,
        "producer_version": "alynxr@0.3.2",
        "producer_hostname": "titanix01",
    }

    marker = adapter.to_done_marker(alynxr_record)
    assert marker.schema_version == "campaign_done_marker_v3"
    expected_row_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert marker.manifest_row_hash == expected_row_hash
    assert marker.producer.version == "alynxr@0.3.2"
    assert marker.producer.hostname == "titanix01"

    # Export back
    exported = adapter.from_done_marker(marker)
    assert exported["manifest_row_hash"] == marker.manifest_row_hash
    assert exported["input_hash"] == marker.input_hash
    assert exported["output_locator"] == marker.output_locator
    assert exported["content_digest_sha256"] == marker.content_digest_sha256

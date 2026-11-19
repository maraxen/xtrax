"""Unit tests for Campaign EDA coverage reconciliation and run telemetry (Q4, Q5)."""

from xtrax.campaign.manifest import CampaignManifest, ManifestMetadata
from xtrax.eda.campaign.coverage import aggregate_visual_cell_status
from xtrax.eda.campaign.telemetry import PACK_TELEMETRY_BANNER, extract_campaign_telemetry
from xtrax.eda.campaign.types import CellStatus, ReconciliationRecord


def test_visual_cell_aggregation_rollup_rule():
    """Verify 100% committed visual cell aggregation rollup rule (MAJ-04)."""
    # 1. 100% committed -> COMMITTED
    assert (
        aggregate_visual_cell_status([CellStatus.COMMITTED, CellStatus.COMMITTED])
        == CellStatus.COMMITTED
    )

    # 2. 1 partial, 1 committed -> PARTIAL
    assert (
        aggregate_visual_cell_status([CellStatus.COMMITTED, CellStatus.PARTIAL])
        == CellStatus.PARTIAL
    )

    # 3. 1 failed -> FAILED immediately
    assert (
        aggregate_visual_cell_status([CellStatus.COMMITTED, CellStatus.FAILED]) == CellStatus.FAILED
    )
    assert (
        aggregate_visual_cell_status([CellStatus.PARTIAL, CellStatus.FAILED]) == CellStatus.FAILED
    )

    # 4. 100% missing -> MISSING
    assert (
        aggregate_visual_cell_status([CellStatus.MISSING, CellStatus.MISSING]) == CellStatus.MISSING
    )

    # 5. Empty -> MISSING
    assert aggregate_visual_cell_status([]) == CellStatus.MISSING


def test_pack_telemetry_banner_debt_3300():
    """Verify explicit pack telemetry seam banner (closing debt #3300)."""
    assert PACK_TELEMETRY_BANNER == "Pack Telemetry: not instrumented (debt #3300)"

    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=ManifestMetadata(
            campaign_id="test_camp",
            created_at_utc="2026-11-06T20:30:00Z",
            planner_version="1.0",
            spec_sha256="abc",
            total_rows=1,
        ),
        rows=[],
    )
    records = [
        ReconciliationRecord(
            manifest_row_hash="hash1",
            input_hash="hash2",
            grid_coords={},
            status=CellStatus.COMMITTED,
            duration_s=1.5,
            is_packed=True,
        )
    ]

    telemetry = extract_campaign_telemetry(manifest, records)
    assert telemetry["pack_telemetry_banner"] == "Pack Telemetry: not instrumented (debt #3300)"
    assert telemetry["pack_telemetry_status"] == "uninstrumented"
    assert telemetry["packed_units"] == 1
    assert telemetry["total_wall_seconds"] == 1.5


def test_pack_telemetry_aggregation_states():
    """Verify pack telemetry deterministic 4-state transitions and aggregates (spec §4.2)."""
    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=ManifestMetadata(
            campaign_id="test_camp",
            created_at_utc="2026-11-06T20:30:00Z",
            planner_version="1.0",
            spec_sha256="abc",
            total_rows=2,
        ),
        rows=[],
    )

    # 1. No packed units
    loose_records = [
        ReconciliationRecord(
            manifest_row_hash="h1",
            input_hash="in1",
            grid_coords={},
            status=CellStatus.COMMITTED,
            duration_s=1.0,
            is_packed=False,
        )
    ]
    telem_no_pack = extract_campaign_telemetry(manifest, loose_records)
    assert telem_no_pack["pack_telemetry_status"] == "no_packed_units"
    assert "no packed units" in telem_no_pack["pack_telemetry_banner"]

    # 2. Instrumented units
    inst_records = [
        ReconciliationRecord(
            manifest_row_hash="h1",
            input_hash="in1",
            grid_coords={},
            status=CellStatus.COMMITTED,
            duration_s=1.0,
            is_packed=True,
            pack_telemetry={
                "schema": "xtrax.pack_telemetry/1",
                "pack_wall_ms": 10.5,
                "raw_bytes": 1000,
                "compressed_bytes": 250,
                "compression_ratio": 4.0,
                "member_count": 2,
                "backend": "zstandard",
            },
        ),
        ReconciliationRecord(
            manifest_row_hash="h2",
            input_hash="in2",
            grid_coords={},
            status=CellStatus.COMMITTED,
            duration_s=2.0,
            is_packed=True,
            pack_telemetry={
                "schema": "xtrax.pack_telemetry/1",
                "pack_wall_ms": 15.5,
                "raw_bytes": 2000,
                "compressed_bytes": 500,
                "compression_ratio": 4.0,
                "member_count": 3,
                "backend": "zstandard",
            },
        ),
    ]
    telem_inst = extract_campaign_telemetry(manifest, inst_records)
    assert telem_inst["pack_telemetry_status"] == "instrumented"
    assert (
        "Pack Telemetry: instrumented (2 units, 4.00x compression"
        in telem_inst["pack_telemetry_banner"]
    )
    assert telem_inst["pack_telemetry"]["total_raw_bytes"] == 3000
    assert telem_inst["pack_telemetry"]["total_compressed_bytes"] == 750
    assert telem_inst["pack_telemetry"]["overall_compression_ratio"] == 4.0
    assert telem_inst["pack_telemetry"]["mean_pack_wall_ms"] == 13.0
    assert telem_inst["pack_telemetry"]["backends"] == ["zstandard"]

    # 3. Mixed units
    mixed_records = [
        inst_records[0],
        ReconciliationRecord(
            manifest_row_hash="h3",
            input_hash="in3",
            grid_coords={},
            status=CellStatus.COMMITTED,
            duration_s=1.2,
            is_packed=True,
            pack_telemetry=None,
        ),
    ]
    telem_mixed = extract_campaign_telemetry(manifest, mixed_records)
    assert telem_mixed["pack_telemetry_status"] == "mixed"
    assert (
        telem_mixed["pack_telemetry_banner"]
        == "Pack Telemetry: mixed (1 of 2 units instrumented, debt #3300)"
    )

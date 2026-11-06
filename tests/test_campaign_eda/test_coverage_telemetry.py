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
    assert telemetry["packed_units"] == 1
    assert telemetry["total_wall_seconds"] == 1.5

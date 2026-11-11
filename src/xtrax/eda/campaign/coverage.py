"""Coverage reconciliation and sweep grid status analytics (Question 4).

Implements Q4 coverage analytics per approved specification 261106_campaign-eda-redesign.md.
Differentiates COMMITTED, PARTIAL, FAILED (killed_incomplete), and MISSING cells, enforcing
the 100% committed visual aggregation rollup rule (MAJ-04) without silent omission.
"""

import logging
from collections.abc import Sequence
from typing import Any

from xtrax.campaign.manifest import CampaignManifest
from xtrax.eda.campaign.types import CellStatus, ReconciliationRecord

logger = logging.getLogger("xtrax.eda.campaign.coverage")


def aggregate_visual_cell_status(constituent_statuses: Sequence[CellStatus]) -> CellStatus:
    """Cartesian grid visual cell aggregation rollup rule (MAJ-04).

    - COMMITTED iff 100% of constituent rows are COMMITTED.
    - PARTIAL if at least one row is PARTIAL and no row is FAILED.
    - FAILED if any constituent row is FAILED or quarantined.
    - MISSING if 100% of constituent rows are MISSING.
    """
    if not constituent_statuses:
        return CellStatus.MISSING

    statuses_set = set(constituent_statuses)

    if CellStatus.FAILED in statuses_set:
        return CellStatus.FAILED

    if CellStatus.PARTIAL in statuses_set:
        return CellStatus.PARTIAL

    if all(s == CellStatus.COMMITTED for s in constituent_statuses):
        return CellStatus.COMMITTED

    if all(s == CellStatus.MISSING for s in constituent_statuses):
        return CellStatus.MISSING

    # Mixed missing / committed without failure
    return CellStatus.PARTIAL


def reconcile_sweep_grid(
    manifest: CampaignManifest,
    reconciliation_records: Sequence[ReconciliationRecord],
    grid_axes: Sequence[str] = ("arm_id", "temperature"),
) -> dict[str, Any]:
    """Reconcile full planned Cartesian sweep grid against persistent storage.

    Returns:
        Structured coverage analysis dictionary with visual cells, row reconciliation,
        rollup statuses, completion percentages, and failure diagnostic banners.
    """
    rec_by_hash: dict[str, ReconciliationRecord] = {
        r.manifest_row_hash: r for r in reconciliation_records
    }

    # Group planned rows by visual grid axes
    visual_grid_cells: dict[str, dict[str, Any]] = {}
    banners: list[str] = []

    for row in manifest.rows:
        # Determine cell key from grid axes
        cell_coords: dict[str, Any] = {}
        for axis in grid_axes:
            if hasattr(row, axis):
                cell_coords[axis] = getattr(row, axis)
            elif axis in row.parameters:
                cell_coords[axis] = row.parameters[axis]
            else:
                cell_coords[axis] = "default"

        coord_key = "-".join(f"{k}:{v}" for k, v in sorted(cell_coords.items()))

        if coord_key not in visual_grid_cells:
            visual_grid_cells[coord_key] = {
                "coords": cell_coords,
                "row_hashes": [],
                "statuses": [],
                "failure_reasons": [],
            }

        rec = rec_by_hash.get(row.manifest_row_hash)
        status = rec.status if rec else CellStatus.MISSING
        visual_grid_cells[coord_key]["row_hashes"].append(row.manifest_row_hash)
        visual_grid_cells[coord_key]["statuses"].append(status)
        if rec and rec.failure_reason:
            visual_grid_cells[coord_key]["failure_reasons"].append(rec.failure_reason)
            banners.append(f"Row {row.manifest_row_hash[:12]} ({coord_key}): {rec.failure_reason}")

    # Evaluate rollup status for each visual grid cell
    cells_summary: list[dict[str, Any]] = []
    committed_cell_count = 0
    total_cells = len(visual_grid_cells)

    for coord_key, cell_data in sorted(visual_grid_cells.items()):
        statuses = cell_data["statuses"]
        rollup_status = aggregate_visual_cell_status(statuses)
        if rollup_status == CellStatus.COMMITTED:
            committed_cell_count += 1

        n_committed = sum(1 for s in statuses if s == CellStatus.COMMITTED)
        n_total = len(statuses)
        pct_committed = float(n_committed / float(n_total)) if n_total > 0 else 0.0

        cells_summary.append(
            {
                "cell_key": coord_key,
                "coords": cell_data["coords"],
                "status": rollup_status.value,
                "total_rows": n_total,
                "committed_rows": n_committed,
                "pct_committed": pct_committed,
                "failure_reasons": cell_data["failure_reasons"],
            }
        )

    total_rows = len(manifest.rows)
    committed_rows = sum(1 for r in reconciliation_records if r.status == CellStatus.COMMITTED)
    partial_rows = sum(1 for r in reconciliation_records if r.status == CellStatus.PARTIAL)
    failed_rows = sum(1 for r in reconciliation_records if r.status == CellStatus.FAILED)
    missing_rows = sum(1 for r in reconciliation_records if r.status == CellStatus.MISSING)

    return {
        "total_planned_rows": total_rows,
        "committed_rows": committed_rows,
        "partial_rows": partial_rows,
        "failed_rows": failed_rows,
        "missing_rows": missing_rows,
        "total_visual_cells": total_cells,
        "committed_visual_cells": committed_cell_count,
        "is_complete": (committed_rows == total_rows and total_rows > 0),
        "cells": cells_summary,
        "banners": banners,
    }

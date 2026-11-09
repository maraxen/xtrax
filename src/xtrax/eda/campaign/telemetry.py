"""Secondary Run Telemetry view analytics (Question 5).

Implements Q5 telemetry analytics per approved specification 261106_campaign-eda-redesign.md.
Surfaces unit compute wall-time distributions, reuse log audit trails, tiling plans,
and explicitly banners uninstrumented pack telemetry seams (debt #3300).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xtrax.campaign.manifest import CampaignManifest
from xtrax.eda.campaign.types import CellStatus, ReconciliationRecord

logger = logging.getLogger("xtrax.eda.campaign.telemetry")

# Mandatory seam banner closing debt #3300
PACK_TELEMETRY_BANNER = "Pack Telemetry: not instrumented (debt #3300)"


def parse_reuse_log(reuse_log_path: Path | str | None) -> list[dict[str, Any]]:
    """Parse append-only JSONL reuse log file if present."""
    if not reuse_log_path:
        return []
    p = Path(reuse_log_path)
    if not p.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.warning("Failed to read reuse log %s: %s", reuse_log_path, exc)
    return entries


def extract_campaign_telemetry(
    manifest: CampaignManifest,
    reconciliation_records: Sequence[ReconciliationRecord],
    reuse_log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Extract and compile secondary run view telemetry across the campaign.

    Returns:
        Structured telemetry dictionary containing duration distributions, total wall time,
        reuse events, static archive statuses, and mandatory debt #3300 banner.
    """
    durations: list[float] = []
    packed_count = 0
    loose_count = 0
    unit_telemetry_list: list[dict[str, Any]] = []
    instrumented_units: list[dict[str, Any]] = []

    for r in reconciliation_records:
        if r.status == CellStatus.COMMITTED and r.duration_s is not None:
            durations.append(float(r.duration_s))

        if r.is_packed:
            packed_count += 1
            if r.pack_telemetry is not None:
                instrumented_units.append(dict(r.pack_telemetry))
        elif r.status == CellStatus.COMMITTED:
            loose_count += 1

        u_entry: dict[str, Any] = {
            "manifest_row_hash": r.manifest_row_hash,
            "status": r.status.value,
            "is_packed": r.is_packed,
            "duration_s": r.duration_s,
            "completed_at_utc": r.completed_at_utc,
            "failure_reason": r.failure_reason,
        }
        if r.pack_telemetry is not None:
            u_entry["pack_telemetry"] = dict(r.pack_telemetry)
        unit_telemetry_list.append(u_entry)

    total_wall_s = float(sum(durations))
    mean_duration = float(total_wall_s / len(durations)) if durations else 0.0
    min_duration = float(min(durations)) if durations else 0.0
    max_duration = float(max(durations)) if durations else 0.0

    reuse_entries = parse_reuse_log(reuse_log_path)
    cache_hits = len(reuse_entries)

    instrumented_count = len(instrumented_units)
    total_raw = sum(int(u.get("raw_bytes", 0)) for u in instrumented_units)
    total_compressed = sum(int(u.get("compressed_bytes", 0)) for u in instrumented_units)
    overall_ratio = round(float(total_raw / total_compressed), 4) if total_compressed > 0 else 1.0
    mean_wall_ms = (
        float(
            sum(float(u.get("pack_wall_ms", 0.0)) for u in instrumented_units) / instrumented_count
        )
        if instrumented_count > 0
        else 0.0
    )
    backends = sorted(set(str(u["backend"]) for u in instrumented_units if "backend" in u))
    backends_str = ", ".join(backends) if backends else "unknown"

    if packed_count == 0:
        pack_status = "no_packed_units"
        pack_banner = "Pack Telemetry: no packed units in campaign"
    elif instrumented_count == 0:
        pack_status = "uninstrumented"
        pack_banner = PACK_TELEMETRY_BANNER
    elif instrumented_count < packed_count:
        pack_status = "mixed"
        pack_banner = (
            f"Pack Telemetry: mixed "
            f"({instrumented_count} of {packed_count} units instrumented, debt #3300)"
        )
    else:
        pack_status = "instrumented"
        pack_banner = (
            f"Pack Telemetry: instrumented ({instrumented_count} units, "
            f"{overall_ratio:.2f}x compression, mean {mean_wall_ms:.1f}ms, {backends_str})"
        )

    return {
        "pack_telemetry_banner": pack_banner,
        "pack_telemetry_status": pack_status,
        "total_wall_seconds": total_wall_s,
        "mean_unit_duration_seconds": mean_duration,
        "min_unit_duration_seconds": min_duration,
        "max_unit_duration_seconds": max_duration,
        "units_measured": len(durations),
        "packed_units": packed_count,
        "loose_units": loose_count,
        "cache_reuse_events": cache_hits,
        "reuse_log_entries": reuse_entries,
        "units": unit_telemetry_list,
        "pack_telemetry": {
            "status": pack_status,
            "instrumented_units": instrumented_count,
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "overall_compression_ratio": overall_ratio,
            "mean_pack_wall_ms": mean_wall_ms,
            "backends": backends,
        },
    }

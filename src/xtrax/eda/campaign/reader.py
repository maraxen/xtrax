"""Storage-parity sink reader for xtrax Campaign EDA.

Implements P5-02 per approved specification 261106_campaign-eda-redesign.md (Rev 1.1).
Guarantees bit-identical reads across loose filesystem directories and compressed
packed archives (members.tar.zst) via dual-marker support (done.json and unit.json).
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xtrax.campaign.manifest import CampaignManifest, ManifestRow
from xtrax.campaign.marker import (
    validate_unit_done,
)
from xtrax.eda.campaign.types import CellStatus, ReconciliationRecord
from xtrax.pack import is_packed as _pack_is_packed
from xtrax.pack import read_unit_member
from xtrax.run.unit_index import (
    Corrupt,
    IndexSnapshot,
    IndexStale,
    Missing,
    SchemaMismatch,
    load_index,
    lookup,
)

logger = logging.getLogger("xtrax.eda.campaign.reader")


def get_marker_record_name(unit_dir: Path | str) -> str:
    """Detect whether unit directory uses done.json (runner marker) or unit.json (pack marker)."""
    p = Path(unit_dir)
    if (p / "done.json").exists():
        return "done.json"
    return "unit.json"


def read_campaign_unit_member(unit_dir: Path | str, relpath: str) -> bytes:
    """Transparently read member bytes from loose filesystem or compressed zstd archive frame.

    Inspects unit directory for either 'done.json' (campaign runner marker) or 'unit.json'
    (pack tool marker) before reading, preventing member lookup failures on swept units (BLK-07).
    """
    p = Path(unit_dir)
    record_name = get_marker_record_name(p)
    return read_unit_member(p, relpath, record_name=record_name)


def is_campaign_unit_packed(unit_dir: Path | str) -> bool:
    """Determine whether unit is stored as a packed archive (.tar.zst / .tar) or loose directory."""
    p = Path(unit_dir)
    if not p.exists() or not p.is_dir():
        return False
    record_name = get_marker_record_name(p)
    try:
        return _pack_is_packed(p, record_name=record_name)
    except Exception:
        # Fallback filesystem check
        return (p / "members.tar.zst").exists() or (p / "members.tar").exists()


def read_done_marker(unit_dir: Path | str) -> dict[str, Any] | None:
    """Read completion marker payload from unit directory (done.json or unit.json)."""
    p = Path(unit_dir)
    marker_name = get_marker_record_name(p)
    marker_file = p / marker_name
    if not marker_file.exists():
        # Check sibling marker (e.g. <unit_dir>.done.json)
        sibling_marker = p.parent / f"{p.name}.done.json"
        if sibling_marker.exists():
            marker_file = sibling_marker
        else:
            return None
    try:
        with open(marker_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse completion marker in %s: %s", unit_dir, exc)
        return None


class CampaignReader:
    """Unified reader for campaign sinks and unit execution artifacts."""

    def __init__(
        self,
        campaign_dir: Path | str,
        index_root: Path | str | None = None,
    ) -> None:
        self.campaign_dir = Path(campaign_dir).resolve()
        if index_root is not None:
            self.index_root = Path(index_root).resolve()
        else:
            self.index_root = self.campaign_dir / ".index"
        self._index_snapshot: IndexSnapshot | None = None

    def get_index_snapshot(self) -> IndexSnapshot | None:
        """Lazily load and cache unit index snapshot across campaign."""
        if self._index_snapshot is None and self.index_root.exists():
            try:
                self._index_snapshot = load_index(self.index_root)
            except Exception as exc:
                logger.warning("Failed to load unit index from %s: %s", self.index_root, exc)
                self._index_snapshot = None
        return self._index_snapshot

    def resolve_unit_directory(self, row: ManifestRow) -> Path:
        """Resolve unit storage directory path from row output locator or default convention."""
        if row.output_locator:
            loc = Path(row.output_locator)
            if loc.is_absolute():
                return loc
            return self.campaign_dir / loc
        # Default layout: units/<manifest_row_hash>
        return self.campaign_dir / "units" / row.manifest_row_hash

    def reconcile_row(
        self,
        row: ManifestRow,
        deep_verify: bool = False,
    ) -> ReconciliationRecord:
        """Reconcile persistence state of planned manifest row (MAJ-03, BLK-07)."""
        unit_dir = self.resolve_unit_directory(row)
        is_packed = is_campaign_unit_packed(unit_dir)

        # 1. Check for active staging directory or partial lock
        staging_dir = Path(str(unit_dir) + ".tmp")
        active_lock_path = unit_dir / "lease.lock"
        if staging_dir.exists():
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.PARTIAL,
                failure_reason="active_staging_directory",
                is_packed=is_packed,
            )
        if active_lock_path.exists():
            # Check lock expiry
            try:
                with open(active_lock_path, encoding="utf-8") as f:
                    lock_data = json.load(f)
                expires = float(lock_data.get("lease_expires_at_unix_s", 0))
                import time

                if expires > time.time():
                    return ReconciliationRecord(
                        manifest_row_hash=row.manifest_row_hash,
                        input_hash=row.input_hash,
                        grid_coords=dict(row.parameters),
                        status=CellStatus.PARTIAL,
                        failure_reason="active_lease_lock",
                        is_packed=is_packed,
                    )
                else:
                    # Expired lock is considered killed incomplete (MAJ-03)
                    return ReconciliationRecord(
                        manifest_row_hash=row.manifest_row_hash,
                        input_hash=row.input_hash,
                        grid_coords=dict(row.parameters),
                        status=CellStatus.FAILED,
                        failure_reason="killed_incomplete (expired lease lock)",
                        is_packed=is_packed,
                    )
            except Exception:
                pass

        # 2. Check if unit directory exists
        if not unit_dir.exists():
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.MISSING,
                failure_reason="never_executed",
                is_packed=False,
            )

        # 3. Check completion marker
        marker_data = read_done_marker(unit_dir)
        if marker_data is None:
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.FAILED,
                failure_reason="killed_incomplete (missing or corrupt completion marker)",
                is_packed=is_packed,
            )

        # Check marker status field if explicitly marked failed
        if marker_data.get("status") == "failed":
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.FAILED,
                failure_reason=marker_data.get("error", "execution_failed"),
                is_packed=is_packed,
                duration_s=marker_data.get("duration_s"),
                completed_at_utc=marker_data.get("completed_at_utc"),
            )

        # 4. Query unit index if available
        snapshot = self.get_index_snapshot()
        if snapshot is not None:
            res = lookup(
                index=snapshot,
                store=unit_dir,
                unit_key=row.manifest_row_hash,
                input_digest=row.input_hash,
                verify=deep_verify,
            )
            if isinstance(res, (Corrupt, SchemaMismatch)):
                return ReconciliationRecord(
                    manifest_row_hash=row.manifest_row_hash,
                    input_hash=row.input_hash,
                    grid_coords=dict(row.parameters),
                    status=CellStatus.FAILED,
                    failure_reason=f"index_verification_failed ({type(res).__name__})",
                    is_packed=is_packed,
                )
            elif isinstance(res, (Missing, IndexStale)):
                return ReconciliationRecord(
                    manifest_row_hash=row.manifest_row_hash,
                    input_hash=row.input_hash,
                    grid_coords=dict(row.parameters),
                    status=CellStatus.FAILED,
                    failure_reason=f"index_stale_or_missing ({type(res).__name__})",
                    is_packed=is_packed,
                )

        # 5. Validate done marker schema and digest
        sibling_marker = unit_dir.parent / f"{unit_dir.name}.done.json"
        if sibling_marker.exists():
            try:
                val_res = validate_unit_done(
                    output_path=unit_dir,
                    expected_row=row,
                    deep_verify=deep_verify,
                    quarantine_on_failure=False,
                )
                if not val_res.is_valid:
                    return ReconciliationRecord(
                        manifest_row_hash=row.manifest_row_hash,
                        input_hash=row.input_hash,
                        grid_coords=dict(row.parameters),
                        status=CellStatus.FAILED,
                        failure_reason=val_res.error_message or "marker_validation_failed",
                        is_packed=is_packed,
                    )
            except Exception as exc:
                logger.debug("validate_unit_done failed: %s", exc)

        # Validate marker contents directly
        if marker_data.get("status") not in (None, "committed", "COMMITTED"):
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.FAILED,
                failure_reason=marker_data.get(
                    "error", f"uncommitted_status ({marker_data.get('status')})"
                ),
                is_packed=is_packed,
                duration_s=marker_data.get("duration_s"),
                completed_at_utc=marker_data.get("completed_at_utc"),
            )
        m_row_hash = marker_data.get("manifest_row_hash")
        if m_row_hash and m_row_hash != row.manifest_row_hash:
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.FAILED,
                failure_reason="manifest_row_hash_mismatch",
                is_packed=is_packed,
            )
        m_input_hash = marker_data.get("input_hash")
        if m_input_hash and row.input_hash and m_input_hash != row.input_hash:
            return ReconciliationRecord(
                manifest_row_hash=row.manifest_row_hash,
                input_hash=row.input_hash,
                grid_coords=dict(row.parameters),
                status=CellStatus.FAILED,
                failure_reason="input_hash_mismatch",
                is_packed=is_packed,
            )

        # All checks passed: unit is COMMITTED
        pack_telem = marker_data.get("pack_telemetry")
        if not isinstance(pack_telem, dict):
            pack_telem = None

        return ReconciliationRecord(
            manifest_row_hash=row.manifest_row_hash,
            input_hash=row.input_hash,
            grid_coords=dict(row.parameters),
            status=CellStatus.COMMITTED,
            failure_reason=None,
            is_packed=is_packed,
            duration_s=marker_data.get("duration_s"),
            completed_at_utc=marker_data.get("completed_at_utc"),
            pack_telemetry=pack_telem,
        )

    def reconcile_campaign(
        self,
        manifest: CampaignManifest,
        deep_verify: bool = False,
    ) -> list[ReconciliationRecord]:
        """Reconcile all planned rows in campaign manifest against persistent storage."""
        return [self.reconcile_row(row, deep_verify=deep_verify) for row in manifest.rows]

    def get_committed_units(
        self,
        manifest: CampaignManifest,
        quarantined_keys: Sequence[str] = (),
    ) -> list[tuple[ManifestRow, Path, dict[str, Any]]]:
        """Return list of (row, unit_dir, marker_data) for COMMITTED units."""
        quarantined_set = set(quarantined_keys)
        committed_units = []
        for row in manifest.rows:
            if row.manifest_row_hash in quarantined_set:
                logger.info(
                    "Refusing quarantined unit %s from statistical pooling", row.manifest_row_hash
                )
                continue
            rec = self.reconcile_row(row, deep_verify=False)
            if rec.status == CellStatus.COMMITTED:
                unit_dir = self.resolve_unit_directory(row)
                marker_data = read_done_marker(unit_dir) or {}
                committed_units.append((row, unit_dir, marker_data))
            else:
                logger.info(
                    "Refusing non-committed unit %s (status=%s) from pooling",
                    row.manifest_row_hash,
                    rec.status.value,
                )
        return committed_units

    def read_unit_array(self, unit_dir: Path | str, relpath: str) -> np.ndarray:
        """Transparently read NumPy array from loose file or packed seekable archive."""
        raw_bytes = read_campaign_unit_member(unit_dir, relpath)
        return np.load(io.BytesIO(raw_bytes))

    def read_unit_json(self, unit_dir: Path | str, relpath: str) -> dict[str, Any]:
        """Transparently read JSON dictionary from loose file or packed seekable archive."""
        raw_bytes = read_campaign_unit_member(unit_dir, relpath)
        return json.loads(raw_bytes.decode("utf-8"))

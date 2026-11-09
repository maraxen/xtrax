"""Done marker schema, validation, adapters, and defensive quarantine.

Covering [T2-04], [T2-05], Section 3.4, and Section 4 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

import errno
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from xtrax.campaign.manifest import ManifestRow

#: Current canonical schema version for campaign done markers.
CAMPAIGN_DONE_MARKER_SCHEMA_VERSION = "campaign_done_marker_v3"


class CampaignMarkerError(Exception):
    """Base exception for all campaign done marker errors."""


class DoneMarkerCorruptError(CampaignMarkerError):
    """Raised when done marker JSON is corrupted or unparseable."""


class DoneMarkerVersionError(CampaignMarkerError):
    """Raised when done marker has an outdated schema version."""


class ManifestRowHashMismatchError(CampaignMarkerError):
    """Raised when manifest_row_hash in marker does not match expected row."""


class InputHashMismatchError(CampaignMarkerError):
    """Raised when input_hash in marker does not match expected row."""


class ContentDigestMismatchError(CampaignMarkerError):
    """Raised when recomputed array content digest disagrees with marker."""


class StoreMissingError(CampaignMarkerError):
    """Raised when output store is missing or unreadable."""


class QuarantineFailedError(CampaignMarkerError):
    """Raised when defensive quarantine procedure fails fatally."""


@dataclass(frozen=True, slots=True)
class ProducerInfo:
    """Provenance and runtime environment of the producing worker."""

    version: str
    git_sha: str
    hostname: str

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "git_sha": self.git_sha,
            "hostname": self.hostname,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProducerInfo":
        return cls(
            version=str(data.get("version", "")),
            git_sha=str(data.get("git_sha", "unknown")),
            hostname=str(data.get("hostname", "")),
        )


@dataclass(frozen=True, slots=True)
class CampaignDoneMarker:
    """Canonical completion marker stamped upon successful unit finalization."""

    schema_version: str
    manifest_row_hash: str
    input_hash: str
    producer: ProducerInfo
    output_locator: str
    content_digest_sha256: str
    completed_at_utc: str
    completed_at_unix_s: float
    duration_s: float
    #: Fence and owner of the lease that committed the unit (#3312). Optional and
    #: informational: a marker older than the unit's ``.fence`` high-water mark was written
    #: by a holder that had been superseded. Absent from markers written before #3312.
    lease_fence: int | None = None
    owner_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "manifest_row_hash": self.manifest_row_hash,
            "input_hash": self.input_hash,
            "producer": self.producer.to_dict(),
            "output_locator": self.output_locator,
            "content_digest_sha256": self.content_digest_sha256,
            "completed_at_utc": self.completed_at_utc,
            "completed_at_unix_s": self.completed_at_unix_s,
            "duration_s": self.duration_s,
        }
        if self.lease_fence is not None:
            data["lease_fence"] = self.lease_fence
        if self.owner_token is not None:
            data["owner_token"] = self.owner_token
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CampaignDoneMarker":
        prod_raw = data.get("producer", {})
        producer = (
            prod_raw if isinstance(prod_raw, ProducerInfo) else ProducerInfo.from_dict(prod_raw)
        )
        return cls(
            schema_version=str(data.get("schema_version", CAMPAIGN_DONE_MARKER_SCHEMA_VERSION)),
            manifest_row_hash=str(data.get("manifest_row_hash", "")),
            input_hash=str(data.get("input_hash", "")),
            producer=producer,
            output_locator=str(data.get("output_locator", "")),
            content_digest_sha256=str(data.get("content_digest_sha256", "")),
            completed_at_utc=str(data.get("completed_at_utc", "")),
            completed_at_unix_s=float(data.get("completed_at_unix_s", 0.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            lease_fence=None if data.get("lease_fence") is None else int(data["lease_fence"]),
            owner_token=None if data.get("owner_token") is None else str(data["owner_token"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "CampaignDoneMarker":
        return cls.from_dict(json.loads(text))

    def write(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "CampaignDoneMarker":
        return cls.from_json(path.read_text(encoding="utf-8"))


def _default_producer_version() -> str:
    """``xtrax@<installed version>``, or ``xtrax@unknown`` when the package is not installed."""
    try:
        return f"xtrax@{importlib.metadata.version('xtrax')}"
    except importlib.metadata.PackageNotFoundError:
        return "xtrax@unknown"


def _default_producer_git_sha() -> str:
    """Git sha from ``XTRAX_PRODUCER_GIT_SHA`` (empty counts as unset), else ``unknown``.

    The sha is never read by shelling out to git and never defaulted to ``clean``.
    """
    env_value = os.environ.get("XTRAX_PRODUCER_GIT_SHA", "").strip()
    return env_value or "unknown"


def create_done_marker(
    manifest_row_hash: str,
    input_hash: str,
    output_locator: str,
    content_digest_sha256: str,
    duration_s: float,
    version: str | None = None,
    git_sha: str | None = None,
    hostname: str | None = None,
) -> CampaignDoneMarker:
    """Create a new CampaignDoneMarker stamped with current environment and timestamps.

    ``version`` defaults to the installed xtrax version; ``git_sha`` defaults to
    ``XTRAX_PRODUCER_GIT_SHA`` or ``unknown``.
    """
    if version is None:
        version = _default_producer_version()
    if git_sha is None:
        git_sha = _default_producer_git_sha()
    host = hostname or socket.getfqdn()
    now_unix = time.time()
    now_utc = datetime.now(UTC).isoformat()
    producer = ProducerInfo(version=version, git_sha=git_sha, hostname=host)

    return CampaignDoneMarker(
        schema_version=CAMPAIGN_DONE_MARKER_SCHEMA_VERSION,
        manifest_row_hash=manifest_row_hash,
        input_hash=input_hash,
        producer=producer,
        output_locator=output_locator,
        content_digest_sha256=content_digest_sha256,
        completed_at_utc=now_utc,
        completed_at_unix_s=now_unix,
        duration_s=duration_s,
    )


@runtime_checkable
class DoneMarkerAdapter(Protocol):
    """Protocol for converting external completion records to and from CampaignDoneMarker."""

    def to_done_marker(self, record: Any) -> CampaignDoneMarker:
        """Convert external record or dict to canonical CampaignDoneMarker."""
        ...

    def from_done_marker(self, marker: CampaignDoneMarker) -> Any:
        """Convert canonical CampaignDoneMarker to external record representation."""
        ...


class AminxDoneMarkerAdapter:
    """Adapter translating aminx campaign v3 done markers to/from CampaignDoneMarker."""

    def to_done_marker(self, record: dict[str, Any]) -> CampaignDoneMarker:
        return CampaignDoneMarker.from_dict(record)

    def from_done_marker(self, marker: CampaignDoneMarker) -> dict[str, Any]:
        return marker.to_dict()


class AlynxrFragmentAdapter:
    """Adapter translating alynxr fold_stats / fold campaign records to/from CampaignDoneMarker."""

    def to_done_marker(self, record: dict[str, Any]) -> CampaignDoneMarker:
        prod_info = ProducerInfo(
            version=str(record.get("producer_version", "alynxr@unknown")),
            git_sha=str(record.get("producer_git_sha", "unknown")),
            hostname=str(record.get("producer_hostname", socket.getfqdn())),
        )
        return CampaignDoneMarker(
            schema_version=CAMPAIGN_DONE_MARKER_SCHEMA_VERSION,
            manifest_row_hash=str(record.get("manifest_row_hash", "")),
            input_hash=str(record.get("input_hash", "")),
            producer=prod_info,
            output_locator=str(record.get("output_locator", record.get("output_path", ""))),
            content_digest_sha256=str(
                record.get("content_digest_sha256", record.get("digest", ""))
            ),
            completed_at_utc=str(record.get("completed_at_utc", "")),
            completed_at_unix_s=float(record.get("completed_at_unix_s", 0.0)),
            duration_s=float(record.get("duration_s", 0.0)),
        )

    def from_done_marker(self, marker: CampaignDoneMarker) -> dict[str, Any]:
        return {
            "prediction_id": Path(marker.output_locator).stem,
            "manifest_row_hash": marker.manifest_row_hash,
            "input_hash": marker.input_hash,
            "output_locator": marker.output_locator,
            "content_digest_sha256": marker.content_digest_sha256,
            "completed_at_utc": marker.completed_at_utc,
            "completed_at_unix_s": marker.completed_at_unix_s,
            "duration_s": marker.duration_s,
            "producer_version": marker.producer.version,
            "producer_hostname": marker.producer.hostname,
        }


@dataclass(frozen=True, slots=True)
class UnitValidationResult:
    """Structured report returned by validate_unit_done."""

    is_valid: bool
    tier: int
    manifest_row_hash: str
    input_hash: str
    error_message: str | None = None
    error_type: str | None = None
    quarantine_path: Path | None = None


def quarantine_output(
    output_path: Path,
    reason: str,
    exc: Exception | None = None,
) -> Path:
    """Safely isolate corrupt output store and marker, defensively handling ENOSPC and EXDEV.

    Per Section 4.4 of SPEC-261105-CAMPAIGN-RUNNER:
    - Default: Atomically moves store and .done.json to
      <campaign_dir>/.quarantine/<YYYYMMDD_HHMMSS>_<reason>_<manifest_row_hash>/
      and writes quarantine_manifest.json with forensic details.
    - ENOSPC Fallback: If creating the quarantine dir or manifest raises ENOSPC (disk/inode full),
      renames BOTH the output store and .done.json in-place to <output_path>.corrupt.<uuid>
      and <done_marker>.corrupt.<uuid> without allocating new directories.
    - EXDEV Fallback: If crossing mount boundaries raises EXDEV, falls back to shutil.move.
    """
    campaign_dir = output_path.parent
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    # The uuid suffix keeps two quarantines of the same unit within one second apart.
    quarantine_dir = (
        campaign_dir
        / ".quarantine"
        / f"{timestamp}_{reason}_{output_path.name}_{uuid.uuid4().hex[:8]}"
    )
    done_marker = output_path.with_name(f"{output_path.name}.done.json")

    try:
        quarantine_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = quarantine_dir / "quarantine_manifest.json"
        manifest_payload = {
            "original_path": str(output_path),
            "reason": reason,
            "detected_at_utc": datetime.now(UTC).isoformat(),
            "exception": str(exc) if exc else None,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        if output_path.exists():
            os.replace(output_path, quarantine_dir / output_path.name)
        if done_marker.exists():
            os.replace(done_marker, quarantine_dir / done_marker.name)
        return quarantine_dir
    except OSError as err:
        if err.errno == errno.ENOSPC:
            # Fallback 1: Disk/inode exhaustion - rename in-place without writing metadata
            unique_suffix = f"corrupt.{uuid.uuid4().hex[:8]}"
            fallback_path = output_path.with_name(f"{output_path.name}.{unique_suffix}")
            if output_path.exists():
                os.replace(output_path, fallback_path)
            if done_marker.exists():
                os.replace(
                    done_marker,
                    output_path.with_name(f"{done_marker.name}.{unique_suffix}"),
                )
            return fallback_path
        if err.errno == errno.EXDEV:
            # Fallback 2: Cross-device link - use copy-and-unlink
            if output_path.exists():
                shutil.move(output_path, quarantine_dir / output_path.name)
            if done_marker.exists():
                shutil.move(done_marker, quarantine_dir / done_marker.name)
            return quarantine_dir
        msg = f"Failed to quarantine corrupt output {output_path}: {err}"
        raise QuarantineFailedError(msg) from err


def compute_content_digest(path: Path) -> str:
    """Compute the content digest of a zarr store directory or a single file.

    A directory is digested with :func:`xtrax.run.zarr_integrity.zarr_content_digest`
    (the logical array digest). There is no fallback to a raw byte hash: the producer
    and the verifier must use the same algorithm, and a directory zarr cannot open is
    an error to surface, not a store to hash differently. A file is digested by streaming
    its bytes with a bounded 64 MB buffer.
    """
    if path.is_dir():
        from xtrax.run.zarr_integrity import zarr_content_digest

        return zarr_content_digest(path, include_provenance=True)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


#: OSErrors that mean the store itself is missing pieces, so the unit is not done. Any
#: other OSError during deep verification is an I/O failure and is raised, not reported
#: as an invalid store.
_STORE_DEFECT_OS_ERRORS: tuple[type[OSError], ...] = (
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
)

#: Zarr root metadata files, v3 first. Any one of them marks a directory as a store.
ZARR_ROOT_METADATA_FILES: tuple[str, ...] = ("zarr.json", ".zgroup", ".zarray")


def validate_unit_done(
    output_path: Path | str,
    expected_row: "ManifestRow",
    deep_verify: bool = False,
    quarantine_on_failure: bool = False,
) -> UnitValidationResult:
    """Validate a completed unit using decoupled Two-Tier Resume verification.

    Tier 1 (< 5 ms): Checks .done.json existence, parses schema v3, verifies
    manifest_row_hash and input_hash, and checks Zarr root metadata (zarr.json for
    v3, .zgroup/.zarray for v2)
    without reading multi-gigabyte array chunk bytes.

    Tier 2 (Opt-in via deep_verify=True): Streams chunk bytes with bounded buffer
    to recompute and verify zarr_content_digest.

    ``quarantine_on_failure`` defaults to False: a read-only check never moves a store.
    Quarantine moves files, so callers that enable it must hold the unit's lease.

    Raises:
        OSError: An I/O error (other than a missing file or directory) while computing
            the Tier-2 digest. The store's state is unknown, so it is neither reported
            invalid nor quarantined.
    """
    path = Path(output_path)
    done_marker_path = path.with_name(f"{path.name}.done.json")

    # 1. Check done marker existence
    if not done_marker_path.exists():
        return UnitValidationResult(
            is_valid=False,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=f"Done marker not found at {done_marker_path}",
            error_type="DoneMarkerNotFoundError",
        )

    # 2. Parse done marker
    try:
        marker_text = done_marker_path.read_text(encoding="utf-8")
        marker_data = json.loads(marker_text)
    except Exception as exc:
        q_path = None
        if quarantine_on_failure:
            q_path = quarantine_output(path, reason="corrupt_done_marker", exc=exc)
        return UnitValidationResult(
            is_valid=False,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=f"Failed to parse done marker JSON: {exc}",
            error_type="DoneMarkerCorruptError",
            quarantine_path=q_path,
        )

    # Check schema version
    schema_ver = marker_data.get("schema_version")
    if schema_ver != CAMPAIGN_DONE_MARKER_SCHEMA_VERSION:
        q_path = None
        if quarantine_on_failure:
            q_path = quarantine_output(
                path,
                reason="outdated_schema_version",
                exc=DoneMarkerVersionError(
                    f"Found schema {schema_ver}, expected {CAMPAIGN_DONE_MARKER_SCHEMA_VERSION}"
                ),
            )
        return UnitValidationResult(
            is_valid=False,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=f"Outdated schema version: {schema_ver}",
            error_type="DoneMarkerVersionError",
            quarantine_path=q_path,
        )

    # Check manifest_row_hash and input_hash
    marker_row_hash = marker_data.get("manifest_row_hash")
    if marker_row_hash != expected_row.manifest_row_hash:
        q_path = None
        if quarantine_on_failure:
            q_path = quarantine_output(path, reason="manifest_row_hash_mismatch")
        return UnitValidationResult(
            is_valid=False,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=(
                f"manifest_row_hash mismatch: expected {expected_row.manifest_row_hash}, "
                f"found {marker_row_hash}"
            ),
            error_type="ManifestRowHashMismatchError",
            quarantine_path=q_path,
        )

    marker_input_hash = marker_data.get("input_hash")
    if marker_input_hash != expected_row.input_hash:
        q_path = None
        if quarantine_on_failure:
            q_path = quarantine_output(path, reason="input_hash_mismatch")
        return UnitValidationResult(
            is_valid=False,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=(
                f"input_hash mismatch: expected {expected_row.input_hash}, "
                f"found {marker_input_hash}"
            ),
            error_type="InputHashMismatchError",
            quarantine_path=q_path,
        )

    # 3. Check store exists
    if not path.exists():
        q_path = None
        if quarantine_on_failure:
            q_path = quarantine_output(path, reason="store_missing")
        return UnitValidationResult(
            is_valid=False,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=f"Output store missing at {path}",
            error_type="StoreMissingError",
            quarantine_path=q_path,
        )

    # 4. Check Zarr root metadata (Tier 1: < 5 ms, NO chunk reading). zarr v3 writes
    # zarr.json; v2 writes .zgroup or .zarray.
    if path.is_dir():
        meta_file = next(
            (path / name for name in ZARR_ROOT_METADATA_FILES if (path / name).exists()),
            None,
        )
        if meta_file is None:
            q_path = None
            if quarantine_on_failure:
                q_path = quarantine_output(path, reason="missing_zarr_root_metadata")
            return UnitValidationResult(
                is_valid=False,
                tier=1,
                manifest_row_hash=expected_row.manifest_row_hash,
                input_hash=expected_row.input_hash,
                error_message=(
                    "Zarr store has no root metadata (expected one of "
                    f"{', '.join(ZARR_ROOT_METADATA_FILES)})"
                ),
                error_type="StoreMissingError",
                quarantine_path=q_path,
            )
        # Verify metadata is valid JSON
        try:
            json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            q_path = None
            if quarantine_on_failure:
                q_path = quarantine_output(path, reason="corrupt_zarr_metadata", exc=exc)
            return UnitValidationResult(
                is_valid=False,
                tier=1,
                manifest_row_hash=expected_row.manifest_row_hash,
                input_hash=expected_row.input_hash,
                error_message=f"Corrupt Zarr metadata in {meta_file}: {exc}",
                error_type="DoneMarkerCorruptError",
                quarantine_path=q_path,
            )

    # If Tier-1 only, we are done and valid
    if not deep_verify:
        return UnitValidationResult(
            is_valid=True,
            tier=1,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
        )

    # 5. Tier 2: Deep Content Verification
    try:
        computed_digest = compute_content_digest(path)
        expected_digest = str(marker_data.get("content_digest_sha256", ""))
        if computed_digest != expected_digest:
            q_path = None
            if quarantine_on_failure:
                q_path = quarantine_output(
                    path,
                    reason="content_digest_mismatch",
                    exc=ContentDigestMismatchError(
                        f"Computed {computed_digest} != expected {expected_digest}"
                    ),
                )
            return UnitValidationResult(
                is_valid=False,
                tier=2,
                manifest_row_hash=expected_row.manifest_row_hash,
                input_hash=expected_row.input_hash,
                error_message=(
                    f"Content digest mismatch: computed {computed_digest} != "
                    f"expected {expected_digest}"
                ),
                error_type="ContentDigestMismatchError",
                quarantine_path=q_path,
            )
    except Exception as exc:
        if isinstance(exc, OSError) and not isinstance(exc, _STORE_DEFECT_OS_ERRORS):
            # EIO, ESTALE, EACCES, ENOSPC, a timed-out network read: the environment failed,
            # not the store. Reporting the store invalid would let callers quarantine a good
            # unit and recompute it, so surface the error instead.
            raise
        q_path = None
        if quarantine_on_failure:
            q_path = quarantine_output(path, reason="deep_verification_exception", exc=exc)
        return UnitValidationResult(
            is_valid=False,
            tier=2,
            manifest_row_hash=expected_row.manifest_row_hash,
            input_hash=expected_row.input_hash,
            error_message=f"Deep verification failed with exception: {exc}",
            error_type=type(exc).__name__,
            quarantine_path=q_path,
        )

    return UnitValidationResult(
        is_valid=True,
        tier=2,
        manifest_row_hash=expected_row.manifest_row_hash,
        input_hash=expected_row.input_hash,
    )

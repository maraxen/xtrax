"""Distributed atomic file leasing, active heartbeat, and singleton commit protocol.

Covering [T2-01], [T2-02], [T2-03], Section 3.3, and Section 5 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xtrax.campaign.marker import CampaignDoneMarker

#: Clock drift safety margin against heterogeneous cluster node clock skew (5 minutes).
CLOCK_DRIFT_GRACE_PERIOD_S: int = 300

#: Schema version for lock files.
CAMPAIGN_LOCK_SCHEMA_VERSION = "campaign_lock_v1"


class CampaignLeaseError(Exception):
    """Base exception for all campaign leasing errors."""


class LockUnavailableError(CampaignLeaseError):
    """Raised when a lock is held by another worker and cannot be acquired or stolen."""


class LeaseLostError(CampaignLeaseError):
    """Raised when an active lease was lost, stolen, or expired during execution or commit."""


class GenerationSealedError(CampaignLeaseError):
    """Raised when a worker attempts to commit output to an already sealed generation."""


@dataclass(frozen=True, slots=True)
class StolenAttemptDiscarded:
    """Sentinel returned when a unit output commit is discarded because it is already done."""

    output_path: Path
    attempt_uuid: str
    reason: str = "Done marker already exists on disk"


@dataclass(frozen=True, slots=True)
class CampaignLock:
    """Persistent on-disk lease metadata for distributed mutual exclusion."""

    schema_version: str
    owner_token: str
    manifest_row_hash: str
    acquired_at_unix_s: float
    lease_expires_at_unix_s: float
    lease_seconds: int = 1800
    heartbeat_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_token": self.owner_token,
            "manifest_row_hash": self.manifest_row_hash,
            "acquired_at_unix_s": self.acquired_at_unix_s,
            "lease_expires_at_unix_s": self.lease_expires_at_unix_s,
            "lease_seconds": self.lease_seconds,
            "heartbeat_count": self.heartbeat_count,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignLock:
        return cls(
            schema_version=str(data.get("schema_version", CAMPAIGN_LOCK_SCHEMA_VERSION)),
            owner_token=str(data.get("owner_token", "")),
            manifest_row_hash=str(data.get("manifest_row_hash", "")),
            acquired_at_unix_s=float(data.get("acquired_at_unix_s", 0.0)),
            lease_expires_at_unix_s=float(data.get("lease_expires_at_unix_s", 0.0)),
            lease_seconds=int(data.get("lease_seconds", 1800)),
            heartbeat_count=int(data.get("heartbeat_count", 0)),
        )

    @classmethod
    def from_json(cls, text: str) -> CampaignLock:
        return cls.from_dict(json.loads(text))


def fsync_file(path: Path) -> None:
    """Force a single file's data to disk."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Durabilize directory entries, gracefully handling NFS client EINVAL/ENOTSUP.

    Per Section 5.2 of SPEC-261105-CAMPAIGN-RUNNER:
    Suppresses EINVAL, ENOTSUP, EOPNOTSUPP, and EBADF on NFS client directory file descriptors.
    Also gracefully suppresses ENOENT or EACCES if directory cannot be opened.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as err:
        if err.errno in (errno.ENOENT, errno.EACCES):
            return
        raise
    try:
        os.fsync(fd)
    except OSError as err:
        if err.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF):
            raise
    finally:
        os.close(fd)


def _clean_scratch(path: Path) -> None:
    """Safely remove a scratch file or directory tree."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with suppress(OSError):
            path.unlink()


class HeartbeatThread(threading.Thread):
    """Refreshes lease_expires_at_unix_s periodically with strict CAS expiry guards.

    Per Section 5.2 of SPEC-261105-CAMPAIGN-RUNNER.
    """

    def __init__(
        self,
        lock_path: Path,
        owner_token: str,
        manifest_row_hash: str,
        lease_seconds: int = 1800,
        interval_seconds: float = 60.0,
        initial_acquired_at: float | None = None,
        expiry_guard_margin: float | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"Heartbeat-{manifest_row_hash[:8]}")
        self.lock_path = lock_path
        self.owner_token = owner_token
        self.manifest_row_hash = manifest_row_hash
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self.acquired_at_unix_s = (
            initial_acquired_at if initial_acquired_at is not None else time.time()
        )
        self.current_lease_expires_at = self.acquired_at_unix_s + lease_seconds
        self.expiry_guard_margin = (
            expiry_guard_margin
            if expiry_guard_margin is not None
            else min(60.0, lease_seconds / 2.0)
        )
        self.abort_event = threading.Event()
        self.stop_event = threading.Event()
        self.heartbeat_count = 0

    def stop(self) -> None:
        """Signal heartbeat thread to terminate cleanly."""
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            if not self._renew_lease_cas():
                self.abort_event.set()
                break

    def _renew_lease_cas(self) -> bool:
        """Atomically renew lease using CAS readback verification and expiration guard."""
        try:
            now = time.time()
            # 1. Expiration Guard: Never attempt renewal if lease is within
            # expiry_guard_margin of expiring or expired
            if now >= self.current_lease_expires_at - self.expiry_guard_margin:
                return False

            # 2. CAS Pre-Check: verify active ownership, immutable epoch, and expected count
            if not self.lock_path.exists():
                return False
            current = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if current.get("owner_token") != self.owner_token:
                return False
            if current.get("acquired_at_unix_s") != self.acquired_at_unix_s:
                return False
            if current.get("heartbeat_count") != self.heartbeat_count:
                return False

            # 3. Stage updated lease
            next_count = self.heartbeat_count + 1
            new_expires = now + self.lease_seconds
            updated_lock = {
                "schema_version": CAMPAIGN_LOCK_SCHEMA_VERSION,
                "owner_token": self.owner_token,
                "manifest_row_hash": self.manifest_row_hash,
                "acquired_at_unix_s": self.acquired_at_unix_s,
                "lease_expires_at_unix_s": new_expires,
                "lease_seconds": self.lease_seconds,
                "heartbeat_count": next_count,
            }
            tmp_path = self.lock_path.with_name(f"{self.lock_path.name}.tmp.{uuid.uuid4().hex[:8]}")
            tmp_path.write_text(json.dumps(updated_lock, indent=2), encoding="utf-8")
            fsync_file(tmp_path)

            # 4. Atomically replace lock file
            os.replace(tmp_path, self.lock_path)
            fsync_directory(self.lock_path.parent)

            # 5. CAS Post-Verification: Confirm our write held
            post_check = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if (
                post_check.get("owner_token") == self.owner_token
                and post_check.get("heartbeat_count") == next_count
                and post_check.get("acquired_at_unix_s") == self.acquired_at_unix_s
            ):
                self.heartbeat_count = next_count
                self.current_lease_expires_at = new_expires
                return True
            return False
        except Exception:
            return False


class FileLock:
    """Atomic POSIX file lease with background heartbeat and stale lease stealing.

    Per Section 5 of SPEC-261105-CAMPAIGN-RUNNER.
    """

    def __init__(
        self,
        output_path: Path,
        owner_token: str,
        manifest_row_hash: str,
        lease_seconds: int = 1800,
        heartbeat_interval: float = 60.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.lock_path = self.output_path.with_name(f"{self.output_path.name}.lock")
        self.owner_token = owner_token
        self.manifest_row_hash = manifest_row_hash
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self.acquired_at_unix_s: float | None = None
        self.heartbeat: HeartbeatThread | None = None
        self.is_locked: bool = False

    def acquire(self) -> bool:
        """Atomically acquire the lease or steal if stale past 300s clock drift margin."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()

        # Step 1: Try atomic creation via O_CREAT | O_EXCL
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            self.acquired_at_unix_s = now
            initial_lock = CampaignLock(
                schema_version=CAMPAIGN_LOCK_SCHEMA_VERSION,
                owner_token=self.owner_token,
                manifest_row_hash=self.manifest_row_hash,
                acquired_at_unix_s=now,
                lease_expires_at_unix_s=now + self.lease_seconds,
                lease_seconds=self.lease_seconds,
                heartbeat_count=0,
            )
            os.write(fd, initial_lock.to_json().encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fsync_directory(self.lock_path.parent)

            self._start_heartbeat(now)
            self.is_locked = True
            return True
        except FileExistsError:
            pass

        # Step 2: Lock exists. Evaluate stale lease stealing with clock drift grace period.
        try:
            lock_stat = self.lock_path.stat()
            current_data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            lock_obj = CampaignLock.from_dict(current_data)
        except Exception as exc:
            msg = f"Failed to read lock file {self.lock_path}: {exc}"
            raise LockUnavailableError(msg) from exc

        # Dual Expiration & Clock Drift Check (Section 5.3)
        is_stale = (now >= lock_obj.lease_expires_at_unix_s + CLOCK_DRIFT_GRACE_PERIOD_S) and (
            now - lock_stat.st_mtime >= lock_obj.lease_seconds + CLOCK_DRIFT_GRACE_PERIOD_S
        )

        if not is_stale:
            msg = (
                f"Lock on {self.lock_path} is currently active or within "
                f"{CLOCK_DRIFT_GRACE_PERIOD_S}s clock drift grace period"
            )
            raise LockUnavailableError(msg)

        # Step 3: Stale Lease Stealing with Atomic Steal Token
        steal_token = self.lock_path.with_name(f"{self.lock_path.name}.steal")
        try:
            steal_fd = os.open(steal_token, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            msg = f"Lost CAS race while attempting to steal lock on {self.lock_path}"
            raise LockUnavailableError(msg) from exc

        try:
            # Re-verify lock is still the same stale lock
            curr_data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if (
                curr_data.get("owner_token") != lock_obj.owner_token
                or curr_data.get("acquired_at_unix_s") != lock_obj.acquired_at_unix_s
            ):
                msg = f"Lock on {self.lock_path} was modified before steal could complete"
                raise LockUnavailableError(msg)

            steal_acquired = now
            stolen_lock = CampaignLock(
                schema_version=CAMPAIGN_LOCK_SCHEMA_VERSION,
                owner_token=self.owner_token,
                manifest_row_hash=self.manifest_row_hash,
                acquired_at_unix_s=steal_acquired,
                lease_expires_at_unix_s=now + self.lease_seconds,
                lease_seconds=self.lease_seconds,
                heartbeat_count=0,
            )
            steal_suffix = f"steal.{uuid.uuid4().hex[:8]}.tmp"
            steal_tmp = self.lock_path.with_name(f"{self.lock_path.name}.{steal_suffix}")
            steal_tmp.write_text(stolen_lock.to_json(), encoding="utf-8")
            fsync_file(steal_tmp)

            os.replace(steal_tmp, self.lock_path)
            fsync_directory(self.lock_path.parent)

            self.acquired_at_unix_s = steal_acquired
            self._start_heartbeat(steal_acquired)
            self.is_locked = True
            return True
        finally:
            with suppress(OSError):
                os.close(steal_fd)
            with suppress(OSError):
                steal_token.unlink()

    def _start_heartbeat(self, acquired_at: float) -> None:
        self.heartbeat = HeartbeatThread(
            lock_path=self.lock_path,
            owner_token=self.owner_token,
            manifest_row_hash=self.manifest_row_hash,
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_interval,
            initial_acquired_at=acquired_at,
        )
        self.heartbeat.start()

    def release(self) -> None:
        """Stop heartbeat and unlink lock file if owned."""
        if self.heartbeat is not None:
            self.heartbeat.stop()
            self.heartbeat.join(timeout=1.0)
            self.heartbeat = None

        if self.is_locked and self.lock_path.exists():
            try:
                curr = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if curr.get("owner_token") == self.owner_token:
                    self.lock_path.unlink()
                    fsync_directory(self.lock_path.parent)
            except Exception:
                pass

        self.is_locked = False

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def commit_output(
        self,
        partial_path: Path,
        output_path: Path,
        done_marker: CampaignDoneMarker,
        attempt_uuid: str,
        generation_dir: Path | None = None,
    ) -> Any:
        """Finalize output according to the Singleton Commit Token Protocol.

        Per Section 5.4 of SPEC-261105-CAMPAIGN-RUNNER:
        1. Check if <output_path>.done.json exists; discard if so.
        2. Check <generation_dir>/.generation_sealed; raise GenerationSealedError if sealed.
        3. Atomically create singleton <output_path>.commit via os.open(path, O_CREAT|O_EXCL).
           NO attempt UUID in path!
        4. Write commit metadata (attempt_uuid, owner_token, committed_at).
        5. Re-verify <output_path>.lock ownership, expiration, and abort event.
        6. Promote .partial.<attempt_uuid> to <output_path> via os.replace.
        7. Write <output_path>.done.json.tmp, fsync_file, and os.replace to .done.json.
        8. Durabilize parent directory and unlink .commit and .lock.
        """
        done_marker_path = output_path.with_name(f"{output_path.name}.done.json")

        # 1. Existing Done Check
        if done_marker_path.exists():
            _clean_scratch(partial_path)
            return StolenAttemptDiscarded(
                output_path=output_path,
                attempt_uuid=attempt_uuid,
                reason=f"Done marker already exists at {done_marker_path}",
            )

        # 2. Generation Seal Check
        if generation_dir is not None:
            seal_file = Path(generation_dir) / ".generation_sealed"
            if seal_file.exists():
                _clean_scratch(partial_path)
                msg = f"Generation directory {generation_dir} is sealed"
                raise GenerationSealedError(msg)

        # 3. Singleton Commit Token Creation (GATE-03)
        commit_token_path = output_path.with_name(f"{output_path.name}.commit")
        try:
            fd = os.open(commit_token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            _clean_scratch(partial_path)
            raise

        try:
            # 4. Write Commit Metadata
            commit_meta = {
                "attempt_uuid": attempt_uuid,
                "owner_token": self.owner_token,
                "committed_at_unix_s": time.time(),
            }
            os.write(fd, json.dumps(commit_meta, indent=2).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

        # 5. Lock Ownership Re-Verification (Zombie Worker Protection)
        lost = False
        try:
            if not self.lock_path.exists():
                lost = True
            else:
                curr_lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if (
                    curr_lock.get("owner_token") != self.owner_token
                    or (
                        self.acquired_at_unix_s is not None
                        and curr_lock.get("acquired_at_unix_s") != self.acquired_at_unix_s
                    )
                    or time.time() >= float(curr_lock.get("lease_expires_at_unix_s", 0.0))
                    or (self.heartbeat is not None and self.heartbeat.abort_event.is_set())
                ):
                    lost = True
        except Exception:
            lost = True

        if lost:
            if commit_token_path.exists():
                with suppress(OSError):
                    commit_token_path.unlink()
            _clean_scratch(partial_path)
            msg = f"Lease for {output_path} was lost, stolen, or expired before commit"
            raise LeaseLostError(msg)

        # 6. Output Promotion
        os.replace(partial_path, output_path)

        # 7. Write and Rename Done Marker
        tmp_done = output_path.with_name(f"{output_path.name}.done.json.tmp.{attempt_uuid}")
        tmp_done.write_text(done_marker.to_json(), encoding="utf-8")
        fsync_file(tmp_done)
        os.replace(tmp_done, done_marker_path)

        # 8. Directory Durabilization & Release
        fsync_directory(output_path.parent)
        if commit_token_path.exists():
            with suppress(OSError):
                commit_token_path.unlink()
        if self.lock_path.exists():
            with suppress(OSError):
                self.lock_path.unlink()
        fsync_directory(output_path.parent)

        if self.heartbeat is not None:
            self.heartbeat.stop()
            self.heartbeat.join(timeout=1.0)
            self.heartbeat = None
        self.is_locked = False

        return done_marker

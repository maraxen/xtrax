"""Distributed atomic file leasing, active heartbeat, and singleton commit protocol.

Covering [T2-01], [T2-02], [T2-03], Section 3.3, and Section 5 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

import dataclasses
import errno
import json
import logging
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

logger = logging.getLogger(__name__)

#: Clock drift safety margin against heterogeneous cluster node clock skew (5 minutes).
CLOCK_DRIFT_GRACE_PERIOD_S: int = 300


def _boottime() -> float:
    """Seconds on a clock that a wall-clock step cannot move and that runs through suspend.

    ``CLOCK_BOOTTIME`` on Linux; ``time.monotonic()`` elsewhere. It bounds a holder's local
    view of its own lease (#3312), so a stalled or suspended holder sees its lease lapse.
    """
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    if clock is not None:
        return time.clock_gettime(clock)
    return time.monotonic()


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
    #: Fencing token (#3312): raised on every acquire and steal, never reused while the
    #: unit's ``.fence`` high-water file exists. 0 in locks written before #3312.
    fence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_token": self.owner_token,
            "manifest_row_hash": self.manifest_row_hash,
            "acquired_at_unix_s": self.acquired_at_unix_s,
            "lease_expires_at_unix_s": self.lease_expires_at_unix_s,
            "lease_seconds": self.lease_seconds,
            "heartbeat_count": self.heartbeat_count,
            "fence": self.fence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CampaignLock":
        return cls(
            schema_version=str(data.get("schema_version", CAMPAIGN_LOCK_SCHEMA_VERSION)),
            owner_token=str(data.get("owner_token", "")),
            manifest_row_hash=str(data.get("manifest_row_hash", "")),
            acquired_at_unix_s=float(data.get("acquired_at_unix_s", 0.0)),
            lease_expires_at_unix_s=float(data.get("lease_expires_at_unix_s", 0.0)),
            lease_seconds=int(data.get("lease_seconds", 1800)),
            heartbeat_count=int(data.get("heartbeat_count", 0)),
            fence=int(data.get("fence", 0)),
        )

    @classmethod
    def from_json(cls, text: str) -> "CampaignLock":
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
        fence: int = 0,
        local_deadline: float | None = None,
    ) -> None:
        if lease_seconds < 10:
            msg = f"lease_seconds must be >= 10, got {lease_seconds}"
            raise ValueError(msg)
        super().__init__(daemon=True, name=f"Heartbeat-{manifest_row_hash[:8]}")
        self.lock_path = lock_path
        self.owner_token = owner_token
        self.manifest_row_hash = manifest_row_hash
        self.lease_seconds = lease_seconds
        self.fence = fence
        #: Boot-clock deadline of this holder's own view of the lease (#3312). Advanced
        #: only after a renewal's post-verify passes, from a time sampled before the write.
        self.local_deadline = (
            local_deadline if local_deadline is not None else _boottime() + lease_seconds
        )
        self.interval_seconds = min(float(interval_seconds), lease_seconds / 3.0)
        self.acquired_at_unix_s = (
            initial_acquired_at if initial_acquired_at is not None else time.time()
        )
        self.current_lease_expires_at = self.acquired_at_unix_s + lease_seconds
        self.expiry_guard_margin = (
            expiry_guard_margin
            if expiry_guard_margin is not None
            else min(60.0, lease_seconds / 6.0)
        )
        self.abort_event = threading.Event()
        self.stop_event = threading.Event()
        # Held across each whole renewal and across stop(). That makes the stop decision and
        # the lock-file replace mutually exclusive: once stop() returns, no renewal can
        # replace the lock file again, even if join() later times out.
        self._renew_lock = threading.Lock()
        self.heartbeat_count = 0

    def stop(self) -> None:
        """Signal heartbeat thread to terminate cleanly.

        Blocks until any renewal already in progress has finished. After this returns the
        thread cannot replace the lock file, so the caller may safely unlink it.
        """
        with self._renew_lock:
            self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            # A renewal refused because stop() was called is not a lost lease.
            if not self._renew_lease_cas() and not self.stop_event.is_set():
                self.abort_event.set()
                break

    def _renew_lease_cas(self) -> bool:
        """Atomically renew lease using CAS readback verification and expiration guard."""
        with self._renew_lock:
            return self._renew_lease_cas_locked()

    def _lock_is_ours(self) -> bool:
        """Re-read the lock: same owner, epoch, fence and heartbeat count as this thread's."""
        try:
            current = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            current.get("owner_token") == self.owner_token
            and current.get("acquired_at_unix_s") == self.acquired_at_unix_s
            and int(current.get("fence", 0)) == self.fence
            and current.get("heartbeat_count") == self.heartbeat_count
        )

    def _renew_lease_cas_locked(self) -> bool:
        """Renewal body. Caller must hold self._renew_lock."""
        try:
            boot_at_start = _boottime()
            now = time.time()
            # 1. Expiration Guard: Never attempt renewal if lease is within
            # expiry_guard_margin of expiring or expired
            if now >= self.current_lease_expires_at - self.expiry_guard_margin:
                return False

            # 2. CAS Pre-Check: verify active ownership, immutable epoch, fence and count
            if not self._lock_is_ours():
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
                "fence": self.fence,
            }
            tmp_path = self.lock_path.with_name(f"{self.lock_path.name}.tmp.{uuid.uuid4().hex[:8]}")
            tmp_path.write_text(json.dumps(updated_lock, indent=2), encoding="utf-8")
            fsync_file(tmp_path)

            # 4. Atomically replace lock file, unless stop() has been decided or the lock
            # changed while the tmp file was written (#3312: a stall there must not let
            # this renewal overwrite a stealer's lock). The lock is held here, so stop()
            # cannot set the event mid-renewal.
            if self.stop_event.is_set() or not self._lock_is_ours():
                with suppress(OSError):
                    tmp_path.unlink()
                return False
            os.replace(tmp_path, self.lock_path)
            fsync_directory(self.lock_path.parent)

            # 5. CAS Post-Verification: Confirm our write held
            post_check = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if (
                post_check.get("owner_token") == self.owner_token
                and post_check.get("heartbeat_count") == next_count
                and post_check.get("acquired_at_unix_s") == self.acquired_at_unix_s
                and int(post_check.get("fence", 0)) == self.fence
            ):
                self.heartbeat_count = next_count
                self.current_lease_expires_at = new_expires
                self.local_deadline = boot_at_start + self.lease_seconds
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
        if lease_seconds < 10:
            msg = f"lease_seconds must be >= 10, got {lease_seconds}"
            raise ValueError(msg)
        self.output_path = Path(output_path)
        self.lock_path = self.output_path.with_name(f"{self.output_path.name}.lock")
        self.owner_token = owner_token
        self.manifest_row_hash = manifest_row_hash
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = min(float(heartbeat_interval), lease_seconds / 3.0)
        self.expiry_guard_margin = min(60.0, lease_seconds / 6.0)
        self.fence_path = self.output_path.with_name(f"{self.output_path.name}.fence")
        self.acquired_at_unix_s: float | None = None
        self.fence: int = 0
        self.local_deadline: float | None = None
        self.heartbeat: HeartbeatThread | None = None
        self.is_locked: bool = False

    # --- fencing (#3312) -------------------------------------------------------------

    def _read_fence_file(self) -> int:
        """The unit's fence high-water mark; 0 when absent or unreadable."""
        try:
            return int(self.fence_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    def _lock_body_is_ours(self, data: dict[str, Any]) -> bool:
        return (
            data.get("owner_token") == self.owner_token
            and data.get("acquired_at_unix_s") == self.acquired_at_unix_s
            and int(data.get("fence", 0)) == self.fence
        )

    def _read_lock_body(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _raise_fence_file(self) -> None:
        """Record this holder's fence as the high-water mark, never lowering it.

        Written only while the lock is still ours, and only when our fence exceeds the
        value on disk, so a holder that stalled between creating its lock and this write
        cannot roll back a later stealer's fence. A failed write is tolerated: identity
        checks compare the whole (owner, fence, acquired_at) triple.
        """
        body = self._read_lock_body()
        if body is None or not self._lock_body_is_ours(body):
            return
        if self.fence <= self._read_fence_file():
            return
        tmp = self.fence_path.with_name(f"{self.fence_path.name}.tmp.{uuid.uuid4().hex[:8]}")
        try:
            tmp.write_text(f"{self.fence}\n", encoding="utf-8")
            os.replace(tmp, self.fence_path)
        except OSError as exc:
            logger.warning("could not record fence %d at %s: %s", self.fence, self.fence_path, exc)
            with suppress(OSError):
                tmp.unlink()

    def current_local_deadline(self) -> float | None:
        """Boot-clock deadline of this holder's lease, advanced by heartbeat renewals."""
        if self.heartbeat is not None:
            return self.heartbeat.local_deadline
        return self.local_deadline

    def assert_holder(self) -> None:
        """Raise ``LeaseLostError`` unless this worker still holds the lease.

        Called immediately before every destructive step (#3312). Holds when the lock
        carries our owner, acquired_at and fence; our fence is at least the unit's
        ``.fence`` high-water mark; the heartbeat has not aborted; the wall-clock lease in
        the lock has not expired; and the boot-clock local deadline has not passed.
        Closes the stall windows down to the gap between this check and the step after it.
        """
        msg = f"Lease for {self.output_path} was lost, stolen, or expired"
        if not self.is_locked or self.acquired_at_unix_s is None:
            raise LeaseLostError(msg)
        body = self._read_lock_body()
        if body is None or not self._lock_body_is_ours(body):
            raise LeaseLostError(msg)
        if self.fence < self._read_fence_file():
            raise LeaseLostError(f"{msg} (fence {self.fence} superseded)")
        if self.heartbeat is not None and self.heartbeat.abort_event.is_set():
            raise LeaseLostError(msg)
        if time.time() >= float(body.get("lease_expires_at_unix_s", 0.0)):
            raise LeaseLostError(msg)
        deadline = self.current_local_deadline()
        if deadline is not None and _boottime() >= deadline:
            raise LeaseLostError(f"{msg} (local lease deadline passed)")

    def _unlink_commit_token_if_ours(self, commit_token_path: Path) -> None:
        """Remove ``.commit`` only when its body names this owner and fence."""
        try:
            body = json.loads(commit_token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(body, dict) or body.get("owner_token") != self.owner_token:
            return
        if int(body.get("fence", self.fence)) != self.fence:
            return
        with suppress(OSError):
            commit_token_path.unlink()

    def _unlink_lock_if_ours(self) -> bool:
        """Remove ``.lock`` only when its body matches this holder's triple."""
        body = self._read_lock_body()
        if body is None or not self._lock_body_is_ours(body):
            return False
        try:
            self.lock_path.unlink()
        except OSError:
            return False
        fsync_directory(self.lock_path.parent)
        return True

    def _stop_heartbeat(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.stop()
            self.heartbeat.join(timeout=1.0)
            self.local_deadline = self.heartbeat.local_deadline
            self.heartbeat = None

    # --- acquire ------------------------------------------------------------------

    def acquire(self) -> bool:
        """Atomically acquire the lease or steal if stale past 300s clock drift margin."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        boot_now = _boottime()

        # Step 1: Try atomic creation via O_CREAT | O_EXCL
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            self.acquired_at_unix_s = now
            self.fence = self._read_fence_file() + 1
            initial_lock = CampaignLock(
                schema_version=CAMPAIGN_LOCK_SCHEMA_VERSION,
                owner_token=self.owner_token,
                manifest_row_hash=self.manifest_row_hash,
                acquired_at_unix_s=now,
                lease_expires_at_unix_s=now + self.lease_seconds,
                lease_seconds=self.lease_seconds,
                heartbeat_count=0,
                fence=self.fence,
            )
            os.write(fd, initial_lock.to_json().encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fsync_directory(self.lock_path.parent)

            self.local_deadline = boot_now + self.lease_seconds
            self._raise_fence_file()
            self._clear_stale_commit_token()
            self._start_heartbeat(now)
            self.is_locked = True
            return True
        except FileExistsError:
            pass

        # Step 2: Lock exists. Evaluate stale lease stealing with clock drift grace period.
        try:
            lock_stat = self.lock_path.stat()
            raw_lock = self.lock_path.read_bytes()
        except FileNotFoundError as exc:
            msg = f"Lock file {self.lock_path} vanished during acquisition; retry"
            raise LockUnavailableError(msg) from exc
        except OSError as exc:
            msg = f"Failed to read lock file {self.lock_path}: {exc}"
            raise LockUnavailableError(msg) from exc

        mtime_age = now - lock_stat.st_mtime
        try:
            lock_obj: CampaignLock | None = CampaignLock.from_json(raw_lock.decode("utf-8"))
        except Exception:
            # A crash between O_EXCL create and write leaves an empty or torn lock.
            # It can never be renewed (the heartbeat CAS needs parseable JSON), so
            # judge it by mtime alone against this worker's lease length.
            lock_obj = None

        # Dual Expiration & Clock Drift Check (Section 5.3)
        if lock_obj is not None:
            is_stale = (now >= lock_obj.lease_expires_at_unix_s + CLOCK_DRIFT_GRACE_PERIOD_S) and (
                mtime_age >= lock_obj.lease_seconds + CLOCK_DRIFT_GRACE_PERIOD_S
            )
        else:
            is_stale = mtime_age >= self.lease_seconds + CLOCK_DRIFT_GRACE_PERIOD_S

        if not is_stale:
            msg = (
                f"Lock on {self.lock_path} is currently active or within "
                f"{CLOCK_DRIFT_GRACE_PERIOD_S}s clock drift grace period"
            )
            raise LockUnavailableError(msg)

        # Step 3: Stale Lease Stealing with Atomic Steal Token
        steal_token = self.lock_path.with_name(f"{self.lock_path.name}.steal")
        steal_fd = self._open_steal_token(steal_token, now)
        token_body = self.owner_token.encode("utf-8")
        steal_tmp: Path | None = None

        def still_stealing() -> bool:
            """Our steal token is still in place and the lock is still the stale one."""
            try:
                if steal_token.read_bytes() != token_body:
                    return False
            except OSError:
                return False
            try:
                return self.lock_path.read_bytes() == raw_lock
            except OSError:
                return False

        try:
            os.write(steal_fd, token_body)
            os.fsync(steal_fd)

            # Re-verify lock is still the same stale lock (byte-for-byte CAS)
            if not still_stealing():
                msg = f"Lock on {self.lock_path} was modified before steal could complete"
                raise LockUnavailableError(msg)

            steal_acquired = now
            stolen_fence = lock_obj.fence if lock_obj is not None else 0
            fence = max(self._read_fence_file(), stolen_fence) + 1
            stolen_lock = CampaignLock(
                schema_version=CAMPAIGN_LOCK_SCHEMA_VERSION,
                owner_token=self.owner_token,
                manifest_row_hash=self.manifest_row_hash,
                acquired_at_unix_s=steal_acquired,
                lease_expires_at_unix_s=now + self.lease_seconds,
                lease_seconds=self.lease_seconds,
                heartbeat_count=0,
                fence=fence,
            )
            steal_suffix = f"steal.{uuid.uuid4().hex[:8]}.tmp"
            steal_tmp = self.lock_path.with_name(f"{self.lock_path.name}.{steal_suffix}")
            steal_tmp.write_text(stolen_lock.to_json(), encoding="utf-8")
            fsync_file(steal_tmp)

            # A stealer that stalled here past the token-reclaim age may have been
            # superseded by a later stealer (#3312); re-check right before the replace.
            if not still_stealing():
                msg = f"Steal of {self.lock_path} was superseded by another stealer"
                raise LockUnavailableError(msg)
            os.replace(steal_tmp, self.lock_path)
            steal_tmp = None
            fsync_directory(self.lock_path.parent)

            self.acquired_at_unix_s = steal_acquired
            self.fence = fence
            self.local_deadline = boot_now + self.lease_seconds
            self._raise_fence_file()
            self._clear_stale_commit_token()
            self._start_heartbeat(steal_acquired)
            self.is_locked = True
            return True
        finally:
            with suppress(OSError):
                os.close(steal_fd)
            if steal_tmp is not None:
                with suppress(OSError):
                    steal_tmp.unlink()
            # Remove the token only if it is still ours: a later stealer may have
            # reclaimed it, and deleting theirs would let a third stealer in.
            with suppress(OSError):
                if steal_token.read_bytes() == token_body:
                    steal_token.unlink()

    def _open_steal_token(self, steal_token: Path, now: float) -> int:
        """Create the singleton steal token, reclaiming one orphaned by a dead stealer.

        A steal completes in milliseconds, so a token older than the clock-drift grace
        period belongs to a stealer that died between create and unlink. Without this
        reclaim the unit would stay locked forever.
        """
        for attempt in range(2):
            try:
                return os.open(steal_token, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError as exc:
                try:
                    token_age = now - steal_token.stat().st_mtime
                except FileNotFoundError:
                    continue  # holder just finished; retry once
                if attempt == 0 and token_age >= CLOCK_DRIFT_GRACE_PERIOD_S:
                    with suppress(FileNotFoundError):
                        steal_token.unlink()
                    continue
                msg = f"Lost CAS race while attempting to steal lock on {self.lock_path}"
                raise LockUnavailableError(msg) from exc
        msg = f"Lost CAS race while attempting to steal lock on {self.lock_path}"
        raise LockUnavailableError(msg)

    def _clear_stale_commit_token(self) -> None:
        """Remove a commit token left by a previous lease holder.

        Called only after this worker holds the lease. A token whose owner is not this
        worker was written by a holder whose lease has since expired or been released
        (it crashed or was preempted between commit steps 3 and 8), so it is stale.
        Left in place it would make every later commit fail with FileExistsError.
        """
        commit_token_path = self.output_path.with_name(f"{self.output_path.name}.commit")
        try:
            raw = commit_token_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError:
            raw = ""
        try:
            owner = json.loads(raw).get("owner_token")
        except Exception:
            owner = None
        if owner != self.owner_token:
            with suppress(FileNotFoundError):
                commit_token_path.unlink()
            fsync_directory(commit_token_path.parent)

    def _start_heartbeat(self, acquired_at: float) -> None:
        self.heartbeat = HeartbeatThread(
            lock_path=self.lock_path,
            owner_token=self.owner_token,
            manifest_row_hash=self.manifest_row_hash,
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_interval,
            initial_acquired_at=acquired_at,
            expiry_guard_margin=self.expiry_guard_margin,
            fence=self.fence,
            local_deadline=self.local_deadline,
        )
        self.heartbeat.start()

    def release(self) -> None:
        """Stop heartbeat and unlink the lock file if it still carries this holder's triple."""
        self._stop_heartbeat()
        if self.is_locked:
            with suppress(Exception):
                self._unlink_lock_if_ours()
        self.is_locked = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def commit_output(
        self,
        partial_path: Path,
        output_path: Path,
        done_marker: "CampaignDoneMarker",
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
        8. Stop and join the heartbeat, then unlink .commit and .lock and durabilize the
           parent directory. The heartbeat stops first so no renewal can recreate .lock.
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
                "fence": self.fence,
                "committed_at_unix_s": time.time(),
            }
            os.write(fd, json.dumps(commit_meta, indent=2).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

        # 5. Lock Ownership Re-Verification (Zombie Worker Protection)
        try:
            self.assert_holder()
        except LeaseLostError:
            # The token may already be a stealer's (they clear ours on steal); remove it
            # only if it is still ours (#3312).
            self._unlink_commit_token_if_ours(commit_token_path)
            _clean_scratch(partial_path)
            msg = f"Lease for {output_path} was lost, stolen, or expired before commit"
            raise LeaseLostError(msg) from None

        marker = dataclasses.replace(
            done_marker, lease_fence=self.fence, owner_token=self.owner_token
        )
        tmp_done = output_path.with_name(f"{output_path.name}.done.json.tmp.{attempt_uuid}")
        try:
            # 6. Output Promotion. An existing store with no done marker (step 1 found
            # none) is an orphan from a holder that crashed between steps 6 and 7; set
            # it aside so os.replace does not fail with ENOTEMPTY on every retry. Each
            # destructive step re-checks the lease first (#3312).
            if output_path.exists():
                from xtrax.campaign.marker import quarantine_output

                self.assert_holder()
                quarantine_output(output_path, reason="orphaned_uncommitted_output")
            self.assert_holder()
            os.replace(partial_path, output_path)

            # 7. Write and Rename Done Marker
            tmp_done.write_text(marker.to_json(), encoding="utf-8")
            fsync_file(tmp_done)
            self.assert_holder()
            os.replace(tmp_done, done_marker_path)

            # 8. Directory Durabilization & Release
            fsync_directory(output_path.parent)
        except LeaseLostError:
            # Nothing of ours is promoted past the failed check; a promoted store with no
            # marker is set aside by the next holder's orphan handling.
            _clean_scratch(partial_path)
            _clean_scratch(tmp_done)
            raise
        finally:
            # The token must not outlive this attempt, success or failure; a stale token
            # blocks every later commit of this unit. Remove it only if it is still ours.
            self._unlink_commit_token_if_ours(commit_token_path)

        # A holder superseded between its last check and the marker rename (residual
        # window R1) reports a lost lease rather than success; the next resume verifies
        # the store against the marker.
        try:
            self.assert_holder()
            lost_after_commit = False
        except LeaseLostError:
            lost_after_commit = True
            logger.warning(
                "lease_lost_after_commit: %s committed under fence %d, unit .fence is now %d",
                output_path,
                self.fence,
                self._read_fence_file(),
            )

        # Stop the heartbeat before unlinking the lock. stop() waits out any renewal in
        # progress and bars later ones, so no renewal can recreate the lock afterwards.
        self._stop_heartbeat()
        self._unlink_lock_if_ours()
        fsync_directory(output_path.parent)
        self.is_locked = False

        if lost_after_commit:
            msg = f"Lease for {output_path} was lost during commit; resume will verify it"
            raise LeaseLostError(msg)
        return marker

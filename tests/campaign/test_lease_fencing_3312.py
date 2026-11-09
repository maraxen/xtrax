"""Debt #3312: a holder that stalls past its lease cannot clobber the worker that stole it.

Spec: ``.praxia/docs/specs/261108_campaign-lease-fencing-token.md``. Each race test parks one
worker at a precise point through a narrow hook, then simulates its lease running out by
rewriting its lock as expired and aged, so a second worker can steal it. No sleeps stand in
for the stall.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.campaign.unit_fns import array_unit
from xtrax.campaign import lease as lease_module
from xtrax.campaign.executors import local as local_module
from xtrax.campaign.executors.local import (
    UNIT_COMMITTED,
    _execute_single_unit,
    _reclaim_dead_worker_leases,
    _resolve_output_path,
    _stage_unit_output,
)
from xtrax.campaign.lease import (
    CampaignLock,
    FileLock,
    HeartbeatThread,
    LeaseLostError,
    LockUnavailableError,
)
from xtrax.campaign.manifest import ManifestRow, build_manifest_row
from xtrax.campaign.marker import compute_content_digest, create_done_marker

ROW = build_manifest_row(
    campaign_id="fence",
    job_id="j0",
    chunk_id=0,
    sample_start=0,
    sample_count=1,
    target_input="t.pdb",
    arm_id="a",
    arm_backend="b",
    checkpoint_id="c",
    seed=3,
)
LEASE = 600  # long enough that no heartbeat renews during a test (interval = 60 s)


# --- helpers ------------------------------------------------------------------------


class _Park:
    """One-shot rendezvous: the parked thread waits until the test calls ``resume``."""

    def __init__(self) -> None:
        self.parked = threading.Event()
        self.resume = threading.Event()

    def __call__(self) -> None:
        self.parked.set()
        assert self.resume.wait(10.0), "test never resumed the parked thread"


class _ParkingEvent(threading.Event):
    """An abort event whose first ``is_set`` from ``thread_name`` parks."""

    def __init__(self, park: _Park, thread_name: str) -> None:
        super().__init__()
        self._park = park
        self._thread_name = thread_name
        self._armed = True

    def is_set(self) -> bool:
        if self._armed and threading.current_thread().name == self._thread_name:
            self._armed = False
            self._park()
        return super().is_set()


def _run_in_thread(name: str, fn: Callable[[], Any]) -> tuple[threading.Thread, dict]:
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test via box
            box["error"] = exc

    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread, box


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "out"
    return output_dir, _resolve_output_path(output_dir, ROW)


def _expire_lock(lock_path: Path) -> None:
    """Make the current lock look like a lease that ran out long ago (the holder stalled)."""
    old = time.time() - 100_000
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    data["lease_expires_at_unix_s"] = old
    lock_path.write_text(json.dumps(data), encoding="utf-8")
    os.utime(lock_path, (old, old))


def _write_store(store: Path, tmp_path: Path, row: ManifestRow = ROW) -> None:
    """Write a valid zarr group at ``store`` the way the executor stages a unit."""
    _stage_unit_output(array_unit(row), tmp_path / "no-such-unit-store.zarr", store)


def _unit_snapshot(out: Path) -> dict[str, str]:
    marker = out.with_name(f"{out.name}.done.json")
    files = sorted(out.rglob("*")) if out.exists() else []
    snap = {
        str(p.relative_to(out.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file()
    }
    if marker.exists():
        snap[marker.name] = hashlib.sha256(marker.read_bytes()).hexdigest()
    return snap


def _owner(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["owner_token"]


def _holder_with_staged_partial(tmp_path: Path) -> tuple[FileLock, Path, Path, Any]:
    _, out = _paths(tmp_path)
    holder = FileLock(out, "holder:1:aaaa", ROW.manifest_row_hash, lease_seconds=LEASE)
    assert holder.acquire()
    partial = out.with_name(f"{out.name}.partial.h1")
    _stage_unit_output(array_unit(ROW), out, partial)
    marker = create_done_marker(
        manifest_row_hash=ROW.manifest_row_hash,
        input_hash=ROW.input_hash,
        output_locator=str(out),
        content_digest_sha256=compute_content_digest(partial),
        duration_s=1.0,
    )
    return holder, out, partial, marker


def _commit(holder: FileLock, partial: Path, out: Path, marker: Any) -> Callable[[], Any]:
    return lambda: holder.commit_output(
        partial_path=partial, output_path=out, done_marker=marker, attempt_uuid="h1"
    )


# --- interleaving A: stall between commit step 5 and step 6 -------------------------


def test_stalled_holder_cannot_quarantine_stealer_store(tmp_path: Path) -> None:
    output_dir, _ = _paths(tmp_path)
    holder, out, partial, marker = _holder_with_staged_partial(tmp_path)
    park = _Park()
    assert holder.heartbeat is not None
    holder.heartbeat.abort_event = _ParkingEvent(park, "holder")
    thread, box = _run_in_thread("holder", _commit(holder, partial, out, marker))
    try:
        assert park.parked.wait(5.0)
        _expire_lock(holder.lock_path)
        outcome = _execute_single_unit(ROW, array_unit, output_dir, owner_token_prefix="stealer")
        assert outcome == UNIT_COMMITTED
        stealer_unit = _unit_snapshot(out)
    finally:
        park.resume.set()
        thread.join(10.0)
        holder.release()

    assert isinstance(box.get("error"), LeaseLostError), box
    assert _unit_snapshot(out) == stealer_unit
    assert not (out.parent / ".quarantine").exists()


# --- interleaving B: stale unlinks of .lock and .commit -----------------------------


def test_stalled_holder_does_not_unlink_stealer_lock(tmp_path: Path) -> None:
    holder, out, partial, marker = _holder_with_staged_partial(tmp_path)
    park = _Park()
    heartbeat = holder.heartbeat
    assert heartbeat is not None
    real_stop = heartbeat.stop

    def parked_stop() -> None:
        park()
        real_stop()

    heartbeat.stop = parked_stop  # ty: ignore[invalid-assignment]
    thread, box = _run_in_thread("holder", _commit(holder, partial, out, marker))
    stealer = FileLock(out, "stealer:1:bbbb", ROW.manifest_row_hash, lease_seconds=LEASE)
    try:
        assert park.parked.wait(5.0)
        _expire_lock(holder.lock_path)
        assert stealer.acquire()
        park.resume.set()
        thread.join(10.0)
        assert box.get("error") is None or isinstance(box["error"], LeaseLostError), box
        assert holder.lock_path.exists()
        assert _owner(holder.lock_path) == stealer.owner_token
    finally:
        park.resume.set()
        stealer.release()


def test_lost_lease_path_does_not_unlink_stealer_commit_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder, out, partial, marker = _holder_with_staged_partial(tmp_path)
    park = _Park()
    real_fsync = os.fsync
    armed = [True]

    def parking_fsync(fd: int) -> None:
        if armed[0] and threading.current_thread().name == "holder":
            armed[0] = False
            park()
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", parking_fsync)  # step 4: the commit token fsync
    thread, box = _run_in_thread("holder", _commit(holder, partial, out, marker))
    stealer = FileLock(out, "stealer:1:bbbb", ROW.manifest_row_hash, lease_seconds=LEASE)
    commit_token = out.with_name(f"{out.name}.commit")
    try:
        assert park.parked.wait(5.0)
        _expire_lock(holder.lock_path)
        assert stealer.acquire()
        commit_token.write_text(
            json.dumps({"attempt_uuid": "s1", "owner_token": stealer.owner_token}),
            encoding="utf-8",
        )
        park.resume.set()
        thread.join(10.0)
        assert isinstance(box.get("error"), LeaseLostError), box
        assert commit_token.exists()
        assert _owner(commit_token) == stealer.owner_token
    finally:
        park.resume.set()
        stealer.release()
        holder.release()


# --- interleaving C and findings 4, 5: executor-side destructive steps --------------


def _stealer_takes_over_with_store(out: Path, tmp_path: Path) -> FileLock:
    lock_path = out.with_name(f"{out.name}.lock")
    _expire_lock(lock_path)
    stealer = FileLock(out, "stealer:1:bbbb", ROW.manifest_row_hash, lease_seconds=LEASE)
    assert stealer.acquire()
    _write_store(out, tmp_path / "stealer")
    return stealer


def _run_parked_unit(
    tmp_path: Path, unit_fn: Callable[[ManifestRow], Any], park: _Park
) -> tuple[Path, dict, FileLock]:
    output_dir, out = _paths(tmp_path)
    thread, box = _run_in_thread(
        "holder",
        lambda: _execute_single_unit(
            ROW, unit_fn, output_dir, lease_seconds=LEASE, owner_token_prefix="holder"
        ),
    )
    try:
        assert park.parked.wait(5.0)
        stealer = _stealer_takes_over_with_store(out, tmp_path)
        before = _unit_snapshot(out)
    finally:
        park.resume.set()
        thread.join(10.0)
    box["stealer_before"] = before
    return out, box, stealer


def test_abort_cleanup_does_not_quarantine_stealer_store(tmp_path: Path) -> None:
    park = _Park()

    def unit_fn(row: ManifestRow) -> Any:
        park()
        raise RuntimeError("unit failed after stalling")

    out, box, stealer = _run_parked_unit(tmp_path, unit_fn, park)
    try:
        assert isinstance(box.get("error"), RuntimeError), box
        assert _unit_snapshot(out) == box["stealer_before"]
        assert not (out.parent / ".quarantine").exists()
        assert _owner(stealer.lock_path) == stealer.owner_token
    finally:
        stealer.release()


def test_stalled_holder_cannot_set_aside_stealer_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    park = _Park()
    real_check = local_module._check_resume_done
    armed = [True]

    def parking_check(*args: Any, **kwargs: Any) -> bool:
        if armed[0] and threading.current_thread().name == "holder":
            armed[0] = False
            park()
            return False
        return real_check(*args, **kwargs)

    monkeypatch.setattr(local_module, "_check_resume_done", parking_check)
    out, box, stealer = _run_parked_unit(tmp_path, array_unit, park)
    try:
        assert isinstance(box.get("error"), LeaseLostError), box
        assert _unit_snapshot(out) == box["stealer_before"]
        assert not (out.parent / ".quarantine").exists()
    finally:
        stealer.release()


def test_stalled_holder_cannot_stage_away_stealer_store(tmp_path: Path) -> None:
    park = _Park()

    def unit_fn(row: ManifestRow) -> Any:
        state = array_unit(row)
        park()
        return state

    out, box, stealer = _run_parked_unit(tmp_path, unit_fn, park)
    try:
        assert isinstance(box.get("error"), LeaseLostError), box
        assert _unit_snapshot(out) == box["stealer_before"]
    finally:
        stealer.release()


# --- interleaving D: heartbeat replace after a steal --------------------------------


def test_heartbeat_replace_after_steal_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "unit.zarr.lock"
    now = time.time()
    holder_token = "holder:1:aaaa"
    lock_path.write_text(
        CampaignLock(
            schema_version="campaign_lock_v1",
            owner_token=holder_token,
            manifest_row_hash=ROW.manifest_row_hash,
            acquired_at_unix_s=now,
            lease_expires_at_unix_s=now + LEASE,
            lease_seconds=LEASE,
        ).to_json(),
        encoding="utf-8",
    )
    park = _Park()
    real_fsync_file = lease_module.fsync_file
    armed = [True]

    def parking_fsync_file(path: Path) -> None:
        if armed[0] and threading.current_thread().name.startswith("Heartbeat"):
            armed[0] = False
            park()
        real_fsync_file(path)

    monkeypatch.setattr(lease_module, "fsync_file", parking_fsync_file)
    hb = HeartbeatThread(
        lock_path=lock_path,
        owner_token=holder_token,
        manifest_row_hash=ROW.manifest_row_hash,
        lease_seconds=LEASE,
        interval_seconds=0.02,
        initial_acquired_at=now,
    )
    hb.start()
    try:
        assert park.parked.wait(5.0)
        stolen = CampaignLock(
            schema_version="campaign_lock_v1",
            owner_token="stealer:1:bbbb",
            manifest_row_hash=ROW.manifest_row_hash,
            acquired_at_unix_s=now + 1,
            lease_expires_at_unix_s=now + 1 + LEASE,
            lease_seconds=LEASE,
        ).to_dict()
        lock_path.write_text(json.dumps(stolen), encoding="utf-8")
        park.resume.set()
        assert hb.abort_event.wait(5.0) or not hb.is_alive()
    finally:
        park.resume.set()
        hb.stop()
        hb.join(2.0)
    assert _owner(lock_path) == "stealer:1:bbbb"


# --- finding 6: a stalled stealer -----------------------------------------------------


def test_stalled_stealer_cannot_clobber_later_stealer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out = _paths(tmp_path)
    out.parent.mkdir(parents=True)
    lock_path = out.with_name(f"{out.name}.lock")
    now = time.time()
    lock_path.write_text(
        CampaignLock(
            schema_version="campaign_lock_v1",
            owner_token="dead:1:dddd",
            manifest_row_hash=ROW.manifest_row_hash,
            acquired_at_unix_s=now - 100_000,
            lease_expires_at_unix_s=now - 100_000,
            lease_seconds=LEASE,
        ).to_json(),
        encoding="utf-8",
    )
    os.utime(lock_path, (now - 100_000, now - 100_000))

    park = _Park()
    real_fsync_file = lease_module.fsync_file
    armed = [True]

    def parking_fsync_file(path: Path) -> None:
        if armed[0] and threading.current_thread().name == "s1" and ".steal." in path.name:
            armed[0] = False
            park()
        real_fsync_file(path)

    monkeypatch.setattr(lease_module, "fsync_file", parking_fsync_file)
    s1 = FileLock(out, "s1:1:aaaa", ROW.manifest_row_hash, lease_seconds=LEASE)
    s2 = FileLock(out, "s2:1:bbbb", ROW.manifest_row_hash, lease_seconds=LEASE)
    thread, box = _run_in_thread("s1", s1.acquire)
    try:
        assert park.parked.wait(5.0)
        token = lock_path.with_name(f"{lock_path.name}.steal")
        os.utime(token, (now - 100_000, now - 100_000))
        assert s2.acquire()
        park.resume.set()
        thread.join(10.0)
        assert isinstance(box.get("error"), LockUnavailableError), box
        assert _owner(lock_path) == s2.owner_token
    finally:
        park.resume.set()
        s2.release()
        s1.release()


# --- reclaim ------------------------------------------------------------------------


def test_reclaim_stops_when_lock_is_rewritten_mid_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, out = _paths(tmp_path)
    out.parent.mkdir(parents=True)
    lock_path = out.with_name(f"{out.name}.lock")
    now = time.time()
    lock_path.write_text(
        json.dumps({"owner_token": "run1:123:aaaa", "acquired_at_unix_s": now}),
        encoding="utf-8",
    )
    for suffix in ("a", "b"):
        partial = out.with_name(f"{out.name}.partial.{suffix}")
        partial.mkdir()
        (partial / "chunk").write_bytes(b"x")

    real_rmtree = shutil.rmtree
    calls = [0]

    def rewriting_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        calls[0] += 1
        if calls[0] == 1:  # another run steals the unit while partials are being removed
            lock_path.write_text(
                json.dumps({"owner_token": "other:9:zzzz", "acquired_at_unix_s": now + 1}),
                encoding="utf-8",
            )
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", rewriting_rmtree)
    reclaimed = _reclaim_dead_worker_leases(output_dir, [ROW], "run1")

    assert reclaimed == []
    assert _owner(lock_path) == "other:9:zzzz"
    assert len(list(out.parent.glob(f"{out.name}.partial.*"))) == 1

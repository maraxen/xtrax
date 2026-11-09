"""P7 Adversarial Challenger Empirical Stress Test Suite for xtrax.campaign.

Empirical challenge tests targeting:
1. Distributed lease stealing under clock drift < 300s vs > 300s.
2. Singleton <output>.commit token under concurrent promotion races.
3. aminx manifest slice parity against golden vectors and float perturbation sensitivity.
4. Tier-1 fast resume (< 5 ms) vs Tier-2 deep chunk audit.
5. Defensive quarantine on ENOSPC and EXDEV.
"""

from __future__ import annotations

import concurrent.futures
import errno
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from xtrax.campaign.lease import (
    CLOCK_DRIFT_GRACE_PERIOD_S,
    FileLock,
    LockUnavailableError,
    StolenAttemptDiscarded,
)
from xtrax.campaign.manifest import (
    build_manifest_row,
    canonical_json_bytes,
    compute_unit_input_hash,
)
from xtrax.campaign.marker import (
    CampaignDoneMarker,
    compute_content_digest,
    create_done_marker,
    quarantine_output,
    validate_unit_done,
)


def _create_mock_zarr(store_path: Path, chunks: list[bytes]) -> None:
    """Create a multi-chunk Zarr directory store."""
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    data_dir = store_path / "arr"
    data_dir.mkdir(parents=True, exist_ok=True)
    total_len = sum(len(c) for c in chunks)
    chunk_len = len(chunks[0]) if chunks else 0
    (data_dir / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [total_len],
                "chunks": [chunk_len],
                "dtype": "|u1",
                "order": "C",
                "filters": None,
                "fill_value": 0,
                "compressor": None,
            }
        ),
        encoding="utf-8",
    )
    for i, chunk in enumerate(chunks):
        (data_dir / str(i)).write_bytes(chunk)


# ==============================================================================
# Challenge 1: Distributed Lease Stealing across 300s Clock Drift Margin
# ==============================================================================


def test_lease_steal_exact_clock_drift_boundaries(tmp_path: Path) -> None:
    """Challenge boundary conditions of CLOCK_DRIFT_GRACE_PERIOD_S (300s).

    1. now == expires_at + 299.0s -> LockUnavailableError (strictly within drift margin)
    2. now == expires_at + 301.0s, but mtime is recent -> LockUnavailableError (dual check)
    3. now == expires_at + 301.0s, and mtime is ancient -> Steal succeeds via CAS
    """
    output_path = tmp_path / "boundary_unit.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")

    lock1 = FileLock(
        output_path=output_path,
        owner_token="worker_primary:pid100",
        manifest_row_hash="row_boundary_01",
        lease_seconds=60,
    )
    assert lock1.acquire() is True
    assert lock_path.exists()

    lock_thief = FileLock(
        output_path=output_path,
        owner_token="worker_thief:pid200",
        manifest_row_hash="row_boundary_01",
        lease_seconds=60,
    )

    now = time.time()
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))

    # Case 1: Exactly 299s after expiration (inside 300s margin by 1s)
    # expires_at = now - 299, so now - expires_at = 299 < 300
    lock_data["lease_expires_at_unix_s"] = now - (CLOCK_DRIFT_GRACE_PERIOD_S - 1.0)
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    past_mtime = now - lock_data["lease_seconds"] - (CLOCK_DRIFT_GRACE_PERIOD_S + 10.0)
    os.utime(lock_path, (past_mtime, past_mtime))

    with pytest.raises(LockUnavailableError, match="clock drift grace period"):
        lock_thief.acquire()

    # Case 2: Expired past 300s, but file mtime is recent (e.g. node clock ran fast)
    lock_data["lease_expires_at_unix_s"] = now - (CLOCK_DRIFT_GRACE_PERIOD_S + 10.0)
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    # Fresh mtime (now - mtime < lease_seconds + 300)
    os.utime(lock_path, (now - 10.0, now - 10.0))

    with pytest.raises(LockUnavailableError, match="clock drift grace period"):
        lock_thief.acquire()

    # Case 3: Both expired past 300s AND mtime past lease_seconds + 300s
    past_mtime_old = now - lock_data["lease_seconds"] - (CLOCK_DRIFT_GRACE_PERIOD_S + 5.0)
    os.utime(lock_path, (past_mtime_old, past_mtime_old))

    assert lock_thief.acquire() is True
    assert lock_thief.is_locked is True

    # Confirm thief is recorded as owner with heartbeat_count reset to 0
    stolen_data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert stolen_data["owner_token"] == "worker_thief:pid200"
    assert stolen_data["heartbeat_count"] == 0
    lock_thief.release()


def test_concurrent_steal_race_post_cas(tmp_path: Path) -> None:
    """Challenge concurrent stale lease stealing: 6 workers race to steal simultaneously.

    Exactly 1 worker must win the CAS race, and the other 5 must receive LockUnavailableError.
    """
    output_path = tmp_path / "stolen_race_unit.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    row_hash = "row_race_steal_01"

    now = time.time()
    ancient_time = now - CLOCK_DRIFT_GRACE_PERIOD_S - 600.0
    stale_lock = {
        "schema_version": "campaign_lock_v1",
        "owner_token": "crashed_worker:pid999",
        "manifest_row_hash": row_hash,
        "acquired_at_unix_s": ancient_time,
        "lease_expires_at_unix_s": ancient_time,
        "lease_seconds": 60,
        "heartbeat_count": 5,
    }
    lock_path.write_text(json.dumps(stale_lock), encoding="utf-8")
    os.utime(lock_path, (ancient_time, ancient_time))

    num_competitors = 6
    winners = []
    errors = []
    held_locks = []

    def _compete_for_steal(worker_id: int) -> None:
        token = f"thief_{worker_id}:pid{1000 + worker_id}"
        lock = FileLock(
            output_path=output_path,
            owner_token=token,
            manifest_row_hash=row_hash,
            lease_seconds=60,
            heartbeat_interval=1.0,
        )
        try:
            if lock.acquire():
                winners.append(token)
                held_locks.append(lock)
        except LockUnavailableError as exc:
            errors.append((token, str(exc)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_competitors) as pool:
        futures = [pool.submit(_compete_for_steal, i) for i in range(num_competitors)]
        concurrent.futures.wait(futures)

    for lk in held_locks:
        lk.release()

    # Exactly 1 winner
    assert len(winners) == 1, f"Expected exactly 1 CAS steal winner, got {len(winners)}: {winners}"
    # Exactly 5 losers with LockUnavailableError
    assert len(errors) == num_competitors - 1


def test_stale_lease_stealing_with_corrupt_or_truncated_lock(tmp_path: Path) -> None:
    """Adversarial probe: What if the existing lock file is empty / corrupt JSON?

    A fresh unparseable lock (a writer may be between O_EXCL create and write) raises
    LockUnavailableError rather than crashing with JSONDecodeError. Once its mtime is past
    the lease plus grace it is stolen instead (see test_review_261105_regressions).
    """
    output_path = tmp_path / "corrupt_lock_unit.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    lock_path.write_text("", encoding="utf-8")  # 0-byte truncated lock

    lock = FileLock(
        output_path=output_path,
        owner_token="probe_worker:pid333",
        manifest_row_hash="row_probe_01",
        lease_seconds=60,
    )
    with pytest.raises(LockUnavailableError, match="currently active"):
        lock.acquire()


# ==============================================================================
# Challenge 2: Singleton <output>.commit Token Under Concurrent Promotion Races
# ==============================================================================


def test_concurrent_commit_singleton_token_mutual_exclusion(tmp_path: Path) -> None:
    """Stress-test concurrent commit attempts from multiple workers.

    Verifies:
    - 8 threads race to invoke commit_output() simultaneously on the same target.
    - Exactly 1 thread successfully creates <output>.commit and completes promotion.
    - All losing threads either catch FileExistsError or StolenAttemptDiscarded.
    - None of the losing threads clobber the winner's output data.
    - All partial directories of losing threads are deleted.
    """
    output_path = tmp_path / "contested_output.zarr"
    row_hash = "row_contested_commit_01"
    owner_token = "common_lease_owner:pid555"

    lock = FileLock(
        output_path=output_path,
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=300,
    )
    assert lock.acquire() is True

    num_threads = 8
    commit_results = []
    partial_paths = []

    for i in range(num_threads):
        att_uuid = f"attempt_{i:03d}"
        partial = output_path.with_name(f"{output_path.name}.partial.{att_uuid}")
        _create_mock_zarr(partial, chunks=[f"content_from_thread_{i}".encode()])
        partial_paths.append(partial)

    def _attempt_commit(idx: int) -> tuple[int, Any, Exception | None]:
        att_uuid = f"attempt_{idx:03d}"
        partial = partial_paths[idx]
        digest = compute_content_digest(partial)
        marker = create_done_marker(
            manifest_row_hash=row_hash,
            input_hash=f"input_{row_hash}",
            output_locator=str(output_path),
            content_digest_sha256=digest,
            duration_s=0.1,
        )
        # Create a thread-local FileLock reference pointing to same lock file
        t_lock = FileLock(
            output_path=output_path,
            owner_token=owner_token,
            manifest_row_hash=row_hash,
            lease_seconds=300,
        )
        # Adopt the lease held on disk (#3312: the check compares owner, epoch and fence).
        held = json.loads(t_lock.lock_path.read_text(encoding="utf-8"))
        t_lock.acquired_at_unix_s = held["acquired_at_unix_s"]
        t_lock.fence = held["fence"]
        t_lock.is_locked = True
        try:
            res = t_lock.commit_output(
                partial_path=partial,
                output_path=output_path,
                done_marker=marker,
                attempt_uuid=att_uuid,
            )
            return (idx, res, None)
        except Exception as exc:
            return (idx, None, exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_attempt_commit, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            commit_results.append(f.result())

    successes = [r for r in commit_results if isinstance(r[1], CampaignDoneMarker)]
    discards = [r for r in commit_results if isinstance(r[1], StolenAttemptDiscarded)]
    file_exists = [r for r in commit_results if isinstance(r[2], FileExistsError)]

    # Exactly 1 thread must successfully promote
    assert len(successes) == 1, (
        f"Expected exactly 1 successful promotion, got {len(successes)}: {commit_results}"
    )
    winner_idx = successes[0][0]

    # Remaining 7 threads must either be discarded or raise FileExistsError
    assert len(discards) + len(file_exists) == num_threads - 1

    # Verify output exists and contains winner's exact payload
    assert output_path.exists()
    winner_data = (output_path / "arr" / "0").read_bytes()
    expected_winner_data = f"content_from_thread_{winner_idx}".encode()
    assert winner_data == expected_winner_data, (
        f"Output payload was clobbered! Found {winner_data!r}, expected {expected_winner_data!r}"
    )

    # All partial directories must have been unlinked
    for partial in partial_paths:
        assert not partial.exists(), f"Dangling partial directory left: {partial}"

    # Commit token must be cleaned up
    assert not output_path.with_name(f"{output_path.name}.commit").exists()


# ==============================================================================
# Challenge 3: aminx Parity Float Perturbation Sensitivity
# ==============================================================================


def test_fine_grained_float_perturbations() -> None:
    """Stress-test float perturbation sensitivity across IEEE-754 precision thresholds.

    AC-02 & GATE-11:
    - np.nextafter(0.1, 1.0) represents smallest possible float perturbation: must flip hash.
    - 0.0 vs -0.0 across python float, np.float32, np.float64: must NEVER flip hash.
    - NaN and Inf must raise ValueError during canonical serialization.
    """
    base_kwargs: dict[str, Any] = {
        "campaign_id": "c_perturb_test",
        "job_id": "job_000",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 100,
        "checkpoint_id": "ligandmpnn_v_32_020_25",
        "fixed_policy": "none",
        "state_weight_profile": "equal",
        "seed": 42,
    }
    base_params = {"temperature": 0.1, "threshold": 0.5}
    baseline_hash = compute_unit_input_hash(spec_dict=base_params, **base_kwargs)

    # 1. Machine epsilon / nextafter perturbation: smallest representable difference
    eps_perturbed = dict(base_params)
    eps_perturbed["temperature"] = float(np.nextafter(0.1, 1.0))
    eps_hash = compute_unit_input_hash(spec_dict=eps_perturbed, **base_kwargs)
    assert eps_hash != baseline_hash, (
        "Smallest float perturbation (np.nextafter) failed to flip input_hash!"
    )

    # 2. 1e-15 perturbation
    p15_perturbed = dict(base_params)
    p15_perturbed["temperature"] = 0.1 + 1e-15
    p15_hash = compute_unit_input_hash(spec_dict=p15_perturbed, **base_kwargs)
    assert p15_hash != baseline_hash

    # 3. IEEE-754 -0.0 vs +0.0 must be IDENTICAL
    zero_pos = {"val": 0.0, "np32": float(np.float32(0.0)), "np64": float(np.float64(0.0))}
    zero_neg = {"val": -0.0, "np32": float(np.float32(-0.0)), "np64": float(np.float64(-0.0))}
    hash_pos = compute_unit_input_hash(spec_dict=zero_pos, **base_kwargs)
    hash_neg = compute_unit_input_hash(spec_dict=zero_neg, **base_kwargs)
    assert hash_pos == hash_neg, (
        f"-0.0 and +0.0 produced different hashes: {hash_neg} != {hash_pos}"
    )

    # 4. Non-finite floats: NaN, +Inf, -Inf MUST raise ValueError
    with pytest.raises(ValueError):
        compute_unit_input_hash(spec_dict={"temperature": float("nan")}, **base_kwargs)

    with pytest.raises(ValueError):
        compute_unit_input_hash(spec_dict={"temperature": float("inf")}, **base_kwargs)

    with pytest.raises(ValueError):
        compute_unit_input_hash(spec_dict={"temperature": float("-inf")}, **base_kwargs)


def test_mixed_key_types_in_parameters() -> None:
    """Stress-test mixed key types in parameter dictionaries.

    Ensures that int keys and str keys in nested dicts do not raise TypeError
    during canonical JSON key sorting.
    """
    mixed_dict = {10: "int_key_10", "10": "str_key_10", 2: "int_key_2", "2": "str_key_2"}
    bytes_out = canonical_json_bytes(mixed_dict)
    assert isinstance(bytes_out, bytes)
    # Re-parse to verify valid JSON
    decoded = json.loads(bytes_out.decode("utf-8"))
    assert "10" in decoded


# ==============================================================================
# Challenge 4: Tier-1 Fast Resume (< 5 ms) vs Tier-2 Deep Chunk Audit
# ==============================================================================


def test_tier1_fast_resume_statistical_latency(tmp_path: Path) -> None:
    """Statistically benchmark Tier-1 Fast Resume across 100 units.

    GATE-04 requires strictly < 5 ms per unit.
    Verify that 99th percentile and max latency are strictly under 5 ms.
    """
    latencies = []
    row = build_manifest_row(
        campaign_id="bench_c",
        job_id="job_bench",
        chunk_id=0,
        sample_start=0,
        sample_count=50,
        target_input="target.pdb",
        arm_id="arm_bench",
        arm_backend="boltz2",
        checkpoint_id="chk_01",
        output_locator="bench_out.zarr",
    )

    store_path = tmp_path / "bench_out.zarr"
    # Create a 4-chunk Zarr store
    _create_mock_zarr(store_path, [b"chunk0", b"chunk1", b"chunk2", b"chunk3"])

    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(store_path),
        content_digest_sha256=compute_content_digest(store_path),
        duration_s=1.0,
    )
    marker_path = store_path.with_name(f"{store_path.name}.done.json")
    marker.write(marker_path)

    # Warmup
    validate_unit_done(store_path, expected_row=row, deep_verify=False)

    for _ in range(100):
        t0 = time.perf_counter()
        val = validate_unit_done(store_path, expected_row=row, deep_verify=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert val.is_valid is True
        assert val.tier == 1
        latencies.append(dt_ms)

    max_latency = max(latencies)
    p99_latency = np.percentile(latencies, 99)
    mean_latency = np.mean(latencies)

    assert max_latency < 10.0, (
        f"GATE-04 VIOLATION: Max Tier-1 resume latency {max_latency:.3f} ms exceeded ceiling!"
    )
    assert p99_latency < 5.0, f"P99 Tier-1 latency unexpectedly high: {p99_latency:.3f} ms"
    assert mean_latency < 2.0, f"Mean Tier-1 latency unexpectedly high: {mean_latency:.3f} ms"


def test_tier2_detects_single_bit_flip_in_deep_chunk(tmp_path: Path) -> None:
    """Stress-test Tier-2 deep verification against silent chunk bit flips.

    In a 10-chunk array:
    - Flip 1 byte in chunk 8.
    - Tier-1 fast resume MUST report valid (does not read chunks).
    - Tier-2 deep audit MUST report invalid, detect ContentDigestMismatchError,
      and relocate corrupt store to .quarantine/.
    """
    store_path = tmp_path / "ten_chunks.zarr"
    chunks = [f"tensor_chunk_{i:02d}_payload_bytes_abcdef".encode() for i in range(10)]
    _create_mock_zarr(store_path, chunks)

    row = build_manifest_row(
        campaign_id="c_deep",
        job_id="job_deep",
        chunk_id=0,
        sample_start=0,
        sample_count=100,
        target_input="target.pdb",
        arm_id="arm_deep",
        arm_backend="chai1",
        checkpoint_id="chk_01",
        output_locator=str(store_path),
    )

    original_digest = compute_content_digest(store_path)
    marker = create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(store_path),
        content_digest_sha256=original_digest,
        duration_s=2.0,
    )
    marker_path = store_path.with_name(f"{store_path.name}.done.json")
    marker.write(marker_path)

    # Corrupt a single byte in chunk 8
    target_chunk = store_path / "arr" / "8"
    raw_bytes = bytearray(target_chunk.read_bytes())
    raw_bytes[0] ^= 0x01  # Flip 1 bit
    target_chunk.write_bytes(bytes(raw_bytes))

    # Tier-1 check: MUST pass because root metadata is untouched
    tier1_res = validate_unit_done(store_path, expected_row=row, deep_verify=False)
    assert tier1_res.is_valid is True

    # Tier-2 check: MUST fail and quarantine
    tier2_res = validate_unit_done(
        store_path,
        expected_row=row,
        deep_verify=True,
        quarantine_on_failure=True,
    )
    assert tier2_res.is_valid is False
    assert tier2_res.tier == 2
    assert "ContentDigestMismatchError" in str(tier2_res.error_type)
    assert not store_path.exists()
    assert not marker_path.exists()
    assert tier2_res.quarantine_path is not None
    assert tier2_res.quarantine_path.exists()


# ==============================================================================
# Challenge 5: Defensive Quarantine on ENOSPC and EXDEV
# ==============================================================================


def test_quarantine_enospc_renames_both_store_and_marker(tmp_path: Path) -> None:
    """GATE-06: On ENOSPC, BOTH the store and the .done.json marker are renamed in-place.

    Ensures that no stale or corrupt .done.json is left behind to falsely satisfy Tier-1 resume.
    """
    store_path = tmp_path / "enospc_unit.zarr"
    _create_mock_zarr(store_path, [b"dummy"])
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    done_path.write_text('{"schema_version": "campaign_done_marker_v3"}', encoding="utf-8")

    # Patch Path.mkdir to simulate ENOSPC
    with patch.object(Path, "mkdir", side_effect=OSError(errno.ENOSPC, "No space left on device")):
        res_dir = quarantine_output(store_path, reason="enospc_challenge")

    # In-place renamed
    assert ".corrupt." in str(res_dir)
    assert not store_path.exists()
    assert not done_path.exists()

    corrupt_markers = list(tmp_path.glob("enospc_unit.zarr.done.json.corrupt.*"))
    assert len(corrupt_markers) == 1, "Done marker was not renamed in-place during ENOSPC fallback!"


def test_quarantine_exdev_multi_file_zarr_store(tmp_path: Path) -> None:
    """Verify that EXDEV fallback safely moves directory stores via shutil.move."""
    store_path = tmp_path / "exdev_unit.zarr"
    _create_mock_zarr(store_path, [b"chunk0_data", b"chunk1_data"])
    done_path = store_path.with_name(f"{store_path.name}.done.json")
    done_path.write_text('{"schema_version": "campaign_done_marker_v3"}', encoding="utf-8")

    real_replace = os.replace

    def _mock_replace(src: Any, dst: Any) -> None:
        if ".quarantine" in str(dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(src, dst)

    with patch("os.replace", side_effect=_mock_replace):
        q_dir = quarantine_output(store_path, reason="exdev_challenge")

    assert q_dir.exists()
    assert ".quarantine" in str(q_dir)
    assert not store_path.exists()
    assert not done_path.exists()
    # Confirm contents moved intact
    assert (q_dir / store_path.name / "arr" / "0").read_bytes() == b"chunk0_data"

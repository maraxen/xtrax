"""Tests for multi-replica and bucketing fields in ProbeRecord (Debt #3196)."""

from __future__ import annotations

import json

import pytest

from xtrax.profiling import ClaimValidityError, ProbeRecord


def _valid_kwargs(**overrides):
    base = {
        "probe_id": "test_replica_probe",
        "stage": 1,
        "n_atoms": 128,
        "platform": "cpu",
        "metrics": {"step_seconds": 0.005},
        "git_sha": "abc1234",
        "timestamp": "2026-11-10T12:00:00Z",
        "x64_enabled": False,
        "jax_version": "0.10.2",
        "jaxlib_version": "0.10.2",
        "xla_flags": "",
        "device_kind": None,
    }
    base.update(overrides)
    return base


def test_probe_record_multi_replica_valid_roundtrip():
    """Positive control: construct, serialize, and deserialize multi-replica and bucket fields."""
    rec = ProbeRecord(
        **_valid_kwargs(
            num_replicas=4,
            replica_id=2,
            bucket_index=1,
            bucket_capacity=256,
        )
    )
    assert rec.num_replicas == 4
    assert rec.replica_id == 2
    assert rec.bucket_index == 1
    assert rec.bucket_capacity == 256

    # JSON round-trip
    payload = rec.to_json()
    loaded = ProbeRecord.from_json(payload)
    assert loaded == rec
    assert loaded.num_replicas == 4
    assert loaded.replica_id == 2
    assert loaded.bucket_index == 1
    assert loaded.bucket_capacity == 256


def test_probe_record_legacy_json_defaults_gracefully():
    """Positive control: legacy records omitting the new fields default cleanly."""
    rec = ProbeRecord(**_valid_kwargs())
    payload = json.loads(rec.to_json())
    # Remove the 4 new fields to simulate legacy JSON
    payload.pop("num_replicas", None)
    payload.pop("replica_id", None)
    payload.pop("bucket_index", None)
    payload.pop("bucket_capacity", None)

    loaded = ProbeRecord.from_json(json.dumps(payload))
    assert loaded.num_replicas == 1
    assert loaded.replica_id is None
    assert loaded.bucket_index is None
    assert loaded.bucket_capacity is None


@pytest.mark.parametrize(
    "bad_replicas",
    [0, -1, True, False, 1.5, "4"],
)
def test_probe_record_rejects_invalid_num_replicas(bad_replicas):
    """Negative control: num_replicas must be integer >= 1 (no bool)."""
    with pytest.raises(ClaimValidityError, match="num_replicas"):
        ProbeRecord(**_valid_kwargs(num_replicas=bad_replicas))


@pytest.mark.parametrize(
    ("bad_replica_id", "num_replicas"),
    [
        (-1, 4),
        (4, 4),
        (5, 4),
        (True, 4),
        (1, 1),  # When num_replicas == 1, replica_id must be None or 0
        ("2", 4),
    ],
)
def test_probe_record_rejects_invalid_replica_id(bad_replica_id, num_replicas):
    """Negative control: replica_id must satisfy 0 <= replica_id < num_replicas."""
    with pytest.raises(ClaimValidityError, match="replica_id"):
        ProbeRecord(**_valid_kwargs(num_replicas=num_replicas, replica_id=bad_replica_id))


@pytest.mark.parametrize(
    "bad_bucket_index",
    [-1, True, False, 1.5, "0"],
)
def test_probe_record_rejects_invalid_bucket_index(bad_bucket_index):
    """Negative control: bucket_index must be integer >= 0 (no bool)."""
    with pytest.raises(ClaimValidityError, match="bucket_index"):
        ProbeRecord(**_valid_kwargs(bucket_index=bad_bucket_index))


@pytest.mark.parametrize(
    "bad_bucket_capacity",
    [0, -1, True, False, 128.0, "256"],
)
def test_probe_record_rejects_invalid_bucket_capacity(bad_bucket_capacity):
    """Negative control: bucket_capacity must be integer >= 1 (no bool)."""
    with pytest.raises(ClaimValidityError, match="bucket_capacity"):
        ProbeRecord(**_valid_kwargs(bucket_capacity=bad_bucket_capacity))


def test_probe_record_rejects_bucket_index_exceeding_capacity():
    """Negative control: bucket_index > bucket_capacity violates ladder bound."""
    with pytest.raises(ClaimValidityError, match="bucket"):
        ProbeRecord(**_valid_kwargs(bucket_index=512, bucket_capacity=256))

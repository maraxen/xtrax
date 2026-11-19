"""Review round 2 regressions: validate_unit_done defaults and done-marker producer fields.

Covers review findings B (validate_unit_done is read-only by default) and C (done markers
record the installed xtrax version and an explicit or "unknown" git sha, never "clean").
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from tests.campaign.executors.test_local import _create_mock_manifest
from xtrax.campaign.executors.local import LocalSequentialExecutor, _resolve_output_path
from xtrax.campaign.marker import (
    CampaignDoneMarker,
    ProducerInfo,
    compute_content_digest,
    create_done_marker,
    validate_unit_done,
)


def _installed_version() -> str:
    return f"xtrax@{importlib.metadata.version('xtrax')}"


def _unit_fn(row: Any) -> dict[str, Any]:
    return {"values": np.arange(4, dtype=np.float32) + float(row.seed)}


def test_validate_unit_done_defaults_leave_mismatched_store_in_place(tmp_path: Path) -> None:
    """A read-only check with default flags must not move a store whose marker mismatches."""
    manifest = _create_mock_manifest(num_rows=1)
    row = manifest.rows[0]
    store = tmp_path / "unit_mismatch.zarr"
    zarr.open_group(str(store), mode="w", zarr_format=3).attrs["probe"] = 1

    marker = CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash="not_the_row_hash",
        input_hash=row.input_hash,
        producer=ProducerInfo(version="xtrax@test", git_sha="test", hostname="localhost"),
        output_locator=str(store),
        content_digest_sha256=compute_content_digest(store),
        completed_at_utc="2026-11-06T00:00:00Z",
        completed_at_unix_s=0.0,
        duration_s=0.0,
    )
    marker.write(store.with_name(f"{store.name}.done.json"))

    res = validate_unit_done(store, expected_row=row)

    assert res.is_valid is False
    assert res.error_type == "ManifestRowHashMismatchError"
    assert res.quarantine_path is None
    assert store.exists(), "read-only validation with default flags moved the store"
    assert store.with_name(f"{store.name}.done.json").exists()
    assert not (tmp_path / ".quarantine").exists()


def test_local_executor_marker_reports_installed_version_not_defaults(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A marker stamped by the local executor names the installed xtrax, not 0.5.0 or clean."""
    monkeypatch.delenv("XTRAX_PRODUCER_GIT_SHA", raising=False)
    manifest = _create_mock_manifest(num_rows=1)
    output_dir = tmp_path / "outputs"

    summary = LocalSequentialExecutor().run(manifest, _unit_fn, output_dir=str(output_dir))
    assert summary.completed_units == 1

    row = manifest.rows[0]
    store = _resolve_output_path(output_dir, row)
    marker_path = store.with_name(f"{store.name}.done.json")
    producer = CampaignDoneMarker.read(marker_path).producer
    assert producer.version == _installed_version()
    assert producer.version not in ("xtrax@0.5.0", "xtrax@unknown")
    assert producer.git_sha == "unknown"
    assert producer.git_sha != "clean"


def test_create_done_marker_git_sha_comes_from_env(monkeypatch: Any) -> None:
    """XTRAX_PRODUCER_GIT_SHA supplies the git sha; an empty value counts as unset."""
    monkeypatch.setenv("XTRAX_PRODUCER_GIT_SHA", "abc1234")
    marker = create_done_marker(
        manifest_row_hash="r",
        input_hash="i",
        output_locator="o",
        content_digest_sha256="d",
        duration_s=0.0,
    )
    assert marker.producer.git_sha == "abc1234"
    assert marker.producer.version == _installed_version()

    monkeypatch.setenv("XTRAX_PRODUCER_GIT_SHA", "  ")
    marker = create_done_marker(
        manifest_row_hash="r",
        input_hash="i",
        output_locator="o",
        content_digest_sha256="d",
        duration_s=0.0,
    )
    assert marker.producer.git_sha == "unknown"

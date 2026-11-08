"""Resume trust: a completed unit is reused only when its content digest still matches.

Covers the 261106 decision: default resume verifies the store's content digest (deep),
the tier1 fast path is opt-in, and every reuse is recorded in ``reuse_log.jsonl``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from tests.campaign.executors.test_local import _create_mock_manifest
from tests.campaign.unit_fns import ARRAY_UNIT_FN, array_unit
from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    _resolve_output_path,
)
from xtrax.campaign.executors.slurm import SlurmArrayExecutor, get_pending_row_indices
from xtrax.campaign.manifest import CampaignManifest
from xtrax.campaign.marker import CampaignDoneMarker
from xtrax.cli.campaign_verb import CampaignArgs, run_campaign
from xtrax.cli.errors import CLIError


def _expected_values(row: Any) -> np.ndarray:
    return np.arange(8, dtype=np.float32) + float(row.seed)


def _marker_path(store: Path) -> Path:
    return store.with_name(f"{store.name}.done.json")


def _corrupt_chunk_in_place(store: Path) -> None:
    """Overwrite one chunk of the array without touching zarr.json or the done marker."""
    zarr.open_group(str(store), mode="r+")["values"][0] = -1.0


def _read_values(store: Path) -> np.ndarray:
    return np.asarray(zarr.open_group(str(store), mode="r")["values"][:])


def _reuse_records(out_dir: Path) -> list[dict[str, Any]]:
    log = out_dir / "reuse_log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_default_resume_recomputes_unit_with_corrupted_chunk(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=3)
    out = tmp_path / "out"
    first = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    assert first.completed_units == 3

    victim = manifest.rows[0]
    store = _resolve_output_path(out, victim)
    _corrupt_chunk_in_place(store)

    second = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))

    assert second.completed_units == 1
    assert second.skipped_units == 2
    np.testing.assert_array_equal(_read_values(store), _expected_values(victim))


def test_tier1_resume_skips_unit_with_corrupted_chunk(tmp_path: Path) -> None:
    """Documents the fast path: tier1 trusts the marker and root metadata, not the bytes."""
    manifest = _create_mock_manifest(num_rows=3)
    out = tmp_path / "out"
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))

    store = _resolve_output_path(out, manifest.rows[0])
    _corrupt_chunk_in_place(store)

    second = LocalSequentialExecutor(resume_verify="tier1").run(
        manifest, array_unit, output_dir=str(out)
    )

    assert second.completed_units == 0
    assert second.skipped_units == 3
    assert _read_values(store)[0] == -1.0


def test_clean_resume_logs_one_reuse_line_per_skipped_unit(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=3)
    out = tmp_path / "out"
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    assert _reuse_records(out) == []  # computed units are not logged

    second = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    assert second.skipped_units == 3

    records = _reuse_records(out)
    assert len(records) == 3
    by_hash = {rec["manifest_row_hash"]: rec for rec in records}
    assert set(by_hash) == {row.manifest_row_hash for row in manifest.rows}
    for row in manifest.rows:
        store = _resolve_output_path(out, row)
        rec = by_hash[row.manifest_row_hash]
        marker = CampaignDoneMarker.read(_marker_path(store))
        assert rec["input_hash"] == row.input_hash
        assert rec["store_path"] == str(store)
        assert rec["content_digest_sha256"] == marker.content_digest_sha256
        assert rec["verified"] == "deep"
        assert rec["reused_by"] == "LocalSequentialExecutor"
        parsed = datetime.fromisoformat(rec["reused_at_utc"])
        assert parsed.utcoffset() is not None


def test_get_pending_default_reports_corrupted_row(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=3)
    out = tmp_path / "out"
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    _corrupt_chunk_in_place(_resolve_output_path(out, manifest.rows[0]))

    assert get_pending_row_indices(manifest, out) == [0]
    assert get_pending_row_indices(manifest, out, resume_verify="tier1") == []


def test_cli_rejects_unknown_resume_verify(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=1)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    args = CampaignArgs(
        action="run",
        manifest=str(manifest_path),
        output_dir=str(tmp_path / "out"),
        fn=ARRAY_UNIT_FN,
        resume_verify="bogus",
    )
    with pytest.raises(CLIError, match="resume_verify"):
        run_campaign(args)


def test_executors_and_pending_reject_unknown_resume_verify(tmp_path: Path) -> None:
    manifest: CampaignManifest = _create_mock_manifest(num_rows=1)
    with pytest.raises(ValueError, match="resume_verify"):
        LocalSequentialExecutor(resume_verify="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resume_verify"):
        LocalProcessPoolExecutor(resume_verify="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resume_verify"):
        SlurmArrayExecutor(resume_verify="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resume_verify"):
        get_pending_row_indices(manifest, tmp_path, resume_verify="bogus")  # type: ignore[arg-type]

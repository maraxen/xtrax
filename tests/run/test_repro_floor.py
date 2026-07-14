"""Tests for xtrax.run.repro_floor (S5 seed-pinned re-run-N reproducibility floor, #3498).

Red-first: these tests are written before `xtrax.run.repro_floor` exists / is complete.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.devtools.freshness import evaluate_freshness
from xtrax.run.repro_floor import (
    ReproFloorResult,
    build_repro_floor_attestation_toml,
    repro_floor_freshness_attestation,
    run_repro_floor,
    write_repro_floor_attestation,
)


def _write_store(path: Path, values: list[int]) -> Path:
    root = zarr.open_group(str(path), mode="a")
    arr = root.create_array(name="data", shape=(len(values),), dtype="int32", overwrite=True)
    arr[...] = np.array(values, dtype=np.int32)
    return path


def _deterministic_compute(tmp_path: Path):
    """Returns a `compute(seed) -> Path` that is genuinely seed-stable."""
    calls = {"n": 0}

    def compute(seed: int) -> Path:
        calls["n"] += 1
        store_path = tmp_path / f"run_{calls['n']}.zarr"
        # deterministic function of the pinned seed only -- same seed => same content
        return _write_store(store_path, [seed, seed * 2, seed * 3])

    return compute


def _flaky_compute(tmp_path: Path, diverge_on_call: int = 2):
    """Returns a `compute(seed) -> Path` that diverges on a later call (simulated
    non-reproducibility) despite receiving the same pinned seed every time."""
    calls = {"n": 0}

    def compute(seed: int) -> Path:
        calls["n"] += 1
        store_path = tmp_path / f"run_{calls['n']}.zarr"
        values = [seed, seed * 2, seed * 3]
        if calls["n"] == diverge_on_call:
            values = [seed, seed * 2, seed * 3 + 1]  # silent nondeterminism
        return _write_store(store_path, values)

    return compute


class TestRunReproFloor:
    def test_all_reruns_match_passes(self, tmp_path: Path) -> None:
        compute = _deterministic_compute(tmp_path)
        result = run_repro_floor(compute, seed=42, rerun_count=3, run_id="run-abc")

        assert isinstance(result, ReproFloorResult)
        assert result.passed is True
        assert result.seed_pin == 42
        assert result.rerun_count == 3
        assert len(result.rerun_digests) == 3
        assert len(set(result.rerun_digests)) == 1, "all reruns must be byte-identical"
        assert result.content_hash == result.rerun_digests[0]
        assert result.mismatched_digests == ()

    def test_divergent_rerun_is_caught_and_denies(self, tmp_path: Path) -> None:
        compute = _flaky_compute(tmp_path, diverge_on_call=2)
        result = run_repro_floor(compute, seed=42, rerun_count=3, run_id="run-abc")

        assert result.passed is False
        assert len(result.rerun_digests) == 3
        # at least one digest differs from the reference content_hash
        assert result.mismatched_digests != ()
        assert all(d != result.content_hash for d in result.mismatched_digests)

    def test_rerun_count_must_be_positive(self, tmp_path: Path) -> None:
        compute = _deterministic_compute(tmp_path)
        with pytest.raises(ValueError, match="rerun_count"):
            run_repro_floor(compute, seed=1, rerun_count=0, run_id="run-x")

    def test_seed_is_pinned_across_every_call(self, tmp_path: Path) -> None:
        seen_seeds: list[int] = []

        def compute(seed: int) -> Path:
            seen_seeds.append(seed)
            return _write_store(tmp_path / f"run_{len(seen_seeds)}.zarr", [seed])

        run_repro_floor(compute, seed=7, rerun_count=4, run_id="run-y")
        assert seen_seeds == [7, 7, 7, 7]


class TestBuildReproFloorAttestationToml:
    def test_passing_result_produces_schema_conformant_toml(self, tmp_path: Path) -> None:
        compute = _deterministic_compute(tmp_path)
        result = run_repro_floor(compute, seed=42, rerun_count=3, run_id="run-abc")

        toml_text = build_repro_floor_attestation_toml(result)
        parsed = tomllib.loads(toml_text)
        section = parsed["attestation"]

        # Schema mirrors bathos.attestation's `[attestation]` shape (spec §3.2) exactly --
        # this is the cross-tool integration seam contract, checked structurally here
        # without importing bathos (xtrax has no bathos dependency).
        assert section["kind"] == "repro_floor"
        assert section["verdict"] == "PASS"
        assert section["attested"]["run_id"] == "run-abc"
        assert section["attested"]["content_hash"] == result.content_hash
        assert section["attested"]["output_path"] == result.output_path
        assert section["seed_pin"] == 42
        assert section["rerun_count"] == 3
        assert section["rerun_digests"] == list(result.rerun_digests)
        # bathos.attestation.validate_attestation's repro_floor invariant:
        # every rerun_digest must equal attested.content_hash.
        assert all(d == section["attested"]["content_hash"] for d in section["rerun_digests"])
        assert section["created_by"]
        assert section["created_at"]

    def test_divergent_result_refuses_to_build_attestation(self, tmp_path: Path) -> None:
        compute = _flaky_compute(tmp_path, diverge_on_call=2)
        result = run_repro_floor(compute, seed=42, rerun_count=3, run_id="run-abc")

        with pytest.raises(ValueError, match="divergent"):
            build_repro_floor_attestation_toml(result)


class TestWriteReproFloorAttestation:
    def test_writes_file_for_passing_result(self, tmp_path: Path) -> None:
        compute = _deterministic_compute(tmp_path / "stores")
        result = run_repro_floor(compute, seed=1, rerun_count=2, run_id="run-z")

        out = tmp_path / "sidecars" / "repro_floor.attestation.bth.toml"
        written = write_repro_floor_attestation(result, out)

        assert written == out
        assert out.exists()
        parsed = tomllib.loads(out.read_text())
        assert parsed["attestation"]["verdict"] == "PASS"

    def test_no_attestation_written_on_divergence(self, tmp_path: Path) -> None:
        compute = _flaky_compute(tmp_path / "stores", diverge_on_call=2)
        result = run_repro_floor(compute, seed=1, rerun_count=3, run_id="run-z")

        out = tmp_path / "sidecars" / "repro_floor.attestation.bth.toml"
        with pytest.raises(ValueError):
            write_repro_floor_attestation(result, out)

        assert not out.exists(), "no attestation must be written on digest divergence"


class TestReproFloorFreshness:
    def test_freshness_attestation_is_fresh_within_ttl(self, tmp_path: Path) -> None:
        compute = _deterministic_compute(tmp_path)
        now = datetime.now(UTC)
        result = run_repro_floor(compute, seed=3, rerun_count=2, run_id="run-f", now=now)

        attestation = repro_floor_freshness_attestation(result, ttl_days=7.0)
        verdict = evaluate_freshness(attestation, now=now + timedelta(days=1))

        assert verdict.fresh is True
        assert verdict.ttl_expired is False

    def test_freshness_attestation_expires_after_ttl(self, tmp_path: Path) -> None:
        compute = _deterministic_compute(tmp_path)
        now = datetime.now(UTC)
        result = run_repro_floor(compute, seed=3, rerun_count=2, run_id="run-f", now=now)

        attestation = repro_floor_freshness_attestation(result, ttl_days=7.0)
        verdict = evaluate_freshness(attestation, now=now + timedelta(days=30))

        assert verdict.fresh is False
        assert verdict.ttl_expired is True

    def test_freshness_attestation_refuses_divergent_result(self, tmp_path: Path) -> None:
        compute = _flaky_compute(tmp_path, diverge_on_call=2)
        result = run_repro_floor(compute, seed=3, rerun_count=3, run_id="run-f")

        with pytest.raises(ValueError):
            repro_floor_freshness_attestation(result, ttl_days=7.0)

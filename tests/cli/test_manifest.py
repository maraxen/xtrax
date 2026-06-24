"""AC6, AC8, inv#1: manifest writer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xtrax.cli.config import ConfigError, TrainConfig
from xtrax.cli.errors import ResumeError
from xtrax.cli.manifest import read_manifest, write_manifest


def _make_cfg(**overrides):
    defaults = dict(
        schema_version=1,
        model={"path": "tests.cli._run_fixtures:make_model", "kwargs": {}},
        optimizer={
            "path": "xtrax.training.optim:adamw_with_schedule",
            "kwargs": {"peak_lr": 1e-3, "warmup_steps": 1, "total_steps": 300},
        },
        loss={"path": "tests.cli._run_fixtures:make_loss", "kwargs": {}},
        data={
            "factory": "tests.cli._run_fixtures:make_dataset",
            "kwargs": {},
            "batch_size": 4,
        },
        seed=42,
        num_epochs=3,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_manifest_written_all_fields(tmp_path) -> None:
    """AC6: manifest is always written with all required fields."""
    run_dir = str(tmp_path / "runs" / "abc123")
    cfg = _make_cfg()
    write_manifest(run_dir, cfg, run_id="abc123", config_hash_val="abc123")

    manifest_path = Path(run_dir) / "manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded["run_id"] == "abc123"
    assert loaded["schema_version"] == 1
    assert loaded["model"]["path"] == "tests.cli._run_fixtures:make_model"
    assert "optimizer" in loaded
    assert "loss" in loaded
    assert "data" in loaded
    assert "checkpoint_dir" in loaded
    assert "config_hash" in loaded
    assert loaded["num_epochs"] == 3
    assert loaded["seed"] == 42
    assert loaded["data"]["batch_size"] == 4


def test_manifest_model_path_non_null(tmp_path) -> None:
    """inv#1: model.path non-optional — None must raise at write site."""
    run_dir = str(tmp_path / "runs" / "abc")
    cfg = _make_cfg(model={"path": None, "kwargs": {}})
    with pytest.raises((ValueError, AssertionError)):
        write_manifest(run_dir, cfg, run_id="abc", config_hash_val="abc")


def test_checkpoint_dir_derived_from_run_id(tmp_path) -> None:
    """AC8 (C1): checkpoint_dir is DERIVED from run_id — NOT a verbatim config scalar.

    The adversarial review INVERTED the original DAG inv#2 wording.
    Correct: checkpoint_dir = .xtrax/runs/<run_id>/checkpoints/
    This is the only formulation that delivers no-clobber.
    """
    run_dir = str(tmp_path / "runs" / "myhash12")
    cfg = _make_cfg()
    m = write_manifest(run_dir, cfg, run_id="myhash12", config_hash_val="myhash12")
    assert m["checkpoint_dir"] == ".xtrax/runs/myhash12/checkpoints/"


def test_config_hash_in_manifest_is_unsuffixed(tmp_path) -> None:
    """AC6: config_hash in manifest is the un-suffixed base hash (stable for resume)."""
    run_dir = str(tmp_path / "runs" / "abc123-x1y2")
    cfg = _make_cfg()
    m = write_manifest(run_dir, cfg, run_id="abc123-x1y2", config_hash_val="abc123")
    assert m["config_hash"] == "abc123"


def test_manifest_schema_version_mirrored(tmp_path) -> None:
    """AC9 (manifest half): schema_version in manifest mirrors config."""
    run_dir = str(tmp_path / "runs" / "v2run")
    cfg = _make_cfg(schema_version=2)
    # Note: write_manifest handles the serialization and write; read_manifest validates schema version.
    m = write_manifest(run_dir, cfg, run_id="v2run", config_hash_val="v2run")
    assert m["schema_version"] == 2


def test_read_manifest_success(tmp_path) -> None:
    """read_manifest successfully parses and validates a valid manifest."""
    run_dir = str(tmp_path / "runs" / "abc123")
    cfg = _make_cfg()
    write_manifest(run_dir, cfg, run_id="abc123", config_hash_val="abc123")
    manifest_path = str(Path(run_dir) / "manifest.json")

    loaded = read_manifest(manifest_path)
    assert loaded["run_id"] == "abc123"
    assert loaded["schema_version"] == 1
    assert loaded["num_epochs"] == 3


def test_read_manifest_missing_file() -> None:
    """read_manifest raises ResumeError if the file does not exist."""
    with pytest.raises(ResumeError, match="manifest file does not exist"):
        read_manifest("/nonexistent/manifest.json")


def test_read_manifest_schema_version_mismatch(tmp_path) -> None:
    """read_manifest raises ConfigError if schema version does not match CURRENT_SCHEMA_VERSION."""
    run_dir = tmp_path / "runs" / "v2run"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    
    manifest_data = {
        "run_id": "v2run",
        "schema_version": 99,  # Mismatch
    }
    manifest_path.write_text(json.dumps(manifest_data))

    with pytest.raises(ConfigError, match="manifest schema_version 99 != expected 1"):
        read_manifest(str(manifest_path))


def test_read_manifest_missing_required_fields(tmp_path) -> None:
    """read_manifest raises ResumeError if any required fields are missing."""
    run_dir = tmp_path / "runs" / "badrun"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    
    # Missing 'seed'
    manifest_data = {
        "run_id": "badrun",
        "schema_version": 1,
        "model": {"path": "some.path"},
        "optimizer": {"path": "some.path"},
        "loss": {"path": "some.path"},
        "data": {"factory": "some.factory", "batch_size": 4},
        "checkpoint_dir": ".xtrax/runs/badrun/checkpoints/",
        "config_hash": "badrun",
        "num_epochs": 3,
    }
    manifest_path.write_text(json.dumps(manifest_data))

    with pytest.raises(ResumeError, match="manifest missing required field: seed"):
        read_manifest(str(manifest_path))

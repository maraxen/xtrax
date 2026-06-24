"""AC6, AC8, inv#1: manifest writer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xtrax.cli.config import TrainConfig
from xtrax.cli.manifest import write_manifest


def _make_cfg(**overrides):
    defaults = dict(
        schema_version=1,
        model={"path": "tests.cli._run_fixtures:make_model", "kwargs": {}},
        optimizer={
            "path": "xtrax.training.optim:adamw_with_schedule",
            "kwargs": {"learning_rate": 1e-3, "total_steps": 300},
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
    m = write_manifest(run_dir, cfg, run_id="v2run", config_hash_val="v2run")
    assert m["schema_version"] == 2

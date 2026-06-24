"""Always-write manifest for xtrax run (AC6, AC8, inv#1)."""

import json
import os
from pathlib import Path

from xtrax.cli.config import TrainConfig


def write_manifest(
    run_dir: str,
    cfg: TrainConfig,
    run_id: str,
    config_hash_val: str,
) -> dict:
    """
    Write manifest.json to .xtrax/runs/<run_id>/manifest.json.

    checkpoint_dir is DERIVED from run_id (not taken from config):
      checkpoint_dir = f".xtrax/runs/{run_id}/checkpoints/"

    This is the no-clobber invariant: distinct run_ids → distinct checkpoint dirs.

    Returns the manifest dict (for tests to inspect).
    """
    model_path = cfg.model.get("path")
    if not model_path:
        raise ValueError("manifest invariant violated: model.path is null or missing")

    checkpoint_dir = f".xtrax/runs/{run_id}/checkpoints/"

    manifest = {
        "run_id": run_id,
        "schema_version": cfg.schema_version,
        "model": {"path": model_path, "kwargs": cfg.model.get("kwargs", {})},
        "optimizer": {
            "path": cfg.optimizer["path"],
            "kwargs": cfg.optimizer.get("kwargs", {}),
        },
        "loss": {"path": cfg.loss["path"], "kwargs": cfg.loss.get("kwargs", {})},
        "data": {
            "factory": cfg.data["factory"],
            "kwargs": cfg.data.get("kwargs", {}),
        },
        "checkpoint_dir": checkpoint_dir,
        "config_hash": config_hash_val,
    }

    manifest_path = Path(run_dir) / "manifest.json"
    os.makedirs(run_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

"""Always-write manifest for xtrax run (AC6, AC8, inv#1)."""

import dataclasses
import json
import os
from pathlib import Path

from xtrax.cli.config import CURRENT_SCHEMA_VERSION, ConfigError, TrainConfig
from xtrax.cli.errors import ResumeError


def write_manifest_dict(
    run_dir: str,
    cfg_dict: dict,
    run_id: str,
    config_hash_val: str,
    resumed_from: str | None = None,
) -> dict:
    """Write manifest from a dict representation of the config."""
    model_path = cfg_dict["model"].get("path")
    if not model_path:
        raise ValueError("manifest invariant violated: model.path is null or missing")

    checkpoint_dir = f".xtrax/runs/{run_id}/checkpoints/"

    manifest = {
        "run_id": run_id,
        "schema_version": cfg_dict["schema_version"],
        "model": {"path": model_path, "kwargs": cfg_dict["model"].get("kwargs", {})},
        "optimizer": {
            "path": cfg_dict["optimizer"]["path"],
            "kwargs": cfg_dict["optimizer"].get("kwargs", {}),
        },
        "loss": {"path": cfg_dict["loss"]["path"], "kwargs": cfg_dict["loss"].get("kwargs", {})},
        "data": {
            "factory": cfg_dict["data"]["factory"],
            "kwargs": cfg_dict["data"].get("kwargs", {}),
            "batch_size": cfg_dict["data"]["batch_size"],
        },
        "checkpoint_dir": checkpoint_dir,
        "config_hash": config_hash_val,
        "num_epochs": cfg_dict["num_epochs"],
        "seed": cfg_dict["seed"],
    }
    if resumed_from is not None:
        manifest["resumed_from"] = resumed_from

    manifest_path = Path(run_dir) / "manifest.json"
    os.makedirs(run_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


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
    return write_manifest_dict(
        run_dir=run_dir,
        cfg_dict=dataclasses.asdict(cfg),
        run_id=run_id,
        config_hash_val=config_hash_val,
    )


def read_manifest(path: str) -> dict:
    """Load and validate the manifest file at path.

    Raises:
        ResumeError: If the manifest does not exist or is missing required fields.
        ConfigError: If the schema version does not match.
    """
    try:
        with open(path, "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise ResumeError(f"manifest file does not exist: {path}")

    version = manifest.get("schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"manifest schema_version {version} != expected {CURRENT_SCHEMA_VERSION}"
        )

    required_fields = [
        ("run_id", lambda m: m.get("run_id")),
        ("model.path", lambda m: m.get("model", {}).get("path")),
        ("optimizer.path", lambda m: m.get("optimizer", {}).get("path")),
        ("loss.path", lambda m: m.get("loss", {}).get("path")),
        ("data.factory", lambda m: m.get("data", {}).get("factory")),
        ("data.batch_size", lambda m: m.get("data", {}).get("batch_size")),
        ("checkpoint_dir", lambda m: m.get("checkpoint_dir")),
        ("config_hash", lambda m: m.get("config_hash")),
        ("schema_version", lambda m: m.get("schema_version")),
        ("num_epochs", lambda m: m.get("num_epochs")),
        ("seed", lambda m: m.get("seed")),
    ]

    for field, accessor in required_fields:
        if accessor(manifest) is None:
            raise ResumeError(
                f"manifest missing required field: {field}. "
                "Was this run created with an older version of xtrax?"
            )

    return manifest


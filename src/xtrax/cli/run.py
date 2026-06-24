"""run_from_config: cli-private glue wiring TrainConfig → Engine.fit_sync (AC5)."""

import dataclasses
import os
import uuid

from xtrax.cli.config import TrainConfig
from xtrax.cli.hash import config_hash as compute_config_hash
from xtrax.cli.loader import CLIImportError, load_fn
from xtrax.cli.manifest import write_manifest
from xtrax.data.module import DataModule
from xtrax.engine.engine import Engine
from xtrax.training import init_state
from xtrax.training.trainer import Trainer


def _resolve(section: str, path: str, kwargs: dict):
    """AC11: wrap load_fn so CLIImportError names the section."""
    try:
        return load_fn(path)(**kwargs)
    except CLIImportError as e:
        raise CLIImportError(f"[{section}] {e}") from e


def run_from_config(cfg: TrainConfig):
    """
    Wire TrainConfig → Engine.fit_sync. cli-private glue (spec decision M).

    AC5: resolves model/optimizer/loss/data → init_state → Engine(Trainer).fit_sync
    AC8/C1: checkpoint_dir derived from run_id (NOT verbatim config)
    AC7: run_id = config_hash; collision → hash-uuid suffix
    AC11: section-labeled CLIImportError on bad import paths
    AC13/M5: callbacks=() — no logging (explicit MVP limitation)
    M4/AC3: DataModule wrap is ALWAYS unconditional (no isinstance branch)
    """
    loss_fn = _resolve("loss", cfg.loss["path"], cfg.loss.get("kwargs", {}))
    model = _resolve("model", cfg.model["path"], cfg.model.get("kwargs", {}))
    optimizer = _resolve(
        "optimizer", cfg.optimizer["path"], cfg.optimizer.get("kwargs", {})
    )

    try:
        dataset = load_fn(cfg.data["factory"])(**cfg.data.get("kwargs", {}))
    except CLIImportError as e:
        raise CLIImportError(f"[data] {e}") from e
    data = DataModule(
        dataset,
        batch_size=cfg.data["batch_size"],
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
        distributed=False,
    )

    state = init_state(model, optimizer, cfg.seed)

    cfg_dict = dataclasses.asdict(cfg)
    hash_val = compute_config_hash(cfg_dict)

    base_run_dir = f".xtrax/runs/{hash_val}"
    run_id = hash_val
    try:
        os.makedirs(base_run_dir, exist_ok=False)
    except FileExistsError:
        suffix = uuid.uuid4().hex[:6]
        run_id = f"{hash_val}-{suffix}"
        os.makedirs(f".xtrax/runs/{run_id}", exist_ok=False)

    checkpoint_dir = f".xtrax/runs/{run_id}/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    engine = Engine(trainer=Trainer(loss_fn, optimizer), callbacks=())
    final_state = engine.fit_sync(
        state,
        data,
        num_epochs=cfg.num_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    run_dir = f".xtrax/runs/{run_id}"
    write_manifest(run_dir, cfg, run_id=run_id, config_hash_val=hash_val)

    return final_state

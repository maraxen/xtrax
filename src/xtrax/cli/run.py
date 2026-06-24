"""run_from_config: cli-private glue wiring TrainConfig → Engine.fit_sync (AC5)."""

import dataclasses
import os
import uuid

from xtrax.cli.config import TrainConfig
from xtrax.cli.hash import config_hash as compute_config_hash
from xtrax.cli.manifest import write_manifest
from xtrax.cli.resolve import resolve_components
from xtrax.engine.engine import Engine
from xtrax.training import ResumableState, init_state
from xtrax.training.trainer import Trainer


def generate_run_id(cfg: TrainConfig) -> str:
    """
    Generate a unique run ID and reserve its base directory.
    Continually generates a new random suffix until the directory is successfully reserved.
    """
    cfg_dict = dataclasses.asdict(cfg)
    hash_val = compute_config_hash(cfg_dict)
    run_id = hash_val
    try:
        os.makedirs(f".xtrax/runs/{run_id}", exist_ok=False)
        return run_id
    except FileExistsError:
        while True:
            suffix = uuid.uuid4().hex[:6]
            candidate_id = f"{hash_val}-{suffix}"
            try:
                os.makedirs(f".xtrax/runs/{candidate_id}", exist_ok=False)
                return candidate_id
            except FileExistsError:
                continue


def run_from_config(cfg: TrainConfig, run_id: str | None = None) -> ResumableState:
    """
    Wire TrainConfig → Engine.fit_sync. cli-private glue (spec decision M).

    AC5: resolves model/optimizer/loss/data → init_state → Engine(Trainer).fit_sync
    AC8/C1: checkpoint_dir derived from run_id (NOT verbatim config)
    AC7: run_id = config_hash; collision → hash-uuid suffix
    AC11: section-labeled CLIImportError on bad import paths
    AC13/M5: callbacks=() — no logging (explicit MVP limitation)
    M4/AC3: DataModule wrap is ALWAYS unconditional (no isinstance branch)
    """
    resolved = resolve_components(dataclasses.asdict(cfg), cfg.num_epochs)
    model = resolved.model
    optimizer = resolved.optimizer
    loss_fn = resolved.loss_fn
    data = resolved.dataset

    state = init_state(model, optimizer, cfg.seed)

    cfg_dict = dataclasses.asdict(cfg)
    hash_val = compute_config_hash(cfg_dict)

    if run_id is None:
        run_id = generate_run_id(cfg)

    checkpoint_dir = f".xtrax/runs/{run_id}/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # AC6: always-write the manifest BEFORE training, not after. The manifest is
    # the contract `resume` consumes; writing it only on success would leave a
    # crashed-but-checkpointed run (the exact resume use-case) unresumable.
    run_dir = f".xtrax/runs/{run_id}"
    write_manifest(run_dir, cfg, run_id=run_id, config_hash_val=hash_val)

    engine = Engine(trainer=Trainer(loss_fn, optimizer), callbacks=())
    final_state = engine.fit_sync(
        state,
        data,
        num_epochs=cfg.num_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    return final_state

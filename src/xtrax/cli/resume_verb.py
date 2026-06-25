import os
import uuid
from dataclasses import dataclass
from typing import Any

from xtrax.cli.config import ConfigError
from xtrax.cli.errors import ResumeError
from xtrax.cli.manifest import read_manifest, write_manifest_dict
from xtrax.cli.resolve import resolve_components
from xtrax.engine.engine import Engine
from xtrax.training import init_state
from xtrax.training.trainer import Trainer

# tyro is bound dynamically in entrypoint.py:main() to keep imports tyro-free
tyro: Any = None


@dataclass
class ResumeArgs:
    """Arguments for the resume verb.

    Attributes:
        run_id: The ID of the run to resume.
        epochs: Number of epochs to train for.
        manifest_path: Optional path to manifest file (if moving/custom).
    """

    run_id: "tyro.conf.Positional[str]"
    epochs: int
    manifest_path: str | None = None


def run_resume(args: ResumeArgs) -> None:
    """Resume training of an existing run from its latest checkpoint.

    AC1/RAC1: Read manifest from run-id.
    AC2/RAC2: Optional manifest-path override.
    AC9/RAC9: Validate epochs > 0.
    """
    if args.epochs <= 0:
        raise ConfigError("--epochs must be a positive integer")

    # Determine manifest path
    if args.manifest_path is not None:
        manifest_path = args.manifest_path
    else:
        manifest_path = f".xtrax/runs/{args.run_id}/manifest.json"

    # Read manifest (handles schema validation and missing fields checking)
    manifest = read_manifest(manifest_path)

    # Re-resolve components
    resolved = resolve_components(manifest, args.epochs)

    # Load checkpoint
    from xtrax.checkpoint.orbax import get_checkpoint_manager, load_checkpoint

    state_template = init_state(resolved.model, resolved.optimizer, manifest["seed"])
    checkpoint_dir = manifest["checkpoint_dir"]
    manager = get_checkpoint_manager(checkpoint_dir)

    try:
        loaded_state = load_checkpoint(manager, state_template)
    except FileNotFoundError as e:
        raise ResumeError(f"No checkpoints found in {checkpoint_dir}") from e

    # Create sibling run id and run dir
    # RAC6: Generates new sibling run-id with config_hash + suffix
    config_hash = manifest["config_hash"]
    suffix = uuid.uuid4().hex[:6]
    new_run_id = f"{config_hash}-{suffix}"
    new_run_dir = f".xtrax/runs/{new_run_id}"

    os.makedirs(new_run_dir, exist_ok=False)

    new_checkpoint_dir = f".xtrax/runs/{new_run_id}/checkpoints/"
    os.makedirs(new_checkpoint_dir, exist_ok=True)

    # Write manifest for the resumed run
    write_manifest_dict(
        run_dir=new_run_dir,
        cfg_dict=manifest,
        run_id=new_run_id,
        config_hash_val=config_hash,
        resumed_from=manifest["run_id"],
    )

    # Run training
    engine = Engine(trainer=Trainer(resolved.loss_fn, resolved.optimizer), callbacks=())
    engine.fit_sync(
        loaded_state,
        resolved.dataset,
        num_epochs=args.epochs,
        checkpoint_dir=new_checkpoint_dir,
        resume=True,
    )

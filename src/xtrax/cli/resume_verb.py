import os
import uuid
from dataclasses import dataclass
from typing import Any

from xtrax.cli.config import ConfigError
from xtrax.cli.errors import ResumeError
from xtrax.cli.manifest import read_manifest, write_manifest_dict
from xtrax.cli.resolve import resolve_components
from xtrax.engine.engine import Engine
from xtrax.telemetry.ledger import RunLedger
from xtrax.telemetry.record import KIND_TRAIN
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

    # Write manifest for the resumed run, carrying forward the closure declaration (#4117)
    # if the original run had one -- write_manifest_dict no longer reads it off cfg_dict,
    # so it must be re-extracted from the manifest we just read and passed through explicitly.
    closure = manifest.get("closure") or {}
    write_manifest_dict(
        run_dir=new_run_dir,
        cfg_dict=manifest,
        run_id=new_run_id,
        config_hash_val=config_hash,
        resumed_from=manifest["run_id"],
        evaluator_paths=closure.get("evaluator_paths"),
        split_paths=closure.get("split_paths"),
        metric_def_paths=closure.get("metric_def_paths"),
    )

    # Run training.
    #
    # derived_from records the parent run in the ledger, so a resumed run's
    # lineage is a queryable edge rather than something a reader has to infer
    # from manifest.json. It is the same single-parent id the manifest already
    # stores as `resumed_from`, matching controller/lineage_interim.py's
    # single-parent contract rather than introducing a second lineage model.
    #
    # This path previously built no sink and recorded no provenance at all --
    # a resumed run was invisible. Engine now opens a ledger for it like any
    # other run; the ledger is opened here only to carry derived_from.
    engine = Engine(trainer=Trainer(resolved.loss_fn, resolved.optimizer), callbacks=())
    with RunLedger.open(
        new_run_id,
        kind=KIND_TRAIN,
        derived_from=manifest["run_id"],
    ) as ledger:
        engine.fit_sync(
            loaded_state,
            resolved.dataset,
            num_epochs=args.epochs,
            checkpoint_dir=new_checkpoint_dir,
            resume=True,
            ledger=ledger,
        )

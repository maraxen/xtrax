"""run_from_config: cli-private glue wiring TrainConfig → Engine.fit_sync (AC5)."""

import dataclasses
import os
import uuid
from pathlib import Path

from xtrax.cli.config import TrainConfig
from xtrax.cli.hash import config_hash as compute_config_hash
from xtrax.cli.manifest import write_manifest
from xtrax.cli.resolve import resolve_components
from xtrax.engine.engine import Engine
from xtrax.run import RunSpec, derive_sink_spec, make_sink
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
    #457(1): output sink built exclusively via derive_sink_spec/make_sink;
    provenance store persisted at .xtrax/runs/<run_id>/metrics.zarr with the
    CLI's run_id as join key (matches manifest.json + checkpoint_dir).
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

    run_dir = f".xtrax/runs/{run_id}"

    # #457(1): first real adoption of the derive_sink_spec seam. The CLI never
    # constructs SinkSpec literally -- precedence (explicit override >
    # spec.run_id > generated) is single-sourced in xtrax.run.sink. The driver
    # RunSpec is axes-free by design: the plain run verb trains without a
    # sparsity axis schedule. Its run_id pins the CLI's config-hash id so the
    # store joins manifest.json and checkpoint_dir on one key.
    #
    # Created BEFORE write_manifest so the sink's git-state capture cannot see
    # the run's own untracked manifest (which would force git_dirty=True on
    # every default-config production run). .xtrax/ is also gitignored, making
    # capture honest regardless of ordering.
    # Created BEFORE fit: missing zarr fails loud before compute is wasted, and
    # a mid-fit crash leaves a root-provenance tombstone (git sha of the code
    # that was running) instead of no trace at all.
    driver_spec = RunSpec(
        seed=cfg.seed,
        axes=[],
        carry_specs=[],
        boundaries=None,
        run_id=run_id,
    )
    sink = make_sink(derive_sink_spec(driver_spec, output_dir=Path(run_dir) / "metrics.zarr"))
    # derive_sink_spec pins format="zarr", so make_sink cannot return None here
    # (None is reserved for format="none"). Narrow for the ty hard CI gate.
    assert sink is not None, "derive_sink_spec pins format='zarr'; make_sink must yield a sink"

    # AC6: always-write the manifest BEFORE training, not after. The manifest is
    # the contract `resume` consumes; writing it only on success would leave a
    # crashed-but-checkpointed run (the exact resume use-case) unresumable.
    write_manifest(run_dir, cfg, run_id=run_id, config_hash_val=hash_val)

    engine = Engine(trainer=Trainer(loss_fn, optimizer), callbacks=())
    final_state = engine.fit_sync(
        state,
        data,
        num_epochs=cfg.num_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    # Post-fit record: echo the manifest's identity fields + resolved component
    # class names into the ('run', 'final') group so one zarr read answers
    # "what ran here". drain() stamps run_id/git_sha onto the key group too.
    sink.stage(
        ("run", "final"),
        attrs={
            "config_hash": hash_val,
            "seed": cfg.seed,
            "num_epochs": cfg.num_epochs,
            "model": type(model).__name__,
            "optimizer": type(optimizer).__name__,
            "loss": type(loss_fn).__name__,
            "data": type(data).__name__,
            "checkpoint_dir": checkpoint_dir,
        },
    )
    sink.drain()
    sink.finalize()

    return final_state

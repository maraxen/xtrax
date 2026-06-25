from xtrax.cli.resolve import ResolvedComponents, resolve_components
from xtrax.data.module import DataModule


def test_resolve_components_success() -> None:
    manifest_dict = {
        "run_id": "test_run",
        "schema_version": 1,
        "model": {"path": "tests.cli._run_fixtures:make_model", "kwargs": {}},
        "optimizer": {
            "path": "xtrax.training.optim:adamw_with_schedule",
            "kwargs": {"peak_lr": 1e-3, "warmup_steps": 1, "total_steps": 300},
        },
        "loss": {"path": "tests.cli._run_fixtures:make_loss", "kwargs": {}},
        "data": {
            "factory": "tests.cli._run_fixtures:make_dataset",
            "kwargs": {},
            "batch_size": 4,
        },
        "checkpoint_dir": ".xtrax/runs/test_run/checkpoints/",
        "config_hash": "test_run",
        "num_epochs": 3,
        "seed": 42,
    }

    resolved = resolve_components(manifest_dict, epochs=3)
    assert isinstance(resolved, ResolvedComponents)
    assert resolved.model is not None
    assert resolved.optimizer is not None
    assert resolved.loss_fn is not None
    assert isinstance(resolved.dataset, DataModule)
    assert resolved.dataset.batch_size == 4
    assert resolved.dataset.num_epochs == 3
    assert resolved.dataset.seed == 42

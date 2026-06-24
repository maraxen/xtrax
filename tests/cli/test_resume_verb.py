from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import jax.numpy as jnp

from xtrax.cli.config import ConfigError, load_config
from xtrax.cli.errors import ResumeError
from xtrax.cli.resume_verb import ResumeArgs, run_resume
from xtrax.cli.run import run_from_config

class DictDataset:
    """Dataset yielding dict batches for real training steps."""
    def __init__(self):
        self.data = [
            {"inputs": jnp.ones(2) * float(i), "targets": jnp.ones(2) * float(i)}
            for i in range(4)
        ]

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self.data[idx]


def make_dict_dataset():
    """Factory function for DictDataset. Must be module-level for loader to resolve."""
    return DictDataset()


FIXTURE_TOML_CONTENT = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3, warmup_steps = 1, total_steps = 10}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli.test_resume_verb:make_dict_dataset"
kwargs = {}
batch_size = 2
"""

@pytest.fixture
def fixture_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(FIXTURE_TOML_CONTENT)
    return str(p)


@pytest.fixture(autouse=True)
def make_checkpoint_paths_absolute():
    """Orbax requires absolute paths for checkpoints.
    Since CLI runs use relative paths, we wrap get_checkpoint_manager to return absolute paths.
    """
    import xtrax.checkpoint.orbax
    original_gcm = xtrax.checkpoint.orbax.get_checkpoint_manager

    def wrapper_gcm(directory, *args, **kwargs):
        return original_gcm(Path(directory).absolute(), *args, **kwargs)

    with patch("xtrax.checkpoint.orbax.get_checkpoint_manager", side_effect=wrapper_gcm):
        yield


def test_resume_end_to_end_success(tmp_path, monkeypatch, fixture_toml):
    monkeypatch.chdir(tmp_path)

    # 1. Start run for 1 epoch
    cfg = load_config(fixture_toml)
    first_state = run_from_config(cfg)
    # DictDataset has 4 items. Iterating it once takes 4 steps (since train_iter yields each item from dataset).
    assert int(first_state.step) == 4

    # Check the first run directory
    run_dirs = sorted(list(Path(".xtrax/runs").iterdir()))
    assert len(run_dirs) == 1
    original_run_dir = run_dirs[0]
    original_run_id = original_run_dir.name

    # Check manifest exists
    manifest_path = original_run_dir / "manifest.json"
    assert manifest_path.exists()
    original_manifest = json.loads(manifest_path.read_text())
    assert original_manifest["run_id"] == original_run_id

    # Check checkpoints exist in the original checkpoints dir
    original_checkpoint_dir = Path(original_manifest["checkpoint_dir"])
    assert original_checkpoint_dir.exists()

    # 2. Call run_resume with ResumeArgs(run_id=original_run_id, epochs=2)
    args = ResumeArgs(run_id=original_run_id, epochs=2)
    run_resume(args)

    # 3. Verify a new sibling run directory exists with manifest containing "resumed_from": original_run_id
    run_dirs = sorted(list(Path(".xtrax/runs").iterdir()))
    assert len(run_dirs) == 2
    new_run_dir = [d for d in run_dirs if d != original_run_dir][0]
    new_run_id = new_run_dir.name

    new_manifest_path = new_run_dir / "manifest.json"
    assert new_manifest_path.exists()
    new_manifest = json.loads(new_manifest_path.read_text())
    assert new_manifest["resumed_from"] == original_run_id
    assert new_manifest["run_id"] == new_run_id

    # 4. Verify new checkpoints directory contains checkpoints
    new_checkpoint_dir = Path(new_manifest["checkpoint_dir"])
    assert new_checkpoint_dir.exists()

    # 5. Verify the final step count is correct (sum of steps = 4 from original + 8 from resuming 2 epochs = 12 steps)
    from xtrax.cli.resolve import resolve_components
    from xtrax.training import init_state
    from xtrax.checkpoint.orbax import get_checkpoint_manager, load_checkpoint

    resolved = resolve_components(new_manifest, epochs=2)
    state_template = init_state(resolved.model, resolved.optimizer, new_manifest["seed"])
    manager = get_checkpoint_manager(new_checkpoint_dir)
    final_state = load_checkpoint(manager, state_template)

    assert int(final_state.step) == 12


def test_resume_custom_manifest_path(tmp_path, monkeypatch, fixture_toml):
    monkeypatch.chdir(tmp_path)

    # Start run for 1 epoch
    cfg = load_config(fixture_toml)
    run_from_config(cfg)

    # Find run dir and manifest
    run_dirs = list(Path(".xtrax/runs").iterdir())
    original_run_id = run_dirs[0].name
    original_manifest_path = run_dirs[0] / "manifest.json"

    # Copy manifest to custom path
    custom_manifest_path = tmp_path / "custom_manifest.json"
    shutil.copy(original_manifest_path, custom_manifest_path)

    # Call run_resume with custom manifest path
    args = ResumeArgs(run_id=original_run_id, epochs=1, manifest_path=str(custom_manifest_path))
    run_resume(args)

    # Verify a new sibling run directory exists
    run_dirs = sorted(list(Path(".xtrax/runs").iterdir()))
    assert len(run_dirs) == 2


def test_resume_schema_validation_failure(tmp_path, monkeypatch, fixture_toml):
    monkeypatch.chdir(tmp_path)

    # Start run
    cfg = load_config(fixture_toml)
    run_from_config(cfg)

    run_dirs = list(Path(".xtrax/runs").iterdir())
    original_run_id = run_dirs[0].name
    manifest_path = run_dirs[0] / "manifest.json"

    # Corrupt schema version in manifest
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 9999
    manifest_path.write_text(json.dumps(manifest))

    # Assert raises ConfigError
    args = ResumeArgs(run_id=original_run_id, epochs=1)
    with pytest.raises(ConfigError, match="schema_version"):
        run_resume(args)


def test_resume_missing_fields_validation_failure(tmp_path, monkeypatch, fixture_toml):
    monkeypatch.chdir(tmp_path)

    # Start run
    cfg = load_config(fixture_toml)
    run_from_config(cfg)

    run_dirs = list(Path(".xtrax/runs").iterdir())
    original_run_id = run_dirs[0].name
    manifest_path = run_dirs[0] / "manifest.json"

    # Delete checkpoint_dir from manifest
    manifest = json.loads(manifest_path.read_text())
    del manifest["checkpoint_dir"]
    manifest_path.write_text(json.dumps(manifest))

    # Assert raises ResumeError
    args = ResumeArgs(run_id=original_run_id, epochs=1)
    with pytest.raises(ResumeError, match="missing required field"):
        run_resume(args)


def test_resume_epochs_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ResumeArgs(run_id="some-run", epochs=0)
    with pytest.raises(ConfigError, match="must be a positive integer"):
        run_resume(args)

    args = ResumeArgs(run_id="some-run", epochs=-5)
    with pytest.raises(ConfigError, match="must be a positive integer"):
        run_resume(args)


def test_resume_empty_checkpoint_dir_failure(tmp_path, monkeypatch, fixture_toml):
    monkeypatch.chdir(tmp_path)

    # Start run
    cfg = load_config(fixture_toml)
    run_from_config(cfg)

    run_dirs = sorted(list(Path(".xtrax/runs").iterdir()))
    original_run_id = run_dirs[0].name
    manifest_path = run_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Delete checkpoints folder
    checkpoint_dir = Path(manifest["checkpoint_dir"])
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    # Assert raises ResumeError
    args = ResumeArgs(run_id=original_run_id, epochs=1)
    with pytest.raises(ResumeError, match="No checkpoints found"):
        run_resume(args)


def test_tyro_isolation():
    sys.modules.pop("tyro", None)
    import xtrax.cli.entrypoint
    assert "tyro" not in sys.modules

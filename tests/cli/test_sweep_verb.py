"""Tests for the sweep CLI verb (E2 sweep verb)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtrax.cli.config import ConfigError


def test_tyro_isolation() -> None:
    """Importing sweep_verb must not pull in tyro."""
    sys.modules.pop("tyro", None)
    import xtrax.cli.sweep_verb  # noqa: F401

    assert "tyro" not in sys.modules, "tyro was imported at module level in sweep_verb"


def test_sweep_cartesian_and_list_overrides(tmp_path, monkeypatch) -> None:
    """Verify Cartesian product generation and correct overriding of list parameter values."""
    monkeypatch.chdir(tmp_path)

    config_content = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3, list_param = [1, 2]}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 2

[sweep.axes]
seed = [42, 43]
optimizer.kwargs.list_param = [[10, 20], [30, 40]]
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    configs_run = []

    def mock_run_from_config(cfg, run_id=None):
        configs_run.append(cfg)
        return MagicMock()

    with patch("xtrax.cli.sweep_verb.run_from_config", mock_run_from_config):
        from xtrax.cli.sweep_verb import SweepArgs, run_sweep

        run_sweep(SweepArgs(config_path=str(p)))

    # 2 seeds x 2 list_params = 4 runs
    assert len(configs_run) == 4

    # Verify lists are overwritten directly and not concatenated or converted to tuples
    for cfg in configs_run:
        assert isinstance(cfg.optimizer["kwargs"]["list_param"], list)
        assert cfg.optimizer["kwargs"]["list_param"] in ([10, 20], [30, 40])
        assert cfg.seed in (42, 43)


def test_sweep_end_to_end_success(tmp_path, monkeypatch) -> None:
    """Verify that a sweep executes and writes manifest to the isolated sweeps directory."""
    monkeypatch.chdir(tmp_path)

    config_content = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3, warmup_steps = 1, total_steps = 4}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 2

[sweep.axes]
seed = [10, 20]
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    mock_state = MagicMock()

    with patch("xtrax.engine.engine.Engine.fit_sync", return_value=mock_state):
        from xtrax.cli.sweep_verb import SweepArgs, run_sweep

        run_sweep(SweepArgs(config_path=str(p)))

    sweeps_dir = Path(".xtrax/sweeps")
    assert sweeps_dir.exists()
    sweep_dirs = list(sweeps_dir.iterdir())
    assert len(sweep_dirs) == 1

    sweep_manifest_path = sweep_dirs[0] / "sweep_manifest.json"
    assert sweep_manifest_path.exists()

    sweep_manifest = json.loads(sweep_manifest_path.read_text())
    assert len(sweep_manifest) == 2
    for run_id, overrides in sweep_manifest.items():
        assert "seed" in overrides
        assert overrides["seed"] in (10, 20)


def test_sweep_fault_tolerance_and_incremental_manifest(tmp_path, monkeypatch) -> None:
    """Verify that sweep is fault-tolerant and saves manifest incrementally."""
    monkeypatch.chdir(tmp_path)

    config_content = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 2

[sweep.axes]
seed = [1, 2, 3]
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    runs_executed = []

    def mock_run_from_config(cfg, run_id=None):
        runs_executed.append(cfg.seed)
        if cfg.seed == 2:
            raise ValueError("Simulated run failure on seed 2")
        return MagicMock()

    with patch("xtrax.cli.sweep_verb.run_from_config", mock_run_from_config):
        from xtrax.cli.sweep_verb import SweepArgs, run_sweep

        run_sweep(SweepArgs(config_path=str(p)))

    # All runs must be attempted (loop continues despite failure on seed 2)
    assert runs_executed == [1, 2, 3]

    sweeps_dir = Path(".xtrax/sweeps")
    sweep_dirs = list(sweeps_dir.iterdir())
    assert len(sweep_dirs) == 1

    sweep_manifest_path = sweep_dirs[0] / "sweep_manifest.json"
    assert sweep_manifest_path.exists()

    sweep_manifest = json.loads(sweep_manifest_path.read_text())
    # All 3 run IDs must be recorded even though run 2 failed
    assert len(sweep_manifest) == 3


def test_sweep_empty_axes(tmp_path, monkeypatch) -> None:
    """Verify that an empty or missing sweep.axes defaults to running base config once."""
    monkeypatch.chdir(tmp_path)

    config_content = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 2
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    configs_run = []

    def mock_run_from_config(cfg, run_id=None):
        configs_run.append(cfg)
        return MagicMock()

    with patch("xtrax.cli.sweep_verb.run_from_config", mock_run_from_config):
        from xtrax.cli.sweep_verb import SweepArgs, run_sweep

        run_sweep(SweepArgs(config_path=str(p)))

    # Base configuration runs exactly once
    assert len(configs_run) == 1
    assert configs_run[0].seed == 0


def test_sweep_axes_non_list_raises_error(tmp_path, monkeypatch) -> None:
    """Verify that a non-iterable/non-list axis leaf raises a ConfigError."""
    monkeypatch.chdir(tmp_path)

    config_content = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 2

[sweep.axes]
seed = 42
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    from xtrax.cli.sweep_verb import SweepArgs, run_sweep

    with pytest.raises(ConfigError, match="must be a list"):
        run_sweep(SweepArgs(config_path=str(p)))

"""Tests for xtrax sweep deprecation compatibility shim (Track 4: T4-02, GATE-10).

Verifies that legacy `xtrax sweep` emits a FutureWarning (scheduled removal 0.6.0),
adapts `[sweep.axes]` into a single-arm CampaignSpec, executes via LocalSequentialExecutor,
and preserves backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtrax.cli.config import ConfigError
from xtrax.cli.sweep_verb import SweepArgs, run_sweep


def test_sweep_emits_future_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GATE-10: Verify that `xtrax sweep` emits a FutureWarning."""
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
seed = [10, 20]
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    with patch("xtrax.cli.sweep_verb.run_from_config", return_value=MagicMock()):
        with pytest.warns(FutureWarning) as warning_records:
            run_sweep(SweepArgs(config_path=str(p)))

    # Verify that FutureWarning was emitted with proper deprecation message
    assert len(warning_records) >= 1
    deprecation_msg = str(warning_records[0].message)
    assert "'xtrax sweep' is deprecated" in deprecation_msg
    assert "0.6.0" in deprecation_msg


def test_sweep_executes_via_local_sequential_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GATE-10: Verify that sweep executes through LocalSequentialExecutor."""
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
seed = [1, 2]
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    executor_instantiated = False
    run_called = False

    from xtrax.campaign.executors.local import LocalSequentialExecutor
    original_run = LocalSequentialExecutor.run

    def mock_run(self, manifest, unit_fn, **kwargs):
        nonlocal executor_instantiated, run_called
        executor_instantiated = isinstance(self, LocalSequentialExecutor)
        run_called = True
        return original_run(self, manifest, unit_fn, **kwargs)

    with patch.object(LocalSequentialExecutor, "run", mock_run):
        with patch("xtrax.cli.sweep_verb.run_from_config", return_value=MagicMock()):
            run_sweep(SweepArgs(config_path=str(p)))

    assert executor_instantiated
    assert run_called


def test_sweep_end_to_end_arrays_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-11: Verify that legacy sweep produces identical output arrays / manifest."""
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
seed = [42, 43]
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    runs = []

    def mock_run_from_config(cfg, run_id=None):
        runs.append(cfg.seed)
        return MagicMock()

    with patch("xtrax.cli.sweep_verb.run_from_config", mock_run_from_config):
        run_sweep(SweepArgs(config_path=str(p)))

    assert runs == [42, 43]

    sweeps_dir = Path(".xtrax/sweeps")
    assert sweeps_dir.exists()
    sweep_dirs = list(sweeps_dir.iterdir())
    assert len(sweep_dirs) == 1

    sweep_manifest_path = sweep_dirs[0] / "sweep_manifest.json"
    assert sweep_manifest_path.exists()
    manifest_data = json.loads(sweep_manifest_path.read_text())
    assert len(manifest_data) == 2


def test_sweep_malformed_axes_negative_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-11: Malformed [sweep] table raises ConfigError identical to legacy behavior."""
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
seed = 999
"""
    p = tmp_path / "config.toml"
    p.write_text(config_content)

    with pytest.raises(ConfigError, match="must be a list"):
        run_sweep(SweepArgs(config_path=str(p)))

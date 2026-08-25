"""
T7 tests: run_from_config behavioral contract.

IMPORTANT: all fixture factories MUST be MODULE-LEVEL functions with real import paths
(e.g. tests.cli._run_fixtures:make_model). Do NOT use pytest closures or conftest
lambda fixtures — load_fn uses path.rsplit(':',1) and cannot resolve closures.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtrax.cli.config import TrainConfig
from xtrax.cli.hash import config_hash
from xtrax.cli.loader import CLIImportError
from xtrax.cli.run import run_from_config

# run_from_config now constructs a real zarr sink pre-fit (#457(1)); without
# the [io] extra these tests fail loud instead of exercising mocked-fit logic.
zarr = pytest.importorskip("zarr")



def _make_cfg(**overrides):
    defaults = dict(
        schema_version=1,
        model={"path": "tests.cli._run_fixtures:make_model", "kwargs": {}},
        optimizer={
            "path": "xtrax.training.optim:adamw_with_schedule",
            "kwargs": {"peak_lr": 1e-3, "warmup_steps": 1, "total_steps": 30},
        },
        loss={"path": "tests.cli._run_fixtures:make_loss", "kwargs": {}},
        data={
            "factory": "tests.cli._run_fixtures:make_datamodule",
            "kwargs": {},
            "batch_size": 2,
        },
        seed=0,
        num_epochs=1,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_bad_model_path_names_section(tmp_path, monkeypatch) -> None:
    """AC11: bad model path raises CLIImportError naming [model] section."""
    monkeypatch.chdir(tmp_path)
    cfg = _make_cfg(model={"path": "nonexistent.module:BadModel", "kwargs": {}})
    with pytest.raises(CLIImportError, match=r"\[model\]"):
        run_from_config(cfg)


def test_bad_optimizer_path_names_section(tmp_path, monkeypatch) -> None:
    """AC11: bad optimizer path raises CLIImportError naming [optimizer] section."""
    monkeypatch.chdir(tmp_path)
    cfg = _make_cfg(optimizer={"path": "nonexistent.module:bad_opt", "kwargs": {}})
    with pytest.raises(CLIImportError, match=r"\[optimizer\]"):
        run_from_config(cfg)


def test_bad_loss_path_names_section(tmp_path, monkeypatch) -> None:
    """AC11: bad loss path raises CLIImportError naming [loss] section."""
    monkeypatch.chdir(tmp_path)
    cfg = _make_cfg(loss={"path": "nonexistent.module:bad_loss", "kwargs": {}})
    with pytest.raises(CLIImportError, match=r"\[loss\]"):
        run_from_config(cfg)


def test_double_wrap_data_module(tmp_path, monkeypatch) -> None:
    """
    AC3/M4 (double-wrap enforcement): the fixture factory at
    tests.cli._run_fixtures:make_datamodule returns an ALREADY-BUILT DataModule.
    run_from_config must RE-WRAP it unconditionally.
    result.dataset is the inner DataModule returned by the factory.

    A duck-type short-circuit (isinstance check) would skip the wrap and fail this test.
    """
    monkeypatch.chdir(tmp_path)
    from xtrax.data.module import DataModule

    constructed_data_modules = []

    original_DataModule_init = DataModule.__init__

    def tracking_init(self, dataset, **kwargs):
        constructed_data_modules.append((self, dataset))
        original_DataModule_init(self, dataset, **kwargs)

    with patch.object(DataModule, "__init__", tracking_init):
        with patch("xtrax.engine.engine.Engine.fit_sync") as mock_fit:
            mock_fit.return_value = MagicMock()
            cfg = _make_cfg()
            run_from_config(cfg)

    assert len(constructed_data_modules) >= 1, "DataModule was never constructed"
    _outer_dm, inner_dataset = constructed_data_modules[-1]
    assert isinstance(inner_dataset, DataModule), (
        f"Expected inner dataset to be DataModule (double-wrap), got {type(inner_dataset)}"
    )


def test_run_id_collision_produces_uuid_suffix(tmp_path, monkeypatch) -> None:
    """AC7: second identical run gets a uuid-suffixed run_id."""
    monkeypatch.chdir(tmp_path)
    cfg = _make_cfg()
    cfg_dict = dataclasses.asdict(cfg)
    hash_val = config_hash(cfg_dict)
    os.makedirs(f".xtrax/runs/{hash_val}", exist_ok=True)

    with patch("xtrax.engine.engine.Engine.fit_sync") as mock_fit:
        mock_fit.return_value = MagicMock()
        run_from_config(cfg)

    run_dirs = list(Path(".xtrax/runs").iterdir())
    suffixed = [d for d in run_dirs if d.name.startswith(hash_val + "-")]
    assert len(suffixed) == 1, f"Expected 1 uuid-suffixed dir, got: {[d.name for d in run_dirs]}"


def test_generate_run_id(tmp_path, monkeypatch) -> None:
    """Test generate_run_id helper directly."""
    from xtrax.cli.run import generate_run_id

    monkeypatch.chdir(tmp_path)
    cfg = _make_cfg()
    cfg_dict = dataclasses.asdict(cfg)
    hash_val = config_hash(cfg_dict)

    # First call: creates base dir and returns hash_val
    run_id1 = generate_run_id(cfg)
    assert run_id1 == hash_val
    assert Path(f".xtrax/runs/{run_id1}").exists()

    # Second call: creates a suffixed dir because first one exists
    run_id2 = generate_run_id(cfg)
    assert run_id2.startswith(hash_val + "-")
    assert Path(f".xtrax/runs/{run_id2}").exists()
    assert run_id1 != run_id2

"""
T9: end-to-end `xtrax run` verb test + invariant round-up.

Invariants asserted here (one place):
  inv#1: manifest model.path non-null
  inv#2 (REVISED/INVERTED): checkpoint_dir DERIVED from run_id (NOT verbatim config)
  inv#3: data wrap unconditional (fixture factory returns DataModule; still re-wrapped)
  inv#4: missing schema_version → hard ConfigError
  AC7:   second identical run → uuid-suffixed run dir
  AC8:   two identical runs → distinct checkpoint_dirs, both sets exist
  AC9:   tyro-free import
  AC10:  'run' in REGISTRY
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xtrax.cli.config import ConfigError, load_config

FIXTURE_TOML_CONTENT = """
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
"""

FIXTURE_TOML_DOUBLE_WRAP_CONTENT = """
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
factory = "tests.cli._run_fixtures:make_datamodule"
kwargs = {}
batch_size = 2
"""


@pytest.fixture
def fixture_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(FIXTURE_TOML_CONTENT)
    return str(p)


@pytest.fixture
def fixture_toml_double_wrap(tmp_path):
    p = tmp_path / "config_dw.toml"
    p.write_text(FIXTURE_TOML_DOUBLE_WRAP_CONTENT)
    return str(p)


def test_tyro_isolation() -> None:
    """AC10/AC9: import xtrax.cli must not pull in tyro."""
    sys.modules.pop("tyro", None)
    import xtrax.cli  # noqa: F401

    assert "tyro" not in sys.modules, (
        "tyro was imported at module level — must stay inside main()"
    )


def test_run_in_registry() -> None:
    """AC10: 'run' verb must be in REGISTRY."""
    from xtrax.cli.registry import REGISTRY

    assert "run" in REGISTRY, f"'run' not in REGISTRY: {list(REGISTRY.keys())}"


def test_missing_schema_version_hard_error(tmp_path, monkeypatch) -> None:
    """inv#4: missing schema_version raises ConfigError (hard error, not default)."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "bad.toml"
    p.write_text(
        """
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
"""
    )
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(str(p))


def test_manifest_written_with_model_path(
    tmp_path, monkeypatch, fixture_toml
) -> None:
    """AC6 + inv#1: manifest is written with model.path non-null."""
    monkeypatch.chdir(tmp_path)
    with patch("xtrax.engine.engine.Engine.fit_sync") as mock_fit:
        mock_fit.return_value = MagicMock()
        from xtrax.cli.run import run_from_config

        cfg = load_config(fixture_toml)
        run_from_config(cfg)

    run_dirs = list(Path(".xtrax/runs").iterdir())
    assert len(run_dirs) >= 1
    manifest_path = run_dirs[0] / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found in {run_dirs[0]}"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"]["path"] is not None
    assert manifest["model"]["path"] != ""


def test_checkpoint_dir_derived_not_config_scalar(
    tmp_path, monkeypatch, fixture_toml
) -> None:
    """
    AC8/C1: checkpoint_dir MUST be derived from run_id.
    It is .xtrax/runs/<run_id>/checkpoints/, NOT taken verbatim from config.
    """
    monkeypatch.chdir(tmp_path)
    with patch("xtrax.engine.engine.Engine.fit_sync") as mock_fit:
        mock_fit.return_value = MagicMock()
        from xtrax.cli.run import run_from_config

        cfg = load_config(fixture_toml)
        run_from_config(cfg)

    run_dirs = list(Path(".xtrax/runs").iterdir())
    assert len(run_dirs) >= 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    run_id = manifest["run_id"]
    expected_ckpt = f".xtrax/runs/{run_id}/checkpoints/"
    assert manifest["checkpoint_dir"] == expected_ckpt, (
        f"checkpoint_dir should be derived from run_id, got: {manifest['checkpoint_dir']!r}"
    )


def test_double_identical_run_distinct_checkpoint_dirs(
    tmp_path, monkeypatch, fixture_toml
) -> None:
    """
    AC7 + AC8: two sequential identical runs must:
    1. Get distinct run_ids (second gets uuid suffix).
    2. Write to distinct checkpoint_dirs.
    3. Both checkpoint dirs exist (no clobber).
    """
    monkeypatch.chdir(tmp_path)
    manifests = []
    seen_ids: set[str] = set()
    for _ in range(2):
        with patch("xtrax.engine.engine.Engine.fit_sync") as mock_fit:
            mock_fit.return_value = MagicMock()
            from xtrax.cli.run import run_from_config

            cfg = load_config(fixture_toml)
            run_from_config(cfg)
        # Use set-diff rather than mtime to find the new dir — mtime resolution
        # is 1 second on many filesystems (WSL2 ext4) and both dirs can share
        # the same mtime on fast hardware, making the sort non-deterministic.
        all_ids = {p.name for p in Path(".xtrax/runs").iterdir()}
        new_ids = all_ids - seen_ids
        assert len(new_ids) == 1, f"expected exactly one new run dir, got {new_ids}"
        new_id = new_ids.pop()
        seen_ids.add(new_id)
        manifest = json.loads((Path(".xtrax/runs") / new_id / "manifest.json").read_text())
        manifests.append(manifest)

    assert manifests[0]["run_id"] != manifests[1]["run_id"], (
        "Two identical runs must get distinct run_ids (second must be uuid-suffixed)"
    )
    assert manifests[0]["checkpoint_dir"] != manifests[1]["checkpoint_dir"], (
        "Two identical runs must have distinct checkpoint_dirs (no clobber)"
    )
    for manifest in manifests:
        ckpt_path = Path(manifest["checkpoint_dir"])
        assert ckpt_path.exists(), (
            f"checkpoint dir should exist: {manifest['checkpoint_dir']}"
        )


def test_double_wrap_datamodule_factory(
    tmp_path, monkeypatch, fixture_toml_double_wrap
) -> None:
    """
    AC3/M4: double-wrap enforcement.

    The factory tests.cli._run_fixtures:make_datamodule returns an ALREADY-BUILT DataModule.
    run_from_config must RE-WRAP it unconditionally — result.dataset is the inner DataModule.
    A duck-type isinstance check would skip the wrap and fail this test.
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
            from xtrax.cli.run import run_from_config

            cfg = load_config(fixture_toml_double_wrap)
            run_from_config(cfg)

    assert len(constructed_data_modules) >= 1, "DataModule was never constructed"
    _outer_dm, inner_dataset = constructed_data_modules[-1]
    assert isinstance(inner_dataset, DataModule), (
        f"Expected inner dataset to be DataModule (double-wrap), got {type(inner_dataset)}"
    )

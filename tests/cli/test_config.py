"""AC9 + AC12 + inv#4: TrainConfig parse and validation tests."""

from __future__ import annotations

import pytest

from xtrax.cli.config import ConfigError, load_config


def _write_toml(tmp_path, content: str) -> str:
    p = tmp_path / "config.toml"
    p.write_text(content)
    return str(p)


VALID_TOML = """
schema_version = 1
seed = 42
num_epochs = 3

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {learning_rate = 1e-3, total_steps = 300}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 4
"""


def test_valid_config_parses(tmp_path) -> None:
    p = _write_toml(tmp_path, VALID_TOML)
    cfg = load_config(p)
    assert cfg.schema_version == 1
    assert cfg.seed == 42
    assert cfg.num_epochs == 3
    assert cfg.loss["path"] == "tests.cli._run_fixtures:make_loss"


def test_missing_schema_version_raises(tmp_path) -> None:
    """inv#4: missing schema_version must raise ConfigError."""
    toml = VALID_TOML.replace("schema_version = 1\n", "")
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(p)


def test_missing_loss_section_raises(tmp_path) -> None:
    """AC-loss: missing [loss] section must raise ConfigError."""
    toml = VALID_TOML.replace(
        """
[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

""",
        "",
    )
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ConfigError, match="loss"):
        load_config(p)


def test_missing_model_section_raises(tmp_path) -> None:
    lines = [
        line
        for line in VALID_TOML.splitlines()
        if not line.startswith("[model]") and "make_model" not in line
    ]
    p = _write_toml(tmp_path, "\n".join(lines))
    with pytest.raises(ConfigError, match="model"):
        load_config(p)


def test_num_epochs_none_raises(tmp_path) -> None:
    """AC12: None num_epochs must raise ConfigError (range(None) TypeError prevented)."""
    toml = VALID_TOML.replace("num_epochs = 3", "")
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ConfigError, match="num_epochs"):
        load_config(p)


def test_num_epochs_zero_raises(tmp_path) -> None:
    """AC12: num_epochs=0 must raise ConfigError."""
    toml = VALID_TOML.replace("num_epochs = 3", "num_epochs = 0")
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ConfigError, match="num_epochs"):
        load_config(p)


def test_num_epochs_negative_raises(tmp_path) -> None:
    toml = VALID_TOML.replace("num_epochs = 3", "num_epochs = -1")
    p = _write_toml(tmp_path, toml)
    with pytest.raises(ConfigError, match="num_epochs"):
        load_config(p)


def test_loss_path_and_kwargs_in_config(tmp_path) -> None:
    """AC-loss: loss section parses to dict with path and kwargs."""
    p = _write_toml(tmp_path, VALID_TOML)
    cfg = load_config(p)
    assert "path" in cfg.loss
    assert "kwargs" in cfg.loss
    assert cfg.loss["path"] == "tests.cli._run_fixtures:make_loss"


def test_loss_kwargs_default_empty(tmp_path) -> None:
    """AC-loss: empty kwargs table parses without error."""
    toml = VALID_TOML  # VALID_TOML already has kwargs = {}
    p = _write_toml(tmp_path, toml)
    cfg = load_config(p)
    assert cfg.loss.get("kwargs") == {} or cfg.loss.get("kwargs") is None

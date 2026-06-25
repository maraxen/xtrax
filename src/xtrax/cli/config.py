"""TrainConfig schema and config-domain errors for the `xtrax run` verb."""

import tomllib
from dataclasses import dataclass

from xtrax.cli.errors import CLIError

CURRENT_SCHEMA_VERSION = 1


@dataclass
class TrainConfig:
    """Parsed training configuration for `xtrax run`.

    Each section (model, optimizer, loss, data) is a raw dict with import-path
    keys and kwargs, hydrated from TOML by ``load_config`` (T3).
    """

    schema_version: int
    model: dict
    optimizer: dict
    loss: dict
    data: dict
    seed: int
    num_epochs: int


class ConfigError(CLIError):
    """Raised when a training config file is invalid or incomplete."""

    pass


def load_config(path: str) -> TrainConfig:
    """Parse and validate a training config TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    if "schema_version" not in raw:
        raise ConfigError("missing required field: schema_version")

    for section in ("model", "optimizer", "loss", "data"):
        if section not in raw:
            raise ConfigError(f"missing required section: [{section}]")

    num_epochs = raw.get("num_epochs")
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ConfigError(f"num_epochs must be a positive int, got: {num_epochs!r}")

    seed = raw.get("seed")
    if not isinstance(seed, int):
        raise ConfigError(f"seed must be an int, got: {seed!r}")

    return TrainConfig(
        schema_version=raw["schema_version"],
        model=raw["model"],
        optimizer=raw["optimizer"],
        loss=raw["loss"],
        data=raw["data"],
        seed=seed,
        num_epochs=num_epochs,
    )

"""TrainConfig schema and config-domain errors for the `xtrax run` verb.

`load_config` composes `xtrax.config`'s domain-agnostic primitives
(the canonical, dog-fooded usage referenced by that module's spec).
"""

from dataclasses import dataclass

from xtrax.cli.errors import CLIError
from xtrax.config import check_schema_version, load_toml_document, require_field, require_sections

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
    raw = load_toml_document(path, ConfigError)
    check_schema_version(raw, CURRENT_SCHEMA_VERSION, ConfigError)
    require_sections(raw, ("model", "optimizer", "loss", "data"), ConfigError)
    num_epochs = require_field(
        raw, "num_epochs", lambda v: isinstance(v, int) and v > 0, ConfigError
    )
    seed = require_field(raw, "seed", lambda v: isinstance(v, int), ConfigError)

    return TrainConfig(
        schema_version=raw["schema_version"],
        model=raw["model"],
        optimizer=raw["optimizer"],
        loss=raw["loss"],
        data=raw["data"],
        seed=seed,
        num_epochs=num_epochs,
    )

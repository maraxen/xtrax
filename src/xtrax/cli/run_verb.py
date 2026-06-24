"""xtrax run verb: RunArgs + run_run for REGISTRY wiring (AC10)."""

from dataclasses import dataclass

from xtrax.cli.config import ConfigError, load_config
from xtrax.cli.run import run_from_config


@dataclass
class RunArgs:
    config: str  # path to config.toml


def run_run(args: RunArgs) -> None:
    """Entry point for `xtrax run config.toml`."""
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        raise SystemExit(f"ConfigError: {e}") from e
    run_from_config(cfg)

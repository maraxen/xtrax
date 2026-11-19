"""xtrax run verb: RunArgs + run_run for REGISTRY wiring (AC10)."""

from dataclasses import dataclass

from xtrax.cli.config import ConfigError, load_config
from xtrax.cli.run import run_from_config
from xtrax.compilation_cache import enable_compilation_cache


@dataclass
class RunArgs:
    config: str  # path to config.toml
    #: JAX persistent compilation cache: a directory, or 'off'. Unset resolves from
    #: configuration (see xtrax.compilation_cache).
    compilation_cache: str | None = None


def run_run(args: RunArgs) -> None:
    """Entry point for `xtrax run config.toml`."""
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        raise SystemExit(f"ConfigError: {e}") from e
    try:
        enable_compilation_cache(args.compilation_cache)
    except ValueError as e:
        raise SystemExit(f"compilation cache: {e}") from e
    run_from_config(cfg)

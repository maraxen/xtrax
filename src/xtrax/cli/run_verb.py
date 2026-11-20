"""xtrax run verb: RunArgs + run_run for REGISTRY wiring (AC10)."""

from dataclasses import dataclass

from xtrax import jax_cache
from xtrax.cli.config import ConfigError, load_config
from xtrax.cli.run import run_from_config


@dataclass
class RunArgs:
    config: str  # path to config.toml
    #: JAX persistent compilation cache: a directory, or 'off'. Unset resolves through
    #: xtrax.jax_cache (directory from env / ~/.config/xtrax; on/off from [tool.xtrax]).
    compilation_cache: str | None = None


def run_run(args: RunArgs) -> None:
    """Entry point for `xtrax run config.toml`."""
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        raise SystemExit(f"ConfigError: {e}") from e
    try:
        jax_cache.configure(args.compilation_cache)
    except (ValueError, OSError) as e:
        raise SystemExit(f"JAX compilation cache: {e}") from e
    run_from_config(cfg)

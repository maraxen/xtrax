"""Sweep CLI verb implementation for xtrax.

Provides grid search parameter sweeping functionality, sequentially running
combinations in a single process while isolating compilation and execution memory.
"""

from __future__ import annotations

import dataclasses
import datetime
import gc
import itertools
import json
import os
import sys
import tempfile
import tomllib
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from xtrax.cli.config import ConfigError, TrainConfig, load_config
from xtrax.cli.run import generate_run_id, run_from_config

# Initialize tyro at module level for entrypoint's lazy resolver to inject it
tyro: Any = None


@dataclass
class SweepArgs:
    """Arguments for the sweep verb."""

    config_path: tyro.conf.Positional[str]


def _validate_and_flatten_axes(
    d: dict[str, Any], path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], list[Any]]]:
    """Validate that every leaf in sweep.axes is a list, and flatten into tuple-paths."""
    flat = []
    for k, v in d.items():
        current_path = path + (k,)
        if isinstance(v, dict):
            flat.extend(_validate_and_flatten_axes(v, current_path))
        else:
            if not isinstance(v, list):
                raise ConfigError(
                    f"sweep axis at {'.'.join(current_path)} must be a list, got {type(v).__name__}"
                )
            flat.append((current_path, v))
    return flat


def run_sweep(args: SweepArgs) -> None:
    """Run a grid-search parameter sweep sequentially in the same process."""
    # 1. Load and validate base config
    base_cfg = load_config(args.config_path)

    # 2. Parse config file to extract sweep.axes
    with open(args.config_path, "rb") as f:
        raw_toml = tomllib.load(f)

    sweep_sec = raw_toml.get("sweep")
    if sweep_sec is not None:
        if not isinstance(sweep_sec, dict):
            raise ConfigError("[sweep] section must be a dictionary")
        axes = sweep_sec.get("axes")
        if axes is not None:
            if not isinstance(axes, dict):
                raise ConfigError("sweep.axes must be a dictionary")
        else:
            axes = {}
    else:
        axes = {}

    # Validate leaves are lists and flatten axes
    if not axes:
        paths: list[tuple[str, ...]] = []
        combinations: list[tuple[Any, ...]] = [()]
    else:
        flat = _validate_and_flatten_axes(axes)
        paths = [item[0] for item in flat]
        value_lists = [item[1] for item in flat]
        combinations = list(itertools.product(*value_lists))

    # 3. Configure JAX compilation cache
    jax.config.update("jax_compilation_cache_dir", ".xtrax/jax_cache")

    # 4. Determine sweep directory
    sweep_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_id = f"sweep_{sweep_timestamp}_{uuid.uuid4().hex[:6]}"
    sweep_dir = f".xtrax/sweeps/{sweep_id}"
    os.makedirs(sweep_dir, exist_ok=True)

    # 5. Track run_id to overrides mappings
    sweep_manifest: dict[str, dict[str, Any]] = {}

    # 6. Sequential execution loop
    for comb in combinations:
        cfg_dict = dataclasses.asdict(base_cfg)

        # Traverse cfg_dict and set override leaf values
        for path, val in zip(paths, comb):
            current = cfg_dict
            for key in path[:-1]:
                current = current.setdefault(key, {})
            current[path[-1]] = val

        # Re-instantiate TrainConfig
        cfg_obj = TrainConfig(**cfg_dict)

        # Generate run ID and reserve directory
        run_id = generate_run_id(cfg_obj)

        # Record overrides
        overrides = {".".join(p): v for p, v in zip(paths, comb)}
        sweep_manifest[run_id] = overrides

        # Write sweep_manifest.json incrementally and atomically
        temp_fd, temp_path = tempfile.mkstemp(dir=sweep_dir, prefix="manifest_", suffix=".json")
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(sweep_manifest, f, indent=2)
            os.replace(temp_path, os.path.join(sweep_dir, "sweep_manifest.json"))
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # Execute run with exception handling
        try:
            state = run_from_config(cfg_obj, run_id=run_id)
            print(f"Sweep run {run_id} completed successfully with overrides {overrides}")

            # Flush device execution queues
            jax.tree_util.tree_map(lambda x: getattr(x, "block_until_ready", lambda: x)(), state)

            del state
            gc.collect()
            jax.clear_caches()

        except Exception as e:
            print(f"Sweep run {run_id} failed with error: {e}", file=sys.stderr)
            traceback.clear_frames(e.__traceback__)

            # Run dummy computation on all devices to flush queues
            try:
                jax.tree_util.tree_map(
                    lambda d: jax.device_put(jnp.zeros(1), d).block_until_ready(),
                    jax.devices(),
                )
            except Exception as dummy_err:
                print(f"Failed to flush JAX device queues on error: {dummy_err}", file=sys.stderr)

            gc.collect()
            jax.clear_caches()


__all__ = ["SweepArgs", "run_sweep"]

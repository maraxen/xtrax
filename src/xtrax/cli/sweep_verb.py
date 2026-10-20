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


def _read_sweep_axes(config_path: str) -> dict[str, Any]:
    """Return the ``[sweep.axes]`` table of a config file (empty when absent)."""
    with open(config_path, "rb") as f:
        raw_toml = tomllib.load(f)

    sweep_sec = raw_toml.get("sweep")
    if sweep_sec is None:
        return {}
    if not isinstance(sweep_sec, dict):
        raise ConfigError("[sweep] section must be a dictionary")
    axes = sweep_sec.get("axes")
    if axes is None:
        return {}
    if not isinstance(axes, dict):
        raise ConfigError("sweep.axes must be a dictionary")
    return axes


def _sweep_grid(axes: dict[str, Any]) -> tuple[list[tuple[str, ...]], list[tuple[Any, ...]]]:
    """Return ``(leaf paths, cartesian product of leaf values)`` for the axes."""
    if not axes:
        return [], [()]
    flat = _validate_and_flatten_axes(axes)
    paths = [item[0] for item in flat]
    value_lists = [item[1] for item in flat]
    return paths, list(itertools.product(*value_lists))


def _write_sweep_manifest(sweep_dir: str, sweep_manifest: dict[str, dict[str, Any]]) -> None:
    """Write ``sweep_manifest.json`` atomically (temp file, then replace)."""
    temp_fd, temp_path = tempfile.mkstemp(dir=sweep_dir, prefix="manifest_", suffix=".json")
    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(sweep_manifest, f, indent=2)
        os.replace(temp_path, os.path.join(sweep_dir, "sweep_manifest.json"))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _flush_device_queues() -> None:
    """Run a dummy computation on every device to drain queued work after a failure."""
    try:
        jax.tree_util.tree_map(
            lambda d: jax.device_put(jnp.zeros(1), d).block_until_ready(),
            jax.devices(),
        )
    except Exception as dummy_err:
        print(f"Failed to flush JAX device queues on error: {dummy_err}", file=sys.stderr)


def _run_sweep_point(
    cfg_obj: TrainConfig,
    run_id: str,
    sweep_id: str,
    overrides: dict[str, Any],
) -> None:
    """Run one sweep configuration; record and survive its failure."""
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
        _record_sweep_failure(run_id, sweep_id, overrides, e)
        traceback.clear_frames(e.__traceback__)
        _flush_device_queues()
        gc.collect()
        jax.clear_caches()


def run_sweep(args: SweepArgs) -> None:
    """Run a grid-search parameter sweep sequentially in the same process."""
    # 1. Load and validate base config
    base_cfg = load_config(args.config_path)

    # 2. Parse config file to extract sweep.axes; validate leaves and flatten
    paths, combinations = _sweep_grid(_read_sweep_axes(args.config_path))

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
        _write_sweep_manifest(sweep_dir, sweep_manifest)

        _run_sweep_point(cfg_obj, run_id, sweep_id, overrides)


def _record_sweep_failure(
    run_id: str,
    sweep_id: str,
    overrides: dict,
    exc: BaseException,
) -> None:
    """Record a sweep run that died before Engine could open its own ledger.

    A failure inside ``fit_sync`` is already written by the Engine's ledger, so
    this only fills the earlier window -- a config that fails to resolve, a
    dataset that will not construct -- where the run would otherwise leave no
    trace at all beyond a line on stderr. ``find_run`` is what keeps the two from
    double-writing.

    Fail-open: a sweep must continue to its next configuration even if this
    bookkeeping cannot be written.
    """
    from xtrax.telemetry.ledger import RunLedger, find_run
    from xtrax.telemetry.record import KIND_TRAIN, STATUS_FAILED

    try:
        if find_run(run_id) is not None:
            return
        with RunLedger.open(
            run_id,
            kind=KIND_TRAIN,
            context={"sweep_id": sweep_id, "overrides": str(overrides)},
        ) as ledger:
            ledger.set_status(
                STATUS_FAILED,
                f"sweep run failed before training began: {type(exc).__name__}: {exc}",
            )
    except Exception as ledger_exc:  # noqa: BLE001 - never break the sweep loop
        print(
            f"warning: could not record sweep failure for {run_id}: {ledger_exc}",
            file=sys.stderr,
        )


__all__ = ["SweepArgs", "run_sweep"]

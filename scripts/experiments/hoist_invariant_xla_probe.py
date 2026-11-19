#!/usr/bin/env python3
"""Probe whether XLA hoists a truly key-invariant encoder out of a map.

Pre-registered in ``hoist_invariant_xla_probe.bth.toml`` (committed before this
script was written; thresholds 1.25 / 2.0 / 4.0 are fixed there). The question:
does a jitted map over N elements whose per-element function calls an expensive
encoder that does NOT read the mapped input cost the same as a hand-hoisted twin
(encoder called once outside the map)? Two map paths are measured: ``jax.vmap``
and ``xtrax.chunked_map`` with a batch size.

Arms, for path in {vmap, chunked}:

- ``inv_naive``: ``map(lambda x: per_element(encoder(coords), x))``.
- ``inv_hoisted``: ``enc = encoder(coords)`` outside the map.
- ``fd_naive``: the encoder reads a per-element PRNG key (coords plus noise
  scaled by ``scale``, passed as an array). This is the instrument control: a
  genuinely replicated encoder must show a large naive/hoisted ratio.
- ``fd_hoisted``: ``fd_encoder(coords, keys[0], scale)`` once, outside the map.

At ``scale = 0`` the naive and hoisted outputs must agree; a mismatch fails the
pair. Timing is the median of ``--reps`` calls after ``--warmup`` calls, each
followed by ``block_until_ready``, on the compiled executable. Compiled
``temp_size_in_bytes`` and the optimized-HLO count of ``' sine('`` are reported
for context; they are not gated.

Usage (through bathos, from the worktree root):
    bth run --project-slug xtrax -- uv run --no-sync python3 \\
        scripts/experiments/hoist_invariant_xla_probe.py --out PATH

Smoke check, not a run that gets cited:
    uv run python scripts/experiments/hoist_invariant_xla_probe.py --smoke --out PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import xtrax
from xtrax import chunked_map

log = logging.getLogger("hoist_invariant_xla_probe")

PATHS = ("vmap", "chunked")
KINDS = ("inv", "fd")
SMOKE = {"n": 8, "length": 32, "channels": 8, "passes": 4, "reps": 3}


def encoder(coords: jax.Array, w: jax.Array, passes: int) -> jax.Array:
    """Pairwise-distance encoder: (L, 3) -> (L, L, C) with chained transcendentals."""
    diff = coords[:, None, :] - coords[None, :, :]
    d = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)
    h = d[..., None] * w
    for _ in range(passes):
        h = jnp.sin(h) * 1.1 + jnp.cos(h * 0.7)
    return h


def fd_encoder(
    coords: jax.Array, key: jax.Array, scale: jax.Array, w: jax.Array, passes: int
) -> jax.Array:
    """Encoder made falsely dependent on a per-element key (noise scaled by ``scale``)."""
    noisy = coords + scale * jax.random.normal(key, coords.shape)
    return encoder(noisy, w, passes)


def per_element(enc: jax.Array, x: jax.Array) -> jax.Array:
    """Cheap per-element reduction of the encoder output against a length-L vector."""
    return jnp.einsum("ijc,j->ic", enc, x).sum()


def _map(fn: Callable[[Any], jax.Array], xs: Any, path: str, batch_size: int) -> jax.Array:
    if path == "vmap":
        return jax.vmap(fn)(xs)
    return chunked_map(fn, xs, batch_size=batch_size)


def build_arm(
    kind: str, role: str, path: str, *, w: jax.Array, passes: int, batch_size: int
) -> Callable[..., jax.Array]:
    """Return a jitted arm. ``role`` is ``naive`` or ``hoisted``."""
    if kind == "inv":

        def naive(coords, xs, keys, scale):
            del keys, scale
            return _map(lambda x: per_element(encoder(coords, w, passes), x), xs, path, batch_size)

        def hoisted(coords, xs, keys, scale):
            del keys, scale
            enc = encoder(coords, w, passes)
            return _map(lambda x: per_element(enc, x), xs, path, batch_size)

    else:

        def naive(coords, xs, keys, scale):
            def one(xk):
                x, key = xk
                return per_element(fd_encoder(coords, key, scale, w, passes), x)

            return _map(one, (xs, keys), path, batch_size)

        def hoisted(coords, xs, keys, scale):
            enc = fd_encoder(coords, keys[0], scale, w, passes)
            return _map(lambda x: per_element(enc, x), xs, path, batch_size)

    return jax.jit(naive if role == "naive" else hoisted)


def _median_seconds(compiled: Any, args: tuple[Any, ...], warmup: int, reps: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


def measure_arm(
    name: str, fn: Callable[..., jax.Array], args: tuple[Any, ...], warmup: int, reps: int
) -> dict[str, Any]:
    """Compile once, record temp bytes and sine count, then time the executable."""
    compiled = fn.lower(*args).compile()
    mem = compiled.memory_analysis()
    temp = float(mem.temp_size_in_bytes) if mem is not None else -1.0
    sines = compiled.as_text().count(" sine(")
    median = _median_seconds(compiled, args, warmup, reps)
    out = jax.block_until_ready(compiled(*args))
    log.info("%-26s median=%.6fs temp_bytes=%.0f sine_ops=%d", name, median, temp, sines)
    return {"median": median, "temp": temp, "sines": sines, "out": np.asarray(out)}


def _ratio(num: float, den: float) -> float:
    if num < 0 or den < 0 or den == 0:
        return -1.0
    return float(num / den)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    n, length, channels, passes = args.n, args.length, args.channels, args.passes
    rng = np.random.default_rng(0)
    coords = jnp.asarray(np.random.default_rng(1).normal(size=(length, 3)) * 3.0, jnp.float32)
    xs = jnp.asarray(rng.normal(size=(n, length)), jnp.float32)
    keys = jax.random.split(jax.random.key(0), n)
    w = jnp.asarray(np.linspace(0.5, 2.0, channels), jnp.float32)
    scale_zero = jnp.float32(0.0)

    failed: list[str] = []
    medians: dict[str, float] = {}
    temps: dict[str, float] = {}
    ratio: dict[str, float] = {}

    for kind in KINDS:
        for path in PATHS:
            pair_ok = True
            outs: dict[str, np.ndarray] = {}
            for role in ("naive", "hoisted"):
                arm = f"{kind}_{role}_{path}"
                try:
                    fn = build_arm(kind, role, path, w=w, passes=passes, batch_size=args.batch_size)
                    m = measure_arm(arm, fn, (coords, xs, keys, scale_zero), args.warmup, args.reps)
                    medians[arm] = m["median"]
                    temps[arm] = m["temp"]
                    outs[role] = m["out"]
                except Exception:  # compile, OOM (RESOURCE_EXHAUSTED) or runtime failure
                    log.exception("arm %s failed", arm)
                    failed.append(arm)
                    pair_ok = False
            if pair_ok and not np.allclose(outs["naive"], outs["hoisted"], rtol=1e-4, atol=1e-3):
                log.error("%s_%s: naive and hoisted outputs disagree", kind, path)
                failed.append(f"{kind}_hoisted_{path}:mismatch")
                pair_ok = False
            if pair_ok:
                ratio[f"{kind}_wall_ratio_{path}"] = _ratio(
                    medians[f"{kind}_naive_{path}"], medians[f"{kind}_hoisted_{path}"]
                )
                ratio[f"{kind}_temp_ratio_{path}"] = _ratio(
                    temps[f"{kind}_naive_{path}"], temps[f"{kind}_hoisted_{path}"]
                )
            else:
                ratio[f"{kind}_wall_ratio_{path}"] = -1.0
                ratio[f"{kind}_temp_ratio_{path}"] = -1.0

    inv_walls = [ratio[f"inv_wall_ratio_{p}"] for p in PATHS]
    fd_walls = [ratio[f"fd_wall_ratio_{p}"] for p in PATHS]
    device = jax.devices()[0]
    return {
        "n_failed": len(failed),
        "failed_arms": ",".join(failed),
        "platform": device.platform,
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "xtrax_file": str(xtrax.__file__),
        "inv_wall_ratio_vmap": ratio["inv_wall_ratio_vmap"],
        "inv_wall_ratio_chunked": ratio["inv_wall_ratio_chunked"],
        "fd_wall_ratio_vmap": ratio["fd_wall_ratio_vmap"],
        "fd_wall_ratio_chunked": ratio["fd_wall_ratio_chunked"],
        "max_inv_wall_ratio": float(max(inv_walls)),
        "min_fd_wall_ratio": float(min(fd_walls)),
        "inv_temp_ratio_vmap": ratio["inv_temp_ratio_vmap"],
        "inv_temp_ratio_chunked": ratio["inv_temp_ratio_chunked"],
        "fd_temp_ratio_vmap": ratio["fd_temp_ratio_vmap"],
        "fd_temp_ratio_chunked": ratio["fd_temp_ratio_chunked"],
        "_medians": medians,
        "_temps": temps,
    }


def _markdown_table(medians: dict[str, float], temps: dict[str, float]) -> str:
    lines = ["| arm | median wall (s) | temp bytes |", "|---|---:|---:|"]
    for arm in sorted(medians):
        lines.append(f"| {arm} | {medians[arm]:.6f} | {temps.get(arm, -1.0):.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny shapes for a wiring check, not a result."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.smoke:
        for key, value in SMOKE.items():
            setattr(args, key, value)

    result = run_probe(args)
    medians = result.pop("_medians")
    temps = result.pop("_temps")
    print(_markdown_table(medians, temps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", args.out)

    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

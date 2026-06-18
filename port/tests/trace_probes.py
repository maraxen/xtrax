"""Trace-count probes for port manifest kernels (AC-5)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

def probe_safe_map(target: Callable[..., object]) -> None:
    """Exercise jitted safe_map with a tiny batch (trace-count gate)."""
    xs = jnp.arange(4, dtype=jnp.float32)

    def _double(x: jax.Array) -> jax.Array:
        return x * 2.0

    _ = target(_double, xs, batch_size=2)

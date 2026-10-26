"""Trace probes for audit performance gate (D4 / #1584)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


# `filter_jit_linear_kernel` replaced `sparse_filter_jit_kernel` when xtrax.sparse was removed
# (#6107). `sparse_filter_jit` was `eqx.filter_jit` unchanged, so the transform under the gate
# is the same; the model passed in is a dense `eqx.nn.Linear` instead of a BCOO-sparsified one.
@eqx.filter_jit
@chex.assert_max_traces(n=1)
def filter_jit_linear_kernel(m: eqx.nn.Linear, x: jnp.ndarray) -> tuple[int, ...]:
    """Representative `eqx.filter_jit` kernel over an Equinox module for the trace-count gate."""
    return m.weight.shape


def probe_filter_jit_linear_kernel(guarded: Callable[..., Any]) -> None:
    """Exercise the `eqx.filter_jit` kernel twice with stable inputs."""
    key = jax.random.PRNGKey(0)
    model = eqx.nn.Linear(4, 4, key=key)
    x = jnp.ones((4,))
    guarded(model, x)
    guarded(model, x)


def probe_stable_jnp_kernel(guarded: Callable[..., Any]) -> None:
    """Minimal stable-input probe for integration tests."""
    x = jnp.zeros(3)
    guarded(x)
    guarded(x)

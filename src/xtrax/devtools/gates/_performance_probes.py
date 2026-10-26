"""Trace probes for audit performance gate (D4 / #1584)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp

from xtrax._tree import model_jit


# `filter_jit_linear_kernel` replaced `sparse_filter_jit_kernel` when xtrax.sparse was removed
# (#6107). The name is kept for audit/performance_targets.toml. The transform under the gate is
# now `xtrax.model_jit`, the one place xtrax models enter a trace, over a two-array MLP held as a
# plain dict (a non-array leaf, the activation tag, rides along as a static compile key).
@model_jit
@chex.assert_max_traces(n=1)
def filter_jit_linear_kernel(m: dict[str, Any], x: jnp.ndarray) -> tuple[int, ...]:
    """Representative `model_jit` kernel over a plain-dict MLP for the trace-count gate."""
    h = jnp.tanh(x @ m["w1"] + m["b1"]) if m["act"] == "tanh" else x @ m["w1"] + m["b1"]
    return (h @ m["w2"]).shape


def probe_filter_jit_linear_kernel(guarded: Callable[..., Any]) -> None:
    """Exercise the `model_jit` kernel twice with stable inputs."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    model = {
        "w1": jax.random.normal(k1, (4, 4)),
        "b1": jnp.zeros((4,)),
        "w2": jax.random.normal(k2, (4, 2)),
        "act": "tanh",
    }
    x = jnp.ones((4,))
    guarded(model, x)
    guarded(model, x)


def probe_stable_jnp_kernel(guarded: Callable[..., Any]) -> None:
    """Minimal stable-input probe for integration tests."""
    x = jnp.zeros(3)
    guarded(x)
    guarded(x)

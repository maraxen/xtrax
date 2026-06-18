"""Fixture module for beartype import-hook smoke tests."""

import jax
import jax.numpy as jnp
from jaxtyping import Float


def strict_vec(x: Float[jax.Array, "3"]) -> Float[jax.Array, "3"]:
    return x


def bad_call() -> jax.Array:
    return strict_vec(jnp.ones(5))

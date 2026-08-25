"""Demo functions for CSE CLI tests."""
import jax.numpy as jnp

from xtrax.inference import AxisOverride, axis_config


def duplicated_compute(x):
    y = jnp.sin(x) * 2.0
    z = jnp.sin(x) * 2.0
    return y + z + jnp.exp(y)


def clean_compute(x):
    return jnp.sin(x) + jnp.cos(x)


@axis_config(AxisOverride(name="batch", default_batch_size=64))
def annotated_compute(x):
    return x + 1.0

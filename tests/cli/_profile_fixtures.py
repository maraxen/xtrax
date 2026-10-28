"""Callables for the ``xtrax profile`` verb tests, addressed as ``module:symbol``.

``host_loop`` is the negative control (one dispatch per element); ``vmapped_twin``
is its positive twin (one dispatch for the batch); ``fresh_jit_per_call`` rebuilds
its jit on every call, so it recompiles at steady state.
"""

import jax
import jax.numpy as jnp

_step = jax.jit(lambda row: jnp.tanh(row) * 2.0)
_vmapped = jax.jit(jax.vmap(lambda row: jnp.tanh(row) * 2.0))


def host_loop(x):
    """Host loop over the leading axis: several dispatches per element."""
    return jnp.stack([_step(x[b]) for b in range(x.shape[0])])


def vmapped_twin(x):
    """The batched twin of ``host_loop``: one executable for the whole batch."""
    return _vmapped(x)


def fresh_jit_per_call(x):
    """Builds a new jit on every call, so every steady call recompiles."""
    return jax.jit(lambda a: jnp.tanh(a) * 2.0)(x)


def identity(x):
    """No device work at all: the trace has zero executions."""
    return x


def raises(x):
    """Raises on every call, so the probe cannot measure anything."""
    raise RuntimeError("fixture failure")


def make_inputs(scale):
    """Factory form: ``(args, kwargs)`` with the scale axis as the kwarg ``x``."""
    return (), {"x": jnp.zeros((scale, 128), jnp.float32)}

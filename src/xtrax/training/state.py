import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.training.types import ResumableState


def init_state(model, optimizer, seed: int) -> ResumableState:
    return ResumableState(
        step=jnp.asarray(0, jnp.int32),
        key=jax.random.PRNGKey(seed),
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
        extras={},
    )

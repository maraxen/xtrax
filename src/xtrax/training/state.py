import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from xtrax.training.types import ResumableState


def init_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    seed: int,
) -> ResumableState:
    return ResumableState(
        step=jnp.asarray(0, jnp.int32),
        key=jax.random.PRNGKey(seed),
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
        extras={},
    )

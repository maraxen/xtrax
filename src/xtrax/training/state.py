import jax
import jax.numpy as jnp
import optax

from xtrax._tree import partition
from xtrax.training.types import PyTree, ResumableState


def init_state(
    model: PyTree,
    optimizer: optax.GradientTransformation,
    seed: int,
) -> ResumableState:
    return ResumableState(
        step=jnp.asarray(0, jnp.int32),
        key=jax.random.PRNGKey(seed),
        model=model,
        opt_state=optimizer.init(partition(model)[0]),
        extras={},
    )

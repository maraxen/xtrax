import jax
import jax.numpy as jnp

try:
    import optax
except ImportError:
    optax = None  # type: ignore

from xtrax._tree import partition
from xtrax.training.optim import GradientTransformation, require_optax
from xtrax.training.types import PyTree, ResumableState


def init_state(
    model: PyTree,
    optimizer: GradientTransformation,
    seed: int,
) -> ResumableState:
    require_optax()
    return ResumableState(
        step=jnp.asarray(0, jnp.int32),
        key=jax.random.PRNGKey(seed),
        model=model,
        opt_state=optimizer.init(partition(model)[0]),
        extras={},
    )

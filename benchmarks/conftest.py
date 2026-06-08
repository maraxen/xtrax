import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


class _TinyMLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(64, 64, key=k1),
            eqx.nn.Linear(64, 1, key=k2),
        ]

    def __call__(self, x):
        # eqx.nn.Linear operates on rank-1 input — vmap over the batch axis.
        def _forward(xi):
            xi = jax.nn.tanh(self.layers[0](xi))
            return self.layers[1](xi)
        return jax.vmap(_forward)(x)


@pytest.fixture(scope="module")
def tiny_model():
    return _TinyMLP(jax.random.key(0))


@pytest.fixture(scope="module")
def synthetic_batch():
    k1, k2 = jax.random.split(jax.random.key(42))
    return {
        "inputs": jax.random.normal(k1, (32, 64)),
        "targets": jax.random.normal(k2, (32, 1)),
    }


@pytest.fixture(scope="module")
def trainer(tiny_model):
    def mse(pred, target):
        return jnp.mean((pred - target) ** 2)
    return Trainer(loss_fn=mse, optimizer=optax.adam(1e-3))


@pytest.fixture(scope="module")
def trainer_state(tiny_model, trainer):
    # Trainer has no init_state method — build ResumableState explicitly.
    opt_state = trainer.optimizer.init(eqx.filter(tiny_model, eqx.is_array))
    return ResumableState(
        step=jnp.int32(0),
        key=jax.random.key(0),
        model=tiny_model,
        opt_state=opt_state,
    )

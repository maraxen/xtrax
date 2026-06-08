import jax.numpy as jnp
import pytest

from xtrax.training.grad import accumulate_grads


@pytest.mark.parametrize("n_microbatches", [1, 2, 4, 8])
def test_accumulate_grads_scaling(
    benchmark, n_microbatches, tiny_model, synthetic_batch
):
    inputs = jnp.stack([synthetic_batch["inputs"]] * n_microbatches)
    targets = jnp.stack([synthetic_batch["targets"]] * n_microbatches)
    microbatches = {"inputs": inputs, "targets": targets}

    def loss_fn(params, batch):
        pred = params(batch["inputs"])
        return jnp.mean((pred - batch["targets"]) ** 2)

    accumulate_grads(loss_fn, tiny_model, microbatches)  # warmup

    benchmark(accumulate_grads, loss_fn, tiny_model, microbatches)

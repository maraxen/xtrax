import jax.numpy as jnp
import pytest

from xtrax.training.grad import accumulate_grads


@pytest.mark.parametrize("n_microbatches", [1, 2, 4, 8])
def test_accumulate_grads_scaling(benchmark, n_microbatches, tiny_model, synthetic_batch):
    # Declaration protocol for XTRAX_BENCH_RECORD_DIR emission (see
    # xtrax.profiling.bench); the n_microbatches scaling axis lands in the
    # record's config automatically via params.
    benchmark.extra_info.update(
        {
            "xtrax_stage": 1,
            "xtrax_n_atoms": 32,
            "xtrax_scale_basis": "batch_rows",
        }
    )
    inputs = jnp.stack([synthetic_batch["inputs"]] * n_microbatches)
    targets = jnp.stack([synthetic_batch["targets"]] * n_microbatches)
    microbatches = {"inputs": inputs, "targets": targets}

    def loss_fn(params, batch):
        pred = params(batch["inputs"])
        return jnp.mean((pred - batch["targets"]) ** 2)

    accumulate_grads(loss_fn, tiny_model, microbatches)  # warmup

    benchmark(accumulate_grads, loss_fn, tiny_model, microbatches)

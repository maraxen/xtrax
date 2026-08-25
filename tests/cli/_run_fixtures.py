"""
Module-level fixture factories for xtrax run e2e tests.
These MUST be module-level functions — load_fn resolves by module import path.
Do NOT use pytest closures or conftest lambdas for factories used in TOML configs.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.data.module import DataModule


def make_model():
    """Minimal eqx.Module for testing. Called by load_fn as factory."""
    return eqx.nn.Linear(2, 2, key=jax.random.PRNGKey(0))


def make_loss():
    """Returns a LossFunction: (predictions, targets) -> scalar."""

    def loss_fn(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

    return loss_fn


class _TinyDataset:
    """Minimal dataset: returns (x, y) pairs."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        x = jnp.ones(2) * idx
        y = jnp.ones(2) * idx
        return x, y


def make_dataset():
    """Returns a plain dataset (NOT a DataModule). run_from_config ALWAYS wraps it."""
    return _TinyDataset()


class _DictBatchDataset:
    """Minimal dataset yielding trainer-contract dict batches.

    Trainer.step requires batch["inputs"] / batch["targets"] (PyTree dict);
    the tuple-yielding _TinyDataset predates real-fit usage and is only
    exercised through mocked fit_sync in existing CLI tests.
    """

    def __len__(self):
        return 4

    def __iter__(self):
        # REQUIRED for real iteration: without __iter__ (or IndexError in
        # __getitem__), Python's legacy sequence-protocol fallback loops
        # forever -- jnp ones*idx never raises IndexError.
        yield from (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        x = jnp.ones(2) * idx
        y = jnp.ones(2) * idx
        return {"inputs": x, "targets": y}


def make_dict_dataset():
    """Returns a raw dataset yielding trainer-contract dict batches (real-fit safe)."""
    return _DictBatchDataset()


def make_datamodule():
    """
    Returns an ALREADY-BUILT DataModule.

    Used for AC3/M4 double-wrap test: run_from_config must RE-WRAP this DataModule
    unconditionally. result.dataset must be this DataModule (the inner one).
    A duck-type isinstance check in run_from_config would skip the wrap and fail the test.
    """
    inner_dataset = _TinyDataset()
    return DataModule(
        inner_dataset,
        batch_size=2,
        num_epochs=1,
        seed=0,
        distributed=False,
    )

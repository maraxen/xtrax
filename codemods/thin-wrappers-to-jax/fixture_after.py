# ruff: noqa -- a codemod fixture: its imports are deliberately unused
# The expected rewrite is byte-exact, so the formatter must not touch it.
# fmt: off
"""A module using thin wrappers deprecated in the xtrax #6107 probe release."""

import jax
import xtrax.engine.io
from xtrax.distributed import get_device_mesh, init_dist
from xtrax.engine.io import BoundedCallbackHandler, async_indexed_stream
from xtrax.engine.io import BoundedCallbackHandler as Handler
from xtrax.training import WeightedLoss, create_train_step
from xtrax.training.optim import partition_labels


def setup(model, is_embedding):
    jax.distributed.initialize()
    jax.distributed.initialize(coordinator_address="host:1234", num_processes=2, process_id=0)
    mesh = jax.make_mesh((2, 4), ("data", "model"))
    mesh2 = jax.make_mesh((8,), ("data",))
    labels = jax.tree.map(lambda leaf: "frozen" if is_embedding(leaf) else "train", model)
    labels2 = jax.tree.map(lambda leaf: "frozen" if (lambda leaf: leaf.ndim == 1)(leaf) else "train", model)
    handler = xtrax.engine.io.BoundedCallbackHandler(max_concurrent=2)
    plain = create_train_step(loss, opt)
    checked = create_train_step(loss, opt, safety=True)
    other = unrelated.init_dist()  # a method of something else: untouched
    kept = partition_labels(model, is_embedding, frozen_label="f")  # extra args: untouched
    return mesh, mesh2, labels, labels2, handler, plain, checked, other, kept

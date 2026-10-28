# ruff: noqa -- a codemod fixture: its imports are deliberately unused
# The expected rewrite is byte-exact, so the formatter must not touch it.
# fmt: off
"""A module using thin wrappers deprecated in the xtrax #6107 probe release."""

import jax
import xtrax.io
from xtrax.distributed import get_device_mesh, init_dist
from xtrax.io import BoundedCallbackHandler, async_indexed_stream
from xtrax.io.callbacks import BoundedCallbackHandler as Handler
from xtrax.training import WeightedLoss, create_train_step
from xtrax.training.optim import partition_labels


def setup(model, is_embedding):
    init_dist()
    init_dist(coordinator_address="host:1234", num_processes=2, process_id=0)
    mesh = get_device_mesh((2, 4), ("data", "model"))
    mesh2 = get_device_mesh(shape=(8,), axis_names=("data",))
    labels = partition_labels(model, is_embedding)
    labels2 = partition_labels(model, lambda leaf: leaf.ndim == 1)
    handler = xtrax.io.BoundedCallbackHandler(max_concurrent=2)
    plain = create_train_step(loss, opt)
    checked = create_train_step(loss, opt, safety=True)
    other = unrelated.init_dist()  # a method of something else: untouched
    kept = partition_labels(model, is_embedding, frozen_label="f")  # extra args: untouched
    return mesh, mesh2, labels, labels2, handler, plain, checked, other, kept

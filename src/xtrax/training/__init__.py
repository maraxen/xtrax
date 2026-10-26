"""Training module for supervised learning with JAX."""

from xtrax._renamed import deprecated_reexports
from xtrax.training.loss import ComposedLoss
from xtrax.training.optim import adamw_with_schedule, make_optimizer
from xtrax.training.state import init_state
from xtrax.training.step import create_train_step
from xtrax.training.trainer import Trainer
from xtrax.training.types import Callback, LossFunction, ResumableState

__all__ = [
    "Trainer",
    "SafetyTrainStep",
    "create_train_step",
    "init_state",
    "LossFunction",
    "Callback",
    "ResumableState",
    "WeightedLoss",
    "MultiTaskLoss",
    "ComposedLoss",
    "make_optimizer",
    "adamw_with_schedule",
]

# Deprecated in the #6107 probe release. They are not imported eagerly: the home module's
# `__getattr__` hook warns (naming the caller) and returns the unchanged object.
__getattr__ = deprecated_reexports(
    __name__,
    {
        "SafetyTrainStep": "xtrax.training.step",
        "WeightedLoss": "xtrax.training.loss",
        "MultiTaskLoss": "xtrax.training.loss",
    },
)

"""Training module for supervised learning with JAX."""

from xtrax.training.loss import MultiTaskLoss, WeightedLoss
from xtrax.training.optim import adamw_with_schedule, make_optimizer
from xtrax.training.step import SafetyTrainStep, create_train_step
from xtrax.training.trainer import Trainer
from xtrax.training.types import Callback, LossFunction, ResumableState

__all__ = [
    "Trainer",
    "SafetyTrainStep",
    "create_train_step",
    "LossFunction",
    "Callback",
    "ResumableState",
    "WeightedLoss",
    "MultiTaskLoss",
    "make_optimizer",
    "adamw_with_schedule",
]

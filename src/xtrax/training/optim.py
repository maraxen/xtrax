from collections.abc import Callable
from typing import Any

import jax
import optax

PyTree = Any


def no_bias_wd_mask(params: PyTree) -> PyTree:
    """Create a mask for selective weight decay (skip 1-D parameters like biases).

    Returns a PyTree with the same structure as params where each leaf is True
    (apply weight decay) for ndim != 1, and False (skip weight decay) for 1-D arrays.

    Args:
        params: A PyTree of parameters (typically JAX arrays).

    Returns:
        A PyTree with boolean leaves indicating whether to apply weight decay.
    """
    return jax.tree.map(lambda x: x.ndim != 1, params)


def make_optimizer(
    base: optax.GradientTransformation,
    clip_norm: float | None = 1.0,
) -> optax.GradientTransformation:
    """Wrap a base optimizer with optional gradient clipping.

    Args:
        base: The base optimizer (e.g., adamw, sgd).
        clip_norm: Global gradient norm clipping threshold. If None, returns base
                   unchanged. If provided, gradient clipping is applied BEFORE
                   the base optimizer.

    Returns:
        The base optimizer (if clip_norm=None) or a chained optimizer with
        clipping first.
    """
    if clip_norm is None:
        return base
    return optax.chain(optax.clip_by_global_norm(clip_norm), base)


def adamw_with_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    weight_decay: float = 1e-2,
    clip_norm: float | None = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    wd_mask: Callable = no_bias_wd_mask,
) -> optax.GradientTransformation:
    """Create an AdamW optimizer with a cosine decay schedule and optional clipping.

    Args:
        peak_lr: Peak learning rate (after warmup).
        warmup_steps: Number of warmup steps.
        total_steps: Total number of steps INCLUDING warmup (passed as decay_steps
                     to schedule).
        weight_decay: L2 weight decay coefficient. Default: 1e-2.
        clip_norm: Gradient clipping threshold. If None, no clipping.
        b1: Adam beta1 (exponential decay for first moment). Default: 0.9.
        b2: Adam beta2 (exponential decay for second moment). Default: 0.999.
        eps: Small constant for numerical stability. Default: 1e-8.
        wd_mask: Callable to mask which parameters get weight decay applied.

    Returns:
        A GradientTransformation combining warmup-cosine decay schedule, AdamW,
        and optional clipping.
    """
    # Create warmup_cosine_decay schedule with total_steps as decay_steps
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,  # Total steps INCLUDING warmup
        end_value=0.0,
    )

    # Create AdamW with the schedule and mask
    base = optax.adamw(
        learning_rate=schedule,
        weight_decay=weight_decay,
        b1=b1,
        b2=b2,
        eps=eps,
        mask=wd_mask,
    )

    # Wrap with optional clipping
    return make_optimizer(base, clip_norm=clip_norm)


def partition_labels(
    model: PyTree,
    frozen_filter: Callable[[Any], bool],
    frozen_label: str = "frozen",
    train_label: str = "train",
) -> PyTree:
    """Create a string-label pytree for parameter partitioning.

    Returns a pytree with the same structure as model where each leaf is either
    frozen_label or train_label depending on the frozen_filter predicate.

    This is distinct from eqx.partition which returns (frozen, train) tuples.
    This function returns a string-label pytree suitable for use with
    optax.partition(transforms_dict, label_tree).

    Args:
        model: An equinox Module (or any PyTree).
        frozen_filter: A callable that takes a leaf and returns True if it
                       should be frozen.
        frozen_label: The label string for frozen parameters. Default: "frozen".
        train_label: The label string for trainable parameters. Default: "train".

    Returns:
        A PyTree with the same structure as model, where each leaf is a string
        label.
    """
    return jax.tree.map(
        lambda leaf: frozen_label if frozen_filter(leaf) else train_label,
        model,
    )

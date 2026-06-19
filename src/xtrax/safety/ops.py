import jax.numpy as jnp
from jaxtyping import Array, Float


def safe_norm(
    x: Float[Array, "..."],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    eps: float = 1e-8,
) -> Float[Array, "..."]:
    """Numerically stable norm computation.

    Computes jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + eps**2).
    This ensures the gradient at x=0 is finite (not NaN).

    Args:
        x: Input array.
        axis: Axis or axes along which to compute the norm. None means all axes.
        keepdims: Whether to keep reduced dimensions.
        eps: Epsilon value for numerical stability. Adds eps**2 to sum of squares.

    Returns:
        Array with shape matching the reduction specified by axis and keepdims.
    """
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + eps**2)


def safe_reciprocal(
    x: Float[Array, "..."] | float, eps: float = 1e-8
) -> Float[Array, "..."]:
    """Numerically stable reciprocal computation.

    Computes 1.0 / (x + eps) using a smooth, unconditional formula with no branching.
    This ensures the result is well-defined even at x=0.

    Args:
        x: Input array or scalar.
        eps: Epsilon value for numerical stability. Added unconditionally to x.

    Returns:
        Array with same shape as x, containing 1.0 / (x + eps).
    """
    return jnp.asarray(1.0 / (x + eps))

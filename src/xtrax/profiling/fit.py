"""Fit pre-check: does a unit's compiled program fit device memory? (spec 261016 X6)

``fit_precheck`` lowers ``jax.jit(fn)`` on ``jax.ShapeDtypeStruct`` versions of
the inputs, compiles it, and reads ``memory_analysis()``. Lowering and compiling
never allocate the real inputs, so the check is safe to run before a
production-sized measurement.

The required bytes are the sum of the argument, output and temporary buffer
sizes reported by XLA. The limit is the device's ``bytes_limit`` from
``memory_stats()``. Where either is unknown (CPU exposes no ``memory_stats``,
or the backend reports no memory analysis), ``fits`` is ``None`` and the reason
says why. ``None`` is never a pass; the caller decides what it means.

Standard library, JAX and NumPy only: ``profiling/`` is a leaf package.
"""

import dataclasses
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class FitCheck:
    """Result of :func:`fit_precheck`.

    Attributes:
        fits: True if the program fits the device limit, False if it does not,
            None if the answer is unknown (see ``reason``).
        required_bytes: Argument + output + temp bytes, or None if unknown.
        limit_bytes: Device ``bytes_limit``, or None if the device reports none.
        reason: Human-readable explanation of the verdict.
    """

    fits: bool | None
    required_bytes: int | None
    limit_bytes: int | None
    reason: str


def _to_struct(leaf: Any) -> Any:
    import jax

    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return jax.ShapeDtypeStruct(tuple(leaf.shape), leaf.dtype)
    return leaf


def _device_limit_bytes() -> tuple[int | None, str]:
    """``bytes_limit`` of the first device, and the platform name for the reason."""
    import jax

    try:
        device = jax.devices()[0]
    except Exception:  # noqa: BLE001 - no backend at all is an unknown limit, not a crash
        return None, "unknown"
    platform = str(getattr(device, "platform", "unknown"))
    try:
        stats = device.memory_stats()
    except Exception:  # noqa: BLE001 - some backends raise instead of returning None
        stats = None
    if not stats:
        return None, platform
    limit = stats.get("bytes_limit")
    return (int(limit) if limit is not None else None), platform


def _required_bytes(compiled: Any) -> tuple[int | None, str]:
    try:
        analysis = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001 - analysis is optional on some backends
        return None, f"memory_analysis raised {type(exc).__name__}: {exc}"
    if analysis is None:
        return None, "memory_analysis() returned None"
    total = 0
    missing = []
    for attr in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes"):
        value = getattr(analysis, attr, None)
        if value is None:
            missing.append(attr)
        else:
            total += int(value)
    if missing:
        return None, f"memory_analysis lacks {', '.join(missing)}"
    return total, ""


def fit_precheck(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> FitCheck:
    """Decide whether ``fn(*args, **kwargs)`` fits device memory, without running it.

    Args:
        fn: The unit to check. It need not be jitted; it is wrapped in ``jax.jit``.
        args: Positional inputs. Array-like leaves are replaced by
            ``ShapeDtypeStruct``; the real values are never touched.
        kwargs: Keyword inputs, handled the same way.

    Returns:
        A :class:`FitCheck`. Never raises for a function that cannot be lowered;
        that case returns ``fits=None`` with the lowering error as the reason.
    """
    import jax

    # Read the limit before compiling. Compiling a host loop as one program
    # unrolls it once per element, which at production extent can take far
    # longer than the measurement; skip that cost where there is no limit to
    # compare against (CPU).
    limit, platform = _device_limit_bytes()
    if limit is None:
        return FitCheck(
            None,
            None,
            None,
            f"no device memory limit on platform {platform!r}; compile skipped",
        )

    kwargs = dict(kwargs or {})
    structs_args, structs_kwargs = jax.tree_util.tree_map(_to_struct, (tuple(args), kwargs))

    try:
        compiled = jax.jit(fn).lower(*structs_args, **structs_kwargs).compile()
    except Exception as exc:  # noqa: BLE001 - an unjittable unit is an unknown fit, not a crash
        return FitCheck(
            fits=None,
            required_bytes=None,
            limit_bytes=None,
            reason=f"not jittable as a whole: {type(exc).__name__}: {exc}",
        )

    required, why_no_required = _required_bytes(compiled)
    if required is None:
        return FitCheck(None, None, limit, why_no_required)
    fits = required <= limit
    return FitCheck(
        fits=fits,
        required_bytes=required,
        limit_bytes=limit,
        reason=f"needs {required} bytes, device limit {limit} bytes",
    )


__all__ = ["FitCheck", "fit_precheck"]

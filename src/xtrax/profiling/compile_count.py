"""Count XLA backend compilations inside a block (recompilation guard).

``compile_time_clock`` splits one call's wall time into compile vs runtime. It
does not say whether a *loop* kept compiling after the first step. This module
counts JAX's ``/jax/core/compile/backend_compile_duration`` monitoring events
so a test can assert the cache held, and can prove the counter itself fires
when a static argument changes.

jax is imported lazily, inside the functions, so importing this module (and
``xtrax.profiling``) stays free of jax -- matching the package's leaf contract.
"""

from collections.abc import Callable, Iterable
from types import TracebackType
from typing import Any

BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


class BackendCompileCount:
    """Compilations observed inside :func:`count_backend_compiles`.

    ``count`` is the number of backend-compile events. ``seconds`` is their
    summed duration, as JAX reported it.
    """

    def __init__(self) -> None:
        self.count = 0
        self.seconds = 0.0
        # One bound method, stored once: unregister matches by this object.
        self._listener = self._on_duration

    def _on_duration(self, event: str, duration_secs: float, **_: object) -> None:
        if event == BACKEND_COMPILE_EVENT:
            self.count += 1
            self.seconds += float(duration_secs)


def _load_private_unregister() -> Callable[..., None]:
    from jax._src import monitoring

    unregister = getattr(monitoring, "unregister_event_duration_listener", None)
    if unregister is None:
        raise AttributeError("jax._src.monitoring.unregister_event_duration_listener")
    return unregister


def _load_public_unregister() -> Callable[..., None]:
    import jax.monitoring as monitoring

    unregister = getattr(monitoring, "unregister_event_duration_listener", None)
    if unregister is None:
        raise AttributeError("jax.monitoring.unregister_event_duration_listener")
    return unregister


def _unregister_duration_listener(callback: Callable[..., None]) -> None:
    """Detach ``callback`` from JAX's duration-listener list.

    The reference unregisters through ``jax._src.monitoring``. That symbol is
    private, so a JAX move must not leak the listener into later measurements:
    try the private hook, then the public ``jax.monitoring`` alias, and raise
    if both are gone.
    """
    failures: list[str] = []
    for label, loader in (
        ("jax._src.monitoring", _load_private_unregister),
        ("jax.monitoring", _load_public_unregister),
    ):
        try:
            unregister = loader()
        except (ImportError, AttributeError) as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        try:
            unregister(callback)
        except Exception as exc:  # noqa: BLE001 -- private API may move under us
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        return
    raise RuntimeError(
        "JAX moved unregister_event_duration_listener; "
        "count_backend_compiles cannot detach its listener. "
        "Tried jax._src.monitoring, then jax.monitoring. " + " | ".join(failures)
    )


class _BackendCompileCounter:
    def __init__(self) -> None:
        self._stats = BackendCompileCount()

    def __enter__(self) -> BackendCompileCount:
        import jax.monitoring

        jax.monitoring.register_event_duration_secs_listener(self._stats._listener)
        return self._stats

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _unregister_duration_listener(self._stats._listener)


def count_backend_compiles() -> _BackendCompileCounter:
    """Context manager counting XLA backend compiles and their total seconds.

    ``with count_backend_compiles() as cc`` counts every
    ``/jax/core/compile/backend_compile_duration`` event JAX reports while the
    block runs, whichever function caused it. ``cc.count`` is the event count
    and ``cc.seconds`` is the summed duration.

    Returns:
        A context manager yielding a :class:`BackendCompileCount`.

    Raises:
        RuntimeError: JAX no longer exposes ``unregister_event_duration_listener``
            on ``jax._src.monitoring`` or ``jax.monitoring``, so the listener
            cannot be detached.
    """
    return _BackendCompileCounter()


def assert_no_recompile_after(
    step_fn: Callable[..., Any],
    args_iter: Iterable[tuple[Any, ...]],
    warmup: int = 1,
) -> None:
    """Assert ``step_fn`` does not backend-compile after ``warmup`` calls.

    Each item of ``args_iter`` is a tuple of positional arguments. The first
    ``warmup`` calls (default 1) may compile; every later call must hit the
    XLA cache. Results are ``block_until_ready``'d so a compile that finishes
    asynchronously is not charged to the next call.

    Args:
        step_fn: Stepped callable, typically ``jax.jit`` or ``eqx.filter_jit``.
        args_iter: Positional-argument tuples, one per call, in order.
        warmup: How many leading calls are allowed to compile.

    Raises:
        ValueError: ``warmup`` is negative, or ``args_iter`` runs out during warmup.
        AssertionError: One or more backend compilations fired after warmup.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    import jax

    iterator = iter(args_iter)
    for i in range(warmup):
        try:
            args = next(iterator)
        except StopIteration:
            raise ValueError(f"args_iter ended during warmup ({i} of {warmup} calls)") from None
        jax.block_until_ready(step_fn(*args))
    with count_backend_compiles() as recorded:
        n_steps = 0
        for args in iterator:
            jax.block_until_ready(step_fn(*args))
            n_steps += 1
    if recorded.count != 0:
        raise AssertionError(
            f"{recorded.count} backend compile(s) ({recorded.seconds:.3f}s) "
            f"across {n_steps} step(s) after {warmup} warmup call(s)"
        )

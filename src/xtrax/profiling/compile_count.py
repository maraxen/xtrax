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
from dataclasses import dataclass
from types import TracebackType
from typing import Any

BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


@dataclass(frozen=True, slots=True)
class LeafSignature:
    """Signature of a single leaf argument in a JIT call."""

    path: str
    is_array: bool
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    static_value: Any = None
    type_name: str = ""


@dataclass(frozen=True, slots=True)
class CallSignature:
    """Extracted signature of positional arguments for JIT caching."""

    treedef: Any
    leaves: tuple[LeafSignature, ...]


def extract_jit_signature(args: tuple[Any, ...]) -> CallSignature:
    """Extract PyTree structure, leaf shapes, dtypes, and static values."""
    import jax

    leaves, treedef = jax.tree_util.tree_flatten_with_path(args)
    leaf_sigs: list[LeafSignature] = []
    for path, leaf in leaves:
        path_str = jax.tree_util.keystr(path)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            leaf_sigs.append(
                LeafSignature(
                    path=path_str,
                    is_array=True,
                    shape=tuple(int(d) for d in leaf.shape),
                    dtype=str(leaf.dtype),
                    type_name=type(leaf).__name__,
                )
            )
        else:
            leaf_sigs.append(
                LeafSignature(
                    path=path_str,
                    is_array=False,
                    static_value=leaf,
                    type_name=type(leaf).__name__,
                )
            )
    return CallSignature(treedef=treedef, leaves=tuple(leaf_sigs))


extract_signature = extract_jit_signature


def diff_jit_signature(
    warmup_args: tuple[Any, ...],
    failing_args: tuple[Any, ...],
) -> list[str]:
    """Diff two argument tuples and report discrepancies that trigger XLA recompile.

    Returns:
        A list of human-readable divergence descriptions.
    """
    sig_warmup = extract_jit_signature(warmup_args)
    sig_failing = extract_jit_signature(failing_args)
    diffs: list[str] = []

    # 1. PyTree Structure Check
    if sig_warmup.treedef != sig_failing.treedef:
        diffs.append(
            f"PyTree structure mismatch: warmup={sig_warmup.treedef} "
            f"vs current={sig_failing.treedef}"
        )
        return diffs

    # 2. Leaf-by-Leaf Comparison
    for idx, (lw, lf) in enumerate(zip(sig_warmup.leaves, sig_failing.leaves, strict=True)):
        if lw.is_array != lf.is_array:
            diffs.append(
                f"Leaf [{idx}] '{lw.path}': array status changed from is_array={lw.is_array} "
                f"({lw.type_name}) to is_array={lf.is_array} ({lf.type_name})"
            )
            continue

        if lw.is_array:
            if lw.shape != lf.shape:
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': shape changed from {lw.shape} to {lf.shape}"
                )
            if lw.dtype != lf.dtype:
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': dtype changed from {lw.dtype} to {lf.dtype}"
                )
        else:
            # Finding C6: Guard against uncomparable objects or array comparisons
            try:
                is_diff = bool(lw.static_value != lf.static_value)
            except Exception:
                is_diff = repr(lw.static_value) != repr(lf.static_value)

            if is_diff:
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': static value changed from "
                    f"{lw.static_value!r} to {lf.static_value!r}"
                )
            elif type(lw.static_value) is not type(lf.static_value):
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': static type changed from "
                    f"{lw.type_name} to {lf.type_name}"
                )
    return diffs


def explain_cache_miss(
    warmup_args: tuple[Any, ...],
    failing_args: tuple[Any, ...],
    step_index: int | None = None,
) -> str:
    """Format an actionable diagnostic explanation of an XLA cache miss."""
    diffs = diff_jit_signature(warmup_args, failing_args)
    step_prefix = f" at step {step_index}" if step_index is not None else ""
    if not diffs:
        return (
            f"XLA cache miss detected{step_prefix}, but no PyTree, shape, dtype, "
            "or static leaf mismatch was found. Possible cause: internal compilation "
            "cache eviction, non-deterministic hash, or closure variable mutation."
        )
    formatted_diffs = "\n  - ".join(diffs)
    return (
        f"XLA cache miss detected{step_prefix} due to signature divergence:\n  - {formatted_diffs}"
    )


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
    """Assert ``step_fn`` does not backend-compile after ``warmup`` calls, with rich diagnostics.

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
    last_warmup_args: tuple[Any, ...] | None = None
    for i in range(warmup):
        try:
            args = next(iterator)
        except StopIteration:
            raise ValueError(f"args_iter ended during warmup ({i} of {warmup} calls)") from None
        last_warmup_args = args
        jax.block_until_ready(step_fn(*args))

    step_idx = 0
    total_recompiles = 0
    total_seconds = 0.0
    first_args: tuple[Any, ...] | None = None
    first_failure_step: int | None = None
    first_failure_args: tuple[Any, ...] | None = None

    for args in iterator:
        step_idx += 1
        if first_args is None:
            first_args = args
        with count_backend_compiles() as recorded:
            jax.block_until_ready(step_fn(*args))
        if recorded.count > 0:
            total_recompiles += recorded.count
            total_seconds += recorded.seconds
            if first_failure_step is None:
                first_failure_step = step_idx
                first_failure_args = args

    if total_recompiles != 0:
        diag = ""
        baseline_args = last_warmup_args if warmup > 0 else first_args
        if baseline_args is not None and first_failure_args is not None:
            if warmup == 0 and first_failure_step == 1:
                diag = (
                    f"\nFunction compiled at step 1 with warmup=0 (expected pre-warmed cache). "
                    f"Initial invocation signature: {extract_jit_signature(first_failure_args)}"
                )
            else:
                diag = "\n" + explain_cache_miss(
                    baseline_args, first_failure_args, step_index=first_failure_step
                )
        elif warmup == 0 and first_failure_step is not None:
            diag = (
                f"\nFunction compiled at step {first_failure_step} with warmup=0 "
                "(expected pre-warmed cache)."
            )
        raise AssertionError(
            f"{total_recompiles} backend compile(s) ({total_seconds:.3f}s) "
            f"across {step_idx} step(s) after {warmup} warmup call(s).{diag}"
        )

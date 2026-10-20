"""Hoisted map: run a shared stage once per call, then map the per-element stage.

``chunked_map`` runs every stage once per element of the mapped axis. When part
of the work reads only shared inputs, ``HoistedMap`` runs that part once and
hands its result to the mapped stage. Keep a per-element input that the shared
stage never reads (a PRNG key, for instance) out of the shared stage: a jaxpr
that reads such an input cannot be hoisted, even when its value does not matter.

The cache lives on the instance that ``make_hoisted`` returns, so repeated calls
on one ``HoistedMap`` can reuse entries. ``hoisted_map`` is the one-shot form and
builds a fresh instance on every call, so its jit wrappers are rebuilt and the
mapped stage recompiles on every call. Callers that repeat a call should hold a
``HoistedMap``. ``batch_size`` is passed to the mapped stage as given; it is not
read from ``axis``.

``axis.varying_inputs`` is checked by name only. ``shared`` never receives
``xs``, so a jaxpr check on it would be vacuous. ``check_elements`` runs a
parity probe against a monolithic ``reference`` on the first few elements.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from xtrax.inference.errors import HoistParityError, HoistVaryingInputError
from xtrax.inference.memo import MemoPolicy, memoize_jaxpr
from xtrax.tiling.plan import AxisSpec
from xtrax.transforms.map import chunked_map

__all__ = [
    "HoistedMap",
    "hoisted_map",
    "make_hoisted",
]


def _has_tracer(tree: Any) -> bool:
    """True when any leaf of ``tree`` is a JAX tracer (an enclosing transform is active)."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


def _leading_size(xs: Any) -> int:
    """Length N of the mapped axis: at least 1, and the same on every leaf."""
    leaves = jax.tree.leaves(xs)
    if not leaves:
        raise ValueError("xs must contain at least one array leaf")
    sizes = []
    for leaf in leaves:
        shape = jnp.shape(leaf)
        if not shape:
            raise ValueError("every leaf of xs needs a leading mapped axis; got a scalar leaf")
        sizes.append(int(shape[0]))
    if len(set(sizes)) != 1:
        raise ValueError(f"xs leaves disagree on the leading mapped axis: sizes {sizes}")
    if sizes[0] < 1:
        raise ValueError(f"xs has an empty mapped axis (N={sizes[0]}); need N >= 1")
    return sizes[0]


# Python scalars that jit cannot trace as arrays: the uncached path marks them static.
_STATIC_TYPES = (bool, int, float, str)


def _static_names(shared_args: Mapping[str, Any]) -> frozenset[str]:
    """Names of shared arguments that are Python scalars or strings (static under jit)."""
    return frozenset(
        name for name, value in shared_args.items() if isinstance(value, _STATIC_TYPES)
    )


def _check_varying(axis: AxisSpec | None, shared_args: Mapping[str, Any]) -> None:
    """Refuse a shared input that the axis declares as varying (name check only)."""
    if axis is None:
        return
    for name in shared_args:
        if name in axis.varying_inputs:
            raise HoistVaryingInputError(axis.name, name)


def _max_abs_diff(got: np.ndarray, want: np.ndarray) -> float:
    """Largest absolute difference, or inf when the dtypes are not numeric."""
    if got.dtype.kind not in "biufc" or want.dtype.kind not in "biufc":
        return float("inf")
    diff = np.abs(got.astype(np.complex128) - want.astype(np.complex128))
    return float(np.max(diff))


def _leaf_mismatch(got: np.ndarray, want: np.ndarray, rtol: float, atol: float) -> float | None:
    """None when the leaves agree, else the max abs difference.

    Float and complex leaves compare with ``allclose``; every other dtype compares exactly.
    """
    if got.shape != want.shape:
        return float("inf")
    if got.dtype.kind in "fc" or want.dtype.kind in "fc":
        if np.allclose(got, want, rtol=rtol, atol=atol):
            return None
    elif np.array_equal(got, want):
        return None
    return _max_abs_diff(got, want)


def _compare_element(index: int, got: Any, want: Any, rtol: float, atol: float) -> None:
    """Raise HoistParityError on the first leaf of one element that disagrees."""
    if jax.tree.structure(got) != jax.tree.structure(want):
        raise HoistParityError(index, "<tree structure>", float("inf"))
    got_flat, _ = jax.tree_util.tree_flatten_with_path(got)
    for (path, leaf), ref_leaf in zip(got_flat, jax.tree.leaves(want), strict=True):
        diff = _leaf_mismatch(np.asarray(leaf), np.asarray(ref_leaf), rtol, atol)
        if diff is not None:
            raise HoistParityError(index, jax.tree_util.keystr(path), diff)


def _probe(
    y: Any,
    xs: Any,
    shared_args: Mapping[str, Any],
    reference: Callable[[Mapping[str, Any], Any], Any],
    count: int,
    rtol: float,
    atol: float,
) -> None:
    """Compare the first ``count`` hoisted elements with ``reference`` on the same inputs."""
    n = _leading_size(xs)
    for i in range(min(count, n)):
        x_i = jax.tree.map(lambda leaf: leaf[i], xs)
        want = reference(shared_args, x_i)
        got = jax.tree.map(lambda leaf: leaf[i], y)
        _compare_element(i, got, want, rtol, atol)


def _jit_mapped(per_element: Callable[[Any, Any], Any], batch_size: int | None) -> Callable:
    """Jitted ``(shared_result, xs) -> ys`` with the shared result as a jit argument.

    Passing the shared result as an argument, not a closure, keeps repeated calls
    with the same shapes from recompiling.
    """

    def run(shared_result: Any, xs: Any) -> Any:
        """Map ``per_element`` over ``xs`` with the shared result bound."""
        return chunked_map(lambda x: per_element(shared_result, x), xs, batch_size)

    return jax.jit(run)


class HoistedMap:
    """Callable that runs the shared stage once per call and maps the per-element stage.

    Built by ``make_hoisted``; hold one instance across calls, because the jitted
    mapped stage and the memo cache (when ``cache`` was given) live on it.
    """

    def __init__(
        self,
        shared: Callable[..., Any],
        per_element: Callable[[Any, Any], Any],
        *,
        batch_size: int | None,
        axis: AxisSpec | None,
        cache: MemoPolicy | None,
    ) -> None:
        """Store the stages and build the jitted and memoized shared paths once."""
        self._axis = axis
        self._shared = shared
        self._jitted_shared: dict[frozenset[str], Callable[..., Any]] = {}
        self._memo = memoize_jaxpr(shared, policy=cache) if cache is not None else None
        self._mapped = _jit_mapped(per_element, batch_size)
        self._counts = {"shared_calls": 0, "traced_bypass": 0}

    def _jit_for(self, static: frozenset[str]) -> Callable[..., Any]:
        """Jitted ``shared`` with ``static`` names marked static; one wrapper per name set."""
        if static not in self._jitted_shared:
            self._jitted_shared[static] = jax.jit(
                self._shared, static_argnames=tuple(sorted(static))
            )
        return self._jitted_shared[static]

    def _run_shared(self, shared_args: Mapping[str, Any]) -> Any:
        """Run the shared stage, through the memo cache when one was requested."""
        self._counts["shared_calls"] += 1
        static = _static_names(shared_args)
        if self._memo is None:
            return self._jit_for(static)(**shared_args)
        if _has_tracer(shared_args):
            # The memo key needs concrete values; under an enclosing jit it cannot be built.
            self._counts["traced_bypass"] += 1
            return self._jit_for(static)(**shared_args)
        return self._memo(**shared_args)

    def __call__(
        self,
        shared_args: Mapping[str, Any],
        xs: Any,
        *,
        reference: Callable[[Mapping[str, Any], Any], Any] | None = None,
        check_elements: int = 0,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Any:
        """Run the shared stage once, map the per-element stage over ``xs``, return ``ys``.

        ``check_elements > 0`` also compares the first elements with ``reference``,
        which is called as ``reference(shared_args, x_i)`` on each element.

        Args:
            shared_args: Keyword arguments for ``shared``. Must not name an input
                that ``axis.varying_inputs`` declares as varying.
            xs: Pytree whose leaves share a leading mapped axis of length N >= 1.
            reference: Monolithic function for the parity probe.
            check_elements: Number of leading elements to probe; 0 disables the probe.
            rtol: Relative tolerance for float and complex leaves.
            atol: Absolute tolerance for float and complex leaves.

        Returns:
            The mapped outputs, with the same structure as ``per_element``'s output
            stacked along a leading axis of N.

        Raises:
            ValueError: ``xs`` is empty or has N < 1, or the probe is misconfigured.
            HoistVaryingInputError: a shared input is declared as varying on ``axis``.
            HoistParityError: a probed element differs from ``reference``.
        """
        _leading_size(xs)
        if check_elements > 0:
            if reference is None:
                raise ValueError("check_elements > 0 needs a reference callable")
            if _has_tracer(xs):
                raise ValueError("check_elements needs concrete xs values; call outside jit")
        _check_varying(self._axis, shared_args)
        shared_result = self._run_shared(shared_args)
        y = self._mapped(shared_result, xs)
        if check_elements > 0 and reference is not None:
            _probe(y, xs, shared_args, reference, check_elements, rtol, atol)
        return y

    def stats(self) -> dict[str, Any]:
        """Call counters, plus the memo counters when this instance has a cache.

        Returns:
            ``shared_calls`` (every call), ``traced_bypass`` (cached calls run under an
            enclosing jit), and, when cached, the ``memo_get_stats()`` keys.
        """
        out: dict[str, Any] = dict(self._counts)
        if self._memo is not None:
            out.update(self._memo.memo_get_stats())
        return out


def make_hoisted(
    shared: Callable[..., Any],
    per_element: Callable[[Any, Any], Any],
    *,
    batch_size: int | None = None,
    axis: AxisSpec | None = None,
    cache: MemoPolicy | None = None,
) -> HoistedMap:
    """Build a ``HoistedMap`` from a shared stage and a per-element stage.

    ``shared(**shared_args)`` returns the shared result ``s``. ``per_element(s, x)``
    maps over the leading axis of ``xs``. ``cache`` opts into a content-keyed memo
    of ``shared``, owned by the returned instance; its entries can hold large
    arrays, so bound them with ``MemoPolicy.max_entries``.

    Args:
        shared: Called as ``shared(**shared_args)``; returns the shared result.
        per_element: Called as ``per_element(shared_result, x)`` for each element.
        batch_size: Chunk size for the mapped stage; None uses vmap.
        axis: Optional axis whose ``varying_inputs`` are checked against ``shared_args``.
        cache: Optional memo policy for ``shared``.

    Returns:
        A ``HoistedMap`` to call with ``(shared_args, xs)``.
    """
    return HoistedMap(
        shared,
        per_element,
        batch_size=batch_size,
        axis=axis,
        cache=cache,
    )


def hoisted_map(
    shared: Callable[..., Any],
    per_element: Callable[[Any, Any], Any],
    shared_args: Mapping[str, Any],
    xs: Any,
    *,
    batch_size: int | None = None,
    axis: AxisSpec | None = None,
    reference: Callable[[Mapping[str, Any], Any], Any] | None = None,
    check_elements: int = 0,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Any:
    """One-shot: ``make_hoisted(...)(...)``. Repeated calls should hold a ``HoistedMap``.

    Args:
        shared: See ``make_hoisted``.
        per_element: See ``make_hoisted``.
        shared_args: Keyword arguments for ``shared``.
        xs: Pytree with a leading mapped axis of length N >= 1.
        batch_size: See ``make_hoisted``.
        axis: See ``make_hoisted``.
        reference: Monolithic function for the parity probe.
        check_elements: Number of leading elements to probe; 0 disables the probe.
        rtol: Relative tolerance for float and complex leaves.
        atol: Absolute tolerance for float and complex leaves.

    Returns:
        The mapped outputs, as ``HoistedMap.__call__``.
    """
    hoisted = make_hoisted(shared, per_element, batch_size=batch_size, axis=axis)
    return hoisted(
        shared_args,
        xs,
        reference=reference,
        check_elements=check_elements,
        rtol=rtol,
        atol=atol,
    )

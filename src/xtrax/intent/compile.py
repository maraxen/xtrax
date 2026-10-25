"""Compile an intent function: plan once per signature, then jit execution.

The plan pass traces with :func:`jax.make_jaxpr` under a plan session, then
traces :func:`jax.linearize` when any leaf is inexact so diff-path sites are
recorded. :func:`~xtrax.intent.resolve.resolve` turns those records into one
:class:`~xtrax.intent.plan.Plan`. Execution is a cached :func:`jax.jit` that
re-enters an exec session and checks the primal site sequence.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError
from xtrax.intent.plan import Plan, SiteDecision, SiteKey
from xtrax.intent.resolve import SCAN_AROUND_WHILE_BACKENDS, resolve
from xtrax.intent.sites import Session, SiteRecord, _leaf_struct, activate
from xtrax.profiling.jaxpr import iter_jaxpr_eqns, sub_jaxprs

_BYPASS_WARNING = re.compile(r"override:.+ bypassed scan_around_while")
_CacheKey = tuple[
    Any,
    tuple[tuple[Any, Any, bool], ...],
    Any,
    tuple[tuple[Any, Any, bool], ...],
    Context,
]
_Cached = tuple[Callable[..., Any], Plan, tuple[SiteRecord, ...]]


class _ExecSession(Session):
    """Exec session that records the order of entered sites.

    Parameters
    ----------
    records : list of SiteRecord
        Planned records copied into this session.
    decisions : mapping
        Decisions keyed by :class:`~xtrax.intent.plan.SiteKey`.

    Returns
    -------
    None
    """

    seen: list[SiteKey]

    def __init__(
        self,
        records: list[SiteRecord],
        decisions: Mapping[SiteKey, SiteDecision],
    ) -> None:
        """Start an exec session with an empty visit list.

        Parameters
        ----------
        records : list of SiteRecord
            Planned records. Copied so callers keep their own list.
        decisions : mapping
            Site decisions passed to :class:`~xtrax.intent.sites.Session`.

        Returns
        -------
        None
        """
        super().__init__("exec", list(records), decisions)
        self.seen = []

    def enter_site(
        self,
        kind: Any,
        label: str,
        avals: tuple[Any, ...],
        length: int | None,
        fn: Callable[..., object] | None,
        const_avals: tuple[Any, ...] = (),
        xs_tree: Any = None,
    ) -> SiteKey:
        """Enter a site and append its key to :attr:`seen`.

        Parameters
        ----------
        kind : {"loop", "map"}
            Site kind.
        label : str
            User name or ``file:line``.
        avals : tuple
            Abstract values of the site arguments.
        length : int or None
            Loop length, or ``None``.
        fn : callable or None
            Closure-converted site function.
        const_avals : tuple
            Hoisted constants forwarded to the plan record. Exec mode does
            not compare them.
        xs_tree : PyTreeDef or None
            Mapped pytree structure, forwarded to the plan record.

        Returns
        -------
        SiteKey
            Key returned by the exec check.
        """
        key = super().enter_site(kind, label, avals, length, fn, const_avals, xs_tree)
        self.seen.append(key)
        return key


def _is_primal_chain(key: SiteKey) -> bool:
    """Return whether every key in the parent chain is on the primal path.

    Parameters
    ----------
    key : SiteKey
        Site key to walk, parents first in the chain.

    Returns
    -------
    bool
        ``True`` when ``key`` and each parent have ``path == "primal"``.
    """
    current: SiteKey | None = key
    while current is not None:
        if current.path != "primal":
            return False
        current = current.parent
    return True


def _abstract_tree(tree: Any) -> Any:
    """Replace every leaf with a :class:`jax.ShapeDtypeStruct` of the same shape.

    Python scalars are accepted. ``weak_type`` is copied onto the struct so
    the plan trace promotes dtypes the same way the execution trace does.

    Parameters
    ----------
    tree : pytree
        Concrete arrays, tracers, or Python scalars.

    Returns
    -------
    pytree
        Same structure, with a :class:`jax.ShapeDtypeStruct` at each leaf.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    abstracts = tuple(_leaf_struct(leaf) for leaf in leaves)
    return jax.tree_util.tree_unflatten(treedef, abstracts)


def _leaf_signature(tree: Any) -> tuple[tuple[Any, Any, bool], ...]:
    """Return ``(shape, dtype, weak_type)`` for each leaf.

    Parameters
    ----------
    tree : pytree
        Arrays, tracers, or Python scalars.

    Returns
    -------
    tuple
        One ``(shape, dtype, weak_type)`` triple per leaf, in flatten order.
        ``weak_type`` keeps a weak scalar from sharing a plan with a strong
        array of the same dtype.
    """
    return tuple(
        (struct.shape, struct.dtype, struct.weak_type)
        for struct in (_leaf_struct(leaf) for leaf in jax.tree_util.tree_leaves(tree))
    )


def _cache_key(args: tuple[Any, ...], kwargs: dict[str, Any], ctx: Context) -> _CacheKey:
    """Cache key of argument structure, leaf shapes and dtypes, and context.

    Parameters
    ----------
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.
    ctx : Context
        Resolution context. Included so two contexts do not share a plan.

    Returns
    -------
    tuple
        Structure and leaf signature of ``args`` and of ``kwargs``, then ``ctx``.
    """
    return (
        jax.tree_util.tree_structure(args),
        _leaf_signature(args),
        jax.tree_util.tree_structure(kwargs),
        _leaf_signature(kwargs),
        ctx,
    )


def _any_inexact(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Return whether any leaf has an inexact dtype.

    Parameters
    ----------
    args : tuple
        Positional arguments. Leaves may be Python scalars.
    kwargs : dict
        Keyword arguments. Leaves may be Python scalars.

    Returns
    -------
    bool
        ``True`` when a leaf dtype is inexact.
    """
    leaves = jax.tree_util.tree_leaves((args, kwargs))
    return any(jnp.issubdtype(_leaf_struct(leaf).dtype, jnp.inexact) for leaf in leaves)


def partial_over_inexact(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Linearize only over inexact leaves.

    Integer and bool leaves are closed over inside the returned function.
    During the plan pass those closed-over values are the tracers created
    from :class:`jax.ShapeDtypeStruct` placeholders, not primals of
    :func:`jax.linearize`.

    Parameters
    ----------
    fn : callable
        Function being planned.
    args : tuple
        Positional arguments, concrete or traced.
    kwargs : dict
        Keyword arguments, concrete or traced.

    Returns
    -------
    tuple
        ``(partial, inexact_leaves)``. ``partial(*inexact_leaves)`` calls
        ``fn``. ``inexact_leaves`` is empty when every leaf is exact.
    """
    flat, treedef = jax.tree_util.tree_flatten((args, kwargs))
    inexact_flags = tuple(jnp.issubdtype(leaf.dtype, jnp.inexact) for leaf in flat)

    def partial(*inexact_leaves: Any) -> Any:
        """Rebuild the call, closing over exact leaves.

        Parameters
        ----------
        inexact_leaves : tuple
            Inexact arrays, in flatten order.

        Returns
        -------
        pytree
            ``fn(*args, **kwargs)`` with exact leaves closed over.
        """
        rebuilt: list[Any] = []
        cursor = 0
        for leaf, is_inexact in zip(flat, inexact_flags, strict=True):
            if is_inexact:
                rebuilt.append(inexact_leaves[cursor])
                cursor += 1
            else:
                rebuilt.append(leaf)
        call_args, call_kwargs = jax.tree_util.tree_unflatten(treedef, rebuilt)
        return fn(*call_args, **call_kwargs)

    inexact_leaves = tuple(
        leaf for leaf, is_inexact in zip(flat, inexact_flags, strict=True) if is_inexact
    )
    return partial, inexact_leaves


def _merge_records(
    primal_records: list[SiteRecord],
    diff_records: list[SiteRecord],
) -> tuple[SiteRecord, ...]:
    """Keep primal-chain records, then diff-path records without duplicate keys.

    Parameters
    ----------
    primal_records : list of SiteRecord
        Records from the primal plan pass.
    diff_records : list of SiteRecord
        Records from the diff plan pass. Keys already carry ``path="diff"``
        when the visit was on the differentiated path.

    Returns
    -------
    tuple of SiteRecord
        Primal-chain records in visit order, then new diff-path records.
    """
    merged = [record for record in primal_records if _is_primal_chain(record.key)]
    seen = {record.key for record in merged}
    for record in diff_records:
        if _is_primal_chain(record.key) or record.key in seen:
            continue
        seen.add(record.key)
        merged.append(record)
    return tuple(merged)


def _trace_primal(
    fn: Callable[..., Any],
    abstract_args: Any,
    abstract_kwargs: Any,
) -> list[SiteRecord]:
    """Trace ``fn`` once under a fresh plan session.

    Parameters
    ----------
    fn : callable
        Function being planned.
    abstract_args : pytree
        :class:`jax.ShapeDtypeStruct` arguments.
    abstract_kwargs : pytree
        :class:`jax.ShapeDtypeStruct` keyword arguments.

    Returns
    -------
    list of SiteRecord
        Sites recorded during this trace. A new wrapper is built so
        :func:`jax.make_jaxpr` cannot reuse a previous trace.
    """

    def primal(args: Any, kwargs: Any) -> Any:
        """Call ``fn`` so this wrapper is a fresh :func:`jax.make_jaxpr` target.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        pytree
            ``fn(*args, **kwargs)``.
        """
        return fn(*args, **kwargs)

    session = Session("plan")
    with activate(session):
        jax.make_jaxpr(primal)(abstract_args, abstract_kwargs)
    return list(session.records)


def _trace_diff(
    fn: Callable[..., Any],
    abstract_args: Any,
    abstract_kwargs: Any,
) -> tuple[list[SiteRecord], bool, str]:
    """Trace :func:`jax.linearize` over inexact leaves under a fresh plan session.

    Parameters
    ----------
    fn : callable
        Function being planned.
    abstract_args : pytree
        :class:`jax.ShapeDtypeStruct` arguments.
    abstract_kwargs : pytree
        :class:`jax.ShapeDtypeStruct` keyword arguments.

    Returns
    -------
    tuple
        ``(records, diff_available, reason)``. On failure ``diff_available``
        is ``False`` and ``reason`` is the exception text.
    """

    def diff(args: Any, kwargs: Any) -> Any:
        """Linearize ``fn`` over its inexact leaves.

        Parameters
        ----------
        args : tuple
            Positional arguments of this trace.
        kwargs : dict
            Keyword arguments of this trace.

        Returns
        -------
        pytree
            Primal outputs of :func:`jax.linearize`. The tangent function is
            dropped so the jaxpr stays on arrays.
        """
        partial, inexact = partial_over_inexact(fn, args, kwargs)
        primals, _tangent = jax.linearize(partial, *inexact)
        return primals

    session = Session("plan")
    try:
        with activate(session):
            jax.make_jaxpr(diff)(abstract_args, abstract_kwargs)
    except Exception as exc:
        return [], False, str(exc)
    return list(session.records), True, ""


def _plan(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ctx: Context,
    overrides: Mapping[str, Any] | None,
) -> tuple[Plan, tuple[SiteRecord, ...]]:
    """Run the primal pass, the diff pass when needed, and :func:`resolve`.

    Parameters
    ----------
    fn : callable
        Function being planned.
    args : tuple
        Call arguments. Only shapes and dtypes are traced.
    kwargs : dict
        Call keyword arguments.
    ctx : Context
        Constraints passed to :func:`~xtrax.intent.resolve.resolve`.
    overrides : mapping or None
        Label overrides passed to :func:`~xtrax.intent.resolve.resolve`.

    Returns
    -------
    tuple
        ``(plan, records)`` for this signature.
    """
    abstract_args = _abstract_tree(args)
    abstract_kwargs = _abstract_tree(kwargs)
    primal_records = _trace_primal(fn, abstract_args, abstract_kwargs)
    if _any_inexact(args, kwargs):
        diff_records, diff_available, reason = _trace_diff(fn, abstract_args, abstract_kwargs)
    else:
        diff_records = []
        diff_available = True
        reason = ""
    records = _merge_records(primal_records, diff_records)
    plan = resolve(
        records,
        ctx,
        overrides=overrides,
        diff_available=diff_available,
        diff_unavailable_reason=reason,
    )
    return plan, records


def _execute(
    fn: Callable[..., Any],
    records: tuple[SiteRecord, ...],
    decisions: Mapping[SiteKey, SiteDecision],
) -> Callable[..., Any]:
    """Build a jitted wrapper that checks the primal site sequence.

    Parameters
    ----------
    fn : callable
        Function to run under the exec session.
    records : tuple of SiteRecord
        Merged plan records.
    decisions : mapping
        ``plan.by_key()``.

    Returns
    -------
    callable
        :func:`jax.jit` of the wrapper. A new session is built on every trace.
    """
    primal_keys = tuple(record.key for record in records if _is_primal_chain(record.key))
    stored = list(records)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` and reject a primal site sequence that is not the plan.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        pytree
            ``fn(*args, **kwargs)``.

        Raises
        ------
        SiteMismatchError
            The primal-path keys differ from the plan. ``constraint`` is
            ``"site sequence differs"``.
        """
        session = _ExecSession(stored, decisions)
        with activate(session):
            out = fn(*args, **kwargs)
        got = tuple(key for key in session.seen if _is_primal_chain(key))
        if got != primal_keys:
            raise SiteMismatchError(
                "site sequence differs",
                site="compile",
                path="primal",
                constraint="site sequence differs",
            )
        return out

    return jax.jit(wrapper)


def _bypass_warning(plan: Plan) -> bool:
    """Return whether the plan records an override that bypassed scan-around-while.

    Parameters
    ----------
    plan : Plan
        Plan whose decision warnings are scanned.

    Returns
    -------
    bool
        ``True`` when a warning matches ``override:… bypassed scan_around_while``.
    """
    return any(
        _BYPASS_WARNING.search(warning)
        for decision in plan.decisions
        for warning in decision.warnings
    )


def _scan_encloses_while(closed: Any) -> bool:
    """Return whether any ``scan`` equation contains a ``while`` equation.

    Parameters
    ----------
    closed : jaxpr
        Closed jaxpr or bare jaxpr of the executed wrapper.

    Returns
    -------
    bool
        ``True`` when a ``while`` equation sits inside a ``scan`` body.
    """
    for eqn in iter_jaxpr_eqns(closed):
        primitive = getattr(eqn, "primitive", None)
        if getattr(primitive, "name", None) != "scan":
            continue
        for sub in sub_jaxprs(getattr(eqn, "params", {})):
            for inner in iter_jaxpr_eqns(sub):
                inner_primitive = getattr(inner, "primitive", None)
                if getattr(inner_primitive, "name", None) == "while":
                    return True
    return False


def _posthoc(
    jitted: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    plan: Plan,
    ctx: Context,
) -> None:
    """Reject a scan that encloses a while on a scan-around-while backend.

    Parameters
    ----------
    jitted : callable
        Executed wrapper.
    args : tuple
        Arguments of this call.
    kwargs : dict
        Keyword arguments of this call.
    plan : Plan
        Plan for this signature. An override bypass warning skips the check.
    ctx : Context
        Context whose :meth:`~xtrax.intent.context.Context.resolved_backend`
        selects the backend set.

    Returns
    -------
    None

    Raises
    ------
    IntentError
        The jaxpr has a ``while`` inside a ``scan`` and the plan has no
        ``override:… bypassed scan_around_while`` warning. ``constraint`` is
        ``"scan_around_while"``.
    """
    if ctx.resolved_backend() not in SCAN_AROUND_WHILE_BACKENDS:
        return
    if _bypass_warning(plan):
        return

    def probe(*probe_args: Any, **probe_kwargs: Any) -> Any:
        """Trace the executed wrapper without re-entering planning.

        Parameters
        ----------
        probe_args : tuple
            Positional arguments.
        probe_kwargs : dict
            Keyword arguments.

        Returns
        -------
        pytree
            Wrapper output.
        """
        return jitted(*probe_args, **probe_kwargs)

    closed = jax.make_jaxpr(probe)(*args, **kwargs)
    if _scan_encloses_while(closed):
        raise IntentError(
            "scan around while",
            site="compile",
            path="primal",
            constraint="scan_around_while",
        )


class Compiled:
    """A planned, jitted intent function.

    Parameters
    ----------
    fn : callable
        Function containing :func:`~xtrax.intent.loop.loop` and
        :func:`~xtrax.intent.map.map` sites.
    ctx : Context
        Constraints for :func:`~xtrax.intent.resolve.resolve`.
    overrides : mapping or None
        Optional per-label strategy overrides.

    Returns
    -------
    None
    """

    plans: int

    def __init__(
        self,
        fn: Callable[..., Any],
        ctx: Context,
        overrides: Mapping[str, Any] | None,
    ) -> None:
        """Store the function and start an empty signature cache.

        Parameters
        ----------
        fn : callable
            Function to plan and execute.
        ctx : Context
            Resolution context.
        overrides : mapping or None
            Strategy overrides, or ``None``.

        Returns
        -------
        None
        """
        self._fn = fn
        self._ctx = ctx
        self._overrides = overrides
        self._cache: dict[_CacheKey, _Cached] = {}
        self._checked: set[_CacheKey] = set()
        self.plans = 0

    def plan_for(self, *args: Any, **kwargs: Any) -> Plan:
        """Plan this signature without executing the jitted wrapper.

        Parameters
        ----------
        args : tuple
            Positional arguments. Shapes and dtypes select the cache entry.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        Plan
            Resolved plan for this signature.
        """
        _key, _jitted, plan = self._prepare(args, kwargs)
        return plan

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the planned function, checking scan-around-while once per key.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        pytree
            Wrapper output.

        Raises
        ------
        IntentError
            The first call's jaxpr has a scan around a while and the plan did
            not record an override bypass.
        SiteMismatchError
            Execution visits a different primal site sequence, or a site's
            abstract values differ from the plan.
        UnsatisfiableContextError
            No strategy satisfies ``ctx``. Raised while planning, before the
            jitted wrapper runs.
        """
        key, jitted, plan = self._prepare(args, kwargs)
        result = jitted(*args, **kwargs)
        if key not in self._checked:
            _posthoc(jitted, args, kwargs, plan, self._ctx)
            self._checked.add(key)
        return result

    def _prepare(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[_CacheKey, Callable[..., Any], Plan]:
        """Return the cached jit and plan, planning on a cache miss.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        tuple
            ``(key, jitted, plan)``. ``plans`` increments once per new key.
        """
        key = _cache_key(args, kwargs, self._ctx)
        cached = self._cache.get(key)
        if cached is None:
            plan, records = _plan(self._fn, args, kwargs, self._ctx, self._overrides)
            jitted = _execute(self._fn, records, plan.by_key())
            cached = (jitted, plan, records)
            self._cache[key] = cached
            self.plans += 1
        jitted, plan, _records = cached
        return key, jitted, plan


def compile(  # noqa: A001
    fn: Callable[..., Any],
    ctx: Context = Context(),
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Compiled:
    """Plan ``fn`` per input signature and return a jitted execution wrapper.

    Parameters
    ----------
    fn : callable
        Function containing intent sites.
    ctx : Context
        Constraints. Defaults to an empty :class:`~xtrax.intent.context.Context`.
    overrides : mapping or None
        ``label -> (primal strategy, diff strategy or None)``. ``None`` means
        no overrides.

    Returns
    -------
    Compiled
        Callable that plans on first use of each signature, then executes.
        ``plan_for`` plans without calling the jitted wrapper. ``plans`` counts
        plan passes.
    """
    return Compiled(fn, ctx, overrides)

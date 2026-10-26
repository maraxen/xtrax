"""Compile an intent function: plan once per signature, then jit execution.

The plan pass traces with :func:`jax.make_jaxpr` under a plan session, then
traces :func:`jax.linearize` once per inexact argument when any leaf is
inexact so diff-path sites are recorded. :func:`~xtrax.intent.resolve`
turns those records into one :class:`~xtrax.intent.plan.Plan`. Execution is a
cached :func:`jax.jit` that re-enters an exec session and checks the primal
site sequence.
"""

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from xtrax.intent._resolve import (
    SCAN_AROUND_WHILE_BACKENDS,
    _is_scan_around_while_backend,
    resolve,
)
from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError
from xtrax.intent.plan import Plan, SiteDecision, SiteKey, canonical
from xtrax.intent.sites import Session, SiteRecord, _leaf_struct, activate
from xtrax.profiling.jaxpr import iter_jaxpr_eqns, sub_jaxprs
from xtrax.tiling.strategy import ChunkedMap, Scan, StaticWhile, WhileCarry

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
            User name or ``module:line``.
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


def _records_agree(left: SiteRecord, right: SiteRecord) -> bool:
    """Return whether ``left`` and ``right`` share avals and label.

    Parameters
    ----------
    left : SiteRecord
        Record already kept.
    right : SiteRecord
        Record being merged under the same key.

    Returns
    -------
    bool
        ``True`` when both labels and both aval tuples match.
    """
    return left.key.label == right.key.label and left.avals == right.avals


def _merge_records(
    primal_records: list[SiteRecord],
    diff_records: list[SiteRecord],
) -> tuple[SiteRecord, ...]:
    """Keep every primal record, then diff records that add a new key.

    Parameters
    ----------
    primal_records : list of SiteRecord
        Records from the primal plan pass, including diff-path visits made
        while ``fn`` itself differentiated.
    diff_records : list of SiteRecord
        Records from the diff plan pass.

    Returns
    -------
    tuple of SiteRecord
        Primal records in visit order, then diff records whose keys are new.

    Raises
    ------
    SiteMismatchError
        Both passes recorded ``key`` and disagree on avals or label. ``site``
        is that key and ``constraint`` names the two aval tuples.
    """
    merged: list[SiteRecord] = []
    by_key: dict[SiteKey, SiteRecord] = {}
    for record in primal_records:
        by_key[record.key] = record
        merged.append(record)
    for record in diff_records:
        previous = by_key.get(record.key)
        if previous is None:
            by_key[record.key] = record
            merged.append(record)
            continue
        if not _records_agree(previous, record):
            constraint = f"planned {previous.avals}, traced {record.avals}"
            raise SiteMismatchError(
                constraint,
                site=str(record.key),
                path=record.key.path,
                constraint=constraint,
            )
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


def _positional_labels(fn: Callable[..., Any], nargs: int) -> tuple[str, ...]:
    """Name each positional argument, or ``args[i]`` when the signature has none.

    Parameters
    ----------
    fn : callable
        Function being planned.
    nargs : int
        Number of positional arguments in this call.

    Returns
    -------
    tuple of str
        One label per positional argument, in call order.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return tuple(f"args[{index}]" for index in range(nargs))
    names: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            names.append(param.name)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            break
    labels: list[str] = []
    for index in range(nargs):
        if index < len(names):
            labels.append(names[index])
        else:
            labels.append(f"args[{index}]")
    return tuple(labels)


def _tree_has_inexact(tree: Any) -> bool:
    """Return whether any leaf of ``tree`` has an inexact dtype.

    Parameters
    ----------
    tree : pytree
        One argument. Leaves may be Python scalars.

    Returns
    -------
    bool
        ``True`` when a leaf dtype is inexact.
    """
    return any(
        jnp.issubdtype(_leaf_struct(leaf).dtype, jnp.inexact)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _trace_one_argument(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    index: int | None,
    keyword: str | None,
) -> tuple[list[SiteRecord], Exception | None]:
    """Linearize ``fn`` with respect to one argument under a fresh plan session.

    Other arguments stay the concrete call values, so a non-differentiable
    input is not part of this linearize. Exact leaves of the chosen argument
    are closed over.

    Parameters
    ----------
    fn : callable
        Function being planned.
    args : tuple
        Concrete positional arguments.
    kwargs : dict
        Concrete keyword arguments.
    index : int or None
        Positional index to differentiate. ``None`` selects ``keyword``.
    keyword : str or None
        Keyword name to differentiate. ``None`` selects ``index``.

    Returns
    -------
    tuple
        ``(records, error)``. ``error`` is the caught exception when
        linearize fails, and ``records`` is empty in that case.
    """
    target = args[index] if index is not None else kwargs[keyword or ""]
    abstract = _abstract_tree(target)

    def diff(traced_target: Any) -> Any:
        """Linearize ``fn`` over the inexact leaves of this argument.

        Parameters
        ----------
        traced_target : pytree
            Abstract value of the argument being differentiated.

        Returns
        -------
        pytree
            Primal outputs of :func:`jax.linearize`.
        """
        flat, treedef = jax.tree_util.tree_flatten(traced_target)
        flags = tuple(jnp.issubdtype(_leaf_struct(leaf).dtype, jnp.inexact) for leaf in flat)
        inexact = tuple(leaf for leaf, flag in zip(flat, flags, strict=True) if flag)

        def partial(*inexact_leaves: Any) -> Any:
            """Rebuild the argument and call ``fn``.

            Parameters
            ----------
            inexact_leaves : tuple
                Inexact leaves of this argument, in flatten order.

            Returns
            -------
            pytree
                ``fn`` applied to the concrete call with this argument replaced.
            """
            rebuilt: list[Any] = []
            cursor = 0
            for leaf, flag in zip(flat, flags, strict=True):
                if flag:
                    rebuilt.append(inexact_leaves[cursor])
                    cursor += 1
                else:
                    rebuilt.append(leaf)
            rebuilt_target = jax.tree_util.tree_unflatten(treedef, rebuilt)
            call_args = list(args)
            call_kwargs = dict(kwargs)
            if index is not None:
                call_args[index] = rebuilt_target
            elif keyword is not None:
                call_kwargs[keyword] = rebuilt_target
            return fn(*call_args, **call_kwargs)

        primals, _tangent = jax.linearize(partial, *inexact)
        return primals

    session = Session("plan")
    try:
        with activate(session):
            jax.make_jaxpr(diff)(abstract)
    except Exception as exc:
        return [], exc
    return list(session.records), None


def _trace_diff(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[SiteRecord], bool, str]:
    """Record diff sites by linearizing each inexact argument on its own.

    Parameters
    ----------
    fn : callable
        Function being planned.
    args : tuple
        Concrete positional arguments.
    kwargs : dict
        Concrete keyword arguments.

    Returns
    -------
    tuple
        ``(records, diff_available, reason)``. An argument whose linearize
        fails contributes no records. ``reason`` names each failed argument
        and the exception type. ``diff_available`` is ``True`` when at least
        one inexact argument linearized, or when there were no inexact
        arguments. It is ``False`` only when every inexact argument failed.
        A partial failure leaves ``diff_available`` true and puts the failed
        arguments in ``reason`` as
        ``not differentiable w.r.t. <label> (<ExceptionName>)``.
    """
    collected: list[SiteRecord] = []
    failures: list[str] = []
    labels = _positional_labels(fn, len(args))
    targets: list[tuple[str, int | None, str | None]] = []
    for index, arg in enumerate(args):
        if _tree_has_inexact(arg):
            targets.append((labels[index], index, None))
    for name, value in kwargs.items():
        if _tree_has_inexact(value):
            targets.append((name, None, name))
    for label, index, keyword in targets:
        records, exc = _trace_one_argument(fn, args, kwargs, index=index, keyword=keyword)
        if exc is not None:
            failures.append(f"{label}: {type(exc).__name__}")
            continue
        collected = list(_merge_records(collected, records))
    if not targets or not failures:
        return collected, True, ""
    if len(failures) == len(targets):
        return collected, False, "; ".join(failures)
    return collected, True, _partial_diff_note(failures)


def _partial_diff_note(failures: list[str]) -> str:
    """Name each argument whose linearize failed while another succeeded.

    Parameters
    ----------
    failures : list of str
        ``label: ExceptionName`` entries from :func:`_trace_diff`.

    Returns
    -------
    str
        ``not differentiable w.r.t. <label> (<ExceptionName>)``, joined by
        ``"; "`` when more than one argument failed.
    """
    parts: list[str] = []
    for item in failures:
        label, separator, exc_name = item.partition(": ")
        if separator:
            parts.append(f"not differentiable w.r.t. {label} ({exc_name})")
        else:
            parts.append(f"not differentiable w.r.t. {item}")
    return "; ".join(parts)


def _annotate_partial_diff(plan: Plan, note: str) -> Plan:
    """Record ``note`` on the plan, and on the diff reason of loops that have a diff strategy.

    The plan-level warning is the one place every plan names the failed
    arguments: a conditional loop has no diff strategy and a map has no loop
    row, so a plan with only those sites would otherwise never say which
    argument failed. Loops with a diff strategy keep the note on
    ``diff_reason`` as well, next to the strategy it qualifies.

    Parameters
    ----------
    plan : Plan
        Plan whose loop decisions may claim a diff strategy.
    note : str
        Partial-failure text from :func:`_partial_diff_note`.

    Returns
    -------
    Plan
        The same plan when ``note`` is empty. Otherwise ``note`` is appended
        once to ``plan.warnings``, and each loop whose ``diff`` is not
        ``None`` names the failed arguments on ``diff_reason``.
    """
    if not note:
        return plan
    decisions = tuple(
        replace(decision, diff_reason=f"{decision.diff_reason}; {note}")
        if decision.kind == "loop" and decision.diff is not None
        else decision
        for decision in plan.decisions
    )
    return replace(plan, decisions=decisions, warnings=(*plan.warnings, note))


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
        Constraints passed to :func:`~xtrax.intent.resolve`.
    overrides : mapping or None
        Label overrides passed to :func:`~xtrax.intent.resolve`.

    Returns
    -------
    tuple
        ``(plan, records)`` for this signature.
    """
    abstract_args = _abstract_tree(args)
    abstract_kwargs = _abstract_tree(kwargs)
    primal_records = _trace_primal(fn, abstract_args, abstract_kwargs)
    if _any_inexact(args, kwargs):
        diff_records, diff_available, reason = _trace_diff(fn, args, kwargs)
    else:
        diff_records = []
        diff_available = True
        reason = ""
    records = _merge_records(primal_records, diff_records)
    unavailable = "" if diff_available else reason
    plan = resolve(
        records,
        ctx,
        overrides=overrides,
        diff_available=diff_available,
        diff_unavailable_reason=unavailable,
    )
    if diff_available and reason:
        plan = _annotate_partial_diff(plan, reason)
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


def _site_bypassed(decision: SiteDecision) -> bool:
    """Return whether this decision's warnings exempt its own scan-around-while.

    Parameters
    ----------
    decision : SiteDecision
        One planned site.

    Returns
    -------
    bool
        ``True`` when a warning matches ``override:… bypassed scan_around_while``.
    """
    return any(_BYPASS_WARNING.search(warning) for warning in decision.warnings)


def _encloses_while(key: SiteKey, plan: Plan) -> bool:
    """Return whether a descendant primal is ``StaticWhile`` or ``WhileCarry``.

    Parameters
    ----------
    key : SiteKey
        Candidate outer map or scan loop.
    plan : Plan
        Plan whose decisions are searched.

    Returns
    -------
    bool
        ``True`` when some decision under ``key`` runs a while loop.
    """
    ancestor = canonical(key)
    for other in plan.decisions:
        if not isinstance(other.primal, (StaticWhile, WhileCarry)):
            continue
        current = other.key.parent
        while current is not None:
            if canonical(current) == ancestor:
                return True
            current = current.parent
    return False


def _scan_while_sources(plan: Plan) -> tuple[SiteDecision, ...]:
    """Return primal sites that can place a while loop inside a scan.

    Parameters
    ----------
    plan : Plan
        Plan to scan. Diff-path copies are skipped.

    Returns
    -------
    tuple of SiteDecision
        ``ChunkedMap`` sites, and ``Scan`` loop sites, that enclose a
        ``StaticWhile`` or ``WhileCarry``.
    """
    sources: list[SiteDecision] = []
    for decision in plan.decisions:
        if not _is_primal_chain(decision.key):
            continue
        chunked = isinstance(decision.primal, ChunkedMap)
        scan_loop = decision.kind == "loop" and isinstance(decision.primal, Scan)
        if (chunked or scan_loop) and _encloses_while(decision.key, plan):
            sources.append(decision)
    return tuple(sources)


def _pattern_is_exempt(plan: Plan) -> bool:
    """Return whether every plan-level scan-around-while source is bypassed.

    A bypass warning exempts that site only. An empty source list is not
    an exemption: a scan around a while that the plan does not name is
    still rejected.

    Parameters
    ----------
    plan : Plan
        Plan whose warnings are consulted.

    Returns
    -------
    bool
        ``True`` when at least one source exists and each of them carries
        ``override:… bypassed scan_around_while``.
    """
    sources = _scan_while_sources(plan)
    if not sources:
        return False
    return all(_site_bypassed(source) for source in sources)


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
        Plan for this signature. A bypass warning exempts only the site that
        carries it. Any other scan around a while is still rejected.
    ctx : Context
        Context whose :meth:`~xtrax.intent.context.Context.resolved_backend`
        selects the backend set. ``"gpu"`` and ``"cuda"`` both select it.

    Returns
    -------
    None

    Raises
    ------
    IntentError
        The jaxpr has a ``while`` inside a ``scan`` and some source of that
        pattern does not carry ``override:… bypassed scan_around_while``.
        ``constraint`` is ``"scan_around_while"``.
    """
    if not _is_scan_around_while_backend(ctx.resolved_backend(), SCAN_AROUND_WHILE_BACKENDS):
        return
    if _pattern_is_exempt(plan):
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
        Function containing :func:`~xtrax.intent.loop` and
        :func:`~xtrax.intent.map` sites.
    ctx : Context
        Constraints for :func:`~xtrax.intent.resolve`.
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
        self._posthoc_verdict: dict[_CacheKey, IntentError | None] = {}
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
        """Execute the planned function after the scan-around-while check.

        The check only traces. It runs before the jitted program, and both
        a pass and a failure are cached on the signature key. A rejected key
        raises the cached error without tracing or executing again.

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
            The jaxpr has a scan around a while and that pattern is not
            exempted by a bypass warning on the site that produces it.
            Raised before the jitted program runs.
        SiteMismatchError
            Execution visits a different primal site sequence, or a site's
            abstract values differ from the plan.
        UnsatisfiableContextError
            No strategy satisfies ``ctx``. Raised while planning, before the
            jitted wrapper runs.
        """
        key, jitted, plan = self._prepare(args, kwargs)
        if key not in self._posthoc_verdict:
            try:
                _posthoc(jitted, args, kwargs, plan, self._ctx)
            except IntentError as exc:
                self._posthoc_verdict[key] = exc
            else:
                self._posthoc_verdict[key] = None
        verdict = self._posthoc_verdict[key]
        if verdict is not None:
            raise verdict
        return jitted(*args, **kwargs)

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

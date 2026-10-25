"""Experimental. Shared tracing helpers for intent sites.

Registration is one flag so a loop and a map suspend the same way: the
closure conversion that only hoists constants does not record nested sites.
"""

import dataclasses
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from xtrax.intent.plan import SiteDecision, SiteKey, canonical
from xtrax.intent.sites import Session, current_session

_REGISTER: ContextVar[bool] = ContextVar("xtrax.intent.register", default=True)


@contextmanager
def _suspend_registration():
    """Hide the active session while a closure is converted for its record."""
    token = _REGISTER.set(False)
    try:
        yield
    finally:
        _REGISTER.reset(token)


def _active_session() -> Session | None:
    """Return the session that should record this call, if registration is on."""
    if not _REGISTER.get():
        return None
    return current_session()


def _retrace(body: Callable[..., Any]) -> Callable[..., Any]:
    """Return a new callable so a later closure conversion traces ``body`` again."""

    def _body(*args: Any) -> Any:
        return body(*args)

    return _body


def _has_diff_edge(key: SiteKey) -> bool:
    """Return whether ``key`` or one of its parents is on the diff path."""
    current: SiteKey | None = key
    while current is not None:
        if current.path == "diff":
            return True
        current = current.parent
    return False


def decision_for(session: Session, key: SiteKey) -> SiteDecision | None:
    """Return the decision for ``key``, falling back to its canonical twin.

    An explicit decision on ``key`` wins, including when that decision's
    ``diff`` is ``None``. ``None`` means the site has no differentiated
    strategy. It is not a missing site, and callers must not raise
    :class:`~xtrax.intent.errors.SiteMismatchError` for it. When the lookup
    misses and ``key`` or any parent is on the diff path, the decision stored
    on :func:`~xtrax.intent.plan.canonical` of ``key`` is returned. Any other
    miss is ``None``.

    Parameters
    ----------
    session : Session
        Session whose ``decisions`` are consulted.
    key : SiteKey
        Site being resolved. A key with a diff edge and no decision of its
        own uses the primal-chain twin.

    Returns
    -------
    SiteDecision or None
        ``session.decision_for(key)`` when that is not ``None``. Otherwise
        the canonical twin's decision when the chain contains a diff edge.
        Otherwise ``None``.
    """
    decision = session.decision_for(key)
    if decision is not None:
        return decision
    if _has_diff_edge(key):
        return session.decision_for(canonical(key))
    return None


def _store_running_fn(session: Session, key: SiteKey, fn: Callable[..., Any]) -> None:
    """In plan mode, keep the conversion that ran, not the registration placeholder."""
    if session.mode != "plan":
        return
    for index, record in enumerate(session.records):
        if record.key == key:
            session.records[index] = dataclasses.replace(record, fn=fn)
            return

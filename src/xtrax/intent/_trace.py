"""Experimental. Shared tracing helpers for intent sites.

Registration is one flag so a loop and a map suspend the same way: the
closure conversion that only hoists constants does not record nested sites.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from xtrax.intent.plan import SiteDecision, SiteKey
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


def decision_for(session: Session, key: SiteKey) -> SiteDecision | None:
    """Return the decision for ``key``, falling back to its primal twin.

    An explicit decision on ``key`` wins. When ``key`` is on the diff path and
    that lookup misses, the decision stored on the same parent, label, and
    ordinal with ``path="primal"`` is returned. Any other miss is ``None``.

    Parameters
    ----------
    session : Session
        Session whose ``decisions`` are consulted.
    key : SiteKey
        Site being resolved. A diff key with no decision of its own uses the
        primal twin.

    Returns
    -------
    SiteDecision or None
        ``session.decision_for(key)`` when that is not ``None``. Otherwise the
        primal twin's decision when ``key.path == "diff"``. Otherwise ``None``.
    """
    decision = session.decision_for(key)
    if decision is not None:
        return decision
    if key.path == "diff":
        return session.decision_for(dataclasses.replace(key, path="primal"))
    return None


def _store_running_fn(session: Session, key: SiteKey, fn: Callable[..., Any]) -> None:
    """In plan mode, keep the conversion that ran, not the registration placeholder."""
    if session.mode != "plan":
        return
    for index, record in enumerate(session.records):
        if record.key == key:
            session.records[index] = dataclasses.replace(record, fn=fn)
            return

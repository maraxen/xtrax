"""Experimental. Intent resolution for loops and maps.

``xtrax.intent`` records caller constraints (``Context``), the errors
raised when a context or site cannot be satisfied, the ``Plan`` of
per-site decisions, the :class:`~xtrax.intent.sites.Session` that visits
those sites, :func:`~xtrax.intent.loop.loop`, and
:func:`~xtrax.intent.map.map`. A later task adds compile.
"""

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError, UnsatisfiableContextError
from xtrax.intent.loop import loop
from xtrax.intent.map import map
from xtrax.intent.plan import PLAN_SCHEMA_VERSION, Plan, SiteDecision, SiteKey
from xtrax.intent.sites import Session, SiteRecord, activate, call_label, current_session

__all__ = [
    "Context",
    "IntentError",
    "PLAN_SCHEMA_VERSION",
    "Plan",
    "Session",
    "SiteDecision",
    "SiteKey",
    "SiteMismatchError",
    "SiteRecord",
    "UnsatisfiableContextError",
    "activate",
    "call_label",
    "current_session",
    "loop",
    "map",
]

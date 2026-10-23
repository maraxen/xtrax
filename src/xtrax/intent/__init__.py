"""Experimental. Intent resolution for loops and maps.

``xtrax.intent`` records caller constraints (``Context``), the errors
raised when a context or site cannot be satisfied, the ``Plan`` of
per-site decisions, and the :class:`~xtrax.intent.sites.Session` that
visits those sites. Later tasks add loop, map, and compile.
"""

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError, UnsatisfiableContextError
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
]

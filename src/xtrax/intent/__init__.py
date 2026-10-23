"""Experimental. Intent resolution for loops and maps.

``xtrax.intent`` records caller constraints (``Context``), the errors
raised when a context or site cannot be satisfied, and the ``Plan`` of
per-site decisions. Later tasks add loop, map, and compile.
"""

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError, UnsatisfiableContextError
from xtrax.intent.plan import PLAN_SCHEMA_VERSION, Plan, SiteDecision, SiteKey

__all__ = [
    "Context",
    "IntentError",
    "PLAN_SCHEMA_VERSION",
    "Plan",
    "SiteDecision",
    "SiteKey",
    "SiteMismatchError",
    "UnsatisfiableContextError",
]

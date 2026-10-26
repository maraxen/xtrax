"""Intent resolution for loops and maps.

``xtrax.intent`` records caller constraints (``Context``), the errors
raised when a context or site cannot be satisfied, the ``Plan`` of
per-site decisions, the :class:`~xtrax.intent.sites.Session` that visits
those sites, :func:`~xtrax.intent.loop`,
:func:`~xtrax.intent.map`, :func:`~xtrax.intent.resolve`, and
:func:`~xtrax.intent.compile`.

The implementation modules are ``_compile``, ``_loop``, ``_map`` and
``_resolve``. Their public names are functions on this package, so a module
with the same name would be shadowed by the function and could not be reached
by ``import xtrax.intent.map as m`` or a ``mock.patch`` string target.
"""

from xtrax.intent._compile import compile  # noqa: A001
from xtrax.intent._loop import loop
from xtrax.intent._map import map
from xtrax.intent._resolve import HAZARDS, SiteStrategyOverride, default_estimator, resolve
from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError, UnsatisfiableContextError
from xtrax.intent.plan import PLAN_SCHEMA_VERSION, Plan, SiteDecision, SiteKey
from xtrax.intent.sites import Session, SiteRecord, activate, call_label, current_session

__all__ = [
    "Context",
    "HAZARDS",
    "IntentError",
    "PLAN_SCHEMA_VERSION",
    "Plan",
    "Session",
    "SiteDecision",
    "SiteKey",
    "SiteMismatchError",
    "SiteRecord",
    "SiteStrategyOverride",
    "UnsatisfiableContextError",
    "activate",
    "call_label",
    "compile",
    "current_session",
    "default_estimator",
    "loop",
    "map",
    "resolve",
]

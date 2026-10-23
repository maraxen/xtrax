"""Experimental. Intent resolution for loops and maps.

``xtrax.intent`` records caller constraints (``Context``) and the errors
raised when a context or site cannot be satisfied. Later tasks add loop,
map, and compile.
"""

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, SiteMismatchError, UnsatisfiableContextError

__all__ = [
    "Context",
    "IntentError",
    "SiteMismatchError",
    "UnsatisfiableContextError",
]

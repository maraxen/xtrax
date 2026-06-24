"""CLI verb registry for xtrax (T3).

Maps verb names to (ArgsClass, run_fn) pairs. The registry is the single
source of truth for all supported CLI verbs. Verbs can be added here as
new modules are implemented.

Note: run_fn imports are eager — these are xtrax's own CLI verb functions,
not user code. Only user-provided --fn arguments are loaded lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xtrax.cli.explain import ExplainArgs, run_explain
from xtrax.cli.export import ExportArgs, run_export
from xtrax.cli.plan import PlanArgs, run_plan
from xtrax.cli.run_verb import RunArgs, run_run

REGISTRY: dict[str, tuple[type[Any], Callable[..., None]]] = {
    "plan": (PlanArgs, run_plan),
    "explain": (ExplainArgs, run_explain),
    "export": (ExportArgs, run_export),
    "run": (RunArgs, run_run),
}

__all__ = ["REGISTRY"]

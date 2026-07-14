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
from xtrax.cli.graph_author_verb import GraphAuthorArgs, run_graph_author
from xtrax.cli.graph_plan_verb import GraphPlanArgs, run_graph_plan
from xtrax.cli.graph_verb import GraphValidateArgs, run_graph_validate
from xtrax.cli.plan import PlanArgs, run_plan
from xtrax.cli.resume_verb import ResumeArgs, run_resume
from xtrax.cli.run_verb import RunArgs, run_run
from xtrax.cli.sweep_verb import SweepArgs, run_sweep

REGISTRY: dict[str, tuple[type[Any], Callable[..., None]]] = {
    "plan": (PlanArgs, run_plan),
    "explain": (ExplainArgs, run_explain),
    "export": (ExportArgs, run_export),
    "run": (RunArgs, run_run),
    "resume": (ResumeArgs, run_resume),
    "sweep": (SweepArgs, run_sweep),
    "graph-validate": (GraphValidateArgs, run_graph_validate),
    "graph-plan": (GraphPlanArgs, run_graph_plan),
    "graph-author": (GraphAuthorArgs, run_graph_author),
}

__all__ = ["REGISTRY"]

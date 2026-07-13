"""D5 graph validation gate: chains eval_shape-consistency + plan-topology + jaxlint into a
per-node `audit_verdict` (T1-10, #2181 entry-edge item).

Two scoping decisions worth recording explicitly (per the AskUserQuestion resolution this
session, matching T1-09's own precedent of documenting scope narrowing):

1. `eval_shape`-consistency is NOT a full `extract_schema`/`jax.eval_shape` trace -- the D4 IR
   document (`xtrax.composition.serialize`) carries no per-node abstract-input shapes for a
   real trace to run against (deliberately deferred by T1-09's own audit review). Instead,
   `_check_callable_resolvable` verifies each node's already-resolved `callable_ref` is
   genuinely introspectable (`inspect.signature` succeeds) -- a real, narrower proxy check with
   its own real (if rare) failure mode, not a synthesized/guessed-shape trace that would risk
   spurious FAILs on legitimate non-scalar callables.
2. `validate_plan_topology` is called for real every time, with `decisions=()` and
   `axis_boundaries={}` -- there is nothing in the D4 document for it to violate yet, so this
   call is trivially a no-op today, but it is wired correctly and will start catching real
   topology violations the moment a future item adds per-node axis/boundary data to
   `HostPrepGraphNode`.

Both narrowings should be revisited by whichever future item grows `HostPrepGraphNode` an
axis/shape/boundary-config field -- widen `_check_callable_resolvable` into a real
`extract_schema` trace and pass real `decisions`/`axis_boundaries` to `validate_plan_topology`
at that point, not before (there is no real data to trace against today).
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.composition.graph import HostPrepGraph, HostPrepGraphNode
from xtrax.devtools.gates._jaxlint import run_jaxlint_json as _run_jaxlint_json
from xtrax.stages.topology import PlanTopologyError, validate_plan_topology

PASS = "PASS"
FAIL = "FAIL"
NEEDS_WORK = "NEEDS_WORK"

_VERDICT_RANK = {PASS: 0, NEEDS_WORK: 1, FAIL: 2}


@dataclass(frozen=True, slots=True)
class NodeCheckFailure:
    node_id: str
    check: str
    message: str


@dataclass(frozen=True, slots=True)
class GraphValidationResult:
    graph: HostPrepGraph
    failures: tuple[NodeCheckFailure, ...]
    passed: bool


def _check_callable_resolvable(fn: Callable[..., Any]) -> str | None:
    """Return an error message if `fn` cannot be introspected, else None.

    A narrower proxy for full eval_shape-consistency (see module docstring) -- its own real
    failure mode is a resolved-but-uninspectable callable (e.g. certain C-extension objects),
    not a shape mismatch (which needs real abstract inputs this document doesn't carry).
    """
    try:
        inspect.signature(fn)
    except (ValueError, TypeError) as exc:
        return f"callable {fn!r} is not introspectable: {exc}"
    return None


def _finding_line(finding: dict[str, Any]) -> int | None:
    line = finding.get("line")
    return line if isinstance(line, int) and not isinstance(line, bool) else None


def _jaxlint_errors_for(
    fn: Callable[..., Any],
    root: Path,
    cache: dict[Path, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return JL-series error-severity jaxlint findings attributable to `fn` specifically.

    Returns [] (not a failure) when source is unavailable -- e.g. builtins have no source
    file to lint, which is a real absence of data, not a lint violation.

    File-wide findings are cached per resolved path (`cache`) to avoid redundant subprocess
    calls when multiple nodes share a module, then filtered to `fn`'s own source-line range --
    without this filter, a single JL-error finding anywhere in a shared file would wrongly
    downgrade EVERY node whose callable lives in that file, not just the one the finding
    actually concerns. A finding with no usable `line` (can't be attributed precisely) is kept
    conservatively rather than silently dropped.
    """
    try:
        source_file = inspect.getsourcefile(fn)
    except TypeError:
        return []
    if source_file is None:
        return []

    resolved = Path(source_file).resolve()
    if resolved not in cache:
        findings = _run_jaxlint_json(resolved, root=root)
        cache[resolved] = [
            f
            for f in findings
            if str(f.get("rule_id", "")).startswith("JL")
            and str(f.get("severity", "")).lower() == "error"
        ]
    file_findings = cache[resolved]

    try:
        source_lines, start_line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return file_findings
    end_line = start_line + len(source_lines) - 1

    return [
        f
        for f in file_findings
        if (line := _finding_line(f)) is None or start_line <= line <= end_line
    ]


def _rebuild_node(node: HostPrepGraphNode, verdict: str) -> HostPrepGraphNode:
    """Construct a NEW node carrying the computed audit_verdict.

    Never mutates node.metadata in place -- that would silently bypass the frozen-node
    __setattr__ guard, which only intercepts attribute reassignment, not in-place dict
    mutation of an already-set attribute's value.
    """
    return HostPrepGraphNode(
        id=node.id,
        callable_ref=node.callable_ref,
        metadata={**dict(node.metadata), "audit_verdict": verdict},
        frozen=node.frozen,
    )


def validate_graph(graph: HostPrepGraph, *, root: Path | None = None) -> GraphValidationResult:
    """Chain eval_shape-consistency + plan-topology + jaxlint into a per-node audit_verdict.

    Every node in the returned graph carries a written `audit_verdict` (PASS included, not
    just on failure) -- "writes audit_verdict into node metadata" is an unconditional effect
    of running this gate, not conditional on failure.
    """
    resolved_root = root if root is not None else Path.cwd()
    jaxlint_cache: dict[Path, list[dict[str, Any]]] = {}

    try:
        validate_plan_topology(decisions=(), axis_boundaries={})
        topology_failure: NodeCheckFailure | None = None
    except PlanTopologyError as exc:
        topology_failure = NodeCheckFailure(node_id="", check="plan_topology", message=str(exc))

    # Graph-wide: a topology violation fails every node, but is reported once (node_id="").
    failures: list[NodeCheckFailure] = []
    if topology_failure is not None:
        failures.append(topology_failure)

    rebuilt_nodes: list[HostPrepGraphNode] = []
    for node in graph.nodes:
        verdict = FAIL if topology_failure is not None else PASS

        resolvable_error = _check_callable_resolvable(node.callable_ref)
        if resolvable_error is not None:
            failures.append(
                NodeCheckFailure(
                    node_id=node.id,
                    # Named "_proxy", not "eval_shape_consistency" -- this checks callable
                    # introspectability, not a real extract_schema/jax.eval_shape trace (see
                    # module docstring). A downstream consumer of the JSON envelope must be
                    # able to tell the two apart from the check name alone.
                    check="eval_shape_consistency_proxy",
                    message=resolvable_error,
                )
            )
            verdict = FAIL

        for finding in _jaxlint_errors_for(node.callable_ref, resolved_root, jaxlint_cache):
            failures.append(
                NodeCheckFailure(
                    node_id=node.id,
                    check="jaxlint",
                    message=str(finding.get("message", "jaxlint JL error")),
                )
            )
            if _VERDICT_RANK[NEEDS_WORK] > _VERDICT_RANK[verdict]:
                verdict = NEEDS_WORK

        rebuilt_nodes.append(_rebuild_node(node, verdict))

    rebuilt_graph = HostPrepGraph(nodes=tuple(rebuilt_nodes), edges=graph.edges)
    return GraphValidationResult(
        graph=rebuilt_graph,
        failures=tuple(failures),
        passed=len(failures) == 0,
    )


__all__ = [
    "FAIL",
    "NEEDS_WORK",
    "PASS",
    "GraphValidationResult",
    "NodeCheckFailure",
    "validate_graph",
]

#!/usr/bin/env python3
"""Walk lowered composition graphs and validate per-node metadata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from load_capability_registry import (
    NodeMetadataSchema,
    load_node_metadata_schema,
    validate_node_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / ".praxia" / "composition" / "samples" / "valid_graph.json"
AUDIT_VERDICTS_REQUIRING_BATHOS = frozenset({"PASS", "FAIL"})


@dataclass(frozen=True)
class GraphNode:
    id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CompositionGraph:
    nodes: tuple[GraphNode, ...]


@dataclass(frozen=True)
class GraphAuditFinding:
    node_id: str
    rule_id: str
    message: str
    severity: str


@dataclass(frozen=True)
class GraphAuditResult:
    passed: bool
    findings: tuple[GraphAuditFinding, ...]


def _fail_gate_severities() -> frozenset[str]:
    return frozenset({"error", "critical"})


def load_graph_json(path: Path) -> CompositionGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("graph JSON must contain a non-empty 'nodes' list")

    nodes: list[GraphNode] = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"nodes[{index}] must be an object")
        node_id = raw.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(f"nodes[{index}]: missing or empty 'id'")
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                f"nodes[{index}] ({node_id}): 'metadata' must be an object"
            )
        nodes.append(GraphNode(id=node_id.strip(), metadata=metadata))

    return CompositionGraph(nodes=tuple(nodes))


def _check_duplicate_node_ids(graph: CompositionGraph) -> list[GraphAuditFinding]:
    counts: dict[str, int] = {}
    for node in graph.nodes:
        counts[node.id] = counts.get(node.id, 0) + 1

    findings: list[GraphAuditFinding] = []
    for node_id, count in sorted(counts.items()):
        if count > 1:
            findings.append(
                GraphAuditFinding(
                    node_id=node_id,
                    rule_id="duplicate_node_id",
                    message=(
                        f"node '{node_id}' appears {count} times; "
                        "node ids must be unique"
                    ),
                    severity="critical",
                )
            )
    return findings


def _check_node_metadata(
    node: GraphNode,
    schema: NodeMetadataSchema,
) -> list[GraphAuditFinding]:
    findings: list[GraphAuditFinding] = []
    try:
        validate_node_metadata(node.metadata, schema)
    except ValueError as exc:
        findings.append(
            GraphAuditFinding(
                node_id=node.id,
                rule_id="validate_node_metadata",
                message=f"node '{node.id}': {exc}",
                severity="error",
            )
        )
    return findings


def _check_bathos_sidecar_ref(node: GraphNode) -> list[GraphAuditFinding]:
    verdict = node.metadata.get("audit_verdict")
    if verdict not in AUDIT_VERDICTS_REQUIRING_BATHOS:
        return []

    bathos_ref = node.metadata.get("bathos_sidecar_ref")
    if isinstance(bathos_ref, str) and bathos_ref.strip():
        return []

    return [
        GraphAuditFinding(
            node_id=node.id,
            rule_id="missing_bathos_sidecar_ref",
            message=(
                f"node '{node.id}': audit_verdict={verdict!r} "
                "without bathos_sidecar_ref"
            ),
            severity="minor",
        )
    ]


def audit_graph(
    graph: CompositionGraph,
    schema: NodeMetadataSchema | None = None,
) -> GraphAuditResult:
    resolved_schema = schema or load_node_metadata_schema()
    findings: list[GraphAuditFinding] = []

    findings.extend(_check_duplicate_node_ids(graph))
    for node in graph.nodes:
        findings.extend(_check_node_metadata(node, resolved_schema))
        findings.extend(_check_bathos_sidecar_ref(node))

    passed = not any(f.severity in _fail_gate_severities() for f in findings)
    return GraphAuditResult(passed=passed, findings=tuple(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph",
        nargs="?",
        default=str(DEFAULT_SAMPLE),
        help="Path to lowered composition graph JSON",
    )
    args = parser.parse_args(argv)

    graph_path = Path(args.graph).resolve()
    if not graph_path.is_file():
        print(f"graph not found: {graph_path}", file=sys.stderr)
        return 2

    graph = load_graph_json(graph_path)
    result = audit_graph(graph)

    status = "PASS" if result.passed else "FAIL"
    fail_gate = [f for f in result.findings if f.severity in _fail_gate_severities()]
    minor = [f for f in result.findings if f.severity == "minor"]
    print(
        f"{status}: nodes={len(graph.nodes)} "
        f"findings={len(result.findings)} "
        f"(gate={len(fail_gate)}, minor={len(minor)})"
    )
    for finding in result.findings:
        print(
            f"  [{finding.severity}] {finding.node_id} "
            f"{finding.rule_id}: {finding.message}"
        )

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

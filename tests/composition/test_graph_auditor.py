"""Graph-auditor validation harness for lowered composition graphs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from graph_auditor import (  # noqa: E402
    CompositionGraph,
    GraphNode,
    audit_graph,
    load_graph_json,
)

SAMPLE_GRAPH = ROOT / ".praxia" / "composition" / "samples" / "valid_graph.json"


def test_valid_graph_passes(valid_graph: CompositionGraph) -> None:
    result = audit_graph(valid_graph)
    assert result.passed
    assert not any(f.severity in {"error", "critical"} for f in result.findings)


def test_missing_nl_description_fails_with_node_id(
    invalid_graph_missing_nl: CompositionGraph,
) -> None:
    result = audit_graph(invalid_graph_missing_nl)
    assert not result.passed
    metadata_findings = [
        f for f in result.findings if f.rule_id == "validate_node_metadata"
    ]
    assert len(metadata_findings) == 1
    assert metadata_findings[0].node_id == "bad-node"
    assert "bad-node" in metadata_findings[0].message
    assert "nl_description" in metadata_findings[0].message


def test_duplicate_ids_fail(graph_duplicate_node_ids: CompositionGraph) -> None:
    result = audit_graph(graph_duplicate_node_ids)
    assert not result.passed
    dup_findings = [f for f in result.findings if f.rule_id == "duplicate_node_id"]
    assert len(dup_findings) == 1
    assert dup_findings[0].node_id == "dup"
    assert dup_findings[0].severity == "critical"


def test_pass_verdict_without_bathos_emits_minor_finding() -> None:
    graph = CompositionGraph(
        nodes=(
            GraphNode(
                id="audited",
                metadata={
                    "nl_description": "Node with PASS verdict but no bathos ref.",
                    "audit_verdict": "PASS",
                },
            ),
        )
    )
    result = audit_graph(graph)
    assert result.passed
    minor = [f for f in result.findings if f.rule_id == "missing_bathos_sidecar_ref"]
    assert len(minor) == 1
    assert minor[0].severity == "minor"
    assert minor[0].node_id == "audited"


def test_load_graph_json_round_trip(
    tmp_path: Path, valid_graph: CompositionGraph
) -> None:
    payload = {
        "nodes": [
            {"id": node.id, "metadata": node.metadata} for node in valid_graph.nodes
        ]
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded = load_graph_json(graph_path)
    assert loaded == valid_graph
    assert audit_graph(loaded).passed


def test_committed_sample_graph_passes() -> None:
    graph = load_graph_json(SAMPLE_GRAPH)
    result = audit_graph(graph)
    assert result.passed

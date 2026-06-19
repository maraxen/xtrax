"""Shared fixtures for composition-layer contract tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from graph_auditor import CompositionGraph, GraphNode  # noqa: E402

VALID_SAMPLE = {
    "nl_description": "Prepare host-side tensors before JIT lowering.",
    "mathjax_label": r"\mathbf{x} \in \mathbb{R}^n",
    "citations": [{"doi": "10.1234/example"}],
    "script_usage": {"language": "python", "excerpt": "x = jnp.asarray(x_host)"},
    "audit_verdict": "PASS",
    "bathos_sidecar_ref": ".praxia/experiments/run_001.toml",
}

MINIMAL_METADATA = {
    "nl_description": "Minimal node with required nl_description only.",
}


@pytest.fixture
def valid_graph() -> CompositionGraph:
    return CompositionGraph(
        nodes=(
            GraphNode(id="minimal-node", metadata=MINIMAL_METADATA),
            GraphNode(id="full-node", metadata=VALID_SAMPLE),
        )
    )


@pytest.fixture
def invalid_graph_missing_nl() -> CompositionGraph:
    return CompositionGraph(
        nodes=(
            GraphNode(
                id="bad-node",
                metadata={"mathjax_label": r"\alpha"},
            ),
        )
    )


@pytest.fixture
def graph_duplicate_node_ids() -> CompositionGraph:
    return CompositionGraph(
        nodes=(
            GraphNode(id="dup", metadata=MINIMAL_METADATA),
            GraphNode(id="dup", metadata=MINIMAL_METADATA),
        )
    )

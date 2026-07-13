"""Tests for the T1-10 graph validation gate (composition-layer core logic)."""

import inspect
from pathlib import Path
from unittest.mock import patch

from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode
from xtrax.composition.validate import (
    FAIL,
    NEEDS_WORK,
    PASS,
    _check_callable_resolvable,
    validate_graph,
)
from xtrax.stages.topology import PlanTopologyError, validate_plan_topology

VALID_METADATA = {"nl_description": "A node under test."}


def _clean_fn(x):
    return x


def _other_clean_fn(x):
    return x


class _UnintrospectableCallable:
    """A callable whose __signature__ raises -- inspect.signature propagates that."""

    def __call__(self, x):
        return x

    @property
    def __signature__(self):
        raise ValueError("no signature available")


class TestCheckCallableResolvable:
    def test_plain_function_is_resolvable(self) -> None:
        assert _check_callable_resolvable(_clean_fn) is None

    def test_unintrospectable_callable_fails_with_message(self) -> None:
        error = _check_callable_resolvable(_UnintrospectableCallable())
        assert error is not None
        assert "not introspectable" in error


class TestValidateGraphAllPass:
    def test_clean_graph_passes_and_writes_pass_verdict(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            result = validate_graph(graph, root=Path.cwd())

        assert result.passed is True
        assert result.failures == ()
        assert result.graph.nodes[0].metadata["audit_verdict"] == PASS


class TestValidateGraphEvalShapeFailure:
    def test_unintrospectable_callable_gets_fail_verdict(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(
                id="bad-node",
                callable_ref=_UnintrospectableCallable(),
                metadata=VALID_METADATA,
            )
            graph = HostPrepGraph(nodes=(node,))
            result = validate_graph(graph, root=Path.cwd())

        assert result.passed is False
        assert result.graph.nodes[0].metadata["audit_verdict"] == FAIL
        assert len(result.failures) == 1
        assert result.failures[0].node_id == "bad-node"
        assert result.failures[0].check == "eval_shape_consistency_proxy"


class TestValidateGraphJaxlintFailure:
    def test_jl_error_finding_gets_needs_work_not_fail(self) -> None:
        clean_fn_line = inspect.getsourcelines(_clean_fn)[1]
        mock_findings = [
            {
                "rule_id": "JL001",
                "severity": "error",
                "message": "side effect in jit",
                "path": "tests/composition/test_validate.py",
                "line": clean_fn_line,
            }
        ]
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=mock_findings):
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            result = validate_graph(graph, root=Path.cwd())

        assert result.passed is False
        assert result.graph.nodes[0].metadata["audit_verdict"] == NEEDS_WORK
        assert len(result.failures) == 1
        assert result.failures[0].check == "jaxlint"

    def test_non_error_severity_finding_is_ignored(self) -> None:
        mock_findings = [
            {"rule_id": "JL002", "severity": "warning", "message": "minor style"},
        ]
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=mock_findings):
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            result = validate_graph(graph, root=Path.cwd())

        assert result.passed is True
        assert result.graph.nodes[0].metadata["audit_verdict"] == PASS

    def test_jaxlint_results_cached_per_source_file(self) -> None:
        """Two nodes sharing a source file (this test module) trigger only one subprocess call."""
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]) as mock_run:
            n1 = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            n2 = HostPrepGraphNode(id="n2", callable_ref=_other_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(n1, n2))
            validate_graph(graph, root=Path.cwd())

        assert mock_run.call_count == 1

    def test_finding_attributed_only_to_owning_node_not_every_node_in_shared_file(self) -> None:
        """A single JL-error finding scoped to _clean_fn's lines must NOT downgrade
        _other_clean_fn's node just because both live in this test module.
        """
        clean_fn_line = inspect.getsourcelines(_clean_fn)[1]
        mock_findings = [
            {
                "rule_id": "JL001",
                "severity": "error",
                "message": "in _clean_fn",
                "line": clean_fn_line,
            }
        ]
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=mock_findings):
            n1 = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            n2 = HostPrepGraphNode(id="n2", callable_ref=_other_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(n1, n2))
            result = validate_graph(graph, root=Path.cwd())

        assert result.graph.nodes[0].metadata["audit_verdict"] == NEEDS_WORK
        assert result.graph.nodes[1].metadata["audit_verdict"] == PASS
        assert len(result.failures) == 1
        assert result.failures[0].node_id == "n1"

    def test_finding_with_no_line_info_conservatively_attributed_to_all_sharing_nodes(self) -> None:
        """A finding lacking a usable `line` can't be attributed precisely -- kept for every
        node in the shared file rather than silently dropped.
        """
        mock_findings = [{"rule_id": "JL001", "severity": "error", "message": "no line info"}]
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=mock_findings):
            n1 = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            n2 = HostPrepGraphNode(id="n2", callable_ref=_other_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(n1, n2))
            result = validate_graph(graph, root=Path.cwd())

        assert result.graph.nodes[0].metadata["audit_verdict"] == NEEDS_WORK
        assert result.graph.nodes[1].metadata["audit_verdict"] == NEEDS_WORK


class TestValidateGraphTopologyFailureFansOutToEveryNode:
    def test_topology_failure_marks_every_node_fail_with_one_graph_level_record(self) -> None:
        with (
            patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]),
            patch(
                "xtrax.composition.validate.validate_plan_topology",
                side_effect=PlanTopologyError("synthetic topology violation"),
            ),
        ):
            n1 = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            n2 = HostPrepGraphNode(id="n2", callable_ref=_other_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(n1, n2))
            result = validate_graph(graph, root=Path.cwd())

        assert result.passed is False
        assert result.graph.nodes[0].metadata["audit_verdict"] == FAIL
        assert result.graph.nodes[1].metadata["audit_verdict"] == FAIL
        topology_failures = [f for f in result.failures if f.check == "plan_topology"]
        assert len(topology_failures) == 1
        assert topology_failures[0].node_id == ""


class TestValidatePlanTopologyIsReallyCalled:
    def test_validate_plan_topology_invoked_with_empty_inputs(self) -> None:
        with (
            patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]),
            patch(
                "xtrax.composition.validate.validate_plan_topology",
                wraps=validate_plan_topology,
            ) as spy,
        ):
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            validate_graph(graph, root=Path.cwd())

        spy.assert_called_once_with(decisions=(), axis_boundaries={})


class TestFrozenNodeWriteBack:
    def test_frozen_node_gets_verdict_without_mutating_original(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(
                id="frozen-node", callable_ref=_clean_fn, metadata=VALID_METADATA, frozen=True
            )
            graph = HostPrepGraph(nodes=(node,))
            result = validate_graph(graph, root=Path.cwd())

        assert "audit_verdict" not in node.metadata
        assert result.graph.nodes[0].metadata["audit_verdict"] == PASS
        assert result.graph.nodes[0].frozen is True
        assert result.graph.nodes[0] is not node


class TestValidateGraphPreservesEdges:
    def test_edges_carried_through_to_rebuilt_graph(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            n1 = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            n2 = HostPrepGraphNode(id="n2", callable_ref=_other_clean_fn, metadata=VALID_METADATA)
            edges = (GraphEdge(src="n1", dst="n2"),)
            graph = HostPrepGraph(nodes=(n1, n2), edges=edges)
            result = validate_graph(graph, root=Path.cwd())

        assert result.graph.edges == edges

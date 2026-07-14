"""Tests for the #2181 loop admission gate (T2-01, #3069, AC-E1)."""

from pathlib import Path
from unittest.mock import patch

from xtrax.composition.graph import HostPrepGraph, HostPrepGraphNode
from xtrax.composition.validate import validate_graph
from xtrax.loop.admission import admit_candidate

VALID_METADATA = {"nl_description": "A node under test."}


def _clean_fn(x):
    return x


class _UnintrospectableCallable:
    """A callable whose __signature__ raises -- inspect.signature propagates that."""

    def __call__(self, x):
        return x

    @property
    def __signature__(self):
        raise ValueError("no signature available")


class TestAdmitCandidate:
    def test_clean_graph_is_admitted(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            result = admit_candidate(graph, root=Path.cwd())

        assert result.passed is True
        assert result.failures == ()

    def test_uninspectable_callable_is_refused_with_named_check(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(
                id="bad-node", callable_ref=_UnintrospectableCallable(), metadata=VALID_METADATA
            )
            graph = HostPrepGraph(nodes=(node,))
            result = admit_candidate(graph, root=Path.cwd())

        assert result.passed is False
        assert [f.check for f in result.failures] == ["eval_shape_consistency_proxy"]

    def test_admit_candidate_is_byte_identical_to_validate_graph(self) -> None:
        """The anti-vacuous-wrapper test: a hardcoded passed=True stub, or any divergence
        between the wrapper and the real gate, fails here immediately.
        """
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(
                id="bad-node", callable_ref=_UnintrospectableCallable(), metadata=VALID_METADATA
            )
            graph_a = HostPrepGraph(nodes=(node,))
            graph_b = HostPrepGraph(nodes=(node,))
            direct = validate_graph(graph_a, root=Path.cwd())
            via_wrapper = admit_candidate(graph_b, root=Path.cwd())

        assert via_wrapper.passed == direct.passed
        assert via_wrapper.failures == direct.failures
        assert (
            via_wrapper.graph.nodes[0].metadata["audit_verdict"]
            == direct.graph.nodes[0].metadata["audit_verdict"]
        )

    def test_admission_writes_audit_verdict_into_returned_graph(self) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            result = admit_candidate(graph, root=Path.cwd())

        assert result.graph.nodes[0].metadata["audit_verdict"] == "PASS"

    def test_root_kwarg_is_forwarded(self, tmp_path: Path) -> None:
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]) as mock_run:
            node = HostPrepGraphNode(id="n1", callable_ref=_clean_fn, metadata=VALID_METADATA)
            graph = HostPrepGraph(nodes=(node,))
            admit_candidate(graph, root=tmp_path)

        assert mock_run.call_args.kwargs["root"] == tmp_path

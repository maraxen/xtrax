"""Tests for T1-12's default free-generation authoring path (#3067)."""

from pathlib import Path

import pytest

from xtrax.composition.author import TemplateGenerator
from xtrax.composition.serialize import serialize_graph
from xtrax.composition.validate import validate_graph

ROOT = Path(__file__).resolve().parents[2]


class TestTemplateGeneratorDeterminism:
    def test_same_seed_produces_identical_graph(self) -> None:
        graph_a = TemplateGenerator().generate(seed=42, num_nodes=4)
        graph_b = TemplateGenerator().generate(seed=42, num_nodes=4)
        assert serialize_graph(graph_a) == serialize_graph(graph_b)

    def test_different_seeds_can_produce_different_graphs(self) -> None:
        """Not a hard guarantee for every pair of seeds, but seeds 0 and 1 across a 4-callable
        pool should not collide -- proves generation is actually seed-driven, not a constant.
        """
        graph_a = TemplateGenerator().generate(seed=0, num_nodes=8)
        graph_b = TemplateGenerator().generate(seed=1, num_nodes=8)
        assert serialize_graph(graph_a) != serialize_graph(graph_b)


class TestTemplateGeneratorShape:
    def test_respects_num_nodes(self) -> None:
        graph = TemplateGenerator().generate(seed=0, num_nodes=5)
        assert len(graph.nodes) == 5

    def test_single_node_has_no_edges(self) -> None:
        graph = TemplateGenerator().generate(seed=0, num_nodes=1)
        assert graph.edges == ()

    def test_edges_form_a_linear_chain(self) -> None:
        graph = TemplateGenerator().generate(seed=0, num_nodes=4)
        assert [(e.src, e.dst) for e in graph.edges] == [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
        ]

    def test_node_ids_are_unique(self) -> None:
        graph = TemplateGenerator().generate(seed=0, num_nodes=6)
        ids = [node.id for node in graph.nodes]
        assert len(ids) == len(set(ids))

    def test_rejects_num_nodes_less_than_one(self) -> None:
        with pytest.raises(ValueError, match="num_nodes"):
            TemplateGenerator().generate(seed=0, num_nodes=0)

    def test_rejects_non_int_num_nodes(self) -> None:
        """A non-int num_nodes (e.g. a str from a misused direct caller) must raise the same
        clean ValueError as num_nodes<1, not a raw TypeError from the internal comparison.
        """
        with pytest.raises(ValueError, match="num_nodes"):
            TemplateGenerator().generate(seed=0, num_nodes="3")  # type: ignore[arg-type]

    def test_rejects_bool_num_nodes(self) -> None:
        """bool is a subclass of int in Python -- num_nodes=True must not silently pass as 1."""
        with pytest.raises(ValueError, match="num_nodes"):
            TemplateGenerator().generate(seed=0, num_nodes=True)  # type: ignore[arg-type]


class TestGeneratedGraphPassesValidation:
    """The actual AC success metric: 'generate-then-validate authors >=1 graph passing
    graph validate.' Parametrized across many seeds (not just one lucky seed) since a
    template generator that only happens to pass validation for seed=0 wouldn't be a
    reliable default authoring path.
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_passes_validate_graph(self, seed: int) -> None:
        graph = TemplateGenerator().generate(seed=seed, num_nodes=3)
        result = validate_graph(graph, root=ROOT)
        assert result.passed is True
        assert result.failures == ()

    def test_every_node_gets_audit_verdict_pass(self) -> None:
        graph = TemplateGenerator().generate(seed=7, num_nodes=3)
        result = validate_graph(graph, root=ROOT)
        for node in result.graph.nodes:
            assert node.metadata["audit_verdict"] == "PASS"

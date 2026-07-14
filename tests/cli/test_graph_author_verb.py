"""Tests for the `graph-author` CLI verb (T1-12, #3067): the default generate-then-validate
authoring front-end.
"""

import json
from pathlib import Path

import pytest

from xtrax.cli.graph_author_verb import (
    AUTHOR_ENVELOPE_SCHEMA_VERSION,
    GraphAuthorArgs,
    run_graph_author,
)
from xtrax.cli.registry import REGISTRY
from xtrax.composition.serialize import load_graph


class TestGraphAuthorInRegistry:
    def test_graph_author_registered(self) -> None:
        assert "graph-author" in REGISTRY
        args_cls, run_fn = REGISTRY["graph-author"]
        assert args_cls is GraphAuthorArgs
        assert run_fn is run_graph_author


class TestRunGraphAuthor:
    def test_authors_a_graph_that_passes_validation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_path = tmp_path / "candidate.json"

        run_graph_author(GraphAuthorArgs(out_path=str(out_path), seed=0, num_nodes=3))

        envelope = json.loads(capsys.readouterr().out)
        assert envelope == {
            "schema_version": AUTHOR_ENVELOPE_SCHEMA_VERSION,
            "failure_count": 0,
            "failures": [],
        }

        assert out_path.exists()
        graph = load_graph(out_path)
        assert len(graph.nodes) == 3
        for node in graph.nodes:
            assert node.metadata["audit_verdict"] == "PASS"

    def test_default_seed_and_num_nodes_succeed(self, tmp_path: Path) -> None:
        out_path = tmp_path / "candidate.json"
        run_graph_author(GraphAuthorArgs(out_path=str(out_path)))
        assert out_path.exists()

    def test_rejects_num_nodes_less_than_one_with_clean_message(self, tmp_path: Path) -> None:
        out_path = tmp_path / "candidate.json"

        with pytest.raises(SystemExit) as exc_info:
            run_graph_author(GraphAuthorArgs(out_path=str(out_path), seed=0, num_nodes=0))

        assert "num_nodes" in str(exc_info.value)
        assert not out_path.exists()

    def test_rejects_non_int_num_nodes_with_clean_message_not_raw_traceback(
        self, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "candidate.json"

        with pytest.raises(SystemExit) as exc_info:
            run_graph_author(GraphAuthorArgs(out_path=str(out_path), seed=0, num_nodes="3"))  # type: ignore[arg-type]

        assert "num_nodes" in str(exc_info.value)
        assert not out_path.exists()

    def test_unwritable_out_path_raises_clean_system_exit_not_raw_traceback(
        self, tmp_path: Path
    ) -> None:
        """A missing parent directory is a realistic first-run mistake for graph-author
        specifically -- unlike graph-validate/graph-plan, which only ever write back to a path
        that was just successfully read, graph-author is the first verb writing to a brand-new
        user-supplied out_path.
        """
        out_path = tmp_path / "does-not-exist" / "candidate.json"

        with pytest.raises(SystemExit) as exc_info:
            run_graph_author(GraphAuthorArgs(out_path=str(out_path), seed=0, num_nodes=3))

        assert str(out_path) in str(exc_info.value)

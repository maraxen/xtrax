"""End-to-end tests for the `graph-validate` CLI verb (T1-10, #2181 entry-edge item)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.cli.graph_verb import (
    VALIDATE_ENVELOPE_SCHEMA_VERSION,
    GraphValidateArgs,
    run_graph_validate,
)
from xtrax.cli.registry import REGISTRY
from xtrax.composition.graph import HostPrepGraph, HostPrepGraphNode
from xtrax.composition.serialize import dump_graph

VALID_METADATA = {"nl_description": "A node under test."}


def _reference_fn(x):
    return x


class TestGraphValidateInRegistry:
    def test_graph_validate_registered(self) -> None:
        assert "graph-validate" in REGISTRY
        args_cls, run_fn = REGISTRY["graph-validate"]
        assert args_cls is GraphValidateArgs
        assert run_fn is run_graph_validate


class TestRunGraphValidateEndToEnd:
    def test_clean_graph_exits_zero_and_writes_pass_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        graph = HostPrepGraph(
            nodes=(HostPrepGraphNode(id="n1", callable_ref=_reference_fn, metadata=VALID_METADATA),)
        )
        ir_path = tmp_path / "ir.json"
        dump_graph(graph, ir_path)

        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=[]):
            run_graph_validate(GraphValidateArgs(ir_path=str(ir_path)))

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["schema_version"] == VALIDATE_ENVELOPE_SCHEMA_VERSION
        assert envelope["failure_count"] == 0
        assert envelope["failures"] == []

        on_disk = json.loads(ir_path.read_text(encoding="utf-8"))
        assert on_disk["nodes"][0]["metadata"]["audit_verdict"] == "PASS"

    def test_failing_node_exits_one_with_named_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        graph = HostPrepGraph(
            nodes=(HostPrepGraphNode(id="n1", callable_ref=_reference_fn, metadata=VALID_METADATA),)
        )
        ir_path = tmp_path / "ir.json"
        dump_graph(graph, ir_path)

        mock_findings = [
            {"rule_id": "JL001", "severity": "error", "message": "side effect in jit"},
        ]
        with patch("xtrax.composition.validate._run_jaxlint_json", return_value=mock_findings):
            with pytest.raises(SystemExit) as exc_info:
                run_graph_validate(GraphValidateArgs(ir_path=str(ir_path)))

        assert exc_info.value.code == 1
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["failure_count"] == 1
        assert envelope["failures"][0]["node_id"] == "n1"
        assert envelope["failures"][0]["check"] == "jaxlint"

        on_disk = json.loads(ir_path.read_text(encoding="utf-8"))
        assert on_disk["nodes"][0]["metadata"]["audit_verdict"] == "NEEDS_WORK"

    def test_malformed_document_raises_clean_system_exit(self, tmp_path: Path) -> None:
        ir_path = tmp_path / "ir.json"
        ir_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")  # missing schema_version

        with pytest.raises(SystemExit) as exc_info:
            run_graph_validate(GraphValidateArgs(ir_path=str(ir_path)))

        assert "schema_version" in str(exc_info.value)

    def test_missing_file_raises_clean_system_exit(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does-not-exist.json"

        with pytest.raises(SystemExit):
            run_graph_validate(GraphValidateArgs(ir_path=str(missing_path)))

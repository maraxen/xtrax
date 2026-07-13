"""Tests for the `graph-plan` CLI verb, including T1-11 AC1's graph->plan parity capstone
(#2181 entry-edge item): a graph-sourced plan must be byte/structure-identical to a plan
produced by calling infer_bundle/BatchPlanner directly on the same callable + shapes.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.cli.errors import CLIError
from xtrax.cli.graph_plan_verb import GraphPlanArgs, run_graph_plan
from xtrax.cli.registry import REGISTRY
from xtrax.cli.shapes import parse_shapes
from xtrax.composition.graph import HostPrepGraph, HostPrepGraphNode
from xtrax.composition.serialize import dump_graph
from xtrax.inference.api import infer_bundle
from xtrax.inference.config import AxisOverride, axis_config
from xtrax.tiling.plan import BatchPlanner

VALID_METADATA = {"nl_description": "A node under test."}


@axis_config(AxisOverride(name="batch", default_batch_size=2))
def _traced_fn(x):
    return x * 2


@axis_config(AxisOverride(name="other_batch", default_batch_size=5))
def _other_node_fn(y):
    return y + 1


@axis_config(
    AxisOverride(name="a_axis", default_batch_size=2),
    AxisOverride(name="b_axis", default_batch_size=2),
)
def _multi_input_fn(a, b):
    return a + b[..., :1]


@contextmanager
def _capture_batch_plan():
    """Patch BatchPlanner.plan with a plain function, not a Mock -- Mock objects aren't
    descriptors, so `patch.object(BatchPlanner, "plan", wraps=BatchPlanner.plan)` would call
    the unbound original without `self` the moment it's invoked as `instance.plan(...)`. A
    plain function IS a descriptor (normal Python method-binding), so this correctly captures
    the real BatchPlan the CLI path produces without altering its behavior.
    """
    captured = []
    original_plan = BatchPlanner.plan

    def _capturing_plan(self, specs):
        result = original_plan(self, specs)
        captured.append(result)
        return result

    with patch.object(BatchPlanner, "plan", _capturing_plan):
        yield captured


class TestGraphPlanInRegistry:
    def test_graph_plan_registered(self) -> None:
        assert "graph-plan" in REGISTRY
        args_cls, run_fn = REGISTRY["graph-plan"]
        assert args_cls is GraphPlanArgs
        assert run_fn is run_graph_plan


class TestRunGraphPlanErrors:
    def test_node_not_found_raises_cli_error_naming_the_id(self, tmp_path: Path) -> None:
        graph = HostPrepGraph(
            nodes=(HostPrepGraphNode(id="n1", callable_ref=_traced_fn, metadata=VALID_METADATA),)
        )
        ir_path = tmp_path / "ir.json"
        dump_graph(graph, ir_path)

        with pytest.raises(CLIError) as exc_info:
            run_graph_plan(
                GraphPlanArgs(ir_path=str(ir_path), node_id="missing", shapes="x=(4,)f32")
            )

        assert "missing" in str(exc_info.value)
        assert "n1" in str(exc_info.value)

    def test_dangling_edge_raises_clean_system_exit_not_raw_traceback(self, tmp_path: Path) -> None:
        doc = {
            "schema_version": 1,
            "nodes": [],
            "edges": [{"src": "missing", "dst": "also-missing"}],
        }
        ir_path = tmp_path / "ir.json"
        ir_path.write_text(json.dumps(doc), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            run_graph_plan(GraphPlanArgs(ir_path=str(ir_path), node_id="n1", shapes=""))

        assert "unknown node id" in str(exc_info.value)


class TestAC1GraphPlanParity:
    """T1-11's capstone: graph -> serialize -> CLI-consume -> BatchPlan must be exactly
    identical to calling infer_bundle/BatchPlanner directly on the same callable + shapes.
    """

    def _build_two_node_graph(self, tmp_path: Path) -> Path:
        graph = HostPrepGraph(
            nodes=(
                HostPrepGraphNode(id="traced", callable_ref=_traced_fn, metadata=VALID_METADATA),
                HostPrepGraphNode(id="other", callable_ref=_other_node_fn, metadata=VALID_METADATA),
            )
        )
        ir_path = tmp_path / "ir.json"
        dump_graph(graph, ir_path)
        return ir_path

    def test_single_input_plan_is_byte_identical(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ir_path = self._build_two_node_graph(tmp_path)
        shapes = "x=(4,)f32"

        # Path A: ground truth, direct infer_bundle + BatchPlanner -- same parse_shapes
        # call plan_from_fn uses internally, so the only variable is fn provenance.
        abstract_inputs = list(parse_shapes(shapes).values())
        _schema, axes = infer_bundle(_traced_fn, abstract_inputs)
        plan_direct = BatchPlanner().plan(axes)

        # Path B: via the CLI, sourcing the callable from the serialized graph.
        with _capture_batch_plan() as captured:
            run_graph_plan(GraphPlanArgs(ir_path=str(ir_path), node_id="traced", shapes=shapes))
        plan_via_cli = captured[-1]

        assert plan_via_cli == plan_direct
        assert plan_via_cli is not plan_direct
        assert len(capsys.readouterr().out) > 0

    def test_node_id_selects_the_correct_node_not_just_the_first(self, tmp_path: Path) -> None:
        """The 2-node graph exists specifically to prove node-id selection is real, not a
        coincidence: request the SECOND node explicitly and confirm the CLI plans it (not the
        first node it happens to share a graph with). `_other_node_fn`'s default_batch_size=5
        (vs `_traced_fn`'s 2) makes their plans structurally distinct -- cardinality 4 <=
        batch_size 5 selects Vmap for "other", while cardinality 4 > batch_size 2 selects
        SafeMap for "traced" -- so a bug that always plans the first node would produce a
        strategy-type mismatch here, not just a coincidentally-equal plan.
        """
        ir_path = self._build_two_node_graph(tmp_path)
        shapes = "y=(4,)f32"

        abstract_inputs = list(parse_shapes(shapes).values())
        _schema, other_axes = infer_bundle(_other_node_fn, abstract_inputs)
        plan_direct_other = BatchPlanner().plan(other_axes)
        _schema, traced_axes = infer_bundle(_traced_fn, abstract_inputs)
        plan_direct_traced = BatchPlanner().plan(traced_axes)

        with _capture_batch_plan() as captured:
            run_graph_plan(GraphPlanArgs(ir_path=str(ir_path), node_id="other", shapes=shapes))
        plan_via_cli = captured[-1]

        assert plan_via_cli == plan_direct_other
        assert plan_via_cli != plan_direct_traced
        assert type(plan_via_cli.decisions[0].strategy) is not type(
            plan_direct_traced.decisions[0].strategy
        )

    def test_multi_input_plan_is_byte_identical(self, tmp_path: Path) -> None:
        """Non-degenerate case: two distinct input arrays, proving parity isn't an artifact
        of a single-scalar-input edge case.
        """
        graph = HostPrepGraph(
            nodes=(
                HostPrepGraphNode(
                    id="traced", callable_ref=_multi_input_fn, metadata=VALID_METADATA
                ),
            )
        )
        ir_path = tmp_path / "multi_ir.json"
        dump_graph(graph, ir_path)
        shapes = "a=(4,3)f32 b=(4,5)f32"

        abstract_inputs = list(parse_shapes(shapes).values())
        _schema, axes = infer_bundle(_multi_input_fn, abstract_inputs)
        plan_direct = BatchPlanner().plan(axes)

        with _capture_batch_plan() as captured:
            run_graph_plan(GraphPlanArgs(ir_path=str(ir_path), node_id="traced", shapes=shapes))
        plan_via_cli = captured[-1]

        assert plan_via_cli == plan_direct
        assert plan_via_cli is not plan_direct

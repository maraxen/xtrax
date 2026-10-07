"""The ``onnx`` export target: gate routing, conversion, and exact-integer parity.

Two halves. ``TestWithoutToolchain`` needs nothing beyond the base install and
runs everywhere, including the CI job with no export extra. Every other class
importorskips the ``onnx`` extra; export-toolchain-tests installs it and fails on
any skip, so there they run for real.

Oracles for index-producing ops are computed with NumPy, never with JAX: a JAX
oracle compared against a jax2onnx graph of the same JAX function would share
any tracing-level mistake. The tie-rich fixtures are checked to be tie-rich --
a descending-index tiebreak must change each answer -- so an "exact" result
cannot come from the fixture having no ties to break.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export import onnx as onnx_mod
from xtrax.export.compile import CompileError, compile_for_target
from xtrax.export.onnx import LeafParityResult, convert_to_onnx, verify_onnx_parity
from xtrax.export.parity import compare
from xtrax.export.pipeline import export_pipeline
from xtrax.export.safety import (
    UnsupportedOperationError,
    check_export_safety,
    validate_export_safe,
)
from xtrax.export.targets import NATIVE, ONNX
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

N = 64
TOP_K = 8


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _vmap_plan(rows: int) -> _Plan:
    return _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name="batch", cardinality=rows, default_batch_size=0),
                batch_size=0,
                reasoning="onnx test",
                strategy=Vmap(),
            )
        ]
    )


def _tie_rich_keys(rows: int = 4, seed: int = 0) -> np.ndarray:
    """Four distinct int32 values over N slots per row: ties everywhere."""
    return np.random.default_rng(seed).integers(0, 4, size=(rows, N)).astype(np.int32)


def _tie_rich_vals(rows: int = 4, seed: int = 1) -> np.ndarray:
    """Eight distinct float32 values over N slots: every top-k slot is tied."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 8, size=(rows, N)) / 8.0).astype(np.float32)


def _sort_2key(k):
    return jax.lax.sort((k, jnp.arange(N, dtype=jnp.int32)), num_keys=2, is_stable=False)[1]


def _stable_argsort(k):
    return jnp.argsort(k, stable=True).astype(jnp.int32)


def _top_k(v):
    vals, idx = jax.lax.top_k(v, TOP_K)
    return vals, idx.astype(jnp.int32)


def _uniform_from_key(key):
    return jax.random.uniform(key, (4,))


# --------------------------------------------------------------------------- #
# Base install: no jax2onnx, onnx or onnxruntime needed.
# --------------------------------------------------------------------------- #


class TestWithoutToolchain:
    def _safety(self, fn, spec, target):
        return check_export_safety(_vmap_plan(4).decisions, {}, (spec,), fn, target)

    def test_iree_rules_do_not_apply_to_onnx(self):
        """Stable sort and lax.top_k are IREE blockers, measured exact on ORT."""
        spec = jax.ShapeDtypeStruct((N,), jnp.int32)
        fspec = jax.ShapeDtypeStruct((N,), jnp.float32)
        assert self._safety(_stable_argsort, spec, ONNX) == []
        assert self._safety(_top_k, fspec, ONNX) == []

    def test_iree_rules_still_apply_to_iree_targets(self):
        """The control: the same programs ARE blocked for native."""
        spec = jax.ShapeDtypeStruct((N,), jnp.int32)
        fspec = jax.ShapeDtypeStruct((N,), jnp.float32)
        assert {b.rule for b in self._safety(_stable_argsort, spec, NATIVE)} == {"sort-stability"}
        assert {b.rule for b in self._safety(_top_k, fspec, NATIVE)} == {"unlegalizable-op"}

    def test_in_graph_rng_from_an_input_key_is_blocked_for_onnx(self):
        spec = jax.ShapeDtypeStruct((2,), jnp.uint32)
        rules = {b.rule for b in self._safety(_uniform_from_key, spec, ONNX)}
        assert rules == {"onnx-in-graph-rng"}

    def test_in_graph_rng_from_a_constant_key_is_blocked_too(self):
        """A baked-in key still draws inside the graph; ONNX still cannot."""
        key = jax.random.PRNGKey(0)

        def fn(x):
            return x + jax.random.uniform(key, x.shape)

        spec = jax.ShapeDtypeStruct((4,), jnp.float32)
        assert "onnx-in-graph-rng" in {b.rule for b in self._safety(fn, spec, ONNX)}

    def test_in_graph_rng_cannot_be_acknowledged_away(self):
        spec = jax.ShapeDtypeStruct((2,), jnp.uint32)
        blockers = check_export_safety(
            _vmap_plan(4).decisions,
            {},
            (spec,),
            _uniform_from_key,
            ONNX,
            acknowledged=frozenset({"onnx-in-graph-rng"}),
        )
        assert [b.rule for b in blockers] == ["onnx-in-graph-rng"]

    def test_validate_raises_on_in_graph_rng(self):
        spec = jax.ShapeDtypeStruct((2,), jnp.uint32)
        with pytest.raises(UnsupportedOperationError, match="onnx-in-graph-rng"):
            validate_export_safe(_vmap_plan(4).decisions, {}, (spec,), _uniform_from_key, ONNX)

    def test_rng_free_program_has_no_onnx_blockers(self):
        spec = jax.ShapeDtypeStruct((4,), jnp.float32)
        assert self._safety(jnp.tanh, spec, ONNX) == []

    def test_convert_accepts_a_model_name_and_an_embed_flag(self):
        import inspect

        params = inspect.signature(convert_to_onnx).parameters
        assert params["model_name"].default == "xtrax_export"
        assert params["embed_external_data"].default is False

    def test_rng_audit_descends_into_function_subgraphs_and_censuses_domains(self):
        """A RandomUniform nested in a function's Loop body is not on function.node."""

        class _Attr:
            def __init__(self, graph):
                self._graph = graph
                self.graphs = ()

            def HasField(self, name: str) -> bool:
                return name == "g" and self._graph is not None

            @property
            def g(self):
                return self._graph

        class _Node:
            def __init__(self, op_type, name, domain="", attributes=()):
                self.op_type = op_type
                self.name = name
                self.domain = domain
                self.attribute = attributes

        class _Graph:
            def __init__(self, nodes):
                self.node = nodes

        class _Model:
            def __init__(self, graph, functions):
                self.graph = graph
                self.functions = functions

        body = _Graph(
            [
                _Node("Identity", "std", domain="ai.onnx"),
                _Node("RandomUniform", "draw"),
                _Node("CustomOp", "custom", domain="com.example"),
            ]
        )
        function = _Graph([_Node("Loop", "loop", attributes=(_Attr(body),))])
        model = _Model(_Graph([_Node("Identity", "top")]), (function,))
        assert onnx_mod.find_onnx_rng_ops(model) == ["RandomUniform:draw"]
        assert onnx_mod.onnx_unknown_domain_census(model) == {"com.example": 1}

    def test_convert_refuses_a_non_onnx_target(self):
        spec = jax.ShapeDtypeStruct((4,), jnp.float32)
        with pytest.raises(ValueError, match="onnx-backend target"):
            convert_to_onnx(jnp.tanh, (spec,), NATIVE)

    def test_convert_refuses_x64(self):
        spec = jax.ShapeDtypeStruct((4,), jnp.float32)
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            with pytest.raises(CompileError, match="jax_enable_x64"):
                convert_to_onnx(jnp.tanh, (spec,), ONNX)
        finally:
            jax.config.update("jax_enable_x64", previous)

    @pytest.mark.parametrize(
        ("module", "require"),
        [
            ("jax2onnx", "_require_jax2onnx"),
            ("onnxruntime", "_require_ort"),
            ("onnx", "_require_onnx"),
        ],
    )
    def test_missing_toolchain_names_the_extra(self, monkeypatch, module, require):
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(CompileError, match=r"xtrax\[onnx\]"):
            getattr(onnx_mod, require)()

    def test_compile_for_target_refuses_the_onnx_target(self):
        """The IREE entry point must not pass ONNX an empty backend flag."""
        with pytest.raises(ValueError, match="IREE-backend target"):
            compile_for_target("module {}", ONNX)

    def test_gates_run_before_the_oracle(self):
        """A rejected export fails fast and never pays for (or is masked by) the oracle.

        The RNG program here is shape-agnostic on purpose: the op gate traces
        ``fn`` with the batched abstract inputs and swallows trace failures, so a
        program that only traces per-element would not reach the gate at all.
        """
        calls: list[object] = []
        key = jax.random.PRNGKey(0)

        def fn(x):
            return x + jax.random.uniform(key, x.shape)

        def reference(inputs):
            calls.append(inputs)
            msg = "the oracle must not run for a gated-out export"
            raise AssertionError(msg)

        xs = np.zeros((4, 3), np.float32)
        with pytest.raises(UnsupportedOperationError, match="onnx-in-graph-rng"):
            export_pipeline(
                fn,
                _vmap_plan(4),
                (jax.ShapeDtypeStruct(xs.shape, xs.dtype),),
                (xs,),
                targets=(ONNX,),
                reference_fn=reference,
            )
        assert calls == []

    def test_namespace_restoration_undoes_a_replacement(self):
        """The guard around to_onnx restores what a conversion replaces."""
        original = jnp.cumsum
        with onnx_mod._restoring_jax_namespaces():
            jnp.cumsum = lambda *a, **k: None  # the shape of jax2onnx's leaked patch
        assert jnp.cumsum is original

    def test_namespace_restoration_is_a_no_op_when_nothing_changed(self):
        before = dict(vars(jnp))
        with onnx_mod._restoring_jax_namespaces():
            pass
        assert all(vars(jnp)[k] is v for k, v in before.items())


class TestParityRulesWithAFakeRuntime:
    """verify_onnx_parity's comparison rules, driven without ORT."""

    @pytest.fixture
    def fake_outputs(self, monkeypatch):
        holder: dict = {"outputs": []}
        monkeypatch.setattr(onnx_mod, "run_onnx", lambda *_a, **_k: holder["outputs"])
        return holder

    def test_large_integers_are_compared_exactly(self, fake_outputs):
        """The negative control: allclose would call these equal (rtol 1e-5)."""
        expected = np.array([1_000_000], dtype=np.int32)
        fake_outputs["outputs"] = [np.array([1_000_009], dtype=np.int32)]
        assert np.allclose(expected, fake_outputs["outputs"][0], atol=1e-5, rtol=1e-5)
        result = verify_onnx_parity(expected, Path("unused.onnx"), ())
        assert not result.passed
        assert result.max_abs_diff == 9.0

    def test_an_index_swap_fails(self, fake_outputs):
        expected = np.arange(8, dtype=np.int32)
        swapped = expected.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        fake_outputs["outputs"] = [swapped]
        assert not verify_onnx_parity(expected, Path("unused.onnx"), ()).passed

    def test_identical_integers_pass(self, fake_outputs):
        expected = np.arange(8, dtype=np.int32)
        fake_outputs["outputs"] = [expected.copy()]
        assert verify_onnx_parity(expected, Path("unused.onnx"), ()).passed

    def test_a_widened_dtype_fails_even_with_equal_values(self, fake_outputs):
        expected = np.arange(4, dtype=np.int32)
        fake_outputs["outputs"] = [expected.astype(np.int64)]
        result = verify_onnx_parity(expected, Path("unused.onnx"), ())
        assert not result.passed
        assert result.dtype_mismatches == ("leaf 0: expected int32, got int64",)

    def test_a_leaf_count_mismatch_fails(self, fake_outputs):
        fake_outputs["outputs"] = [np.zeros(2, np.float32), np.zeros(2, np.float32)]
        result = verify_onnx_parity(np.zeros(2, np.float32), Path("unused.onnx"), ())
        assert not result.passed
        assert "expected 1 output leaves, got 2" in result.dtype_mismatches[0]

    def test_floats_use_the_tolerance(self, fake_outputs):
        expected = np.ones(4, np.float32)
        fake_outputs["outputs"] = [expected + np.float32(1e-7)]
        assert verify_onnx_parity(expected, Path("unused.onnx"), ()).passed
        fake_outputs["outputs"] = [expected + np.float32(1e-3)]
        assert not verify_onnx_parity(expected, Path("unused.onnx"), ()).passed

    def test_a_numpy_float64_oracle_is_narrowed_like_compare(self, fake_outputs):
        """One NumPy reference_fn must verify alike on native and onnx."""
        expected = np.linspace(0.0, 1.0, 4)  # float64, as NumPy produces
        fake_outputs["outputs"] = [expected.astype(np.float32)]
        assert compare(expected, fake_outputs["outputs"][0]).passed
        assert verify_onnx_parity(expected, Path("unused.onnx"), ()).passed

    def test_a_numpy_int64_oracle_is_narrowed(self, fake_outputs):
        expected = np.argsort(np.array([3, 1, 2, 0]))  # int64 from NumPy
        fake_outputs["outputs"] = [expected.astype(np.int32)]
        assert verify_onnx_parity(expected, Path("unused.onnx"), ()).passed

    def test_summary_names_a_dtype_change(self, fake_outputs):
        fake_outputs["outputs"] = [np.arange(4, dtype=np.int64)]
        text = verify_onnx_parity(np.arange(4, dtype=np.int32), Path("unused.onnx"), ()).summary()
        assert text == "FAIL: leaf 0: expected int32, got int64"

    def test_summary_names_a_leaf_count_mismatch(self, fake_outputs):
        fake_outputs["outputs"] = [np.zeros(2, np.float32)] * 2
        text = verify_onnx_parity(np.zeros(2, np.float32), Path("unused.onnx"), ()).summary()
        assert "expected 1 output leaves, got 2" in text
        assert "shape" not in text

    def test_summary_names_an_exact_integer_failure(self, fake_outputs):
        expected = (np.ones(3, np.float32), np.arange(3, dtype=np.int32))
        fake_outputs["outputs"] = [expected[0].copy(), np.array([0, 1, 5], np.int32)]
        text = verify_onnx_parity(expected, Path("unused.onnx"), ()).summary()
        assert text.startswith("FAIL: leaf 1 of 2:")
        assert "exact comparison" in text
        assert "atol" not in text

    def test_every_leaf_is_reported(self, fake_outputs):
        expected = (np.ones(3, np.float32), np.arange(3, dtype=np.int32))
        fake_outputs["outputs"] = [expected[0].copy(), expected[1].copy()]
        result = verify_onnx_parity(expected, Path("unused.onnx"), ())
        assert isinstance(result, LeafParityResult)
        assert result.passed
        assert len(result.leaf_results) == 2


# --------------------------------------------------------------------------- #
# Real toolchain: jax2onnx + onnx + onnxruntime.
# --------------------------------------------------------------------------- #


@pytest.fixture
def toolchain():
    pytest.importorskip("jax2onnx")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


def _export(fn, xs, reference, **kwargs):
    rows = xs.shape[0]
    return export_pipeline(
        fn,
        _vmap_plan(rows),
        (jax.ShapeDtypeStruct(xs.shape, xs.dtype),),
        (xs,),
        targets=(ONNX,),
        reference_fn=lambda _inputs: reference,
        **kwargs,
    )["onnx"]


@pytest.mark.usefixtures("toolchain")
class TestEndToEnd:
    def _mlp(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        w1 = jax.random.normal(k1, (8, 16), dtype=jnp.float32) * 0.1
        w2 = jax.random.normal(k2, (16, 4), dtype=jnp.float32) * 0.1
        return (lambda x: jnp.tanh(x @ w1) @ w2), np.asarray(w1), np.asarray(w2)

    def test_mlp_is_verified_against_a_numpy_oracle(self):
        fn, w1, w2 = self._mlp()
        xs = (np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / 256.0).astype(np.float32)
        reference = np.tanh(xs @ w1) @ w2
        result = _export(fn, xs, reference)
        assert result.verified, result.parity.summary()
        assert result.path.suffix == ".onnx"
        assert result.size_bytes == len(result.vmfb_bytes) == result.path.stat().st_size
        assert result.artifact_bytes == result.vmfb_bytes
        assert result.onnx_census is not None
        assert result.onnx_census.graph_inputs == ("float32",)
        assert result.onnx_census.graph_outputs == ("float32",)
        assert result.spirv_bytes is None

    def test_a_wrong_reference_is_not_verified(self):
        """Negative control: the parity check must be able to fail."""
        fn, w1, w2 = self._mlp()
        xs = np.ones((32, 8), np.float32)
        wrong = np.tanh(xs @ (w1 + 1e-3)) @ w2
        result = _export(fn, xs, wrong)
        assert not result.verified
        assert result.parity.max_abs_diff > 1e-5


@pytest.mark.usefixtures("toolchain")
class TestTiedIndexOpsAreExact:
    """The op classes that diverge silently on IREE, against NumPy oracles."""

    def test_fixtures_are_tie_rich(self):
        """Without this, an exact match could mean there were no ties."""
        keys, vals = _tie_rich_keys(), _tie_rich_vals()
        desc_idx = -np.arange(N)
        for row in keys:
            asc = np.lexsort((np.arange(N), row))
            assert not np.array_equal(asc, np.lexsort((desc_idx, row)))
        for row in vals:
            asc = np.lexsort((np.arange(N), -row))[:TOP_K]
            assert not np.array_equal(asc, np.lexsort((desc_idx, -row))[:TOP_K])

    def test_two_key_tiebreak_sort(self):
        keys = _tie_rich_keys()
        reference = np.stack([np.lexsort((np.arange(N), r)) for r in keys]).astype(np.int32)
        result = _export(_sort_2key, keys, reference)
        assert result.verified, result.parity.summary()

    def test_stable_argsort(self):
        """Blocked for IREE (sort-stability); must be exact here."""
        keys = _tie_rich_keys()
        reference = np.argsort(keys, axis=1, kind="stable").astype(np.int32)
        result = _export(_stable_argsort, keys, reference)
        assert result.verified, result.parity.summary()

    def test_lax_top_k(self):
        """Unlegalizable on IREE; lower index wins a tie, as in JAX."""
        vals = _tie_rich_vals()
        idx = np.argsort(-vals, axis=1, kind="stable")[:, :TOP_K].astype(np.int32)
        reference = (np.take_along_axis(vals, idx, axis=1), idx)
        result = _export(_top_k, vals, reference)
        assert result.verified, result.parity.summary()
        assert isinstance(result.parity, LeafParityResult)
        assert len(result.parity.leaf_results) == 2

    def test_index_graphs_report_their_internal_int64(self):
        keys = _tie_rich_keys()
        reference = np.argsort(keys, axis=1, kind="stable").astype(np.int32)
        result = _export(_stable_argsort, keys, reference)
        census = result.onnx_census
        assert census is not None
        assert census.n_int64 > 0
        assert "TopK" in census.int64_producers
        assert census.graph_outputs == ("int32",), "jax2onnx must cast back at the boundary"
        assert any("int64" in d and "WebGPU" in d for d in result.diagnostics)


@pytest.mark.usefixtures("toolchain")
class TestToolchainBehaviour:
    @staticmethod
    def _first_conversion_in_a_fresh_process(*, disable_guard: bool) -> str:
        """Run convert_to_onnx as a fresh interpreter's FIRST conversion.

        In-process, an earlier test's conversion has already spent jax2onnx's
        one-time cumsum patch, so a check there passes whether or not
        convert_to_onnx restores anything (#5690).
        """
        guard = (
            "import contextlib; onnx_mod._restoring_jax_namespaces = contextlib.nullcontext; "
            if disable_guard
            else ""
        )
        code = (
            "import jax, jax.numpy as jnp; "
            "from xtrax.export import ONNX; from xtrax.export import onnx as onnx_mod; "
            f"{guard}o = jnp.cumsum; "
            "f = lambda a: jnp.argsort(a, stable=True); "
            "onnx_mod.convert_to_onnx(f, (jax.ShapeDtypeStruct((4,), jnp.int32),), ONNX); "
            "print('leaked' if jnp.cumsum is not o else 'clean')"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr[-2000:]
        return result.stdout.strip().splitlines()[-1]

    def test_a_first_conversion_leaves_jnp_as_it_found_it(self):
        assert self._first_conversion_in_a_fresh_process(disable_guard=False) == "clean"

    def test_without_the_guard_the_same_first_conversion_leaks(self):
        """Control: the guard, not luck, is what restores jnp.cumsum."""
        assert self._first_conversion_in_a_fresh_process(disable_guard=True) == "leaked"

    def test_the_restoration_is_load_bearing(self):
        """Red control: a process's FIRST bare to_onnx leaves jnp.cumsum replaced.

        jax2onnx patches once per process, at its first conversion, so this has
        to run in a fresh interpreter -- in-process, an earlier test's
        conversion has already spent the one-time patch and the check would
        pass vacuously. If this starts failing, jax2onnx stopped leaking and the
        guard in convert_to_onnx is merely redundant; update the module
        docstring.
        """
        code = (
            "import jax, jax.numpy as jnp, jax2onnx; o = jnp.cumsum; "
            "f = lambda a: jnp.argsort(a, stable=True); "
            "jax2onnx.to_onnx(f, [jax.ShapeDtypeStruct((4,), jnp.int32)]); "
            "print('leaked' if jnp.cumsum is not o else 'clean')"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().endswith("leaked"), result.stdout

    def test_cumsum_still_converts_after_the_restoration(self):
        """Restoring the original must not break jax2onnx's own cumsum path."""
        spec = jax.ShapeDtypeStruct((N,), jnp.int32)
        convert_to_onnx(_stable_argsort, (spec,), ONNX)  # spends any one-time patch
        x = np.arange(8, dtype=np.float32)
        compiled, _ = convert_to_onnx(jnp.cumsum, (jax.ShapeDtypeStruct(x.shape, x.dtype),), ONNX)
        result = verify_onnx_parity(np.cumsum(x), compiled.path, (x,))
        assert result.passed, result.summary()

    def test_embedding_drops_external_data_and_keeps_the_model_name(self, monkeypatch, tmp_path):
        """The spill path (limit forced to 0) must not leave external refs when embedding."""
        import onnx

        monkeypatch.setattr(onnx_mod, "_PROTOBUF_LIMIT_BYTES", 0)
        weights = np.arange(4096, dtype=np.float32)
        x = np.ones((4096,), np.float32)
        out = tmp_path / "embedded.onnx"
        convert_to_onnx(
            lambda v: v * weights,
            (jax.ShapeDtypeStruct(x.shape, x.dtype),),
            ONNX,
            out_path=out,
            model_name="per_graph",
            embed_external_data=True,
        )
        model = onnx.load(str(out), load_external_data=False)
        assert model.graph.name == "per_graph"
        assert not (tmp_path / "embedded.onnx.data").exists()
        for tensor in onnx_mod._model_tensors(model):
            assert tensor.data_location != onnx.TensorProto.EXTERNAL
            assert len(tensor.external_data) == 0

    def test_rng_ops_are_found_inside_subgraphs(self):
        """The graph-level backstop walks If/Loop bodies, not just the top."""
        import onnx
        from onnx import TensorProto, helper

        draw = helper.make_node("RandomUniform", [], ["r"], shape=[2], name="draw")
        branch = helper.make_graph(
            [draw], "then", [], [helper.make_tensor_value_info("r", TensorProto.FLOAT, [2])]
        )
        cond = helper.make_node("If", ["c"], ["o"], then_branch=branch, else_branch=branch)
        graph = helper.make_graph(
            [cond],
            "g",
            [helper.make_tensor_value_info("c", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("o", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph)
        assert isinstance(model, onnx.ModelProto)
        assert onnx_mod.find_onnx_rng_ops(model) == ["RandomUniform:draw", "RandomUniform:draw"]

    def test_the_census_never_raises_on_undefined_types(self):
        """elem_type 0 (UNDEFINED) makes onnx's dtype helper raise KeyError."""
        from onnx import TensorProto, helper

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.UNDEFINED, [2])],
            [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, [2])],
        )
        census = onnx_mod.onnx_dtype_census(helper.make_model(graph))
        assert census.graph_inputs == ("unknown",)
        assert census.graph_outputs == ("unknown",)
        assert census.node_output_dtypes == {"unknown": 1}

    def test_a_clean_graph_has_no_rng_ops(self):
        spec = jax.ShapeDtypeStruct((4,), jnp.float32)
        compiled, _ = convert_to_onnx(jnp.tanh, (spec,), ONNX)
        import onnx

        assert onnx_mod.find_onnx_rng_ops(onnx.load(str(compiled.path))) == []

    def test_an_rng_graph_is_refused_even_without_the_gate(self):
        """convert_to_onnx alone (no validate_export_safe) still refuses RNG."""
        key_spec = jax.ShapeDtypeStruct((2,), jnp.uint32)
        with pytest.raises(CompileError, match="random-number ops"):
            convert_to_onnx(_uniform_from_key, (key_spec,), ONNX)

    def test_pytree_inputs_and_outputs_round_trip(self):
        def fn(tree):
            return {"s": tree["a"] + tree["b"], "p": tree["a"] * tree["b"]}

        a = np.arange(4, dtype=np.float32)
        b = np.full(4, 2.0, np.float32)
        spec = {
            "a": jax.ShapeDtypeStruct((4,), jnp.float32),
            "b": jax.ShapeDtypeStruct((4,), jnp.float32),
        }
        compiled, _ = convert_to_onnx(fn, (spec,), ONNX)
        expected = {"s": a + b, "p": a * b}
        result = verify_onnx_parity(expected, compiled.path, ({"a": a, "b": b},))
        assert result.passed, result.summary()


@pytest.mark.usefixtures("toolchain")
class TestDtypeEnvelope:
    """Re-measures targets._ONNX_DTYPES against the pinned toolchain."""

    _NUMPY = {
        "f32": np.float32,
        "f16": np.float16,
        "i32": np.int32,
        "i16": np.int16,
        "i8": np.int8,
        "u32": np.uint32,
        "u16": np.uint16,
        "u8": np.uint8,
        "bool": np.bool_,
    }

    def test_the_table_covers_the_envelope(self):
        assert set(self._NUMPY) == set(ONNX.supported_dtypes)

    @pytest.mark.parametrize("name", sorted(ONNX.supported_dtypes))
    def test_each_supported_dtype_round_trips(self, name):
        dtype = self._NUMPY[name]
        if dtype is np.bool_:
            x = np.array([True, False] * 4)
            fn, expected = jnp.logical_not, np.logical_not(x)
        else:
            x = np.arange(8).astype(dtype)
            fn, expected = (lambda a: a + a), (x + x).astype(dtype)
        compiled, census = convert_to_onnx(fn, (jax.ShapeDtypeStruct(x.shape, x.dtype),), ONNX)
        result = verify_onnx_parity(expected, compiled.path, (x,))
        assert result.passed, (name, result.summary(), result.dtype_mismatches)
        assert census.graph_inputs == (str(np.dtype(dtype)),)

    def test_bf16_is_excluded_because_ort_cannot_run_it(self):
        """The exclusion is measured, not assumed: this must keep failing."""
        import onnxruntime as ort

        x = np.asarray(jnp.arange(8).astype(jnp.bfloat16))
        compiled, _ = convert_to_onnx(
            lambda a: a + a, (jax.ShapeDtypeStruct(x.shape, x.dtype),), ONNX
        )
        with pytest.raises(Exception, match="NOT_IMPLEMENTED|Could not find an implementation"):
            ort.InferenceSession(str(compiled.path), providers=["CPUExecutionProvider"])

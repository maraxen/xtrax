"""#5689 / #5690 / #5691: the export gate judges the program actually exported, and
ONNX failures reach the caller as the documented error types.

From the code review of PR #174, each reproduced 2026-10-01 before the fix:

- a bf16 INTERMEDIATE passed the dtype gate, then ORT raised a raw
  ``onnxruntime ... NOT_IMPLEMENTED`` instead of ``DtypeNotSupportedError``;
- a float64 NumPy concrete input to an f32 graph raised a raw ORT ``InvalidArgument``;
- a per-element RNG ``fn`` under a Vmap plan skipped the "unsuppressible"
  onnx-in-graph-rng gate, because the gate traced ``fn`` against the BATCHED inputs
  and swallowed the failure; only the graph backstop caught it, after conversion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export import onnx as onnx_mod
from xtrax.export import safety
from xtrax.export.compile import CompileError
from xtrax.export.pipeline import export_pipeline
from xtrax.export.safety import (
    DtypeNotSupportedError,
    UnsupportedOperationError,
    check_export_safety,
    trace_for_export_safety,
)
from xtrax.export.targets import NATIVE, ONNX
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _vmap_plan(rows: int) -> _Plan:
    spec = AxisSpec(name="batch", cardinality=rows, default_batch_size=0)
    return _Plan([AxisDecision(spec=spec, batch_size=0, reasoning="t", strategy=Vmap())])


def _spec(x):
    return jax.ShapeDtypeStruct(np.shape(x), x.dtype)


def _no_oracle(_inputs):
    msg = "a gated-out export must not run the oracle"
    raise AssertionError(msg)


# --- #5689: the dtype gate covers pytree leaves and the onnx program's own dtypes ---


class TestDtypeGate:
    def test_a_bf16_intermediate_is_refused_before_conversion(self):
        xs = np.ones((4, 3), np.float32)
        with pytest.raises(DtypeNotSupportedError, match=r"program:.*'bf16'"):
            export_pipeline(
                lambda x: (x.astype(jnp.bfloat16) + 1).astype(jnp.float32),
                _vmap_plan(4),
                (_spec(xs),),
                (xs,),
                targets=(ONNX,),
                reference_fn=_no_oracle,
            )

    def test_an_f16_intermediate_is_accepted(self):
        """Control: f16 runs on ORT's CPU EP (spiked 2026-10-01), so it is no blocker."""
        xs = np.ones((4, 3), np.float32)
        fn = lambda x: (x.astype(jnp.float16) + 1).astype(jnp.float32)  # noqa: E731
        top = trace_for_export_safety(fn, (_spec(xs),))
        assert check_export_safety([], {}, (_spec(xs),), fn, ONNX, traced_jaxpr=top) == []

    def test_a_bf16_intermediate_is_not_an_iree_blocker(self):
        """IREE compiles bf16 intermediates; only its I/O boundary is judged."""
        xs = np.ones((4, 3), np.float32)
        fn = lambda x: (x.astype(jnp.bfloat16) + 1).astype(jnp.float32)  # noqa: E731
        top = trace_for_export_safety(fn, (_spec(xs),))
        assert check_export_safety([], {}, (_spec(xs),), fn, NATIVE, traced_jaxpr=top) == []

    def test_a_pytree_input_leaf_is_judged(self):
        ab = ({"a": jax.ShapeDtypeStruct((4,), jnp.bfloat16)},)
        blockers = check_export_safety([], {}, ab, lambda d: d["a"], ONNX)
        assert [(b.axis, b.rule) for b in blockers][0] == ("abstract_inputs[0]['a']", "dtype")

    def test_a_bare_input_keeps_its_plain_name(self):
        ab = (jax.ShapeDtypeStruct((4,), jnp.float64),)
        blockers = check_export_safety([], {}, ab, lambda x: x, NATIVE)
        assert blockers[0].axis == "abstract_inputs[0]"


# --- #5689: ORT and serialization failures surface as CompileError -----------------


@pytest.fixture
def onnx_toolchain():
    pytest.importorskip("jax2onnx")
    pytest.importorskip("onnxruntime")


@pytest.mark.usefixtures("onnx_toolchain")
class TestOnnxErrorsAndInputs:
    def test_a_float64_numpy_input_runs_and_verifies(self):
        """jnp.asarray narrows it, as on the way into the traced program."""
        xs = np.ones((4, 3))  # float64
        result = export_pipeline(
            lambda x: x * 2,
            _vmap_plan(4),
            (jax.ShapeDtypeStruct((4, 3), jnp.float32),),
            (xs,),
            targets=(ONNX,),
            reference_fn=lambda inp: np.asarray(inp[0], np.float32) * 2,
        )["onnx"]
        assert result.verified, result.parity.summary()

    def test_an_ort_run_failure_is_a_compile_error(self):
        x = np.ones((4,), np.float32)
        compiled, _ = onnx_mod.convert_to_onnx(lambda v: v + 1, (_spec(x),), ONNX)
        with pytest.raises(CompileError, match=r"could not run .*declared"):
            onnx_mod.run_onnx(compiled.path, np.ones((5, 2), np.float32))

    def test_an_input_count_mismatch_is_a_compile_error(self):
        x = np.ones((4,), np.float32)
        compiled, _ = onnx_mod.convert_to_onnx(lambda v: v + 1, (_spec(x),), ONNX)
        with pytest.raises(CompileError, match="declares 1 input"):
            onnx_mod.run_onnx(compiled.path, x, x)

    def test_a_large_model_is_written_with_external_data(self, monkeypatch, tmp_path):
        """Above protobuf's 2 GiB limit the weights go to <model>.onnx.data. Simulated by
        lowering the limit: a real 2 GiB model is too big for a unit test."""
        monkeypatch.setattr(onnx_mod, "_PROTOBUF_LIMIT_BYTES", 0)
        weights = np.arange(4096, dtype=np.float32)  # 16 KiB, above the size threshold
        x = np.ones((4096,), np.float32)
        out = tmp_path / "big.onnx"
        compiled, _ = onnx_mod.convert_to_onnx(
            lambda v: v * weights, (_spec(x),), ONNX, out_path=out
        )
        data = tmp_path / "big.onnx.data"
        assert data.exists() and data.stat().st_size >= weights.nbytes
        assert compiled.size_bytes == out.stat().st_size + data.stat().st_size
        result = onnx_mod.verify_onnx_parity(x * weights, out, (x,))
        assert result.passed, result.summary()

    def test_a_write_failure_is_a_compile_error(self, tmp_path):
        x = np.ones((4,), np.float32)
        out = tmp_path / "as_dir.onnx"
        out.mkdir()  # writing a file over a directory fails
        with pytest.raises(CompileError, match="could not write the ONNX model"):
            onnx_mod.convert_to_onnx(lambda v: v + 1, (_spec(x),), ONNX, out_path=out)


# --- #5690: the op gate judges the batched program actually exported ----------------


class TestOpGateSeesTheExportedProgram:
    @staticmethod
    def _per_element_rng(key_row):
        return jax.random.uniform(key_row, (4,))

    def test_a_per_element_rng_fn_is_gated_before_conversion(self):
        """Red before #5690: the gate traced fn on the batched (B, 2) keys, the trace
        failed and was swallowed, and the export reached jax2onnx."""
        keys = np.asarray(jax.random.split(jax.random.PRNGKey(0), 4))
        with pytest.raises(UnsupportedOperationError, match="onnx-in-graph-rng"):
            export_pipeline(
                self._per_element_rng,
                _vmap_plan(4),
                (_spec(keys),),
                (keys,),
                targets=(ONNX,),
                reference_fn=_no_oracle,
            )

    def test_without_the_exported_jaxpr_the_per_element_fn_is_not_judged(self):
        """Control documenting the direct-call fallback: tracing fn itself against the
        batched inputs fails, so the op rules see nothing. export_pipeline always passes
        traced_jaxpr; a direct caller can too."""
        keys = np.asarray(jax.random.split(jax.random.PRNGKey(0), 4))
        assert check_export_safety([], {}, (_spec(keys),), self._per_element_rng, ONNX) == []

    def test_one_trace_serves_every_target(self, monkeypatch):
        """#5691: export_pipeline traces the composed callable once, not once per target."""
        calls = []
        real = safety.trace_for_export_safety

        def spy(fn, abstract_inputs):
            calls.append(fn)
            return real(fn, abstract_inputs)

        monkeypatch.setattr("xtrax.export.pipeline.trace_for_export_safety", spy)
        xs = np.ones((4, 3), np.float64)
        with pytest.raises(DtypeNotSupportedError):  # f64 is refused by every target
            export_pipeline(
                lambda x: x,
                _vmap_plan(4),
                (_spec(xs),),
                (xs,),
                targets=(NATIVE, ONNX),
                reference_fn=_no_oracle,
            )
        assert len(calls) == 1


# --- #5690 part 7: the namespace guard restores deleted names too --------------------


def test_the_namespace_guard_restores_a_deleted_public_name():
    original = jax.nn.relu
    try:
        with onnx_mod._restoring_jax_namespaces():
            del jax.nn.relu
        assert jax.nn.relu is original
    finally:
        jax.nn.relu = original


def test_the_namespace_guard_still_restores_a_replaced_name():
    original = jnp.cumsum
    try:
        with onnx_mod._restoring_jax_namespaces():
            jnp.cumsum = lambda *a, **k: None
        assert jnp.cumsum is original
    finally:
        jnp.cumsum = original

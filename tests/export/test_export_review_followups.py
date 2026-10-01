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
        """Control: f16 runs on ORT's CPU EP.

        Pinned by test_envelope_dtype_runs_as_an_intermediate_on_ort.
        """
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

        def spy(fn, abstract_inputs, **kw):
            calls.append(fn)
            return real(fn, abstract_inputs, **kw)

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


# --- review of #180: fixes for what the first version of this PR got wrong ----------


def test_an_input_dtype_is_reported_once():
    """The program-dtype rule does not re-judge inputs (was: two blockers for one)."""
    ab = (jax.ShapeDtypeStruct((4,), jnp.bfloat16),)
    fn = lambda x: x  # noqa: E731
    top = trace_for_export_safety(fn, ab)
    blockers = check_export_safety([], {}, ab, fn, ONNX, traced_jaxpr=top)
    assert [b.axis for b in blockers] == ["abstract_inputs[0]"]


def test_float0_is_not_a_dtype_blocker():
    """issubdtype(float0, extended) is False, so float0 needs its own skip."""
    from types import SimpleNamespace as NS

    var = NS(aval=NS(dtype=jax.dtypes.float0))
    eqn = NS(primitive=NS(name="grad_int"), invars=[], outvars=[var], params={})
    top = NS(eqns=[eqn], invars=[], outvars=[var])
    assert safety._program_dtype_blockers(top, ONNX, frozenset(), frozenset()) == []


def test_trace_for_export_safety_is_exported():
    from xtrax.export import trace_for_export_safety as exported

    assert exported is trace_for_export_safety


def test_a_composition_error_does_not_hide_the_gate():
    """A Scan axis with no init cannot compose; a bf16 input must still be reported
    first, as it was before composition moved ahead of the gates."""
    from xtrax.export.composer import ComposerError
    from xtrax.tiling.strategy import Scan

    spec = AxisSpec(name="t", cardinality=4, default_batch_size=0)
    plan = _Plan([AxisDecision(spec=spec, batch_size=0, reasoning="t", strategy=Scan())])

    def step(carry, x):
        return carry, x

    bad = np.ones((4, 3), np.float32).astype(jnp.bfloat16)
    with pytest.raises(DtypeNotSupportedError):
        export_pipeline(step, plan, (_spec(bad),), (bad,), targets=(ONNX,), reference_fn=_no_oracle)
    ok = np.ones((4, 3), np.float32)
    with pytest.raises(ComposerError, match="initial carry"):
        export_pipeline(step, plan, (_spec(ok),), (ok,), targets=(ONNX,), reference_fn=_no_oracle)


@pytest.mark.usefixtures("onnx_toolchain")
def test_exporting_twice_does_not_grow_the_external_data(monkeypatch, tmp_path):
    """onnx appends to an existing <model>.onnx.data; a stale one must be removed."""
    monkeypatch.setattr(onnx_mod, "_PROTOBUF_LIMIT_BYTES", 0)
    weights = np.arange(4096, dtype=np.float32)
    x = np.ones((4096,), np.float32)
    out = tmp_path / "big.onnx"
    sizes = [
        onnx_mod.convert_to_onnx(lambda v: v * weights, (_spec(x),), ONNX, out_path=out)[
            0
        ].size_bytes
        for _ in range(2)
    ]
    assert sizes[0] == sizes[1]


@pytest.mark.usefixtures("onnx_toolchain")
def test_an_external_data_export_says_the_artifact_is_two_files(monkeypatch):
    monkeypatch.setattr(onnx_mod, "_PROTOBUF_LIMIT_BYTES", 0)
    weights = np.arange(4096, dtype=np.float32)
    xs = np.ones((2, 4096), np.float32)
    result = export_pipeline(
        lambda v: v * weights,
        _vmap_plan(2),
        (_spec(xs),),
        (xs,),
        targets=(ONNX,),
        reference_fn=lambda inp: inp[0] * weights,
    )["onnx"]
    assert result.verified, result.parity.summary()
    assert any(".onnx.data" in note for note in result.diagnostics)


# --- second review of #180 (xhigh) ---------------------------------------------------


def test_a_consumed_input_dtype_is_reported_once():
    """Was: a bf16 input with fn = x + 1 gave 'abstract_inputs[0]' AND 'program:add'."""
    ab = (jax.ShapeDtypeStruct((4,), jnp.bfloat16),)
    fn = lambda x: x + 1  # noqa: E731
    blockers = check_export_safety(
        [], {}, ab, fn, ONNX, traced_jaxpr=trace_for_export_safety(fn, ab)
    )
    assert [b.axis for b in blockers] == ["abstract_inputs[0]"]


def test_the_bf16_precision_emulation_idiom_passes_the_onnx_gate():
    """Was refused: casts are data movement, which ORT runs in bf16."""
    xs = np.ones((4, 3), np.float32)
    fn = lambda x: x.astype(jnp.bfloat16).astype(jnp.float32)  # noqa: E731
    top = trace_for_export_safety(fn, (_spec(xs),))
    assert check_export_safety([], {}, (_spec(xs),), fn, ONNX, traced_jaxpr=top) == []


def test_a_bf16_comparison_is_refused_although_its_result_is_bool():
    """Operands are judged too: gt on bf16 returns bool but has no ORT bf16 kernel."""
    xs = np.ones((4, 3), np.float32)
    fn = lambda x: (x.astype(jnp.bfloat16) > 1).astype(jnp.float32)  # noqa: E731
    top = trace_for_export_safety(fn, (_spec(xs),))
    blockers = check_export_safety([], {}, (_spec(xs),), fn, ONNX, traced_jaxpr=top)
    assert [b.axis for b in blockers] == ["program:gt"]


def test_a_native_gate_refuses_a_bf16_output():
    """Was: only inputs were judged, so a bf16-returning program passed `native`."""
    xs = np.ones((4,), np.float32)
    fn = lambda x: x.astype(jnp.bfloat16)  # noqa: E731
    top = trace_for_export_safety(fn, (_spec(xs),))
    blockers = check_export_safety([], {}, (_spec(xs),), fn, NATIVE, traced_jaxpr=top)
    assert [(b.axis, b.rule) for b in blockers] == [("program:output", "dtype")]


def test_failed_tracing_is_not_repeated_per_target(monkeypatch):
    """Was: when both traces failed, every target's gate re-traced fn."""
    calls = []
    real = safety.trace_for_export_safety

    def spy(fn, abstract_inputs, **kw):
        calls.append(fn)
        return real(fn, abstract_inputs, **kw)

    monkeypatch.setattr(safety, "trace_for_export_safety", spy)
    monkeypatch.setattr("xtrax.export.pipeline.trace_for_export_safety", spy)

    def untraceable(x):
        msg = "cannot be traced"
        raise TypeError(msg)

    xs = np.ones((4, 3), np.float32)
    with pytest.raises(Exception):  # noqa: B017, PT011 - export itself fails afterwards
        export_pipeline(
            untraceable,
            _vmap_plan(4),
            (_spec(xs),),
            (xs,),
            targets=(NATIVE, ONNX),
            reference_fn=lambda inp: inp[0],
        )
    assert len(calls) == 2  # the composed callable, then fn: never once per target


def test_a_composer_error_raised_while_tracing_is_a_composer_error(monkeypatch):
    """Was: swallowed by the safety trace, then surfaced from conversion as CompileError."""
    from xtrax.export.composer import MultiAxisCompositionError

    def composes_but_fails_on_trace(*_args, **_kwargs):
        def callable_(*_a):
            msg = "lane-dependent ordered sink"
            raise MultiAxisCompositionError(msg)

        return callable_

    monkeypatch.setattr(
        "xtrax.export.pipeline.build_traceable_callable", composes_but_fails_on_trace
    )
    xs = np.ones((4, 3), np.float32)
    with pytest.raises(MultiAxisCompositionError, match="lane-dependent"):
        export_pipeline(
            lambda x: x,
            _vmap_plan(4),
            (_spec(xs),),
            (xs,),
            targets=(ONNX,),
            reference_fn=_no_oracle,
        )


# --- ORT facts the dtype rules rest on, pinned (were cited as dated spikes) ------------


_BF16_CASES = {
    "convert_element_type": lambda v: v.astype(jnp.bfloat16).astype(jnp.float32),
    "reshape": lambda v: v.astype(jnp.bfloat16).reshape(4, 3).astype(jnp.float32),
    "transpose": lambda v: v.astype(jnp.bfloat16).T.astype(jnp.float32),
    "squeeze": lambda v: jnp.squeeze(v.astype(jnp.bfloat16)[None]).astype(jnp.float32),
    "slice": lambda v: v.astype(jnp.bfloat16)[1:, :2].astype(jnp.float32),
    "dynamic_slice": lambda v: jax.lax.dynamic_slice(v.astype(jnp.bfloat16), (1, 1), (2, 2)).astype(
        jnp.float32
    ),
    "concatenate": lambda v: jnp.concatenate([v.astype(jnp.bfloat16)] * 2).astype(jnp.float32),
    "gather": lambda v: v.astype(jnp.bfloat16)[jnp.array([2, 0])].astype(jnp.float32),
    "rev": lambda v: v.astype(jnp.bfloat16)[::-1].astype(jnp.float32),
    "reduce_sum": lambda v: v.astype(jnp.bfloat16).sum(0).astype(jnp.float32),
}


@pytest.mark.usefixtures("onnx_toolchain")
@pytest.mark.parametrize("primitive", sorted(_BF16_CASES))
def test_allowlisted_primitive_runs_in_bf16_on_ort(primitive):
    """Each data-movement primitive the onnx dtype rule lets through really runs on
    ORT's CPU EP in bf16, and every non-wrapper allowlist entry has a case here."""
    wrappers = {
        "pjit",
        "jit",
        "closed_call",
        "custom_jvp_call",
        "custom_vjp_call",
        "remat",
        "checkpoint",
    }
    assert set(_BF16_CASES) == safety._ONNX_DTYPE_AGNOSTIC_PRIMITIVES - wrappers
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    fn = _BF16_CASES[primitive]
    compiled, _ = onnx_mod.convert_to_onnx(fn, (_spec(x),), ONNX)
    out = onnx_mod.run_onnx(compiled.path, x)[0]
    np.testing.assert_allclose(out, np.asarray(jax.jit(fn)(x)), atol=0.5)


@pytest.mark.usefixtures("onnx_toolchain")
@pytest.mark.parametrize(
    "fn",
    [lambda v: v.astype(jnp.bfloat16) + 1, lambda v: -v.astype(jnp.bfloat16)],
    ids=["add", "neg"],
)
def test_bf16_arithmetic_has_no_ort_kernel(fn):
    """Control: why arithmetic is NOT allowlisted. If this starts passing, ORT gained
    bf16 kernels and the allowlist can grow."""
    x = np.ones((3, 4), np.float32)
    wrapped = lambda v: fn(v).astype(jnp.float32)  # noqa: E731
    compiled, _ = onnx_mod.convert_to_onnx(wrapped, (_spec(x),), ONNX)
    with pytest.raises(CompileError, match="NOT_IMPLEMENTED"):
        onnx_mod.run_onnx(compiled.path, x)


@pytest.mark.usefixtures("onnx_toolchain")
@pytest.mark.parametrize(
    "dtype", ["float16", "int32", "int16", "int8", "uint32", "uint16", "uint8"]
)
def test_envelope_dtype_runs_as_an_intermediate_on_ort(dtype):
    """Every non-f32 dtype in the onnx envelope runs on ORT's CPU EP as arithmetic."""
    x = np.ones((4,), np.float32)
    dt = jnp.dtype(dtype)
    fn = lambda v: (v.astype(dt) + jnp.asarray(1, dt) * 2).astype(jnp.float32)  # noqa: E731
    compiled, _ = onnx_mod.convert_to_onnx(fn, (_spec(x),), ONNX)
    np.testing.assert_array_equal(onnx_mod.run_onnx(compiled.path, x)[0], np.full(4, 3.0))


# --- external data: attribute tensors, and no CWD collision ---------------------------


@pytest.mark.usefixtures("onnx_toolchain")
def test_every_large_tensor_goes_external_including_attributes(monkeypatch, tmp_path):
    import onnx

    monkeypatch.setattr(onnx_mod, "_PROTOBUF_LIMIT_BYTES", 0)
    weights = np.arange(4096, dtype=np.float32)
    x = np.ones((4096,), np.float32)
    out = tmp_path / "m.onnx"
    onnx_mod.convert_to_onnx(lambda v: v * weights, (_spec(x),), ONNX, out_path=out)
    model = onnx.load(str(out), load_external_data=False)
    inline = [
        t.name
        for t in onnx_mod._model_tensors(model)
        if t.data_location != onnx.TensorProto.EXTERNAL
        and len(t.raw_data) >= onnx_mod._EXTERNAL_DATA_THRESHOLD_BYTES
    ]
    assert inline == []


@pytest.mark.usefixtures("onnx_toolchain")
def test_an_unrelated_data_file_in_the_cwd_does_not_fail_the_export(monkeypatch, tmp_path):
    """Was: onnx checked os.path.exists(location) against the CWD, not the model dir."""
    monkeypatch.setattr(onnx_mod, "_PROTOBUF_LIMIT_BYTES", 0)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "m.onnx.data").write_bytes(b"unrelated")
    monkeypatch.chdir(cwd)
    weights = np.arange(4096, dtype=np.float32)
    x = np.ones((4096,), np.float32)
    out = tmp_path / "dest" / "m.onnx"
    onnx_mod.convert_to_onnx(lambda v: v * weights, (_spec(x),), ONNX, out_path=out)
    assert (cwd / "m.onnx.data").read_bytes() == b"unrelated"
    assert onnx_mod.verify_onnx_parity(x * weights, out, (x,)).passed

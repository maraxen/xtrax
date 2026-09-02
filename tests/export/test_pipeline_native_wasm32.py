"""export_pipeline end to end against a fake toolchain, plus the materialize strip."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from xtrax.export import compile as compile_mod
from xtrax.export.compile import CompileError
from xtrax.export.parity import ParityResult
from xtrax.export.pipeline import (
    ExportResult,
    _boundaries_for_export,
    _StrippedSink,
    _verified_for,
    export_pipeline,
)
from xtrax.export.spirv import SpirvValidationResult
from xtrax.export.targets import NATIVE, WASM32, Target, VerificationLevel
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.topology import PlanTopologyError

from .conftest import FakeCompilerTools

# PR1 registers no VALIDATED target; this stands in for one so the guard that
# refuses them can be exercised.
VALIDATED_TARGET = Target(
    name="spirv-probe",
    iree_backend="vulkan-spirv",
    verification_level=VerificationLevel.VALIDATED,
    supported_dtypes=frozenset({"f32", "i32", "bool"}),
)


class _Sink:
    """A sink that records nothing; only its presence and .ordered matter here."""

    def __init__(self, *, ordered: bool = False) -> None:
        self.ordered = ordered

    def __call__(self, x) -> None:
        return None


class TestArgumentGuards:
    def test_executed_target_without_concrete_inputs_raises(
        self, model, plan, abstract_inputs, reference_fn
    ):
        with pytest.raises(ValueError, match="concrete_inputs is required"):
            export_pipeline(
                model, plan, abstract_inputs, None, targets=(NATIVE,), reference_fn=reference_fn
            )

    def test_executed_target_without_reference_fn_raises(self, model, plan, abstract_inputs, xs):
        with pytest.raises(ValueError, match="reference_fn is required"):
            export_pipeline(model, plan, abstract_inputs, [xs], targets=(NATIVE,))

    def test_the_error_explains_what_a_valid_oracle_is(self, model, plan, abstract_inputs, xs):
        """Passing the callable under test as its own oracle verifies nothing."""
        with pytest.raises(ValueError, match="compares the composed callable against itself"):
            export_pipeline(model, plan, abstract_inputs, [xs], targets=(NATIVE,))

    def test_codegen_only_target_needs_neither(self, model, plan, abstract_inputs, fake_compiler):
        results = export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))
        assert set(results) == {"wasm32"}

    def test_validated_target_is_refused_rather_than_silently_unverified(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        """SPIR-V validation is not wired in, so `verified` could only be False.

        Emitting a result that reports False while nothing was actually checked
        is the failure mode this package exists to avoid, so refuse instead.
        """
        with pytest.raises(NotImplementedError, match="not wired into export_pipeline"):
            export_pipeline(model, plan, abstract_inputs, None, targets=(VALIDATED_TARGET,))

    def test_the_validated_guard_names_the_offending_target(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        with pytest.raises(NotImplementedError, match="spirv-probe"):
            export_pipeline(model, plan, abstract_inputs, None, targets=(VALIDATED_TARGET,))

    def test_the_validated_guard_fires_before_any_compile(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        with pytest.raises(NotImplementedError):
            export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32, VALIDATED_TARGET))
        assert fake_compiler.calls == [], "the guard must run before the compiler"


class TestVerifiedFor:
    """`_verified_for` is the per-level contract; VALIDATED lands with PR2."""

    def test_executed_mirrors_parity(self):
        passing = ParityResult(
            passed=True,
            max_abs_diff=0.0,
            atol=1e-5,
            rtol=1e-5,
            shape_expected=(2,),
            shape_actual=(2,),
        )
        assert _verified_for(VerificationLevel.EXECUTED, passing, None) is True

    def test_executed_is_false_without_a_parity_result(self):
        assert _verified_for(VerificationLevel.EXECUTED, None, None) is False

    def test_validated_mirrors_the_shader_validation(self):
        valid = SpirvValidationResult(
            valid=True, adapter_type="CPU", backend="Vulkan", device_name="llvmpipe", error=None
        )
        assert _verified_for(VerificationLevel.VALIDATED, None, valid) is True

    def test_validated_is_false_when_the_shader_was_rejected(self):
        invalid = SpirvValidationResult(
            valid=False,
            adapter_type="CPU",
            backend="Vulkan",
            device_name="llvmpipe",
            error="push constants are not a WebGPU capability",
        )
        assert _verified_for(VerificationLevel.VALIDATED, None, invalid) is False

    def test_validated_is_false_without_a_validation_result(self):
        assert _verified_for(VerificationLevel.VALIDATED, None, None) is False

    def test_codegen_only_is_false_even_with_a_passing_parity(self):
        """Nothing beyond compilation was established, whatever else is present."""
        passing = ParityResult(
            passed=True,
            max_abs_diff=0.0,
            atol=1e-5,
            rtol=1e-5,
            shape_expected=(2,),
            shape_actual=(2,),
        )
        assert _verified_for(VerificationLevel.CODEGEN_ONLY, passing, None) is False


class TestVerificationSemantics:
    def test_codegen_only_is_never_verified(self, model, plan, abstract_inputs, fake_compiler):
        results = export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))
        result = results["wasm32"]
        assert result.verification_level is VerificationLevel.CODEGEN_ONLY
        assert result.verified is False
        assert result.parity is None

    def test_executed_verified_mirrors_parity(
        self, model, plan, abstract_inputs, xs, reference_fn, fake_compiler, fake_runtime
    ):
        fake_runtime["result"] = np.asarray(reference_fn([xs]))
        results = export_pipeline(
            model, plan, abstract_inputs, [xs], targets=(NATIVE,), reference_fn=reference_fn
        )
        assert results["native"].verified is True
        assert results["native"].parity.passed is True

    def test_executed_not_verified_when_parity_fails(
        self, model, plan, abstract_inputs, xs, reference_fn, fake_compiler, fake_runtime
    ):
        fake_runtime["result"] = np.zeros((32, 4), dtype=np.float32)
        results = export_pipeline(
            model, plan, abstract_inputs, [xs], targets=(NATIVE,), reference_fn=reference_fn
        )
        assert results["native"].verified is False
        assert results["native"].parity.passed is False


class TestExportResultShape:
    def test_path_is_the_compiled_artifact(self, model, plan, abstract_inputs, fake_compiler):
        """AC-17/AC-14b need a route to the real executed output array."""
        results = export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))
        result = results["wasm32"]
        assert isinstance(result, ExportResult)
        assert result.path.exists()

    def test_vmfb_bytes_matches_the_file(self, model, plan, abstract_inputs, fake_compiler):
        result = export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))["wasm32"]
        assert result.vmfb_bytes == result.path.read_bytes()

    def test_results_are_keyed_by_target_name(
        self, model, plan, abstract_inputs, xs, reference_fn, fake_compiler, fake_runtime
    ):
        results = export_pipeline(
            model,
            plan,
            abstract_inputs,
            [xs],
            targets=(NATIVE, WASM32),
            reference_fn=reference_fn,
        )
        assert list(results) == ["native", "wasm32"]

    def test_downgrade_is_surfaced_in_diagnostics(self, model, plan, abstract_inputs, monkeypatch):
        tools = FakeCompilerTools(calls=[], fail_first=True)
        monkeypatch.setattr(compile_mod, "_require_compiler", lambda: tools)
        monkeypatch.setattr(compile_mod, "_downgrade_to_portable", lambda _t: b"portable")

        result = export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))["wasm32"]
        assert any("downgraded" in d for d in result.diagnostics)


class TestAllOrNothing:
    def test_a_later_target_failing_returns_no_partial_dict(
        self, model, plan, abstract_inputs, xs, reference_fn, monkeypatch, fake_runtime
    ):
        """The first target succeeds; the second fails outright, so nothing is returned.

        The second target's failure must survive the portable-artifact retry --
        a fake that fails only once would be repaired by the downgrade and prove
        nothing about all-or-nothing.
        """

        class _FailsAfterFirstTarget(FakeCompilerTools):
            def compile_str(self, source, *, input_type, extra_args):
                self.calls.append({"source": source})
                if len(self.calls) >= 2:
                    msg = "second target fails, and keeps failing"
                    raise RuntimeError(msg)
                return b"FAKE"

        tools = _FailsAfterFirstTarget(calls=[])
        monkeypatch.setattr(compile_mod, "_require_compiler", lambda: tools)
        monkeypatch.setattr(compile_mod, "_downgrade_to_portable", lambda _t: b"portable")

        with pytest.raises(CompileError):
            export_pipeline(
                model,
                plan,
                abstract_inputs,
                [xs],
                targets=(WASM32, NATIVE),
                reference_fn=reference_fn,
            )
        # The first target did compile; the caller still receives nothing.
        assert len(tools.calls) >= 2

    def test_the_gate_runs_before_any_compile(self, model, plan, abstract_inputs, fake_compiler):
        """AC-3: rejection happens with zero compiler invocations."""
        boundaries = {"batch": AxisBoundary(sink=_Sink())}
        with pytest.raises(PlanTopologyError):
            export_pipeline(
                model,
                plan,
                abstract_inputs,
                None,
                axis_boundaries=boundaries,
                targets=(WASM32,),
            )
        assert fake_compiler.calls == []


class TestBoundariesForExport:
    def test_none_passes_through(self):
        assert _boundaries_for_export(None) is None

    def test_empty_passes_through(self):
        empty: dict = {}
        assert _boundaries_for_export(empty) is empty

    def test_no_materializing_axis_returns_the_same_mapping(self):
        """Nothing to strip means nothing is rebuilt."""
        boundaries = {"batch": AxisBoundary(fuse=lambda ys: ys)}
        assert _boundaries_for_export(boundaries) is boundaries

    def test_materializing_axis_gets_a_stripped_sink(self):
        original = _Sink(ordered=True)
        boundaries = {"batch": AxisBoundary(sink=original, materialize=True)}
        stripped = _boundaries_for_export(boundaries)
        assert stripped["batch"].sink is not original
        assert isinstance(stripped["batch"].sink, _StrippedSink)

    def test_stripped_sink_preserves_ordered(self):
        """A bare None would flip an ordered SafeMap onto a different lowering.

        execute_map_axis reads boundary.sink.ordered to choose between
        jax.lax.map and safe_map(..., batch_size=...); the latter raises when
        cardinality is not divisible by batch_size, so a working configuration
        would crash at export.
        """
        boundaries = {"batch": AxisBoundary(sink=_Sink(ordered=True), materialize=True)}
        assert _boundaries_for_export(boundaries)["batch"].sink.ordered is True

    def test_stripped_sink_preserves_unordered(self):
        boundaries = {"batch": AxisBoundary(sink=_Sink(ordered=False), materialize=True)}
        assert _boundaries_for_export(boundaries)["batch"].sink.ordered is False

    def test_stripped_sink_is_a_no_op(self):
        assert _StrippedSink(ordered=True)(object()) is None

    def test_fuse_and_tap_on_the_materializing_axis_are_untouched(self):
        """AC-17g: only the sink slot changes."""
        fuse = None  # a materializing axis cannot carry a fuse; tap is rejected outright
        boundary = AxisBoundary(sink=_Sink(), materialize=True, fuse=fuse)
        stripped = _boundaries_for_export({"batch": boundary})["batch"]
        assert stripped.fuse is boundary.fuse
        assert stripped.tap is boundary.tap
        assert stripped.materialize is True

    def test_other_axes_are_returned_by_identity(self):
        """AC-17g: `is`, not merely `==`."""
        other = AxisBoundary(fuse=lambda ys: ys)
        boundaries = {
            "outer": AxisBoundary(sink=_Sink(), materialize=True),
            "inner": other,
        }
        assert _boundaries_for_export(boundaries)["inner"] is other

    def test_uses_dataclasses_replace_so_subclasses_survive(self):
        """A field-by-field rebuild would downcast a subclass and drop new fields."""

        class Extended(AxisBoundary):
            pass

        boundary = Extended(sink=_Sink(), materialize=True)
        stripped = _boundaries_for_export({"batch": boundary})["batch"]
        assert isinstance(stripped, Extended)
        assert stripped == dataclasses.replace(boundary, sink=stripped.sink)


class TestMaterializeEndToEnd:
    def test_a_materializing_sink_reaches_compilation(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        """AC-17: what the gate rejects undeclared, it accepts declared."""
        boundaries = {"batch": AxisBoundary(sink=_Sink(ordered=True), materialize=True)}
        results = export_pipeline(
            model,
            plan,
            abstract_inputs,
            None,
            axis_boundaries=boundaries,
            targets=(WASM32,),
        )
        assert results["wasm32"].path.exists()
        assert len(fake_compiler.calls) == 1

    def test_the_exported_trace_carries_no_io_callback(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        """The whole point of stripping: the host call is absent from the trace."""
        calls: list[int] = []

        class _Counting:
            ordered = False

            def __call__(self, x) -> None:
                calls.append(1)

        boundaries = {"batch": AxisBoundary(sink=_Counting(), materialize=True)}
        export_pipeline(
            model,
            plan,
            abstract_inputs,
            None,
            axis_boundaries=boundaries,
            targets=(WASM32,),
        )
        assert calls == [], "the stripped sink must never be invoked during tracing"

    def test_the_callers_boundaries_are_not_mutated(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        original = _Sink(ordered=True)
        boundaries = {"batch": AxisBoundary(sink=original, materialize=True)}
        export_pipeline(
            model,
            plan,
            abstract_inputs,
            None,
            axis_boundaries=boundaries,
            targets=(WASM32,),
        )
        assert boundaries["batch"].sink is original


class TestStableHloIsWhatGetsCompiled:
    def test_the_compiler_receives_stablehlo_mentioning_the_entry_point(
        self, model, plan, abstract_inputs, fake_compiler
    ):
        export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))
        source = fake_compiler.calls[0]["source"]
        assert isinstance(source, str)
        assert "stablehlo" in source or "func" in source

    def test_input_shape_appears_in_the_module(self, model, plan, abstract_inputs, fake_compiler):
        export_pipeline(model, plan, abstract_inputs, None, targets=(WASM32,))
        assert "32x8" in fake_compiler.calls[0]["source"]

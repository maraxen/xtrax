"""compile_for_target and the parity primitives, against a fake toolchain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xtrax.export import compile as compile_mod
from xtrax.export.compile import CompileError, compile_for_target, run_native_vmfb
from xtrax.export.parity import ParityResult, compare, verify_native_parity
from xtrax.export.targets import NATIVE, WASM32

from .conftest import FAKE_VMFB, FakeCompilerTools

MLIR = "module { func.func @main() { return } }"


class TestTargetFlagsComeFromTheTarget:
    def test_native_passes_its_backend_and_flags(self, fake_compiler, tmp_path):
        compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")
        args = fake_compiler.calls[0]["extra_args"]
        assert "--iree-hal-target-backends=llvm-cpu" in args
        assert "--iree-llvmcpu-target-cpu=host" in args

    def test_wasm32_passes_the_emscripten_triple(self, fake_compiler, tmp_path):
        compile_for_target(MLIR, WASM32, out_path=tmp_path / "a.vmfb")
        args = fake_compiler.calls[0]["extra_args"]
        assert "--iree-llvmcpu-target-triple=wasm32-unknown-emscripten" in args

    def test_two_targets_differ_only_by_their_own_flags(self, fake_compiler, tmp_path):
        compile_for_target(MLIR, NATIVE, out_path=tmp_path / "n.vmfb")
        compile_for_target(MLIR, WASM32, out_path=tmp_path / "w.vmfb")
        native_args, wasm_args = (c["extra_args"] for c in fake_compiler.calls)
        assert native_args != wasm_args
        assert native_args[0] == wasm_args[0] == "--iree-hal-target-backends=llvm-cpu"

    def test_input_type_is_stablehlo(self, fake_compiler, tmp_path):
        compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")
        assert fake_compiler.calls[0]["input_type"] == "stablehlo"


class TestCompileResult:
    def test_writes_a_real_file(self, fake_compiler, tmp_path):
        out = tmp_path / "nested" / "a.vmfb"
        result = compile_for_target(MLIR, NATIVE, out_path=out)
        assert result.path == out
        assert out.read_bytes() == FAKE_VMFB

    def test_size_matches_the_written_bytes(self, fake_compiler, tmp_path):
        result = compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")
        assert result.size_bytes == len(FAKE_VMFB)

    def test_defaults_to_a_temp_path_when_none_given(self, fake_compiler):
        result = compile_for_target(MLIR, NATIVE)
        assert result.path.exists()
        assert result.path.suffix == ".vmfb"

    def test_spirv_bytes_is_none_for_cpu_targets(self, fake_compiler, tmp_path):
        result = compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")
        assert result.spirv_bytes is None

    def test_clean_compile_is_not_marked_downgraded(self, fake_compiler, tmp_path):
        result = compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")
        assert result.downgraded_stablehlo is False
        assert result.stderr == ""


class TestPortableArtifactDowngrade:
    def test_retry_succeeds_and_is_flagged(self, monkeypatch, tmp_path):
        """Version skew: IREE's bundled StableHLO can be older than jax's."""
        tools = FakeCompilerTools(calls=[], fail_first=True)
        monkeypatch.setattr(compile_mod, "_require_compiler", lambda: tools)
        monkeypatch.setattr(compile_mod, "_downgrade_to_portable", lambda _t: b"portable")

        result = compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")

        assert result.downgraded_stablehlo is True
        assert len(tools.calls) == 2
        assert tools.calls[1]["source"] == b"portable"

    def test_first_failure_is_preserved_in_stderr(self, monkeypatch, tmp_path):
        tools = FakeCompilerTools(calls=[], fail_first=True)
        monkeypatch.setattr(compile_mod, "_require_compiler", lambda: tools)
        monkeypatch.setattr(compile_mod, "_downgrade_to_portable", lambda _t: b"portable")

        result = compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")
        assert "rejected the current-version StableHLO" in result.stderr

    def test_failure_after_downgrade_reports_both_attempts(self, monkeypatch, tmp_path):
        class _AlwaysFails(FakeCompilerTools):
            def compile_str(self, source, *, input_type, extra_args):
                self.calls.append({"source": source})
                msg = "nope"
                raise RuntimeError(msg)

        monkeypatch.setattr(compile_mod, "_require_compiler", lambda: _AlwaysFails(calls=[]))
        monkeypatch.setattr(compile_mod, "_downgrade_to_portable", lambda _t: b"portable")

        with pytest.raises(CompileError) as excinfo:
            compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")

        message = str(excinfo.value)
        assert "direct:" in message
        assert "portable:" in message

    def test_names_the_target_and_backend_on_failure(self, monkeypatch, tmp_path):
        class _AlwaysFails(FakeCompilerTools):
            def compile_str(self, source, *, input_type, extra_args):
                msg = "nope"
                raise RuntimeError(msg)

        monkeypatch.setattr(compile_mod, "_require_compiler", lambda: _AlwaysFails(calls=[]))
        monkeypatch.setattr(compile_mod, "_downgrade_to_portable", lambda _t: b"portable")

        with pytest.raises(CompileError, match="'wasm32'.*'llvm-cpu'"):
            compile_for_target(MLIR, WASM32, out_path=tmp_path / "a.vmfb")


class TestMissingToolchain:
    def test_compiler_import_error_names_the_extra(self, no_toolchain, tmp_path):
        with pytest.raises(CompileError, match=r"pip install xtrax\[export\]"):
            compile_for_target(MLIR, NATIVE, out_path=tmp_path / "a.vmfb")

    def test_runtime_import_error_names_the_extra(self, no_toolchain, tmp_path):
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        with pytest.raises(CompileError, match=r"pip install xtrax\[export\]"):
            run_native_vmfb(artifact)


class TestRunNativeVmfb:
    def test_resolves_the_entry_point_and_returns_its_output(self, fake_runtime, tmp_path):
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        fake_runtime["result"] = np.full((2, 2), 7.0, dtype=np.float32)

        out = run_native_vmfb(artifact, np.zeros((2, 2), dtype=np.float32))

        np.testing.assert_allclose(out, np.full((2, 2), 7.0))
        assert len(fake_runtime["calls"]) == 1

    def test_missing_entry_point_names_it(self, fake_runtime, tmp_path):
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        with pytest.raises(CompileError, match="no entry point 'absent'"):
            run_native_vmfb(artifact, function="absent")


class TestCompare:
    def test_identical_arrays_pass(self):
        a = np.ones((3, 2), dtype=np.float32)
        result = compare(a, a)
        assert result.passed is True
        assert result.max_abs_diff == 0.0

    def test_small_difference_passes_within_tolerance(self):
        a = np.ones((3,), dtype=np.float32)
        result = compare(a, a + 1e-7)
        assert result.passed is True

    def test_large_difference_fails(self):
        a = np.ones((3,), dtype=np.float32)
        result = compare(a, a + 1.0)
        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(1.0)

    def test_shape_mismatch_fails_rather_than_broadcasting(self):
        """A silently broadcast comparison is how a real regression gets missed."""
        result = compare(np.ones((3, 1), dtype=np.float32), np.ones((3, 4), dtype=np.float32))
        assert result.passed is False
        assert result.max_abs_diff == float("inf")
        assert result.shape_expected == (3, 1)
        assert result.shape_actual == (3, 4)

    def test_empty_arrays_compare_equal(self):
        empty = np.zeros((0,), dtype=np.float32)
        assert compare(empty, empty).passed is True

    def test_tolerances_are_recorded(self):
        result = compare(np.ones((2,)), np.ones((2,)), atol=1e-3, rtol=1e-2)
        assert (result.atol, result.rtol) == (1e-3, 1e-2)

    def test_summary_reports_a_shape_mismatch(self):
        result = compare(np.ones((2,)), np.ones((3,)))
        assert "shape mismatch" in result.summary()

    def test_summary_reports_the_max_diff_on_a_pass(self):
        assert "PASS" in compare(np.ones((2,)), np.ones((2,))).summary()


class TestVerifyNativeParity:
    def test_compares_the_artifacts_output_against_the_supplied_expected(
        self, fake_runtime, tmp_path
    ):
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        expected = np.full((2, 2), 3.0, dtype=np.float32)
        fake_runtime["result"] = expected.copy()

        result = verify_native_parity(expected, artifact, [np.zeros((2, 2), dtype=np.float32)])

        assert isinstance(result, ParityResult)
        assert result.passed is True

    def test_detects_a_divergent_artifact(self, fake_runtime, tmp_path):
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        fake_runtime["result"] = np.full((2, 2), 9.0, dtype=np.float32)

        result = verify_native_parity(
            np.full((2, 2), 3.0, dtype=np.float32),
            artifact,
            [np.zeros((2, 2), dtype=np.float32)],
        )
        assert result.passed is False
        assert result.max_abs_diff == pytest.approx(6.0)

    def test_passes_concrete_inputs_through_to_the_artifact(self, fake_runtime, tmp_path):
        artifact = tmp_path / "a.vmfb"
        artifact.write_bytes(FAKE_VMFB)
        fake_runtime["result"] = np.zeros((1,), dtype=np.float32)
        inputs = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]

        verify_native_parity(np.zeros((1,), dtype=np.float32), artifact, inputs)

        assert len(fake_runtime["calls"][0]) == 2


class TestWasm32IsNeverExecuted:
    def test_compiling_wasm32_never_touches_the_runtime(self, fake_compiler, monkeypatch, tmp_path):
        """CODEGEN_ONLY means compiled and nothing more."""

        def _explode(*_a, **_k):
            msg = "wasm32 must never reach the runtime"
            raise AssertionError(msg)

        monkeypatch.setattr(compile_mod, "run_native_vmfb", _explode)
        result = compile_for_target(MLIR, WASM32, out_path=tmp_path / "a.vmfb")
        assert isinstance(result.path, Path)

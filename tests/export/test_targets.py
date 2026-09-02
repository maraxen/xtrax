"""The target registry is plain data and must stay importable without IREE."""

from __future__ import annotations

import subprocess
import sys

import pytest

from xtrax.export.targets import (
    ALL_TARGETS,
    METAL_SPIRV,
    NATIVE,
    VULKAN_SPIRV,
    WASM32,
    Target,
    VerificationLevel,
    target_by_name,
)


class TestRegistryContents:
    def test_the_four_targets_are_registered(self):
        assert ALL_TARGETS == (NATIVE, WASM32, VULKAN_SPIRV, METAL_SPIRV)

    def test_no_target_is_registered_as_validated(self):
        """export_pipeline refuses VALIDATED, having nothing to populate it with."""
        levels = {t.verification_level for t in ALL_TARGETS}
        assert VerificationLevel.VALIDATED not in levels

    def test_only_vulkan_emits_spirv(self):
        """metal-spirv is named for its input dialect; it dumps MSL, not SPIR-V."""
        emitters = {t.name for t in ALL_TARGETS if t.emits_spirv}
        assert emitters == {"vulkan-spirv"}

    @pytest.mark.parametrize("target", ALL_TARGETS)
    def test_no_target_claims_f64(self, target: Target):
        """IREE demotes f64 to f32 everywhere and rewrites the public signature.

        Listing it would promise a dtype no artifact actually carries; the gate
        rejects it up front instead.
        """
        assert "f64" not in target.supported_dtypes
        assert "f64" not in target.optional_dtypes

    def test_executed_target_excludes_bf16(self):
        """IREE's runtime cannot map bf16 buffers to numpy, so it cannot be verified."""
        assert "bf16" not in NATIVE.supported_dtypes

    @pytest.mark.parametrize("target", [WASM32, VULKAN_SPIRV, METAL_SPIRV])
    def test_codegen_only_targets_carry_bf16(self, target: Target):
        """bf16 compiles and its signature is untouched; nothing here is executed."""
        assert "bf16" in target.supported_dtypes

    @pytest.mark.parametrize("target", ALL_TARGETS)
    def test_the_envelope_splits_by_level_not_by_backend(self, target: Target):
        """Measured: every backend compiles the same set; only the runtime differs."""
        expected = NATIVE.supported_dtypes | {"bf16"}
        if target.verification_level is VerificationLevel.EXECUTED:
            expected = NATIVE.supported_dtypes
        assert target.supported_dtypes == expected

    @pytest.mark.parametrize("target", ALL_TARGETS)
    def test_no_optional_dtypes_are_invented(self, target: Target):
        """The machinery stays; nothing populates it without a measured reason."""
        assert target.optional_dtypes == frozenset()

    def test_target_is_not_hashable(self):
        """Documents a real consequence of optional_dtype_features being a Mapping.

        Targets are looked up by name and results are keyed by name, so nothing
        needs to hash one -- but a caller reaching for a set of targets should
        find out here rather than at their own call site.
        """
        with pytest.raises(TypeError, match="unhashable"):
            {NATIVE}  # noqa: B018

    def test_native_is_executed(self):
        assert NATIVE.verification_level is VerificationLevel.EXECUTED

    def test_wasm32_is_codegen_only(self):
        """Executing wasm32 needs an emsdk-built runtime that has no package."""
        assert WASM32.verification_level is VerificationLevel.CODEGEN_ONLY

    def test_both_use_the_llvm_cpu_backend(self):
        assert NATIVE.iree_backend == "llvm-cpu"
        assert WASM32.iree_backend == "llvm-cpu"

    def test_wasm32_carries_the_emscripten_triple(self):
        joined = " ".join(WASM32.extra_compiler_flags)
        assert "wasm32-unknown-emscripten" in joined

    def test_wasm32_sets_target_cpu_explicitly(self):
        """Left unset, IREE falls back to a generic CPU it documents as slow."""
        joined = " ".join(WASM32.extra_compiler_flags)
        assert "--iree-llvmcpu-target-cpu=generic" in joined

    def test_native_targets_the_host(self):
        assert "--iree-llvmcpu-target-cpu=host" in NATIVE.extra_compiler_flags


class TestTargetLookup:
    @pytest.mark.parametrize("target", ALL_TARGETS)
    def test_round_trips_by_name(self, target: Target):
        assert target_by_name(target.name) is target

    def test_unknown_name_raises_and_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="native"):
            target_by_name("cuda")


class TestTargetIsPlainData:
    def test_target_is_frozen(self):
        with pytest.raises(AttributeError):
            NATIVE.name = "other"  # type: ignore[misc]

    def test_importing_the_package_does_not_pull_in_iree(self):
        """AC-1: target selection must work on a base install.

        Run in a fresh interpreter, because the toolchain may well be installed
        in this one -- an in-process check would pass for the wrong reason.
        """
        code = (
            "import sys; import xtrax.export; "
            "mods = [m for m in sys.modules if m == 'iree' or m.startswith('iree.')]; "
            "assert not mods, mods; print('clean')"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "clean" in result.stdout

"""The target registry is plain data and must stay importable without IREE."""

from __future__ import annotations

import subprocess
import sys

import pytest

from xtrax.export.targets import (
    ALL_TARGETS,
    NATIVE,
    WASM32,
    Target,
    VerificationLevel,
    target_by_name,
)


class TestRegistryContents:
    def test_native_and_wasm32_are_registered(self):
        assert ALL_TARGETS == (NATIVE, WASM32)

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

"""SPIR-V identification and extraction from an IREE executable dump.

The dump directory is not homogeneous: metal-spirv writes Metal Shading Language
source alongside nothing else, so the magic filter is what stops a caller being
handed MSL that no shader tool can read.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.export.spirv import (
    SPIRV_MAGIC_BE,
    SPIRV_MAGIC_LE,
    SpirvValidationResult,
    is_spirv,
    spirv_binaries_in,
)

# The first bytes of the MSL IREE dumps: the "#inc" of an #include line.
MSL_HEAD = b"#inc"


class TestIsSpirv:
    @pytest.mark.parametrize("magic", [SPIRV_MAGIC_LE, SPIRV_MAGIC_BE])
    def test_accepts_both_endiannesses(self, magic):
        """The word is stored in the producer's endianness; readers detect which."""
        assert is_spirv(magic + b"rest of the module")

    def test_rejects_metal_shading_language(self):
        assert not is_spirv(MSL_HEAD + b"lude <metal_stdlib>\n")

    def test_rejects_random_bytes(self):
        assert not is_spirv(b"\x00\x01\x02\x03\x04")

    def test_rejects_bytes_too_short_to_carry_magic(self):
        assert not is_spirv(b"\x03\x02")

    def test_rejects_empty(self):
        assert not is_spirv(b"")


class TestSpirvBinariesIn:
    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert spirv_binaries_in(tmp_path / "never-created") == {}

    def test_empty_directory_is_empty(self, tmp_path):
        assert spirv_binaries_in(tmp_path) == {}

    def test_collects_only_the_spirv_files(self, tmp_path):
        (tmp_path / "a.spv").write_bytes(SPIRV_MAGIC_LE + b"aaaa")
        (tmp_path / "b.metal").write_bytes(MSL_HEAD + b"lude <metal_stdlib>")
        (tmp_path / "c.o").write_bytes(b"\x7fELF" + b"cccc")
        found = spirv_binaries_in(tmp_path)
        assert set(found) == {"a.spv"}
        assert found["a.spv"] == SPIRV_MAGIC_LE + b"aaaa"

    def test_ignores_subdirectories(self, tmp_path):
        (tmp_path / "nested").mkdir()
        assert spirv_binaries_in(tmp_path) == {}

    def test_is_sorted_by_name_for_a_stable_mapping(self, tmp_path):
        for name in ("z.spv", "a.spv", "m.spv"):
            (tmp_path / name).write_bytes(SPIRV_MAGIC_LE + name.encode())
        assert list(spirv_binaries_in(tmp_path)) == ["a.spv", "m.spv", "z.spv"]


class TestValidationResultStillExists:
    def test_record_is_constructible(self):
        """Kept for a future validator; no target is registered at VALIDATED."""
        result = SpirvValidationResult(
            valid=False, adapter_type="CPU", backend="Vulkan", device_name="llvmpipe", error="nope"
        )
        assert result.valid is False


class TestRealToolchainExtraction:
    """Against real IREE: which backend actually yields SPIR-V, and which does not."""

    def _mlir(self):
        return jax.export.export(jax.jit(lambda x: x * x + x))(
            jax.ShapeDtypeStruct((8,), jnp.float32)
        ).mlir_module()

    def test_vulkan_spirv_populates_spirv_bytes(self):
        pytest.importorskip("iree.compiler")
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import VULKAN_SPIRV

        result = compile_for_target(self._mlir(), VULKAN_SPIRV)
        assert result.spirv_bytes, "vulkan-spirv must yield at least one module"
        for blob in result.spirv_bytes.values():
            assert is_spirv(blob)

    def test_metal_spirv_yields_none_despite_its_name(self):
        """It dumps MSL, so there is no SPIR-V to extract -- None, not an empty dict."""
        pytest.importorskip("iree.compiler")
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import METAL_SPIRV

        assert compile_for_target(self._mlir(), METAL_SPIRV).spirv_bytes is None

    def test_native_yields_none(self):
        pytest.importorskip("iree.compiler")
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import NATIVE

        assert compile_for_target(self._mlir(), NATIVE).spirv_bytes is None


class TestDumpWiringWithoutTheToolchain:
    """The dump flag and the spirv_bytes contract, exercised through the fake."""

    MLIR = "module {}"

    def test_a_spirv_target_asks_for_the_executable_dump(self, fake_compiler):
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import VULKAN_SPIRV

        compile_for_target(self.MLIR, VULKAN_SPIRV)
        args = fake_compiler.calls[0]["extra_args"]
        assert any(a.startswith("--iree-hal-dump-executable-binaries-to=") for a in args), args

    @pytest.mark.parametrize("target_name", ["native", "wasm32", "metal-spirv"])
    def test_other_targets_do_not_ask_for_it(self, fake_compiler, target_name):
        """llvm-cpu writes object files, and metal-spirv writes MSL. Neither is a shader."""
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import target_by_name

        compile_for_target(self.MLIR, target_by_name(target_name))
        args = fake_compiler.calls[0]["extra_args"]
        assert not any(a.startswith("--iree-hal-dump-executable-binaries-to=") for a in args)

    def test_asked_but_found_none_is_an_empty_dict_not_none(self, fake_compiler):
        """The fake writes no dump, so the distinction is visible.

        None means "this target emits no SPIR-V"; {} means "it does, and this
        compile produced none" -- a real difference when debugging.
        """
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import VULKAN_SPIRV

        assert compile_for_target(self.MLIR, VULKAN_SPIRV).spirv_bytes == {}

    def test_a_non_emitting_target_reports_none(self, fake_compiler):
        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import NATIVE

        assert compile_for_target(self.MLIR, NATIVE).spirv_bytes is None

    def test_the_dump_directory_does_not_outlive_the_compile(self, fake_compiler):
        """It is a TemporaryDirectory, so the bytes must be read before it closes."""
        from pathlib import Path

        from xtrax.export.compile import compile_for_target
        from xtrax.export.targets import VULKAN_SPIRV

        compile_for_target(self.MLIR, VULKAN_SPIRV)
        args = fake_compiler.calls[0]["extra_args"]
        flag = next(a for a in args if a.startswith("--iree-hal-dump-executable-binaries-to="))
        assert not Path(flag.split("=", 1)[1]).exists(), "temp dump dir should be cleaned up"

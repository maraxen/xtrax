"""Loading checkpoint weights into a target's dtype envelope.

No network: the checkpoints here are real safetensors files written to tmp_path,
and the download path is exercised with a fake ``huggingface_hub`` module.
"""

from __future__ import annotations

import sys
import types

import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export.hf_weights import (
    HFWeightsError,
    LoadedWeights,
    WeightReport,
    load_hf_weights,
)
from xtrax.export.safety import DtypeNotSupportedError
from xtrax.export.targets import NATIVE, WASM32, Target, VerificationLevel

safetensors_numpy = pytest.importorskip("safetensors.numpy")


def _write(path, tensors):
    safetensors_numpy.save_file(tensors, str(path))
    return path


@pytest.fixture
def bf16_checkpoint(tmp_path):
    return _write(
        tmp_path / "model.safetensors",
        {
            "layer.0.weight": np.ones((2, 3), dtype=jnp.bfloat16),
            "layer.0.bias": np.zeros((3,), dtype=jnp.bfloat16),
            "layer.1.weight": np.ones((3, 2), dtype=np.float32),
        },
    )


class TestCasting:
    def test_bf16_is_cast_for_the_executed_target(self, bf16_checkpoint):
        """native cannot run bf16 at all, so exporting one means casting first."""
        loaded = load_hf_weights(target=NATIVE, local_path=bf16_checkpoint)
        assert all(t.dtype == jnp.float32 for t in loaded.tensors.values())

    def test_every_cast_leaf_is_reported_not_just_a_sample(self, bf16_checkpoint):
        """The spike truncated this to two entries, making the report a sample."""
        loaded = load_hf_weights(target=NATIVE, local_path=bf16_checkpoint)
        assert len(loaded.report.dtypes_cast) == 2
        assert all("bf16 -> f32" in entry for entry in loaded.report.dtypes_cast)

    def test_accepted_dtypes_are_left_alone(self, bf16_checkpoint):
        loaded = load_hf_weights(target=NATIVE, local_path=bf16_checkpoint)
        assert "layer.1.weight" not in " ".join(loaded.report.dtypes_cast)

    def test_codegen_only_target_keeps_bf16(self, bf16_checkpoint):
        """wasm32 carries bf16 fine; nothing is executed, so nothing needs casting."""
        loaded = load_hf_weights(target=WASM32, local_path=bf16_checkpoint)
        assert loaded.report.dtypes_cast == ()
        assert loaded.tensors["layer.0.weight"].dtype == jnp.bfloat16

    def test_report_counts_every_tensor_seen(self, bf16_checkpoint):
        loaded = load_hf_weights(target=NATIVE, local_path=bf16_checkpoint)
        assert loaded.report.tensors_seen == 3

    def test_report_names_the_source(self, bf16_checkpoint):
        loaded = load_hf_weights(target=NATIVE, local_path=bf16_checkpoint)
        assert str(bf16_checkpoint) == loaded.report.source

    def test_returns_the_documented_records(self, bf16_checkpoint):
        loaded = load_hf_weights(target=NATIVE, local_path=bf16_checkpoint)
        assert isinstance(loaded, LoadedWeights)
        assert isinstance(loaded.report, WeightReport)


class TestF64IsNeverSilentlyDowncast:
    def test_f64_tensor_raises(self, tmp_path):
        path = _write(tmp_path / "m.safetensors", {"w": np.ones((2, 2), dtype=np.float64)})
        with pytest.raises(DtypeNotSupportedError, match="f64"):
            load_hf_weights(target=NATIVE, local_path=path)

    def test_the_message_says_why_rather_than_just_no(self, tmp_path):
        path = _write(tmp_path / "m.safetensors", {"w": np.ones((2, 2), dtype=np.float64)})
        with pytest.raises(DtypeNotSupportedError) as excinfo:
            load_hf_weights(target=NATIVE, local_path=path)
        assert "demotes it" in str(excinfo.value)


class TestArgumentGuards:
    def test_target_is_required(self, bf16_checkpoint):
        with pytest.raises(ValueError, match="target is required"):
            load_hf_weights(local_path=bf16_checkpoint)

    def test_a_source_is_required(self):
        with pytest.raises(ValueError, match="repo_id"):
            load_hf_weights(target=NATIVE)

    def test_an_unusable_cast_target_is_refused(self, bf16_checkpoint):
        """Casting into a dtype the target also rejects would not help."""
        with pytest.raises(DtypeNotSupportedError, match="would not help"):
            load_hf_weights(target=NATIVE, local_path=bf16_checkpoint, cast_to="f64")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(HFWeightsError, match="no such checkpoint"):
            load_hf_weights(target=NATIVE, local_path=tmp_path / "absent.safetensors")

    def test_empty_checkpoint_raises(self, monkeypatch, tmp_path):
        """safetensors will not write a tensor-less file, so stub the read."""
        from xtrax.export import hf_weights as mod

        path = _write(tmp_path / "m.safetensors", {"w": np.zeros((1,), np.float32)})
        monkeypatch.setattr(mod, "_read_safetensors", lambda *a, **k: {})
        with pytest.raises(HFWeightsError, match="no tensors"):
            load_hf_weights(target=NATIVE, local_path=path)


class TestDownloadPath:
    def test_uses_the_hub_when_no_local_path_is_given(self, monkeypatch, bf16_checkpoint):
        """The parent package must exist too, or the import works only by accident."""
        calls: list[dict] = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return str(bf16_checkpoint)

        fake = types.ModuleType("huggingface_hub")
        fake.hf_hub_download = fake_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

        loaded = load_hf_weights("org/model", target=NATIVE)
        assert calls == [{"repo_id": "org/model", "filename": "model.safetensors"}]
        assert loaded.report.source == "org/model/model.safetensors"

    def test_a_hub_failure_is_wrapped_with_the_repo_name(self, monkeypatch):
        fake = types.ModuleType("huggingface_hub")

        def boom(**_kwargs):
            raise RuntimeError("404 not found")

        fake.hf_hub_download = boom
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

        with pytest.raises(HFWeightsError, match="org/missing"):
            load_hf_weights("org/missing", target=NATIVE)


class TestMissingOptionalDependencies:
    """The lazy imports must name the extra, not surface a bare ImportError."""

    def test_missing_safetensors_names_the_extra(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "safetensors.numpy", None)
        with pytest.raises(HFWeightsError, match="xtrax\\[export\\]"):
            load_hf_weights(target=NATIVE, local_path=tmp_path / "m.safetensors")

    def test_missing_hub_names_the_repo_and_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(HFWeightsError, match="org/model"):
            load_hf_weights("org/model", target=NATIVE)


class TestOptionalDtypeUnlocking:
    def test_an_optional_dtype_is_kept_when_its_feature_is_requested(self, tmp_path):
        """The cast rule honours request_features wherever the dtype was found."""
        narrow = Target(
            name="narrow",
            iree_backend="llvm-cpu",
            verification_level=VerificationLevel.CODEGEN_ONLY,
            supported_dtypes=frozenset({"f32"}),
            optional_dtypes=frozenset({"f16"}),
            optional_dtype_features={"f16": "shader-f16"},
        )
        path = _write(tmp_path / "m.safetensors", {"w": np.ones((2,), dtype=np.float16)})

        kept = load_hf_weights(
            target=narrow, local_path=path, request_features=frozenset({"shader-f16"})
        )
        assert kept.report.dtypes_cast == ()
        assert kept.tensors["w"].dtype == jnp.float16

        cast = load_hf_weights(target=narrow, local_path=path)
        assert cast.tensors["w"].dtype == jnp.float32


class TestNumpyNameExpansion:
    """cast_to is a short name; it has to expand back to something JAX accepts."""

    @pytest.mark.parametrize(
        ("short", "dtype"),
        [("f32", jnp.float32), ("f16", jnp.float16), ("bf16", jnp.bfloat16), ("i32", jnp.int32)],
    )
    def test_short_names_round_trip_to_real_dtypes(self, tmp_path, short, dtype):
        target = Target(
            name="t",
            iree_backend="llvm-cpu",
            verification_level=VerificationLevel.CODEGEN_ONLY,
            supported_dtypes=frozenset({short}),
        )
        path = _write(tmp_path / "m.safetensors", {"w": np.ones((2,), dtype=np.int8)})
        loaded = load_hf_weights(target=target, local_path=path, cast_to=short)
        assert loaded.tensors["w"].dtype == jnp.dtype(dtype)

    def test_bool_is_expanded_too(self, tmp_path):
        target = Target(
            name="t",
            iree_backend="llvm-cpu",
            verification_level=VerificationLevel.CODEGEN_ONLY,
            supported_dtypes=frozenset({"bool"}),
        )
        path = _write(tmp_path / "m.safetensors", {"w": np.ones((2,), dtype=np.int8)})
        loaded = load_hf_weights(target=target, local_path=path, cast_to="bool")
        assert loaded.tensors["w"].dtype == jnp.bool_

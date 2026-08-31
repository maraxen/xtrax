"""Tests for the IREE export spike (task 260831_iree-wasm-webgpu-export).

These must pass with **no IREE installed and no network access**, so the toolchain
is faked via `sys.modules` injection -- the same pattern as
`test_smoke_outlines_constrained_decode.py`, chosen over `importorskip` so the logic
is actually verified in normal CI instead of silently skipping.

The composer and gate tests are real, though: they run genuine JAX.
"""

import sys
import types
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from scripts.iree_export_spike.composer import ComposerError, compose_exportable
from scripts.iree_export_spike.export_safety import (
    ExportUnsafeError,
    assert_plan_export_safe,
    check_plan_export_safety,
)
from scripts.iree_export_spike.hf_weights import TinyMLP, random_mlp
from scripts.iree_export_spike.parity import compare
from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling.plan import AxisSpec, BatchPlanner
from xtrax.tiling.strategy import Bucket, SafeMap, Vmap, WhileCarry


def _decision(name: str, strategy: Any) -> Any:
    """A duck-typed stand-in for AxisDecision (the gate is structural)."""
    spec = AxisSpec(name=name, cardinality=8, default_batch_size=4)
    return types.SimpleNamespace(spec=spec, batch_size=4, reasoning="test", strategy=strategy)


class TestExportSafetyGate:
    def test_vmap_axis_with_no_boundary_is_exportable(self) -> None:
        assert check_plan_export_safety([_decision("batch", Vmap())]) == []

    def test_safemap_axis_is_exportable(self) -> None:
        # SafeMap lowers to jax.lax.map, a real XLA loop -- not a Python unroll.
        assert check_plan_export_safety([_decision("batch", SafeMap(batch_size=4))]) == []

    def test_fuse_only_boundary_is_exportable(self) -> None:
        boundary = AxisBoundary(fuse=lambda ys: jnp.sum(ys, axis=0))
        assert check_plan_export_safety([_decision("batch", Vmap())], {"batch": boundary}) == []

    def test_bucket_is_blocked_as_host_tier(self) -> None:
        blockers = check_plan_export_safety([_decision("seq", Bucket(boundaries=(4, 8)))])
        assert len(blockers) == 1
        assert blockers[0].rule == "strategy"
        assert "host-tier" in blockers[0].detail

    def test_whilecarry_is_blocked_as_unbounded(self) -> None:
        blockers = check_plan_export_safety([_decision("iter", WhileCarry())])
        assert len(blockers) == 1
        assert "unbounded" in blockers[0].detail

    @pytest.mark.parametrize("slot", ["tap", "sink"])
    def test_tap_or_sink_pierces_the_boundary(self, slot: str) -> None:
        boundary = AxisBoundary(**{slot: lambda y: y})
        blockers = check_plan_export_safety([_decision("batch", Vmap())], {"batch": boundary})
        assert len(blockers) == 1
        assert blockers[0].rule == "boundary"
        assert "io_callback" in blockers[0].detail

    def test_assert_raises_and_names_every_blocker(self) -> None:
        decisions = [
            _decision("seq", Bucket(boundaries=(4, 8))),
            _decision("iter", WhileCarry()),
        ]
        with pytest.raises(ExportUnsafeError) as exc:
            assert_plan_export_safe(decisions)
        message = str(exc.value)
        assert "seq" in message
        assert "iter" in message

    def test_assert_passes_on_clean_plan(self) -> None:
        assert_plan_export_safe([_decision("batch", Vmap())]) is None


class TestComposer:
    def _plan(self, cardinality: int, batch_size: int) -> Any:
        spec = AxisSpec(name="batch", cardinality=cardinality, default_batch_size=batch_size)
        return BatchPlanner().plan([spec])

    def test_composed_vmap_matches_a_manual_loop(self) -> None:
        model = random_mlp(4, 6, 2, seed=1)
        plan = self._plan(cardinality=8, batch_size=16)  # cardinality <= bs -> Vmap
        assert type(plan.decisions[0].strategy).__name__ == "Vmap"

        forward = compose_exportable(model, plan)
        xs = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4) / 32.0

        got = forward(xs)
        want = jnp.stack([model(x) for x in xs])
        assert jnp.allclose(got, want, atol=1e-6)

    def test_composed_safemap_matches_vmap_semantics(self) -> None:
        model = random_mlp(4, 6, 2, seed=2)
        plan = self._plan(cardinality=16, batch_size=4)  # cardinality > bs -> SafeMap
        assert type(plan.decisions[0].strategy).__name__ == "SafeMap"

        forward = compose_exportable(model, plan)
        xs = jnp.arange(16 * 4, dtype=jnp.float32).reshape(16, 4) / 64.0

        assert jnp.allclose(forward(xs), jax.vmap(model)(xs), atol=1e-6)

    def test_fuse_is_applied_once_after_the_axis(self) -> None:
        model = random_mlp(4, 6, 2, seed=3)
        plan = self._plan(cardinality=8, batch_size=16)
        boundaries = {"batch": AxisBoundary(fuse=lambda ys: jnp.mean(ys, axis=0))}

        forward = compose_exportable(model, plan, boundaries)
        xs = jnp.ones((8, 4), dtype=jnp.float32)

        got = forward(xs)
        assert got.shape == (2,)
        assert jnp.allclose(got, jnp.mean(jax.vmap(model)(xs), axis=0), atol=1e-6)

    def test_tap_bearing_boundary_is_refused_before_tracing(self) -> None:
        model = random_mlp(4, 6, 2, seed=4)
        plan = self._plan(cardinality=8, batch_size=16)
        boundaries = {"batch": AxisBoundary(tap=lambda y: y)}

        with pytest.raises(ExportUnsafeError, match="io_callback"):
            compose_exportable(model, plan, boundaries)

    def test_multi_axis_is_refused_as_a_spike_limitation(self) -> None:
        model = random_mlp(4, 6, 2, seed=5)
        plan = BatchPlanner().plan(
            [
                AxisSpec(name="a", cardinality=4, default_batch_size=8),
                AxisSpec(name="b", cardinality=4, default_batch_size=8),
            ]
        )
        with pytest.raises(ComposerError, match="single axis"):
            compose_exportable(model, plan)


class TestExportability:
    """The composed callable must actually reach StableHLO."""

    def test_composed_callable_lowers_to_stablehlo_with_weights_baked_in(self) -> None:
        model = random_mlp(4, 6, 2, seed=6)
        plan = BatchPlanner().plan([AxisSpec(name="batch", cardinality=8, default_batch_size=16)])
        forward = compose_exportable(model, plan)

        aval = jax.ShapeDtypeStruct((8, 4), jnp.float32)
        exported = jax.export.export(jax.jit(forward))(aval)
        text = exported.mlir_module()

        assert "stablehlo" in text
        # Weights are closure constants, so the traced signature takes ONE input.
        assert len(exported.in_avals) == 1
        assert exported.in_avals[0].shape == (8, 4)

    def test_no_io_callback_survives_in_the_exported_module(self) -> None:
        model = random_mlp(4, 6, 2, seed=7)
        plan = BatchPlanner().plan([AxisSpec(name="batch", cardinality=8, default_batch_size=16)])
        forward = compose_exportable(model, plan)
        exported = jax.export.export(jax.jit(forward))(jax.ShapeDtypeStruct((8, 4), jnp.float32))
        # A host callback lowers to a custom_call; a fuse-only pipeline has none.
        assert "custom_call" not in exported.mlir_module()


class TestParity:
    def test_identical_arrays_pass(self) -> None:
        a = jnp.ones((3, 2))
        assert compare(a, a).passed

    def test_shape_mismatch_fails_rather_than_broadcasting(self) -> None:
        result = compare(jnp.ones((3, 2)), jnp.ones((3,)))
        assert not result.passed
        assert "shape mismatch" in result.summary()

    def test_small_fusion_level_difference_still_passes(self) -> None:
        a = jnp.ones((4,), dtype=jnp.float32)
        b = a + 1e-7
        assert compare(a, b, atol=1e-5).passed

    def test_large_difference_fails(self) -> None:
        a = jnp.zeros((4,), dtype=jnp.float32)
        result = compare(a, a + 0.5, atol=1e-5)
        assert not result.passed
        assert result.max_abs_diff == pytest.approx(0.5)


class TestCompileIREEWithoutToolchain:
    """The IREE layer must fail with a clear, actionable message when absent."""

    def test_compile_reports_missing_compiler(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.iree_export_spike import compile_iree

        monkeypatch.setitem(sys.modules, "iree", None)
        monkeypatch.setitem(sys.modules, "iree.compiler", None)

        with pytest.raises(compile_iree.IREECompileError, match="export-spike"):
            compile_iree.compile_stablehlo("module {}", tmp_path / "x.vmfb")

    def test_unknown_target_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.iree_export_spike import compile_iree

        fake_tools = types.SimpleNamespace(compile_str=lambda *_a, **_k: b"\x00binary")
        monkeypatch.setattr(compile_iree, "_require_compiler", lambda: fake_tools)

        with pytest.raises(compile_iree.IREECompileError, match="unknown target"):
            compile_iree.compile_stablehlo("module {}", tmp_path / "x.vmfb", target="webgpu")

    def test_wasm32_target_passes_the_emscripten_triple(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.iree_export_spike import compile_iree

        seen: dict[str, Any] = {}

        def _compile_str(text: Any, input_type: str, extra_args: list[str]) -> bytes:
            seen["input_type"] = input_type
            seen["extra_args"] = list(extra_args)
            return b"\x00fake-vmfb"

        monkeypatch.setattr(
            compile_iree,
            "_require_compiler",
            lambda: types.SimpleNamespace(compile_str=_compile_str),
        )

        out = tmp_path / "wasm.vmfb"
        result = compile_iree.compile_stablehlo("module {}", out, target=compile_iree.WASM32_TARGET)

        assert seen["input_type"] == "stablehlo"
        assert any("wasm32-unknown-emscripten" in a for a in seen["extra_args"])
        assert result.size_bytes == len(b"\x00fake-vmfb")
        assert out.read_bytes() == b"\x00fake-vmfb"
        assert not result.downgraded_stablehlo

    def test_native_target_pins_the_host_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unset target-cpu makes IREE warn and emit generic, slow code."""
        from scripts.iree_export_spike import compile_iree

        seen: dict[str, Any] = {}

        def _compile_str(text: Any, input_type: str, extra_args: list[str]) -> bytes:
            seen["extra_args"] = list(extra_args)
            return b"\x00native"

        monkeypatch.setattr(
            compile_iree,
            "_require_compiler",
            lambda: types.SimpleNamespace(compile_str=_compile_str),
        )
        compile_iree.compile_stablehlo("module {}", tmp_path / "n.vmfb")
        assert any("target-cpu=host" in a for a in seen["extra_args"])

    def test_runner_resolves_the_module_name_from_the_vmfb(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression: the name is jax-derived (jit_<fn>), never literally 'module'."""
        from scripts.iree_export_spike import compile_iree

        vmfb = tmp_path / "m.vmfb"
        vmfb.write_bytes(b"\x00fake")

        class _Loaded(dict):
            pass

        loaded = _Loaded({"main": lambda *a: "called"})
        fake_module = types.SimpleNamespace(name="jit__run_map")

        fake_ireert = types.ModuleType("iree.runtime")
        fake_ireert.Config = lambda _n: object()  # type: ignore[attr-defined]
        fake_ireert.VmModule = types.SimpleNamespace(  # type: ignore[attr-defined]
            mmap=lambda _inst, _p: fake_module
        )
        fake_ireert.SystemContext = lambda config: types.SimpleNamespace(  # type: ignore[attr-defined]
            instance=object(),
            add_vm_module=lambda _m: None,
            modules={"jit__run_map": loaded},
        )
        monkeypatch.setitem(sys.modules, "iree.runtime", fake_ireert)

        assert compile_iree.run_native_vmfb(vmfb) == "called"

    def test_missing_entry_point_is_reported_with_the_module_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.iree_export_spike import compile_iree

        vmfb = tmp_path / "m.vmfb"
        vmfb.write_bytes(b"\x00fake")
        fake_module = types.SimpleNamespace(name="jit__run_map")

        fake_ireert = types.ModuleType("iree.runtime")
        fake_ireert.Config = lambda _n: object()  # type: ignore[attr-defined]
        fake_ireert.VmModule = types.SimpleNamespace(  # type: ignore[attr-defined]
            mmap=lambda _inst, _p: fake_module
        )
        fake_ireert.SystemContext = lambda config: types.SimpleNamespace(  # type: ignore[attr-defined]
            instance=object(),
            add_vm_module=lambda _m: None,
            modules={"jit__run_map": {}},
        )
        monkeypatch.setitem(sys.modules, "iree.runtime", fake_ireert)

        with pytest.raises(compile_iree.IREECompileError, match="jit__run_map"):
            compile_iree.run_native_vmfb(vmfb)

    def test_portable_artifact_downgrade_is_attempted_on_first_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.iree_export_spike import compile_iree

        calls: list[Any] = []

        def _compile_str(text: Any, input_type: str, extra_args: list[str]) -> bytes:
            calls.append(text)
            if len(calls) == 1:
                raise RuntimeError("unregistered op: stablehlo.brand_new_thing")
            return b"\x00downgraded"

        monkeypatch.setattr(
            compile_iree,
            "_require_compiler",
            lambda: types.SimpleNamespace(compile_str=_compile_str),
        )
        monkeypatch.setattr(compile_iree, "_downgrade_to_portable", lambda _text: b"portable-bytes")

        result = compile_iree.compile_stablehlo("module {}", tmp_path / "x.vmfb")

        assert len(calls) == 2
        assert calls[1] == b"portable-bytes"
        assert result.downgraded_stablehlo


class TestTinyMLP:
    def test_is_an_equinox_module_with_array_leaves(self) -> None:
        model = random_mlp(4, 6, 2, seed=8)
        assert isinstance(model, eqx.Module)
        leaves = jax.tree_util.tree_leaves(model)
        assert len(leaves) == 4
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_forward_shape(self) -> None:
        model = random_mlp(4, 6, 2, seed=9)
        assert model(jnp.ones((4,), dtype=jnp.float32)).shape == (2,)

    def test_is_deterministic_for_a_seed(self) -> None:
        assert jnp.allclose(random_mlp(4, 6, 2, seed=10).w1, random_mlp(4, 6, 2, seed=10).w1)


class TestHFWeightsWithoutNetwork:
    def test_missing_huggingface_hub_is_reported_actionably(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.iree_export_spike import hf_weights

        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(hf_weights.HFWeightsError, match="export-spike"):
            hf_weights.mlp_from_hf("fake/repo", in_dim=4, hidden=6, out_dim=2)

    def test_weights_are_cast_and_fitted_from_a_fake_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from scripts.iree_export_spike import hf_weights

        tensors = {
            "a.weight": np.full((3, 3), 2.0, dtype=np.float16),
            "b.weight": np.full((5, 5), 3.0, dtype=np.float16),
        }
        monkeypatch.setattr(hf_weights, "_load_safetensors", lambda *_a, **_k: tensors)

        model, report = hf_weights.mlp_from_hf("fake/repo", in_dim=4, hidden=6, out_dim=2)

        assert isinstance(model, TinyMLP)
        assert model.w1.shape == (4, 6)
        assert model.w2.shape == (6, 2)
        assert model.w1.dtype == jnp.float32  # explicit cast from f16
        # The real 3x3 block was copied in; the rest is zero-padded.
        assert model.w1[0, 0] == pytest.approx(2.0)
        assert model.w1[3, 5] == pytest.approx(0.0)
        assert report.tensors_seen == 2
        assert any("float16" in c for c in report.dtypes_cast)

    def test_checkpoint_without_enough_matrices_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from scripts.iree_export_spike import hf_weights

        monkeypatch.setattr(
            hf_weights,
            "_load_safetensors",
            lambda *_a, **_k: {"only.bias": np.zeros((4,), dtype=np.float32)},
        )
        with pytest.raises(hf_weights.HFWeightsError, match="need at least 2"):
            hf_weights.mlp_from_hf("fake/repo", in_dim=4, hidden=6, out_dim=2)

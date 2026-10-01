"""Artifact size budgets, to catch a codegen regression that still compiles.

``SIZE_BUDGET_BYTES`` is a single flat 32 KiB ceiling applied to every target,
not a per-target multiple of its measurement. That is deliberate -- the budget
exists to catch an order-of-magnitude codegen blowup, not to track each
target's byte count -- but it means the headroom is whatever the flat ceiling
happens to leave, and it varies: roughly 2.3x for metal-spirv up to 3.1x for
wasm32. (An earlier revision of this docstring described "roughly 2.5x
headroom", which read as a per-target rule that was never implemented.)

The measurements the ceiling was chosen against, from the fixture below
(260902, IREE 3.11.0, linux x86_64):

    native           13889 B
    wasm32           10632 B
    vulkan-spirv     12577 B   (4780 B of extracted SPIR-V)
    metal-spirv      14275 B

    native-portable  11985 B   (260910, same toolchain)

    onnx               775 B   (260930, jax2onnx 0.17.0, opset 23)

``onnx`` is two orders of magnitude smaller because an ONNX file carries no
generated code: it is the graph (three nodes here) plus its initializers, and
this fixture's weights are 48 float32s, 192 B. The 1024 B floor measures "did a
kernel get generated", which an ONNX file never contains, so ``onnx`` gets its
own floor of 512 B instead -- still above the weights alone, so a graph that
kept its initializers but lost its ops falls under it.

``native-portable`` is smaller than ``native`` because x86-64-v2 codegen cannot
use the wider vector ISA the host CPU offers -- on the machine that measured
this, ``native`` resolved to ``cpu = znver5`` with the full AVX-512 feature set
while ``native-portable`` resolved to the twelve-feature v2 baseline. Note the
``native`` row is the original 260902 figure; the same fixture measures 13897 B
today, which is the ordinary drift a generous flat ceiling exists to absorb.

A floor is checked as well as a ceiling. An artifact that suddenly collapses to
a few hundred bytes still "compiles" and would sail past a ceiling-only budget,
while meaning the kernel was optimised away to nothing.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import pytest

from xtrax.export.pipeline import export_pipeline
from xtrax.export.targets import ALL_TARGETS, VULKAN_SPIRV, Backend
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

SIZE_BUDGET_BYTES = {
    "native": 32 * 1024,
    "native-portable": 32 * 1024,
    "wasm32": 32 * 1024,
    "vulkan-spirv": 32 * 1024,
    "metal-spirv": 32 * 1024,
    "onnx": 32 * 1024,
}

# Extracted shader bytes, budgeted separately from the containing vmfb.
SPIRV_BUDGET_BYTES = {"vulkan-spirv": 16 * 1024}

# Anything smaller than this did not compile a real kernel.
SIZE_FLOOR_BYTES = 1024

# Per-target floors where the artifact carries no kernel code (see docstring).
FLOOR_OVERRIDE_BYTES = {"onnx": 512}

W1 = jnp.asarray(jax.random.normal(jax.random.key(0), (4, 8)), dtype=jnp.float32)
W2 = jnp.asarray(jax.random.normal(jax.random.key(1), (8, 2)), dtype=jnp.float32)


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _fixture_model(x):
    return jnp.tanh(x @ W1) @ W2


def _installed(*modules: str) -> bool:
    # find_spec on a dotted name imports the parent, and raises rather than
    # returning None when the parent package itself is missing.
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except ModuleNotFoundError:
        return False


def _available_targets() -> tuple:
    """Every registered target whose toolchain is installed here.

    Each toolchain gates only its own targets: an `export`-only environment keeps
    every IREE budget, and an `onnx`-only one keeps the onnx budget.
    """
    have_iree = _installed("iree.compiler", "iree.runtime")
    have_onnx = _installed("jax2onnx", "onnx", "onnxruntime")
    return tuple(t for t in ALL_TARGETS if (have_onnx if t.backend is Backend.ONNX else have_iree))


@pytest.fixture(scope="module")
def exported():
    targets = _available_targets()
    if not targets:
        pytest.skip("neither the IREE nor the ONNX toolchain is installed")
    plan = _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name="batch", cardinality=8, default_batch_size=0),
                batch_size=0,
                reasoning="size budget",
                strategy=Vmap(),
            )
        ]
    )
    xs = jnp.ones((8, 4), dtype=jnp.float32)
    return export_pipeline(
        _fixture_model,
        plan,
        (jax.ShapeDtypeStruct(xs.shape, xs.dtype),),
        (xs,),
        targets=targets,
        reference_fn=lambda inputs: jax.vmap(_fixture_model)(inputs[0]),
    )


def _result(exported, name):
    if name not in exported:
        pytest.skip(f"{name}'s toolchain is not installed here")
    return exported[name]


class TestSizeBudgets:
    @pytest.mark.parametrize("name", sorted(SIZE_BUDGET_BYTES))
    def test_artifact_is_within_budget(self, exported, name):
        result = _result(exported, name)
        budget = SIZE_BUDGET_BYTES[name]
        assert result.size_bytes <= budget, (
            f"{name} artifact grew to {result.size_bytes} B, over its {budget} B budget"
        )

    @pytest.mark.parametrize("name", sorted(SIZE_BUDGET_BYTES))
    def test_artifact_is_not_suspiciously_empty(self, exported, name):
        """A collapsed artifact compiles fine and passes a ceiling-only budget."""
        result = _result(exported, name)
        floor = FLOOR_OVERRIDE_BYTES.get(name, SIZE_FLOOR_BYTES)
        assert result.size_bytes >= floor, (
            f"{name} artifact is only {result.size_bytes} B -- did the kernel survive?"
        )

    def test_every_registered_target_has_a_budget(self):
        """A new target must arrive with a measured budget, not slip through."""
        assert {t.name for t in ALL_TARGETS} == set(SIZE_BUDGET_BYTES)

    def test_extracted_spirv_is_within_its_own_budget(self, exported):
        blobs = _result(exported, VULKAN_SPIRV.name).spirv_bytes
        assert blobs, "vulkan-spirv must yield SPIR-V to budget"
        total = sum(len(b) for b in blobs.values())
        budget = SPIRV_BUDGET_BYTES[VULKAN_SPIRV.name]
        assert SIZE_FLOOR_BYTES <= total <= budget, f"{total} B of SPIR-V, budget {budget} B"

    def test_only_vulkan_carries_spirv(self, exported):
        _result(exported, VULKAN_SPIRV.name)
        carriers = {name for name, r in exported.items() if r.spirv_bytes}
        assert carriers == {VULKAN_SPIRV.name}

"""Artifact size budgets, to catch a codegen regression that still compiles.

The budgets are seeded from real measurements of the fixture below (260902,
IREE 3.11.0, linux x86_64), with roughly 2.5x headroom so ordinary toolchain
drift does not fail the suite:

    native        13889 B
    wasm32        10632 B
    vulkan-spirv  12577 B   (4780 B of extracted SPIR-V)
    metal-spirv   14275 B

A floor is checked as well as a ceiling. An artifact that suddenly collapses to
a few hundred bytes still "compiles" and would sail past a ceiling-only budget,
while meaning the kernel was optimised away to nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.export.pipeline import export_pipeline
from xtrax.export.targets import ALL_TARGETS, VULKAN_SPIRV
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

SIZE_BUDGET_BYTES = {
    "native": 32 * 1024,
    "wasm32": 32 * 1024,
    "vulkan-spirv": 32 * 1024,
    "metal-spirv": 32 * 1024,
}

# Extracted shader bytes, budgeted separately from the containing vmfb.
SPIRV_BUDGET_BYTES = {"vulkan-spirv": 16 * 1024}

# Anything smaller than this did not compile a real kernel.
SIZE_FLOOR_BYTES = 1024

W1 = jnp.asarray(jax.random.normal(jax.random.key(0), (4, 8)), dtype=jnp.float32)
W2 = jnp.asarray(jax.random.normal(jax.random.key(1), (8, 2)), dtype=jnp.float32)


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _fixture_model(x):
    return jnp.tanh(x @ W1) @ W2


@pytest.fixture(scope="module")
def exported():
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
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
        targets=tuple(ALL_TARGETS),
        reference_fn=lambda inputs: jax.vmap(_fixture_model)(inputs[0]),
    )


class TestSizeBudgets:
    @pytest.mark.parametrize("name", sorted(SIZE_BUDGET_BYTES))
    def test_artifact_is_within_budget(self, exported, name):
        result = exported[name]
        budget = SIZE_BUDGET_BYTES[name]
        assert result.size_bytes <= budget, (
            f"{name} vmfb grew to {result.size_bytes} B, over its {budget} B budget"
        )

    @pytest.mark.parametrize("name", sorted(SIZE_BUDGET_BYTES))
    def test_artifact_is_not_suspiciously_empty(self, exported, name):
        """A collapsed artifact compiles fine and passes a ceiling-only budget."""
        result = exported[name]
        assert result.size_bytes >= SIZE_FLOOR_BYTES, (
            f"{name} vmfb is only {result.size_bytes} B -- did the kernel survive?"
        )

    def test_every_registered_target_has_a_budget(self):
        """A new target must arrive with a measured budget, not slip through."""
        assert {t.name for t in ALL_TARGETS} == set(SIZE_BUDGET_BYTES)

    def test_extracted_spirv_is_within_its_own_budget(self, exported):
        blobs = exported[VULKAN_SPIRV.name].spirv_bytes
        assert blobs, "vulkan-spirv must yield SPIR-V to budget"
        total = sum(len(b) for b in blobs.values())
        budget = SPIRV_BUDGET_BYTES[VULKAN_SPIRV.name]
        assert SIZE_FLOOR_BYTES <= total <= budget, f"{total} B of SPIR-V, budget {budget} B"

    def test_only_vulkan_carries_spirv(self, exported):
        carriers = {name for name, r in exported.items() if r.spirv_bytes}
        assert carriers == {VULKAN_SPIRV.name}

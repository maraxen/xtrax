"""Per-element parity swept across input sizes and across NATIVE / NATIVE_PORTABLE.

**What this establishes, and what it does not.** Every existing export parity
test in this suite (``test_bf16_exactness.py``, ``test_size_budget.py``, the
fake-toolchain tests in ``test_pipeline_native_wasm32.py``) fixes a single
input shape. That is the actual gap this file closes: regression coverage
across sizes, for both the host-tuned ``native`` target and the
``native-portable`` (x86-64-v2) target this sprint added. It is **not**
evidence of a bug caught or fixed -- the sprint spec's own three-length sweep
of a real model (L=17/40/128) found the parity margin is thin and
data-dependent (one length failed the default tolerance while its neighbours
passed comfortably at roughly half the error), which is exactly the kind of
intermittent failure a single-shape test cannot see either way. This file
gives that kind of failure more chances to show up on this repo's own tiny
fixture; it does not prove one doesn't exist.

The oracle is built directly from the model (``model(row)`` per row, stacked),
never from the composed callable under test -- see ``verify_native_parity``'s
own docstring for why that's the only comparison that bounds anything.

This file also checks that ``native-portable`` actually carries the x86-64-v2
feature baseline and not the host's -- reading it back from the compiled
artifact with ``iree-dump-module``, not asserted from the target's own flags
(``targets.py`` is trusted for what flags it *passes*; this checks what
``iree-compile`` actually *did* with them). That is the check that would catch
a build silently using the wrong target for the portable artifact.
"""

from __future__ import annotations

import re
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export.pipeline import export_pipeline
from xtrax.export.targets import NATIVE, NATIVE_PORTABLE
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

INPUT_SIZES = (256, 1024, 4096)

# x86-64-v2's defining feature; a native-portable artifact carrying anything
# host-only (e.g. avx512f) alongside this would mean the wrong target was
# built.
_V2_FEATURES = {"sse4.2"}
_HOST_ONLY_MARKERS = {"avx512f", "avx2", "avx"}


class TinyMLP(eqx.Module):
    """A 4 -> 64 -> 8 MLP, per the sprint's own measurement fixture."""

    w1: jax.Array
    w2: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x @ self.w1) @ self.w2


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _vmap_plan(cardinality: int) -> _Plan:
    return _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name="batch", cardinality=cardinality, default_batch_size=0),
                batch_size=0,
                reasoning="multi-size parity sweep",
                strategy=Vmap(),
            )
        ]
    )


@pytest.fixture(scope="module")
def model() -> TinyMLP:
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    return TinyMLP(
        w1=jax.random.normal(k1, (4, 64), dtype=jnp.float32) * 0.1,
        w2=jax.random.normal(k2, (64, 8), dtype=jnp.float32) * 0.1,
    )


def _oracle(model: TinyMLP, xs: jax.Array) -> jax.Array:
    """An independent, per-element reference built from the model directly."""
    return jnp.stack([model(xs[i]) for i in range(xs.shape[0])])


def _cpu_features(vmfb_path) -> str:
    """Read back the compiled artifact's ``cpu_features`` string.

    ``iree-dump-module``'s metadata view doesn't surface this; it appears in
    the disassembled bytecode as part of an embedded HAL-device-target
    attribute string, so ``--output=all`` is required.
    """
    result = subprocess.run(  # noqa: S603, S607
        ["iree-dump-module", "--output=all", str(vmfb_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r'cpu_features = "([^"]*)"', result.stdout)
    assert match, f"no cpu_features attribute found in {vmfb_path}"
    return match.group(1)


@pytest.fixture(scope="module")
def exported_by_size(model: TinyMLP):
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")

    results = {}
    for size in INPUT_SIZES:
        xs = jnp.asarray(np.random.default_rng(size).normal(size=(size, 4)), dtype=jnp.float32)
        abstract_inputs = (jax.ShapeDtypeStruct(xs.shape, xs.dtype),)
        results[size] = export_pipeline(
            model,
            _vmap_plan(size),
            abstract_inputs,
            (xs,),
            targets=(NATIVE, NATIVE_PORTABLE),
            reference_fn=lambda inputs, model=model: _oracle(model, inputs[0]),
        )
    return results


class TestParityAcrossSizesAndTargets:
    @pytest.mark.parametrize("size", INPUT_SIZES)
    @pytest.mark.parametrize("target_name", [NATIVE.name, NATIVE_PORTABLE.name])
    def test_verified_at_every_size_and_target(self, exported_by_size, size, target_name):
        result = exported_by_size[size][target_name]
        assert result.verified is True

    @pytest.mark.parametrize("size", INPUT_SIZES)
    @pytest.mark.parametrize("target_name", [NATIVE.name, NATIVE_PORTABLE.name])
    def test_per_element_parity_passes(self, exported_by_size, size, target_name):
        parity = exported_by_size[size][target_name].parity
        assert parity is not None
        assert parity.passed is True
        assert parity.shape_expected == (size, 8)
        assert parity.shape_actual == (size, 8)

    @pytest.mark.parametrize("size", INPUT_SIZES)
    @pytest.mark.parametrize("target_name", [NATIVE.name, NATIVE_PORTABLE.name])
    def test_max_abs_diff_stays_in_the_ordinary_float32_range(
        self, exported_by_size, size, target_name
    ):
        """Sanity ceiling, not a tight bound -- ordinary XLA/IREE drift is ~1e-6/1e-7.

        A regression that widened this by orders of magnitude (while still
        sneaking under ``np.allclose``'s per-element rtol on a large-magnitude
        output) would still trip this.
        """
        parity = exported_by_size[size][target_name].parity
        assert parity.max_abs_diff < 1e-4, (
            f"{target_name} @ n={size}: max|diff|={parity.max_abs_diff:.6e} "
            f"(atol={parity.atol:g}, rtol={parity.rtol:g})"
        )


class TestNativePortableCarriesTheV2Baseline:
    """The check that would catch a build silently using the wrong target."""

    @pytest.mark.parametrize("size", INPUT_SIZES)
    def test_portable_artifact_declares_x86_64_v2_features(self, exported_by_size, size):
        path = exported_by_size[size][NATIVE_PORTABLE.name].path
        features = _cpu_features(path)
        for expected in _V2_FEATURES:
            assert f"+{expected}" in features, f"expected +{expected} in {features!r}"

    @pytest.mark.parametrize("size", INPUT_SIZES)
    def test_portable_artifact_excludes_host_only_features(self, exported_by_size, size):
        path = exported_by_size[size][NATIVE_PORTABLE.name].path
        features = _cpu_features(path)
        for marker in _HOST_ONLY_MARKERS:
            assert f"+{marker}" not in features, (
                f"native-portable must not carry host-only +{marker}: {features!r}"
            )

    def test_native_and_portable_features_actually_differ(self, exported_by_size):
        """Guards against a fixture where the host happens to equal x86-64-v2."""
        size = INPUT_SIZES[0]
        native_features = _cpu_features(exported_by_size[size][NATIVE.name].path)
        portable_features = _cpu_features(exported_by_size[size][NATIVE_PORTABLE.name].path)
        assert native_features != portable_features

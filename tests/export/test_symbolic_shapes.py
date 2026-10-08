"""Pin the symbolic-shape boundary for ``export_pipeline``.

Measured this sprint (260910, jax 0.11.1 / IREE 3.11.0): symbolic shapes work
through ``export_pipeline``, but **only for reshape-free programs**.

- **Positive.** A reshape-free program (one matmul, one ``tanh``, no reshape
  anywhere) exported with ``jax.export.symbolic_shape("n, 4")`` compiles,
  re-executes at several concrete values of ``n``, and reports
  ``verified=True``. On the fixture the sprint used to establish this, parity
  came out ``max|diff| = 3.814697e-06`` on a 27025 B artifact -- the numbers
  below are this file's own measurement on its own tiny fixture and will not
  match those exactly (different model, different weights), but the
  qualitative result is the same: it exports, compiles, executes, and
  verifies.
- **Negative.** A reshape whose *target size* depends on the symbolic leading
  dimension (``x.reshape(-1)`` where ``x``'s leading dim is the symbolic
  ``n``) raises ``xtrax.export.compile.CompileError``. The underlying IREE
  diagnostic names the offending op explicitly:
  ``failed to legalize operation 'stablehlo.dynamic_reshape' that was
  explicitly marked illegal``. That is the diagnostic this test pins -- not
  merely "raises something", because a diagnostic that degraded to a bare
  ``RuntimeError`` with no mention of the illegal op would be a real
  regression a looser assertion would miss.

A single passing positive test would certify a capability -- "symbolic shapes
work" -- that does not hold in general for the consumer this exists for
(see ``.praxia/docs/plans/260909_runnable-artifact-and-wasm-price.md``, Phase
A2, for the fuller boundary table: rank-3 batched matmul and gather->reshape
also fail the same way). Both halves of this file are the point; neither one
alone would establish the boundary.

The two headline cases above differ in **two** variables, not one -- where the
symbolic dimension lives, as well as whether a reshape is present (see the note
below). So a third case, ``test_reshape_free_at_the_negative_case_s_own_shape``,
holds the shape fixed at the negative's and removes only the reshape. That
control is what makes the reshape the *identified* cause rather than a
plausible one; without it this file would claim a boundary it had not isolated.

Note on where the symbolic dimension has to live to trigger the negative case:
``export_pipeline`` composes ``fn`` under ``jax.vmap`` over the plan's batch
axis (see ``xtrax.tiling.dispatch``), which strips that axis's dimension out
of what ``fn`` itself sees. So a symbolic *batch* axis alone never reaches a
reshape inside ``fn`` -- the negative fixture below instead gives the batch
axis a concrete cardinality and puts the symbolic dimension inside each
per-element input, which is what ``fn`` actually traces over.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export.compile import CompileError
from xtrax.export.pipeline import export_pipeline
from xtrax.export.targets import NATIVE
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

W1 = jnp.asarray(np.random.default_rng(0).normal(size=(4, 4)), dtype=jnp.float32) * 0.1


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _vmap_plan(name: str, cardinality: int) -> _Plan:
    return _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name=name, cardinality=cardinality, default_batch_size=0),
                batch_size=0,
                reasoning="symbolic-shape boundary test",
                strategy=Vmap(),
            )
        ]
    )


def _reshape_free_fn(x: jax.Array) -> jax.Array:
    """One matmul, one tanh, no reshape anywhere."""
    return jnp.tanh(x @ W1)


def _reference_fn(inputs):
    (arr,) = inputs
    return jnp.stack([_reshape_free_fn(arr[i]) for i in range(arr.shape[0])])


class TestSymbolicShapePositive:
    """Reshape-free: symbolic export round-trips at several concrete sizes."""

    def test_symbolic_export_compiles_and_verifies(self):
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        n, four = jax.export.symbolic_shape("n, 4")
        abstract_inputs = (jax.ShapeDtypeStruct((n, four), jnp.float32),)
        xs = jnp.asarray(np.random.default_rng(1).normal(size=(8, 4)), dtype=jnp.float32)

        results = export_pipeline(
            _reshape_free_fn,
            _vmap_plan("n", 8),
            abstract_inputs,
            (xs,),
            targets=(NATIVE,),
            reference_fn=_reference_fn,
        )

        result = results["native"]
        assert result.verified is True
        assert result.parity is not None
        assert result.parity.passed is True
        assert result.size_bytes > 0

    def test_the_same_artifact_reexecutes_at_other_concrete_sizes(self):
        """The whole point of a symbolic export: one artifact, many shapes."""
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        n, four = jax.export.symbolic_shape("n, 4")
        abstract_inputs = (jax.ShapeDtypeStruct((n, four), jnp.float32),)
        xs = jnp.asarray(np.random.default_rng(1).normal(size=(8, 4)), dtype=jnp.float32)

        results = export_pipeline(
            _reshape_free_fn,
            _vmap_plan("n", 8),
            abstract_inputs,
            (xs,),
            targets=(NATIVE,),
            reference_fn=_reference_fn,
        )
        artifact_path = results["native"].path

        from xtrax.export.compile import run_native_vmfb

        for size in (3, 8, 40):
            other_xs = jnp.asarray(
                np.random.default_rng(size).normal(size=(size, 4)), dtype=jnp.float32
            )
            actual = np.asarray(run_native_vmfb(artifact_path, np.asarray(other_xs)))
            expected = np.asarray(_reference_fn((other_xs,)))
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_reshape_free_at_the_negative_case_s_own_shape(self):
        """The control: the negative fixture's shape, minus the reshape.

        Without this, the positive and negative cases above differ in *two*
        variables, not one -- the positive puts the symbolic dimension on the
        vmap batch axis (which ``export_pipeline`` strips before ``fn`` sees
        it, so ``fn`` traces over a fully concrete ``(4,)``), while the
        negative gives the batch axis a concrete cardinality and puts the
        symbolic dimension inside each per-element input. A pair that differs
        in two variables cannot attribute the failure to either one, so the
        module's "reshape is the boundary" claim would rest on nothing.

        This case holds the shape fixed at the negative's and removes only the
        reshape. It passing is what makes the reshape the identified cause.
        """
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        def summing_fn(x: jax.Array) -> jax.Array:
            # Same (n, 4) per-element input as the negative case, but reduces
            # over the symbolic axis instead of reshaping on it.
            return jnp.sum(x, axis=0)

        def summing_reference_fn(inputs):
            (arr,) = inputs
            return jnp.stack([summing_fn(arr[i]) for i in range(arr.shape[0])])

        n, four = jax.export.symbolic_shape("n, 4")
        abstract_inputs = (jax.ShapeDtypeStruct((8, n, four), jnp.float32),)
        xs = jnp.asarray(np.random.default_rng(2).normal(size=(8, 5, 4)), dtype=jnp.float32)

        results = export_pipeline(
            summing_fn,
            _vmap_plan("batch", 8),
            abstract_inputs,
            (xs,),
            targets=(NATIVE,),
            reference_fn=summing_reference_fn,
        )

        result = results["native"]
        assert result.verified is True
        assert result.parity is not None
        assert result.parity.passed is True


class TestSymbolicShapeNegative:
    """A dynamic reshape on the symbolic dimension fails loud, not silently."""

    def test_reshape_on_symbolic_dim_raises_compile_error(self):
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        def reshaping_fn(x: jax.Array) -> jax.Array:
            # x has shape (n, 4) once vmap strips the concrete batch axis;
            # reshape's target size (n * 4) depends on the symbolic n.
            return x.reshape(-1)

        def reshaping_reference_fn(inputs):
            (arr,) = inputs
            return jnp.stack([reshaping_fn(arr[i]) for i in range(arr.shape[0])])

        n, four = jax.export.symbolic_shape("n, 4")
        abstract_inputs = (jax.ShapeDtypeStruct((8, n, four), jnp.float32),)
        xs = jnp.asarray(np.random.default_rng(2).normal(size=(8, 5, 4)), dtype=jnp.float32)

        with pytest.raises(CompileError) as excinfo:
            export_pipeline(
                reshaping_fn,
                _vmap_plan("batch", 8),
                abstract_inputs,
                (xs,),
                targets=(NATIVE,),
                reference_fn=reshaping_reference_fn,
            )

        message = str(excinfo.value)
        assert "stablehlo.dynamic_reshape" in message, (
            "the diagnostic must name the illegal op -- a bare 'raised something' "
            "assertion would pass even if this degraded to an unhelpful RuntimeError"
        )
        assert "explicitly marked illegal" in message

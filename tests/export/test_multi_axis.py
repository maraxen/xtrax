"""Multi-axis composition: ordering certification, exported parity, and the refusal.

Mirrors ``tests/stages/test_nested_ordering.py``, whose sinks are imported rather
than re-declared so these tests move if the certification harness does.

The three classes certify three different things, and none of them subsumes
another:

- ``TestMultiAxisOrdering`` (AC-14a) certifies the *un-stripped* composed callable
  in pure JAX. It never reaches ``jax.export.export``.
- ``TestMultiAxisExportParity`` (AC-14b) certifies the *stripped* callable twice:
  once in pure JAX, once compiled and executed. The middle leg exists so a red
  result names one suspect instead of three -- without it, AC-14a green plus a red
  artifact leaves a bug in the stripped composition indistinguishable from a bug
  in IREE's lowering.
- ``TestMultiAxisRefusal`` (AC-15) certifies that the shape which would need a
  literal ``jax.vmap`` around lane-dependent ordered IO fails loud, with the
  executor's own guidance text.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.stages.test_executor import HostRecordSink
from tests.stages.test_nested_ordering import BatchedHostRecordSink
from xtrax.export.composer import (
    ComposerError,
    MultiAxisCompositionError,
    UnsupportedStrategyError,
    _init_is_batched,
    build_traceable_callable,
    compose_vmap_of_scan,
)
from xtrax.export.pipeline import _boundaries_for_export, export_pipeline
from xtrax.export.targets import NATIVE, VerificationLevel
from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Scan, Vmap

N_TRIALS = 20


def _decision(name: str, strategy: object, cardinality: int) -> AxisDecision:
    return AxisDecision(
        spec=AxisSpec(name=name, cardinality=cardinality, default_batch_size=0),
        batch_size=0,
        reasoning="test",
        strategy=strategy,
    )


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _vmap_of_scan_plan(batch: int, steps: int) -> _Plan:
    """The certified two-axis shape: outer Vmap lane axis, inner Scan step axis."""
    return _Plan(
        [
            _decision("lane", Vmap(), batch),
            _decision("step", Scan(init=None), steps),
        ]
    )


class _IdentityEncodingSink:
    """Records nothing; exists only to be declared ``materialize=True``.

    ``export_pipeline`` strips this before tracing, so the exported program
    carries the sunk values out as its own output instead. Ordered because the
    values it would sink are lane-dependent, which is exactly the configuration
    that makes ordering meaningful.
    """

    ordered = True

    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, x: object) -> None:
        self.calls.append(x)
        return None


class TestMultiAxisOrdering:
    """AC-14a: ordering only, at the composer layer. No export, no ExportResult."""

    def test_batched_shape_preserves_lane_step_order_under_stress(self) -> None:
        for trial in range(N_TRIALS):
            records: list[list[int]] = []
            boundary = AxisBoundary(sink=BatchedHostRecordSink(records, ordered=True))
            batch = 2 + (trial % 3)
            steps = 2 + (trial % 4)

            plan = _vmap_of_scan_plan(batch, steps)
            init = jnp.arange(batch) * 100
            composed = build_traceable_callable(
                lambda carry, x: (carry + x, carry),
                plan,
                {"step": boundary},
                scan_init=init,
            )

            ys = jax.jit(composed)(jnp.arange(steps))
            jax.block_until_ready(ys)

            running = [i * 100 for i in range(batch)]
            expected: list[list[int]] = []
            for step in range(steps):
                expected.append(list(running))
                running = [r + step for r in running]

            assert records == expected, (
                f"trial {trial} (batch={batch}, steps={steps}): {records} != {expected}"
            )

    def test_no_literal_vmap_runs_for_a_batched_carry(self) -> None:
        """An ordered sink surviving the trace proves no jax.vmap wrapped it.

        Had the composer nested a literal vmap, JAX would have refused the ordered
        callback outright rather than producing a result.
        """
        records: list[list[int]] = []
        boundary = AxisBoundary(sink=BatchedHostRecordSink(records, ordered=True))
        plan = _vmap_of_scan_plan(3, 4)
        composed = build_traceable_callable(
            lambda carry, x: (carry + x, carry),
            plan,
            {"step": boundary},
            scan_init=jnp.arange(3),
        )
        jax.block_until_ready(jax.jit(composed)(jnp.arange(4)))
        assert len(records) == 4, "one ordered host call per step, each carrying all lanes"

    def test_outer_axis_boundary_is_refused_not_dropped(self) -> None:
        plan = _vmap_of_scan_plan(3, 4)
        boundaries = {"lane": AxisBoundary(fuse=lambda ys: jnp.sum(ys, axis=0))}
        with pytest.raises(MultiAxisCompositionError, match="no per-lane point"):
            build_traceable_callable(
                lambda carry, x: (carry + x, carry),
                plan,
                boundaries,
                scan_init=jnp.arange(3),
            )


class TestMultiAxisExportParity:
    """AC-14b: the same shape exported, in three legs that isolate three suspects."""

    BATCH = 3
    STEPS = 4

    def _fixture(self):
        """A transition whose every step encodes its own ``(lane, step)`` identity."""
        plan = _vmap_of_scan_plan(self.BATCH, self.STEPS)
        sink = _IdentityEncodingSink()
        boundaries = {"step": AxisBoundary(sink=sink, materialize=True)}
        # carry is the lane's base (lane * STEPS); y is base + step.
        init = jnp.arange(self.BATCH, dtype=jnp.int32) * self.STEPS
        xs = jnp.arange(self.STEPS, dtype=jnp.int32)

        def fn(carry, x):
            return carry, carry + x

        def reference_fn(inputs):
            """Independently computed: never routes through the composer."""
            steps_in = inputs[0]
            lanes = jnp.arange(self.BATCH, dtype=jnp.int32) * self.STEPS
            return lanes[None, :] + steps_in[:, None]

        return plan, boundaries, init, xs, fn, reference_fn

    def _expected(self) -> np.ndarray:
        return np.array(
            [
                [lane * self.STEPS + step for lane in range(self.BATCH)]
                for step in range(self.STEPS)
            ],
            dtype=np.int32,
        )

    def test_stripped_composition_matches_reference_in_pure_jax(self) -> None:
        """Middle leg: isolates a stripped-composition bug from an IREE lowering bug."""
        plan, boundaries, init, xs, fn, reference_fn = self._fixture()
        stripped = _boundaries_for_export(boundaries)
        composed = build_traceable_callable(fn, plan, stripped, scan_init=init)

        out = np.asarray(jax.jit(composed)(xs))
        np.testing.assert_array_equal(out, np.asarray(reference_fn((xs,))))
        np.testing.assert_array_equal(out, self._expected())

    def test_stripping_silences_the_sink(self) -> None:
        plan, boundaries, init, xs, fn, _ref = self._fixture()
        sink = boundaries["step"].sink
        stripped = _boundaries_for_export(boundaries)
        composed = build_traceable_callable(fn, plan, stripped, scan_init=init)
        jax.block_until_ready(jax.jit(composed)(xs))
        assert sink.calls == [], "export strips the sink; nothing should reach it"

    def test_exports_and_verifies_against_an_independent_oracle(self) -> None:
        pytest.importorskip("iree.compiler")
        plan, boundaries, init, xs, fn, reference_fn = self._fixture()
        abstract = (jax.ShapeDtypeStruct(xs.shape, xs.dtype),)

        results = export_pipeline(
            fn,
            plan,
            abstract,
            (xs,),
            axis_boundaries=boundaries,
            targets=(NATIVE,),
            scan_init=init,
            reference_fn=reference_fn,
        )

        result = results["native"]
        assert result.verification_level is VerificationLevel.EXECUTED
        assert result.verified is True, result.parity.summary() if result.parity else "no parity"

    def test_executed_artifact_reconstructs_the_lane_step_order(self) -> None:
        """Ordering certified on the artifact's own output, not a test double's log."""
        pytest.importorskip("iree.runtime")
        from xtrax.export.compile import run_native_vmfb

        plan, boundaries, init, xs, fn, reference_fn = self._fixture()
        abstract = (jax.ShapeDtypeStruct(xs.shape, xs.dtype),)

        results = export_pipeline(
            fn,
            plan,
            abstract,
            (xs,),
            axis_boundaries=boundaries,
            targets=(NATIVE,),
            scan_init=init,
            reference_fn=reference_fn,
        )

        executed = np.asarray(run_native_vmfb(results["native"].path, np.asarray(xs)))
        np.testing.assert_array_equal(executed, self._expected())


class _Bare:
    """A decision-shaped object, for the fields AxisSpec will not let us omit."""

    def __init__(self, strategy=None, spec=None) -> None:
        self.strategy = strategy
        self.spec = spec


class _BareSpec:
    def __init__(self, name="x", cardinality=None) -> None:
        self.name = name
        self.cardinality = cardinality


class TestMultiAxisGuards:
    """The refusals and the tap path, which the certification shapes do not reach."""

    def test_outer_axis_without_a_strategy_is_refused(self) -> None:
        with pytest.raises(UnsupportedStrategyError, match="nothing to map"):
            compose_vmap_of_scan(
                lambda carry, x: (carry, x),
                _Bare(strategy=None, spec=_BareSpec("lane", 3)),
                _decision("step", Scan(init=None), 4),
                scan_init=jnp.arange(3),
            )

    def test_missing_initial_carry_is_refused(self) -> None:
        plan = _vmap_of_scan_plan(3, 4)
        with pytest.raises(ComposerError, match="initial carry"):
            build_traceable_callable(lambda carry, x: (carry, x), plan)

    def test_inner_tap_replaces_the_step_output(self) -> None:
        class _Tap:
            ordered = False

            def __call__(self, y):
                return y * 10

        plan = _vmap_of_scan_plan(3, 4)
        composed = build_traceable_callable(
            lambda carry, x: (carry, carry),
            plan,
            {"step": AxisBoundary(tap=_Tap())},
            scan_init=jnp.arange(3, dtype=jnp.int32),
        )
        out = np.asarray(jax.jit(composed)(jnp.arange(4, dtype=jnp.int32)))
        np.testing.assert_array_equal(out, np.tile(np.arange(3) * 10, (4, 1)))

    def test_inner_fuse_collapses_the_stack(self) -> None:
        plan = _vmap_of_scan_plan(3, 4)
        composed = build_traceable_callable(
            lambda carry, x: (carry, carry),
            plan,
            {"step": AxisBoundary(fuse=lambda ys: jnp.sum(ys, axis=0))},
            scan_init=jnp.arange(3, dtype=jnp.int32),
        )
        out = np.asarray(jax.jit(composed)(jnp.arange(4, dtype=jnp.int32)))
        np.testing.assert_array_equal(out, np.arange(3) * 4)

    def test_inner_axis_without_cardinality_is_refused_on_the_vmap_route(self) -> None:
        composed = compose_vmap_of_scan(
            lambda carry, x: (carry, x),
            _decision("lane", Vmap(), 3),
            _Bare(strategy=Scan(init=None), spec=_BareSpec("step", None)),
            scan_init=jnp.int32(0),
        )
        with pytest.raises(ComposerError, match="no length to run for"):
            composed(jnp.arange(3))

    @pytest.mark.parametrize(
        ("init", "expected"),
        [
            (jnp.arange(3), True),
            (jnp.zeros((3, 5)), True),
            ({"a": jnp.arange(3), "b": jnp.zeros((3, 2))}, True),
            (jnp.int32(0), False),
            (jnp.arange(4), False),
            ({"a": jnp.arange(3), "b": jnp.arange(4)}, False),
            ({}, False),
        ],
    )
    def test_which_carries_are_batched_to_the_outer_axis(self, init, expected) -> None:
        """The route selector: every leaf must lead with the outer cardinality."""
        assert _init_is_batched(init, 3) is expected


class TestLaneIndependentVmapRoute:
    """The narrow success the executor certifies: vmap is fine when y ignores the lane."""

    def test_composes_when_the_sunk_value_ignores_the_lane(self) -> None:
        records: list[int] = []
        plan = _vmap_of_scan_plan(3, 4)
        composed = build_traceable_callable(
            lambda carry, x: (carry, x),
            plan,
            {"step": AxisBoundary(sink=HostRecordSink(records, ordered=True))},
            scan_init=jnp.int32(0),
        )
        lane_inits = jnp.arange(3) * 100
        out = np.asarray(composed(lane_inits))
        # The transition never updates the carry, so each lane returns its own init.
        np.testing.assert_array_equal(out, np.asarray(lane_inits))
        assert records == [0, 1, 2, 3], "the inner axis's own steps, once, unbatched"


class TestMultiAxisRefusal:
    """AC-15: the shape that would need a literal vmap around lane-dependent ordered IO."""

    def _unbatched_init_plan(self):
        records: list[int] = []
        plan = _vmap_of_scan_plan(3, 4)
        boundary = AxisBoundary(sink=HostRecordSink(records, ordered=True))
        # Scalar init: the outer axis is NOT baked into the carry, so lanes can
        # only be iterated by an actual jax.vmap.
        composed = build_traceable_callable(
            lambda carry, x: (carry + x, carry),
            plan,
            {"step": boundary},
            scan_init=jnp.int32(0),
        )
        return composed

    def test_lane_dependent_ordering_raises_composition_error(self) -> None:
        composed = self._unbatched_init_plan()
        with pytest.raises(MultiAxisCompositionError):
            composed(jnp.arange(3) * 100)

    def test_error_carries_the_certified_guidance_text(self) -> None:
        composed = self._unbatched_init_plan()
        with pytest.raises(MultiAxisCompositionError, match=r"Vmap axis's `fn`"):
            composed(jnp.arange(3) * 100)

    def test_error_chains_the_underlying_executor_error(self) -> None:
        from xtrax.stages.executor import ExecutorError

        composed = self._unbatched_init_plan()
        with pytest.raises(MultiAxisCompositionError) as excinfo:
            composed(jnp.arange(3) * 100)
        assert isinstance(excinfo.value.__cause__, ExecutorError)

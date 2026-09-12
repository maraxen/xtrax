"""The plan-time gate: topology delegation and dtype rejection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.export.safety import (
    DtypeNotSupportedError,
    ExportBlocker,
    UnsupportedOperationError,
    check_export_safety,
    dtype_name,
    find_bcoo_leaves,
    validate_export_safe,
)
from xtrax.export.targets import NATIVE, Target, VerificationLevel
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.topology import PlanTopologyError

# A deliberately narrow target, so a dtype rejection can be exercised without
# waiting for the SPIR-V targets to land.
NARROW = Target(
    name="narrow",
    iree_backend="llvm-cpu",
    verification_level=VerificationLevel.CODEGEN_ONLY,
    supported_dtypes=frozenset({"f32", "i32", "bool"}),
    optional_dtypes=frozenset({"f16"}),
    optional_dtype_features={"f16": "shader-f16"},
)


def _sentinel_fn(x):
    return x


class TestDtypeName:
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (jnp.float32, "f32"),
            (jnp.float64, "f64"),
            (jnp.bfloat16, "bf16"),
            (jnp.float16, "f16"),
            (jnp.int32, "i32"),
            (jnp.bool_, "bool"),
        ],
    )
    def test_renders_short_names(self, dtype, expected):
        assert dtype_name(jnp.dtype(dtype)) == expected


class TestDtypeGate:
    def test_supported_dtype_passes(self, plan, abstract_inputs):
        validate_export_safe(plan.decisions, {}, abstract_inputs, _sentinel_fn, NATIVE)

    def test_unsupported_dtype_raises_naming_dtype_and_target(self, plan):
        inputs = [jax.ShapeDtypeStruct((4, 8), jnp.float64)]
        with pytest.raises(DtypeNotSupportedError, match="'f64'.*'narrow'"):
            validate_export_safe(plan.decisions, {}, inputs, _sentinel_fn, NARROW)

    def test_optional_dtype_without_feature_names_the_feature(self, plan):
        inputs = [jax.ShapeDtypeStruct((4, 8), jnp.float16)]
        with pytest.raises(DtypeNotSupportedError, match="shader-f16"):
            validate_export_safe(plan.decisions, {}, inputs, _sentinel_fn, NARROW)

    def test_optional_dtype_passes_once_the_feature_is_requested(self, plan):
        inputs = [jax.ShapeDtypeStruct((4, 8), jnp.float16)]
        validate_export_safe(
            plan.decisions,
            {},
            inputs,
            _sentinel_fn,
            NARROW,
            request_features=frozenset({"shader-f16"}),
        )

    def test_reports_every_offending_leaf_not_just_the_first(self, plan):
        inputs = [
            jax.ShapeDtypeStruct((4,), jnp.float64),
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.bfloat16),
        ]
        with pytest.raises(DtypeNotSupportedError) as excinfo:
            validate_export_safe(plan.decisions, {}, inputs, _sentinel_fn, NARROW)
        message = str(excinfo.value)
        assert "2 export blocker" in message
        assert "abstract_inputs[0]" in message
        assert "abstract_inputs[2]" in message
        assert "abstract_inputs[1]" not in message


class TestCheckExportSafetyIsTheListReturningTwin:
    def test_returns_blockers_rather_than_raising(self, plan):
        inputs = [jax.ShapeDtypeStruct((4,), jnp.float64)]
        blockers = check_export_safety(plan.decisions, {}, inputs, _sentinel_fn, NARROW)
        assert [type(b) for b in blockers] == [ExportBlocker]
        assert blockers[0].rule == "dtype"

    def test_returns_empty_when_clean(self, plan, abstract_inputs):
        assert check_export_safety(plan.decisions, {}, abstract_inputs, _sentinel_fn, NATIVE) == []

    def test_does_not_apply_topology_rules(self, plan, abstract_inputs):
        """Topology always raises directly; it is never demoted to a blocker."""

        class _Sink:
            ordered = False

            def __call__(self, x) -> None:
                pass

        boundaries = {"batch": AxisBoundary(sink=_Sink())}
        assert (
            check_export_safety(plan.decisions, boundaries, abstract_inputs, _sentinel_fn, NATIVE)
            == []
        )


class TestTopologyPropagatesUnwrapped:
    def test_undeclared_sink_raises_plan_topology_error(self, plan, abstract_inputs):
        class _Sink:
            ordered = False

            def __call__(self, x) -> None:
                pass

        boundaries = {"batch": AxisBoundary(sink=_Sink())}
        with pytest.raises(PlanTopologyError):
            validate_export_safe(plan.decisions, boundaries, abstract_inputs, _sentinel_fn, NATIVE)

    def test_materializing_sink_passes_the_gate(self, plan, abstract_inputs):
        class _Sink:
            ordered = True

            def __call__(self, x) -> None:
                pass

        boundaries = {"batch": AxisBoundary(sink=_Sink(), materialize=True)}
        validate_export_safe(plan.decisions, boundaries, abstract_inputs, _sentinel_fn, NATIVE)


class TestFindBcooLeaves:
    def test_returns_empty_for_a_dense_tree(self, model):
        assert find_bcoo_leaves(model) == []

    def test_finds_a_sparse_leaf(self):
        from jax.experimental.sparse import BCOO

        dense = jnp.eye(4, dtype=jnp.float32)
        tree = {"weight": BCOO.fromdense(dense)}
        assert find_bcoo_leaves(tree) == ["['weight']"]


# ---------------------------------------------------------------------------
# Op blockers: #5092 (unlegalizable top_k) and #5094 (sort-stability).
# ---------------------------------------------------------------------------

_OP_INPUT = [jax.ShapeDtypeStruct((8,), jnp.float32)]
_SCAN_INPUT = [jax.ShapeDtypeStruct((3, 8), jnp.float32)]


def _top_k_fn(x):
    return jax.lax.top_k(x, 4)


def _top_k_in_scan_fn(xs):
    def body(carry, x):
        vals, idx = jax.lax.top_k(x, 4)
        return carry, (vals, idx)

    _, out = jax.lax.scan(body, None, xs)
    return out


def _stable_argsort_fn(x):
    return jnp.argsort(-x, axis=-1, stable=True)[..., :4]


def _unstable_argsort_fn(x):
    return jnp.argsort(x, stable=False)


def _clean_fn(x):
    return jnp.sum(x * 2)


def _permutation_fn(x):
    return jax.random.permutation(jax.random.key(0), x)


def _permutation_in_scan_fn(xs):
    def body(carry, x):
        return carry, jax.random.permutation(jax.random.key(0), x)

    _, out = jax.lax.scan(body, None, xs)
    return out


def _permutation_in_jit_fn(x):
    return jax.jit(lambda v: jax.random.permutation(jax.random.key(0), v))(x)


def _argsort_bits_fn(x):
    return x[jnp.argsort(jax.random.bits(jax.random.key(0), shape=x.shape))]


class TestUnlegalizableOpBlocker:
    def test_top_k_is_refused_at_plan_time(self, plan):
        with pytest.raises(UnsupportedOperationError, match="top_k"):
            validate_export_safe(plan.decisions, {}, _OP_INPUT, _top_k_fn, NATIVE)

    def test_top_k_nested_in_a_scan_is_still_caught(self, plan):
        """Regression test: a top-level-only jaxpr walk finds nothing here."""
        with pytest.raises(UnsupportedOperationError, match="top_k"):
            validate_export_safe(plan.decisions, {}, _SCAN_INPUT, _top_k_in_scan_fn, NATIVE)

    def test_check_export_safety_reports_the_rule(self, plan):
        blockers = check_export_safety(plan.decisions, {}, _OP_INPUT, _top_k_fn, NATIVE)
        assert any(b.rule == "unlegalizable-op" for b in blockers)

    def test_acknowledging_unlegalizable_op_does_not_suppress_it(self, plan):
        blockers = check_export_safety(
            plan.decisions,
            {},
            _OP_INPUT,
            _top_k_fn,
            NATIVE,
            acknowledged=frozenset({"unlegalizable-op"}),
        )
        assert any(b.rule == "unlegalizable-op" for b in blockers)
        with pytest.raises(UnsupportedOperationError):
            validate_export_safe(
                plan.decisions,
                {},
                _OP_INPUT,
                _top_k_fn,
                NATIVE,
                acknowledged=frozenset({"unlegalizable-op"}),
            )


class TestSortStabilityBlocker:
    def test_stable_argsort_produces_a_sort_stability_blocker(self, plan):
        blockers = check_export_safety(plan.decisions, {}, _OP_INPUT, _stable_argsort_fn, NATIVE)
        assert any(b.rule == "sort-stability" for b in blockers)

    def test_unstable_sort_produces_no_blocker(self, plan):
        blockers = check_export_safety(plan.decisions, {}, _OP_INPUT, _unstable_argsort_fn, NATIVE)
        assert blockers == []

    def test_acknowledged_suppresses_sort_stability(self, plan):
        blockers = check_export_safety(
            plan.decisions,
            {},
            _OP_INPUT,
            _stable_argsort_fn,
            NATIVE,
            acknowledged=frozenset({"sort-stability"}),
        )
        assert blockers == []
        validate_export_safe(
            plan.decisions,
            {},
            _OP_INPUT,
            _stable_argsort_fn,
            NATIVE,
            acknowledged=frozenset({"sort-stability"}),
        )

    def test_validate_raises_unsupported_operation_error_and_names_the_fix(self, plan):
        with pytest.raises(UnsupportedOperationError, match="tiebreak"):
            validate_export_safe(plan.decisions, {}, _OP_INPUT, _stable_argsort_fn, NATIVE)


class TestRandomPermutationBlocker:
    def test_permutation_produces_exactly_one_random_permutation_blocker(self, plan):
        blockers = check_export_safety(plan.decisions, {}, _OP_INPUT, _permutation_fn, NATIVE)
        perm = [b for b in blockers if b.rule == "random-permutation"]
        assert len(perm) == 1

    def test_argsort_of_random_bits_is_not_refused(self, plan):
        """Over-refusal guard: the IREE-exact substitute must not match '_shuffle'."""
        blockers = check_export_safety(plan.decisions, {}, _OP_INPUT, _argsort_bits_fn, NATIVE)
        assert not any(b.rule == "random-permutation" for b in blockers)

    def test_permutation_nested_in_a_scan_is_still_caught(self, plan):
        """Regression test: a top-level-only jaxpr walk finds nothing here."""
        blockers = check_export_safety(
            plan.decisions, {}, _SCAN_INPUT, _permutation_in_scan_fn, NATIVE
        )
        assert any(b.rule == "random-permutation" for b in blockers)

    def test_permutation_nested_in_a_jit_is_still_caught(self, plan):
        blockers = check_export_safety(
            plan.decisions, {}, _OP_INPUT, _permutation_in_jit_fn, NATIVE
        )
        assert any(b.rule == "random-permutation" for b in blockers)

    def test_acknowledged_suppresses_random_permutation_not_unlegalizable_op(self, plan):
        suppressed = check_export_safety(
            plan.decisions,
            {},
            _OP_INPUT,
            _permutation_fn,
            NATIVE,
            acknowledged=frozenset({"random-permutation"}),
        )
        assert not any(b.rule == "random-permutation" for b in suppressed)
        still = check_export_safety(
            plan.decisions,
            {},
            _OP_INPUT,
            _top_k_fn,
            NATIVE,
            acknowledged=frozenset({"unlegalizable-op"}),
        )
        assert any(b.rule == "unlegalizable-op" for b in still)

    def test_validate_raises_unsupported_operation_error_naming_the_rule(self, plan):
        with pytest.raises(UnsupportedOperationError, match="random-permutation"):
            validate_export_safe(plan.decisions, {}, _OP_INPUT, _permutation_fn, NATIVE)


class TestOpBlockersDoNotOverreport:
    def test_a_clean_program_has_no_blockers(self, plan):
        assert check_export_safety(plan.decisions, {}, _OP_INPUT, _clean_fn, NATIVE) == []
        validate_export_safe(plan.decisions, {}, _OP_INPUT, _clean_fn, NATIVE)

    def test_dtype_blocker_takes_precedence_in_the_raised_exception_type(self, plan):
        """A dtype blocker alongside an op blocker still raises DtypeNotSupportedError."""
        bad_dtype_input = [jax.ShapeDtypeStruct((8,), jnp.float64)]
        with pytest.raises(DtypeNotSupportedError) as excinfo:
            validate_export_safe(plan.decisions, {}, bad_dtype_input, _top_k_fn, NATIVE)
        message = str(excinfo.value)
        assert "dtype" in message
        assert "unlegalizable-op" in message

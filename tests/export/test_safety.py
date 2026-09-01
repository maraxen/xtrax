"""The plan-time gate: topology delegation and dtype rejection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.export.safety import (
    DtypeNotSupportedError,
    ExportBlocker,
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
        assert "2 dtype blocker" in message
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

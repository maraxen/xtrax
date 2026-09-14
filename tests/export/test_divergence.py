"""Pure-logic tests for xtrax.export.divergence -- no toolchain, no IREE.

Every AC referenced in a test name/comment is quoted from
``.praxia/docs/specs/260911_export-divergence-mapping.md`` SS10.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import jax
import jax.export as jexport
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export import divergence as d

# --------------------------------------------------------------------------
# AC-12: purity
# --------------------------------------------------------------------------


class TestPurity:
    def test_no_import_of_iree_anywhere_in_the_module(self):
        """AC-12: divergence.py imports nothing from iree (asserted by test)."""
        source = Path(inspect.getfile(d)).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("iree"), alias.name
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("iree"), node.module

    def test_module_is_importable_without_toolchain_imports_resolving(self):
        # If divergence.py imported iree at module scope, this import would
        # already have failed at collection time (no `export` extra here).
        assert d.compare_pytree is not None


# --------------------------------------------------------------------------
# AC-1 / AC-13: compare_pytree leaf vs. treedef mismatch
# --------------------------------------------------------------------------


class TestComparePytree:
    def test_matching_pytree_reports_identical_leaves(self):
        expected = {"a": np.array([1.0, 2.0], dtype=np.float32)}
        actual = {"a": np.array([1.0, 2.0], dtype=np.float32)}
        (leaf,) = d.compare_pytree(expected, actual)
        assert leaf.severity == d.Severity.IDENTICAL
        assert leaf.failed is False
        assert leaf.dtype_class == "float"

    def test_ac1_leaf_mismatch_among_matching_leaves_records_and_continues(self):
        """AC-1: a leaf mismatch (matching treedefs, differing leaf shape or
        dtype) yields a per-leaf failure record and continues, asserted on a
        pytree with one mismatched leaf among matching ones.
        """
        expected = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([1, 2, 3], dtype=np.int32),
        }
        actual = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            # shape mismatch on 'b' only
            "b": np.array([1, 2], dtype=np.int32),
        }
        leaves = d.compare_pytree(expected, actual)
        assert len(leaves) == 2
        by_path = {leaf.path: leaf for leaf in leaves}
        assert by_path["['a']"].failed is False
        assert by_path["['a']"].severity == d.Severity.IDENTICAL
        assert by_path["['b']"].failed is True
        assert by_path["['b']"].metrics == {}
        assert "shape" in by_path["['b']"].message
        assert by_path["['b']"].severity == d.Severity.BEYOND_BUDGET

    def test_ac13_treedef_mismatch_raises_naming_both_treedefs(self):
        """AC-13: a treedef mismatch raises ProbeStructureError naming both
        treedefs, while a leaf mismatch under a matching treedef produces a
        per-leaf failure record and continues.
        """
        expected = {"a": 1.0, "b": 2.0}
        actual = {"a": 1.0}
        with pytest.raises(d.ProbeStructureError) as excinfo:
            d.compare_pytree(expected, actual)
        message = str(excinfo.value)
        assert repr(excinfo.value.expected_treedef) in message
        assert repr(excinfo.value.actual_treedef) in message
        assert excinfo.value.expected_treedef != excinfo.value.actual_treedef

    def test_ac13_none_valued_field_varies_arity_and_raises_structure_error(self):
        """AC-13's None-valued-field case: JAX flattens ``None`` as an empty
        node, so a field typed ``jax.Array | None`` (aminx's
        ``node_features_out``) makes output arity configuration-dependent.
        """
        expected = {"node_features_out": np.zeros((3,), dtype=np.float32)}
        actual = {"node_features_out": None}
        with pytest.raises(d.ProbeStructureError):
            d.compare_pytree(expected, actual)

    def test_dtype_mismatch_is_a_leaf_failure_not_a_treedef_failure(self):
        expected = {"a": np.array([1.0, 2.0], dtype=np.float32)}
        actual = {"a": np.array([1.0, 2.0], dtype=np.float64)}
        (leaf,) = d.compare_pytree(expected, actual)
        assert leaf.failed is True
        assert "dtype" in leaf.message

    def test_bool_leaf_metrics(self):
        expected = {"m": np.array([True, False, True])}
        actual = {"m": np.array([True, True, True])}
        (leaf,) = d.compare_pytree(expected, actual)
        assert leaf.dtype_class == "bool"
        assert leaf.metrics["hamming_distance"] == 1.0
        assert leaf.metrics["first_mismatch_index"] == 1.0
        assert leaf.severity == d.Severity.BEYOND_BUDGET

    def test_bool_leaf_identical_has_no_mismatch_index(self):
        expected = {"m": np.array([True, False])}
        actual = {"m": np.array([True, False])}
        (leaf,) = d.compare_pytree(expected, actual)
        assert leaf.metrics["first_mismatch_index"] == -1.0
        assert leaf.severity == d.Severity.IDENTICAL

    def test_integer_leaf_empty_array_is_vacuously_identical(self):
        expected = {"idx": np.zeros((0,), dtype=np.int32)}
        actual = {"idx": np.zeros((0,), dtype=np.int32)}
        (leaf,) = d.compare_pytree(expected, actual)
        assert leaf.metrics["exact_match_fraction"] == 1.0
        assert leaf.severity == d.Severity.IDENTICAL

    def test_float_leaf_nonfinite_mismatch_counted_separately_from_magnitude(self):
        expected = {"x": np.array([1.0, np.inf, np.nan], dtype=np.float32)}
        actual = {"x": np.array([1.0, 5.0, np.nan], dtype=np.float32)}
        (leaf,) = d.compare_pytree(expected, actual)
        # position 0: identical finite; position 1: inf vs finite -> nonfinite
        # mismatch; position 2: nan vs nan -> not a mismatch (both non-finite,
        # same marker).
        assert leaf.metrics["n_nonfinite_mismatch"] == 1.0
        assert leaf.metrics["max_abs_diff"] == 0.0  # only finite-both pair is identical
        assert leaf.severity == d.Severity.BEYOND_BUDGET

    def test_float_leaf_max_ulp_diff_nonzero_for_last_bit_noise(self):
        base = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        nudged = np.nextafter(base, np.float32(np.inf))
        (leaf,) = d.compare_pytree({"x": base}, {"x": nudged})
        assert leaf.metrics["max_ulp_diff"] == 1.0
        assert leaf.severity == d.Severity.BEYOND_BUDGET  # provisional, no budget applied yet

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            d.compare_pytree(
                {"x": np.array([1 + 2j], dtype=np.complex64)},
                {"x": np.array([1 + 2j], dtype=np.complex64)},
            )

    def test_float_leaf_all_nonfinite_has_zero_magnitude_metrics(self):
        """No finite-both pair exists, so the magnitude metrics fall back to
        0.0 rather than an undefined max-of-empty (`_float_metrics`'s "no
        finite pairs" branch).
        """
        expected = {"x": np.array([np.inf, -np.inf], dtype=np.float32)}
        actual = {"x": np.array([np.inf, np.nan], dtype=np.float32)}
        (leaf,) = d.compare_pytree(expected, actual)
        assert leaf.metrics["max_abs_diff"] == 0.0
        assert leaf.metrics["max_rel_diff"] == 0.0
        assert leaf.metrics["max_ulp_diff"] == 0.0
        # position 0: +inf == +inf, not a mismatch; position 1: -inf vs nan is.
        assert leaf.metrics["n_nonfinite_mismatch"] == 1.0

    def test_ulp_ordered_float16_width(self):
        """`_ulp_ordered`'s itemsize == 2 branch (float16 view)."""
        base = np.array([1.0, -2.0], dtype=np.float16)
        nudged = np.nextafter(base, np.float16(np.inf))
        ordered_base = d._ulp_ordered(base)
        ordered_nudged = d._ulp_ordered(nudged)
        assert np.all(np.abs(ordered_nudged - ordered_base) == 1)

    def test_float64_leaf_max_ulp_diff(self):
        """`_ulp_ordered`'s itemsize == 8 branch (float64 view)."""
        base = np.array([1.0, -3.5], dtype=np.float64)
        nudged = np.nextafter(base, np.float64(np.inf))
        (leaf,) = d.compare_pytree({"x": base}, {"x": nudged})
        assert leaf.dtype_class == "float"
        assert leaf.metrics["max_ulp_diff"] == 1.0

    def test_ulp_ordered_falls_back_to_float32_for_an_unrecognised_width(self):
        """`_ulp_ordered`'s else branch: a float width that is not 2/4/8
        bytes (e.g. ``np.longdouble``, 16 bytes on this platform) is upcast
        to float32 rather than left unhandled.
        """
        x = np.array([1.0, 2.5], dtype=np.longdouble)
        ordered = d._ulp_ordered(x)
        assert ordered.dtype == np.int64
        assert ordered.shape == x.shape


# --------------------------------------------------------------------------
# budget_leaf -- AC-7
# --------------------------------------------------------------------------


class TestBudgetLeaf:
    def test_ac7_zero_sensitivity_yields_exactly_the_floor(self):
        assert d.budget_leaf(0.0) == d.ULP_FLOOR

    def test_ac7_monotonic_above_the_floor(self):
        """AC-7: two fixtures with sensitivities
        ``ULP_FLOOR/SLACK < m1 < m2`` yield ``budget1 < budget2``.
        """
        threshold = d.ULP_FLOOR / d.SLACK
        m1 = threshold + 1.0
        m2 = threshold + 2.0
        assert d.budget_leaf(m1) < d.budget_leaf(m2)

    def test_ac7_flat_below_the_floor_not_monotonic(self):
        """max(FLOOR, SLACK*m) violates unconditional monotonicity below the
        floor by construction -- two distinct tiny sensitivities give the
        same budget. Asserting this (not unconditional monotonicity) is the
        point of AC-7's "only above the floor" qualifier.
        """
        threshold = d.ULP_FLOOR / d.SLACK
        m_a = threshold * 0.1
        m_b = threshold * 0.5
        assert m_a != m_b
        assert d.budget_leaf(m_a) == d.budget_leaf(m_b) == d.ULP_FLOOR

    def test_budget_formula_matches_spec_exactly(self):
        m = 10.0
        assert d.budget_leaf(m) == max(d.ULP_FLOOR, d.SLACK * m)
        assert d.budget_leaf(m) == 40.0


# --------------------------------------------------------------------------
# classify_probes -- AC-3, AC-3b, AC-4, AC-5, AC-14, AC-19
# --------------------------------------------------------------------------


def _float_leaf(
    *, severity: d.Severity, max_ulp_diff: float = 0.0, path: str = ""
) -> d.LeafDivergence:
    return d.LeafDivergence(
        path=path,
        dtype_class="float",
        severity=severity,
        metrics={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "max_ulp_diff": max_ulp_diff,
            "n_nonfinite_mismatch": 0.0,
        },
        failed=False,
        message=None,
    )


def _int_leaf(*, mismatched: bool, path: str = "") -> d.LeafDivergence:
    severity = d.Severity.BEYOND_BUDGET if mismatched else d.Severity.IDENTICAL
    return d.LeafDivergence(
        path=path,
        dtype_class="integer",
        severity=severity,
        metrics={
            "exact_match_fraction": 0.0 if mismatched else 1.0,
            "first_mismatch_index": 0.0 if mismatched else -1.0,
            "n_mismatched": 1.0 if mismatched else 0.0,
        },
        failed=False,
        message=None,
    )


class TestClassifyProbes:
    def test_ac3_join_with_one_clean_one_diverged_predecessor_is_amplified_not_injected(self):
        """AC-3: a DAG containing a join with one clean and one diverged
        predecessor is AMPLIFIED (max-over-predecessors), not INJECTED.
        """
        probes = {
            "clean_pred": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "diverged_pred": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=1000.0),),
            "join": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=1000.0),),
        }
        deps = {"join": ("clean_pred", "diverged_pred")}
        budgets = {"diverged_pred": 4.0, "join": 4.0}
        reports = d.classify_probes(probes, deps, budgets)
        by_name = {r.name: r for r in reports}
        assert by_name["join"].divergence_class == d.DivergenceClass.AMPLIFIED
        assert by_name["join"].divergence_class != d.DivergenceClass.INJECTED

    def test_ac3b_discrete_to_float_edge_classifies_to_a_definite_class(self):
        """AC-3b: an edge from a discrete predecessor to a float probe (the
        dogfood's first edge, neighbor_indices -> rbf) classifies by SS5.3's
        severity ordinal, and a test asserts a definite class rather than an
        error.
        """
        probes = {
            "neighbor_indices": (_int_leaf(mismatched=True),),
            "rbf": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=2.0),),
        }
        deps = {"rbf": ("neighbor_indices",)}
        budgets = {"rbf": 100.0}
        reports = d.classify_probes(probes, deps, budgets)
        by_name = {r.name: r for r in reports}
        assert by_name["neighbor_indices"].divergence_class == d.DivergenceClass.DISCRETE_FLIP
        # rbf's predecessor (neighbor_indices) is not clean (severity 2), and
        # rbf itself resolves to WITHIN_BUDGET (severity 1) < 2 -> ATTENUATED.
        assert by_name["rbf"].divergence_class in set(d.DivergenceClass)
        assert by_name["rbf"].divergence_class == d.DivergenceClass.ATTENUATED

    def test_ac4_attenuated_when_strictly_smaller_than_predecessor(self):
        """AC-4: ATTENUATED is produced for a probe whose divergence is
        strictly smaller than its predecessor's.
        """
        probes = {
            "pred": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=1000.0),),
            "renormalized": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=1.0),),
        }
        deps = {"renormalized": ("pred",)}
        budgets = {"pred": 4.0, "renormalized": 4.0}
        reports = d.classify_probes(probes, deps, budgets)
        by_name = {r.name: r for r in reports}
        assert by_name["renormalized"].divergence_class == d.DivergenceClass.ATTENUATED

    def test_ac5_discrete_flip_with_bit_identical_predecessor(self):
        """AC-5: DISCRETE_FLIP is raised for a mismatched int leaf with
        bit-identical predecessor floats, and for the empty-predecessor case.
        """
        probes = {
            "pred": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "idx": (_int_leaf(mismatched=True),),
        }
        deps = {"idx": ("pred",)}
        reports = d.classify_probes(probes, deps, budgets={})
        by_name = {r.name: r for r in reports}
        assert by_name["idx"].divergence_class == d.DivergenceClass.DISCRETE_FLIP

    def test_ac5_discrete_flip_with_empty_predecessor_set(self):
        """AC-5's second half: a probe with an empty predecessor set and a
        mismatched discrete leaf is DISCRETE_FLIP by rule.
        """
        probes = {"idx": (_int_leaf(mismatched=True),)}
        reports = d.classify_probes(probes, probe_deps={}, budgets={})
        assert reports[0].divergence_class == d.DivergenceClass.DISCRETE_FLIP
        assert reports[0].predecessors == ()

    def test_ac14_predecessor_also_discrete_mismatched_is_amplified_not_discrete_flip(self):
        """AC-14: a probe whose predecessor also had a discrete mismatch is
        AMPLIFIED, not DISCRETE_FLIP -- the
        propagation-through-the-discrete-branch regression.
        """
        probes = {
            "pred_idx": (_int_leaf(mismatched=True),),
            "child_idx": (_int_leaf(mismatched=True),),
        }
        deps = {"child_idx": ("pred_idx",)}
        reports = d.classify_probes(probes, deps, budgets={})
        by_name = {r.name: r for r in reports}
        assert by_name["pred_idx"].divergence_class == d.DivergenceClass.DISCRETE_FLIP
        assert by_name["child_idx"].divergence_class == d.DivergenceClass.AMPLIFIED

    def test_ac14_precedence_order_first_match_wins(self):
        """Plus: a probe qualifying under two classes resolves by the SS5.3
        precedence order. This probe has BOTH a fresh discrete mismatch
        (would-be DISCRETE_FLIP) and, independently, severity-2 float
        divergence with clean predecessors (would-be INJECTED) -- precedence
        picks DISCRETE_FLIP.
        """
        probes = {
            "pred": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "probe": (
                _int_leaf(mismatched=True),
                _float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=1000.0),
            ),
        }
        deps = {"probe": ("pred",)}
        budgets = {"probe": 4.0}
        reports = d.classify_probes(probes, deps, budgets)
        by_name = {r.name: r for r in reports}
        assert by_name["probe"].divergence_class == d.DivergenceClass.DISCRETE_FLIP

    def test_ac19_severity_1_with_clean_predecessors_is_clean(self):
        """AC-19: a probe whose leaves are severity 1 (float divergence
        strictly inside budget_leaf) with all predecessors severity 0
        classifies as CLEAN.
        """
        probes = {
            "pred": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "probe": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=2.0),),
        }
        deps = {"probe": ("pred",)}
        budgets = {"probe": 4.0}  # 2.0 <= 4.0 -> resolves to WITHIN_BUDGET (severity 1)
        reports = d.classify_probes(probes, deps, budgets)
        by_name = {r.name: r for r in reports}
        assert by_name["probe"].severity == d.Severity.WITHIN_BUDGET
        assert by_name["probe"].divergence_class == d.DivergenceClass.CLEAN

    def test_ac19_same_probe_at_severity_2_is_injected(self):
        """AC-19's other half: the same probe at severity 2 classifies as
        INJECTED. Required alongside the CLEAN assertion above -- the CLEAN
        assertion alone is satisfied by a classifier that never emits
        INJECTED at all.
        """
        probes = {
            "pred": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "probe": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=100.0),),
        }
        deps = {"probe": ("pred",)}
        budgets = {"probe": 4.0}  # 100.0 > 4.0 -> stays BEYOND_BUDGET (severity 2)
        reports = d.classify_probes(probes, deps, budgets)
        by_name = {r.name: r for r in reports}
        assert by_name["probe"].severity == d.Severity.BEYOND_BUDGET
        assert by_name["probe"].divergence_class == d.DivergenceClass.INJECTED

    def test_ac19_empty_predecessor_severity_1_is_also_clean(self):
        """AC-19: "Add the empty-predecessor case at severity 1, which must
        also be CLEAN."
        """
        probes = {"probe": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=2.0),)}
        budgets = {"probe": 4.0}
        reports = d.classify_probes(probes, probe_deps={}, budgets=budgets)
        assert reports[0].severity == d.Severity.WITHIN_BUDGET
        assert reports[0].divergence_class == d.DivergenceClass.CLEAN

    def test_ac8_integer_leaf_ignores_a_huge_float_budget(self):
        """AC-8: integer leaves are judged by exact match regardless of any
        budget -- asserted by running a calibrated ladder with a
        deliberately huge float budget and confirming a single flipped index
        still fails.
        """
        probes = {
            "probe": (
                _int_leaf(mismatched=True),
                _float_leaf(severity=d.Severity.IDENTICAL),
            )
        }
        # A huge budget on the co-located float leaf must not rescue the
        # integer mismatch -- integers never consult budgets at all.
        budgets = {"probe": 1e18}
        reports = d.classify_probes(probes, probe_deps={}, budgets=budgets)
        int_leaf = next(leaf for leaf in reports[0].leaves if leaf.dtype_class == "integer")
        assert int_leaf.severity == d.Severity.BEYOND_BUDGET
        assert reports[0].divergence_class == d.DivergenceClass.DISCRETE_FLIP

    def test_missing_budget_raises_for_a_diverged_float_leaf(self):
        probes = {"probe": (_float_leaf(severity=d.Severity.BEYOND_BUDGET, max_ulp_diff=10.0),)}
        with pytest.raises(d.MissingBudgetError, match="probe"):
            d.classify_probes(probes, probe_deps={}, budgets={})

    def test_identical_float_leaf_needs_no_budget_entry(self):
        # No KeyError/MissingBudgetError even though budgets is empty --
        # bit-identity is a measured fact, not something a budget defaults.
        probes = {"probe": (_float_leaf(severity=d.Severity.IDENTICAL),)}
        reports = d.classify_probes(probes, probe_deps={}, budgets={})
        assert reports[0].divergence_class == d.DivergenceClass.CLEAN

    def test_nonfinite_mismatch_forces_beyond_budget_regardless_of_ulp_budget(self):
        leaf = d.LeafDivergence(
            path="",
            dtype_class="float",
            severity=d.Severity.BEYOND_BUDGET,
            metrics={
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "max_ulp_diff": 0.0,
                "n_nonfinite_mismatch": 1.0,
            },
            failed=False,
            message=None,
        )
        probes = {"probe": (leaf,)}
        reports = d.classify_probes(probes, probe_deps={}, budgets={"probe": 1e18})
        assert reports[0].leaves[0].severity == d.Severity.BEYOND_BUDGET

    def test_failed_leaf_is_never_budget_adjusted(self):
        leaf = d.LeafDivergence(
            path="",
            dtype_class="float",
            severity=d.Severity.BEYOND_BUDGET,
            metrics={},
            failed=True,
            message="shape mismatch",
        )
        probes = {"probe": (leaf,)}
        reports = d.classify_probes(probes, probe_deps={}, budgets={})
        assert reports[0].leaves[0].failed is True
        assert reports[0].leaves[0].severity == d.Severity.BEYOND_BUDGET

    def test_undeclared_predecessor_probe_raises(self):
        probes = {"a": (_float_leaf(severity=d.Severity.IDENTICAL),)}
        with pytest.raises(ValueError, match="undeclared predecessor|unknown probe"):
            d.classify_probes(probes, probe_deps={"a": ("nonexistent",)}, budgets={})

    def test_probe_deps_key_not_in_probes_raises(self):
        probes = {"a": (_float_leaf(severity=d.Severity.IDENTICAL),)}
        with pytest.raises(ValueError, match="unknown probe"):
            d.classify_probes(probes, probe_deps={"nonexistent": ("a",)}, budgets={})

    def test_dependency_cycle_raises(self):
        probes = {
            "a": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "b": (_float_leaf(severity=d.Severity.IDENTICAL),),
        }
        deps = {"a": ("b",), "b": ("a",)}
        with pytest.raises(ValueError, match="cycle"):
            d.classify_probes(probes, deps, budgets={})

    def test_output_order_matches_probes_iteration_order_not_topological_order(self):
        probes = {
            "child": (_float_leaf(severity=d.Severity.IDENTICAL),),
            "parent": (_float_leaf(severity=d.Severity.IDENTICAL),),
        }
        deps = {"child": ("parent",)}
        reports = d.classify_probes(probes, deps, budgets={})
        assert [r.name for r in reports] == ["child", "parent"]

    def test_budget_key_convention(self):
        assert d.budget_key("rbf", "") == "rbf"
        assert d.budget_key("rbf", "['weights']") == "rbf['weights']"


# --------------------------------------------------------------------------
# probe_resolution / validate_probe_deps -- AC-9, AC-15
# --------------------------------------------------------------------------


def _toy_dag_export() -> tuple[object, dict[str, str]]:
    """A small pytree-output export mirroring aminx's edge-stage DAG shape:
    ``rbf`` and ``encoded_positions`` are independent, ``edges_concat``
    depends on both, and (deliberately, for the negative test) nothing
    depends on ``neighbor_indices`` the way ``rbf`` would in the real model --
    this toy keeps ``rbf`` and ``encoded_positions`` genuinely independent so
    the SS5.3/AC-15 "rbf -> encoded_positions" edge is genuinely invalid.
    """

    def f(x: jax.Array, y: jax.Array) -> dict[str, jax.Array]:
        neighbor_indices = jnp.argsort(x).astype(jnp.int32)
        rbf = jnp.sin(x) + 1.0
        encoded_positions = y * 2.0
        edges_concat = jnp.concatenate([rbf, encoded_positions])
        return {
            "neighbor_indices": neighbor_indices,
            "rbf": rbf,
            "encoded_positions": encoded_positions,
            "edges_concat": edges_concat,
        }

    exported = jexport.export(jax.jit(f))(
        jnp.zeros((4,), jnp.float32), jnp.zeros((4,), jnp.float32)
    )
    probe_result_paths = {
        "neighbor_indices": "result['neighbor_indices']",
        "rbf": "result['rbf']",
        "encoded_positions": "result['encoded_positions']",
        "edges_concat": "result['edges_concat']",
    }
    return exported, probe_result_paths


class TestProbeResolution:
    def test_ac9_slice_delta_and_unattributed_ops_reported(self):
        """AC-9: probe_resolution reports slice_delta per declared edge and
        unattributed_ops.
        """
        exported, paths = _toy_dag_export()
        deps = {"edges_concat": ("rbf", "encoded_positions")}
        result = d.probe_resolution(exported, deps, paths)
        assert set(result.slice_delta) == {"rbf->edges_concat", "encoded_positions->edges_concat"}
        assert all(v > 0 for v in result.slice_delta.values())
        assert result.unattributed_ops >= 0
        assert result.total_ops > 0

    def test_ac9_docstring_states_the_three_caveats(self):
        doc = d.probe_resolution.__doc__ or ""
        assert "overlap" in doc  # sibling slices overlap, do not partition
        assert "static" in doc  # counts are static, loop bodies count once
        assert "instrumented" in doc  # measures pre-IREE-fusion StableHLO

    def test_probe_resolution_requires_every_referenced_probe_in_paths(self):
        exported, paths = _toy_dag_export()
        incomplete = {k: v for k, v in paths.items() if k != "encoded_positions"}
        deps = {"edges_concat": ("rbf", "encoded_positions")}
        with pytest.raises(ValueError, match="encoded_positions"):
            d.probe_resolution(exported, deps, incomplete)

    def test_probe_resolution_rejects_a_result_info_string_with_no_match(self):
        exported, paths = _toy_dag_export()
        wrong = dict(paths)
        wrong["rbf"] = "result['does_not_exist']"
        deps = {"edges_concat": ("rbf", "encoded_positions")}
        with pytest.raises(ValueError, match="does_not_exist"):
            d.probe_resolution(exported, deps, wrong)

    def test_probe_resolution_over_a_multi_func_module_with_a_loop(self):
        """Exercises _module_entry_func's multi-function symbol-name
        resolution and _all_ops_in_region_tree/_backward_slice_ops's nested-
        region walk, via a real `jax.lax.fori_loop` (lowers to a private
        helper function plus a `stablehlo.while` with cond/do regions).
        """

        def f(x: jax.Array) -> dict[str, jax.Array]:
            def body(_i: jax.Array, val: jax.Array) -> jax.Array:
                return val + x

            total = jax.lax.fori_loop(0, 3, body, jnp.zeros_like(x))
            return {"total": total, "x2": x * 2.0}

        exported = jexport.export(jax.jit(f))(jnp.zeros((4,), jnp.float32))
        paths = {"total": "result['total']", "x2": "result['x2']"}
        deps = {"x2": ("total",)}
        # Not a real dataflow edge (x2 does not consume total), so the
        # slice-subset check must reject it -- and doing so requires having
        # walked the while loop's regions to build slice('total') correctly.
        with pytest.raises(d.ProbeDependencyError):
            d.validate_probe_deps(exported, deps, paths)

        result = d.probe_resolution(exported, {}, paths)
        assert result.total_ops > 0


class TestValidateProbeDeps:
    def test_ac15_rejects_the_revision2_error_naming_both_slices(self):
        """AC-15: validate_probe_deps rejects a declared edge whose
        slice-subset relation fails, naming both slices. Asserted on this
        spec's own revision-2 error -- a declared rbf -> encoded_positions
        edge for the aminx graph, which must be rejected because
        encoded_positions does not consume rbf.
        """
        exported, paths = _toy_dag_export()
        bad_deps = {"encoded_positions": ("rbf",)}
        with pytest.raises(d.ProbeDependencyError) as excinfo:
            d.validate_probe_deps(exported, bad_deps, paths)
        message = str(excinfo.value)
        assert "rbf" in message
        assert "encoded_positions" in message
        assert excinfo.value.predecessor == "rbf"
        assert excinfo.value.probe == "encoded_positions"
        assert len(excinfo.value.predecessor_slice) > 0
        assert len(excinfo.value.probe_slice) >= 0

    def test_validate_probe_deps_accepts_a_genuine_edge(self):
        exported, paths = _toy_dag_export()
        good_deps = {"edges_concat": ("rbf", "encoded_positions")}
        assert d.validate_probe_deps(exported, good_deps, paths) is None

    def test_validate_probe_deps_missing_path_entry_raises(self):
        exported, paths = _toy_dag_export()
        incomplete = {k: v for k, v in paths.items() if k != "rbf"}
        deps = {"edges_concat": ("rbf", "encoded_positions")}
        with pytest.raises(ValueError, match="rbf"):
            d.validate_probe_deps(exported, deps, incomplete)


# --------------------------------------------------------------------------
# Severity / DivergenceClass sanity
# --------------------------------------------------------------------------


class TestSeverityOrdering:
    def test_severity_is_ordered(self):
        assert d.Severity.IDENTICAL < d.Severity.WITHIN_BUDGET < d.Severity.BEYOND_BUDGET

    def test_divergence_class_has_exactly_five_members(self):
        assert {c.value for c in d.DivergenceClass} == {
            "DISCRETE_FLIP",
            "INJECTED",
            "AMPLIFIED",
            "ATTENUATED",
            "CLEAN",
        }

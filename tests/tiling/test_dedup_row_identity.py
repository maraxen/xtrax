"""Tests for exact per-leaf row identity in synthesize_dedup_spec (spec 260922 §5,
AC-27..AC-35, AC-44..AC-48), plus the §7.1 structural transfer oracle
(check_device_allowlist / collect_np_sites, AC-45..AC-47).
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.dedup import DedupSpec, get_k_bucket
from xtrax.tiling.dedup_synthesis import (
    synthesize_dedup_spec,
    verify_dedup_spec,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEDUP_SYNTHESIS_PATH = REPO_ROOT / "src" / "xtrax" / "tiling" / "dedup_synthesis.py"


def _fixture(pattern_a, pattern_b, n_each=50, dtype=np.float32):
    """Fixture rule (spec 260922 §7, AC-27..AC-32): two named row patterns,
    each repeated n_each times (default N=100)."""
    a = np.tile(np.asarray(pattern_a, dtype=dtype).reshape(1, -1), (n_each, 1))
    b = np.tile(np.asarray(pattern_b, dtype=dtype).reshape(1, -1), (n_each, 1))
    return np.concatenate([a, b], axis=0)


class TestAC27SignedZero:
    def test_signed_zero_rows_are_distinct(self):
        leaf = _fixture([0.0, 1.0], [-0.0, 1.0])
        result = synthesize_dedup_spec([leaf], threshold=0.1)
        assert result.stage == "synthesized"
        assert result.spec is not None
        spec = result.spec
        assert spec.k == 2
        # the two patterns map to different canonical slots.
        assert spec.index_map[0] != spec.index_map[50]
        verify_dedup_spec(spec, [leaf])


class TestAC28MixedDtypeCollision:
    def test_int32_wide_values_beside_constant_float32_leaf_distinct(self):
        int_leaf = _fixture([2**24], [2**24 + 1], dtype=np.int32)
        float_leaf = _fixture([1.0], [1.0], dtype=np.float32)

        result = synthesize_dedup_spec([int_leaf, float_leaf], threshold=0.1)
        assert result.stage == "synthesized"
        assert result.spec is not None
        assert result.spec.k == 2
        verify_dedup_spec(result.spec, [int_leaf, float_leaf])

    def test_hand_built_merge_of_the_two_patterns_is_unsound(self):
        int_leaf = _fixture([2**24], [2**24 + 1], dtype=np.int32)
        float_leaf = _fixture([1.0], [1.0], dtype=np.float32)
        # Hand-built spec that (incorrectly) claims both patterns are the same row.
        bad_spec = DedupSpec(
            axis_name="batch",
            unique_indices=np.array([0], dtype=np.int32),
            index_map=np.zeros(100, dtype=np.int32),
            k=1,
        )
        with pytest.raises(Exception) as exc_info:
            verify_dedup_spec(bad_spec, [int_leaf, float_leaf])
        from xtrax.tiling.dedup_synthesis import DedupSpecVerificationError

        assert isinstance(exc_info.value, DedupSpecVerificationError)
        assert exc_info.value.check == "row_mismatch"


class TestAC29NumpyInt64WideValues:
    def test_x64_off_wide_int64_values_distinct(self):
        leaf = _fixture([5], [2**32 + 5], dtype=np.int64)
        result = synthesize_dedup_spec([leaf], threshold=0.1)
        assert result.stage == "synthesized"
        assert result.spec is not None
        assert result.spec.k == 2


class TestAC30IdenticalNaN:
    def test_bitwise_identical_nan_rows_dedup(self):
        leaf = _fixture([np.nan], [1.0])
        result = synthesize_dedup_spec([leaf], threshold=0.1)
        assert result.stage == "synthesized"
        assert result.spec is not None
        assert result.spec.k == 2


class TestAC31AxisNotZero:
    def test_axis1_batch_columns(self):
        pattern_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pattern_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        cols = [pattern_a] * 50 + [pattern_b] * 50
        leaf = np.stack(cols, axis=1)  # shape (3, 100)
        assert leaf.shape == (3, 100)

        result = synthesize_dedup_spec([leaf], axis=1, threshold=0.1)
        assert result.stage == "synthesized"
        assert result.spec is not None
        assert len(result.spec.index_map) == 100
        assert result.spec.k == 2
        verify_dedup_spec(result.spec, [leaf], axis=1)


class TestAC32MixedDtypeAccounting:
    def test_true_byte_widths_not_promoted(self):
        int_leaf = _fixture([1], [2], dtype=np.int8)
        float_leaf = _fixture([1.0], [2.0], dtype=np.float32)

        result = synthesize_dedup_spec([int_leaf, float_leaf], threshold=0.1, max_sample_rows=4096)
        assert result.stage == "synthesized"
        spec = result.spec
        assert spec is not None
        assert spec.k == 2
        assert result.k_bucket_bytes == get_k_bucket(2) * 5 == 10
        assert result.transfer_bytes_spent == 5 * (100 + 100) == 1000


class TestAC33SampleGatherOnDevice:
    def _all_unique_batch(self, N=1000):
        return jnp.asarray(np.arange(N, dtype=np.float32).reshape(-1, 1))

    def test_only_sampled_rows_transferred(self, monkeypatch):
        import xtrax.tiling.dedup_synthesis as ds

        batch = self._all_unique_batch()
        original = ds._to_host
        calls = []

        def wrapper(x):
            calls.append((type(x), getattr(x, "shape", None)))
            return original(x)

        monkeypatch.setattr(ds, "_to_host", wrapper)

        result = synthesize_dedup_spec([batch], threshold=0.5, max_sample_rows=64)

        assert result.stage == "no_duplication"
        assert len(calls) == 1
        arg_type, arg_shape = calls[0]
        assert issubclass(arg_type, jax.Array)
        assert arg_shape[0] <= 64

    def test_mutation_control_host_slice_after_full_transfer_is_red(self, monkeypatch):
        """§7.1(6): replacing the device gather with a host-side slice after a
        full transfer must break the AC-33 shape assertion."""
        import xtrax.tiling.dedup_synthesis as ds

        def mutated_sample_stage(stacked, N, max_sample_rows):
            sample_count = min(N, max_sample_rows)
            idx = np.round(np.linspace(0, N - 1, sample_count)).astype(np.int64)
            idx = np.unique(idx)
            # MUTATION: full transfer, then host-side slice (spec §7.1(6)).
            sampled_rows = ds._to_host(stacked)[idx]
            sampled_ratio = ds._estimate_duplication_ratio(sampled_rows)
            element_width = ds._element_width_bytes(stacked)
            transfer_bytes = len(idx) * element_width
            n_unique_sampled = round(len(idx) * (1.0 - sampled_ratio))
            return sampled_ratio, n_unique_sampled, transfer_bytes

        monkeypatch.setattr(ds, "_sample_stage", mutated_sample_stage)

        batch = self._all_unique_batch()
        original = ds._to_host
        calls = []

        def wrapper(x):
            calls.append((type(x), getattr(x, "shape", None)))
            return original(x)

        monkeypatch.setattr(ds, "_to_host", wrapper)

        synthesize_dedup_spec([batch], threshold=0.5, max_sample_rows=64)

        # Under the mutation, the single _to_host call transfers ALL N rows,
        # not <=64 sampled rows -- the AC-33 shape assertion would fail.
        assert len(calls) == 1
        _, arg_shape = calls[0]
        assert arg_shape[0] == 1000
        assert arg_shape[0] > 64  # this is what makes AC-33's own assertion red


class TestAC34SharedLeafRowBytes:
    def test_leaf_row_bytes_called_once_per_leaf_by_both_functions(self, monkeypatch):
        import xtrax.tiling.dedup_synthesis as ds

        leaf = _fixture([1.0], [2.0])
        original = ds._leaf_row_bytes
        counts = {"n": 0}

        def wrapper(leaf_arr, axis, N):
            counts["n"] += 1
            return original(leaf_arr, axis, N)

        monkeypatch.setattr(ds, "_leaf_row_bytes", wrapper)

        result = synthesize_dedup_spec([leaf], threshold=0.1)
        assert counts["n"] == 1

        counts["n"] = 0
        verify_dedup_spec(result.spec, [leaf])
        assert counts["n"] == 1


class TestAC35RegressionFileUnmodified:
    def test_104_test_file_diff_against_origin_main_is_empty(self):
        proc = subprocess.run(
            ["git", "diff", "origin/main", "--", "tests/tiling/test_dedup_synthesis.py"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == "", proc.stdout


class TestAC44ZeroWidthLeaves:
    def test_zero_width_leaf_needs_no_special_case(self):
        N = 100
        a_pattern = np.array([1.0], dtype=np.float32)
        b_pattern = np.array([2.0], dtype=np.float32)
        leaf_a = np.concatenate([np.tile(a_pattern, (50, 1)), np.tile(b_pattern, (50, 1))], axis=0)
        leaf_z = np.zeros((N, 0), dtype=np.float32)

        result_az = synthesize_dedup_spec([leaf_a, leaf_z], threshold=0.1)
        result_a = synthesize_dedup_spec([leaf_a], threshold=0.1)

        assert result_az.stage == "synthesized"
        assert result_a.stage == "synthesized"
        np.testing.assert_array_equal(result_az.spec.unique_indices, result_a.spec.unique_indices)
        np.testing.assert_array_equal(result_az.spec.index_map, result_a.spec.index_map)

        result_z = synthesize_dedup_spec([leaf_z], threshold=0.1)
        assert result_z.stage == "synthesized"
        assert result_z.spec.k == 1
        verify_dedup_spec(result_z.spec, [leaf_z])


# ---------------------------------------------------------------------------
# §7.1 structural transfer oracle (AC-45..AC-47): AST checkers over the
# device-only helper set D and the module-wide np.*/builtin-conversion pin.
# ---------------------------------------------------------------------------

D = {
    "_device_rows",
    "_device_bytes_plain",
    "_device_bytes_bool",
    "_device_bytes_complex",
    "_device_bytes_subbyte_signed",
    "_device_bytes_subbyte_unsigned",
    "_concat_row_bytes",
    "_take_rows_device",
    "_mismatch_mask_device",
}

_ALLOWED_ATTR_METHODS = {"reshape", "astype", "real", "imag"}
_FORBIDDEN_NODE_TYPES = (
    ast.If,
    ast.While,
    ast.Assert,
    ast.Compare,
    ast.BoolOp,
    ast.JoinedStr,
    ast.IfExp,
    ast.Match,
)
_BUILTIN_CONVERSIONS = {
    "int",
    "float",
    "bool",
    "complex",
    "bytes",
    "bytearray",
    "memoryview",
    "list",
    "tuple",
}


def _is_allowed_call_func(func: ast.expr) -> bool:
    """§14 OBJ-R3-01: rules (a)/(b)/(c) for an allowed D-function Call.func."""
    if isinstance(func, ast.Name):
        return func.id in D or func.id == "_to_host"
    if isinstance(func, ast.Attribute):
        if func.attr in _ALLOWED_ATTR_METHODS:
            return True
        # Rule (a): chain of only Attribute nodes, rooted at Name jnp/lax.
        # Any Call or Subscript along the way leaves `node` as something
        # other than a bare ast.Name, which fails the final check below.
        node: ast.expr = func
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in {"jnp", "lax"}
    return False


def check_device_allowlist(source: str) -> list[str]:
    """§14 OBJ-R3-01/02: scan every D-named function for disallowed calls and
    forbidden node types. Returns a list of violation descriptions (empty ==
    clean).

    Walks the function's decorators and body (including nested lambdas and
    comprehensions), but not its parameter/return type annotations -- those
    are inert generic-subscript syntax (e.g. `Sequence[jax.Array]`), not
    runtime behaviour, and `from __future__ import annotations` does not
    change how `ast.parse` sees them.
    """
    tree = ast.parse(source)
    violations: list[str] = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name in D):
            continue
        scan_roots = list(node.decorator_list) + list(node.body)
        for root in scan_roots:
            for sub in ast.walk(root):
                if isinstance(sub, _FORBIDDEN_NODE_TYPES):
                    violations.append(f"{node.name}: forbidden {type(sub).__name__}")
                elif isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
                    violations.append(f"{node.name}: forbidden UnaryOp(Not)")
                elif isinstance(sub, ast.comprehension) and sub.ifs:
                    violations.append(f"{node.name}: forbidden comprehension filter")
                elif isinstance(sub, ast.Subscript):
                    if not (isinstance(sub.value, ast.Attribute) and sub.value.attr == "shape"):
                        violations.append(f"{node.name}: forbidden Subscript")
                elif isinstance(sub, ast.Call):
                    if not _is_allowed_call_func(sub.func):
                        violations.append(f"{node.name}: disallowed call {ast.dump(sub.func)}")
    return violations


def collect_np_sites(source: str) -> set[tuple[str, str]]:
    """§14 OBJ-R3-01/03: module-wide (function, attr) pairs for every
    `np.<attr>(...)` call and every builtin-conversion call, keyed by the
    enclosing module-level function's name."""
    tree = ast.parse(source)
    sites: set[tuple[str, str]] = set()
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "np"
            ):
                sites.add((fn.name, func.attr))
            elif isinstance(func, ast.Name) and func.id in _BUILTIN_CONVERSIONS:
                sites.add((fn.name, func.id))
    return sites


EXPECTED_NP_SITES = {
    ("_validate_batch_leaves", "asarray"),
    ("_host_leaf_row_bytes", "ascontiguousarray"),
    ("_host_leaf_row_bytes", "moveaxis"),
    ("_to_host", "asarray"),
    ("_sample_stage", "linspace"),
    ("_sample_stage", "round"),
    ("_sample_stage", "unique"),
    ("_exact_stage", "unique"),
    ("_exact_stage", "where"),
    ("_exact_stage", "array"),
    ("_exact_stage", "argsort"),
    ("_exact_stage", "empty"),
    ("_exact_stage", "arange"),
    ("_exact_stage", "asarray"),
    ("_estimate_duplication_ratio", "unique"),
    ("_check_spec_structure", "issubdtype"),
}


class TestAC45DeviceAllowlist:
    def test_every_d_function_exists_at_module_level(self):
        source = DEDUP_SYNTHESIS_PATH.read_text()
        tree = ast.parse(source)
        top_level_funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        missing = D - top_level_funcs
        assert not missing, f"D functions missing at module level: {missing}"

    def test_no_forbidden_call_or_node_in_real_source(self):
        source = DEDUP_SYNTHESIS_PATH.read_text()
        violations = check_device_allowlist(source)
        assert violations == []


class TestAC46NpSitePin:
    def test_np_sites_equal_pinned_literal(self):
        source = DEDUP_SYNTHESIS_PATH.read_text()
        sites = collect_np_sites(source)
        assert sites == EXPECTED_NP_SITES

    def test_verify_dedup_spec_has_no_np_sites(self):
        """§14 OBJ-R3-04: verify_dedup_spec itself uses no np.* calls."""
        source = DEDUP_SYNTHESIS_PATH.read_text()
        tree = ast.parse(source)
        verify_fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_dedup_spec"
        )
        for sub in ast.walk(verify_fn):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                assert not (isinstance(sub.func.value, ast.Name) and sub.func.value.id == "np"), (
                    "verify_dedup_spec must contain no np.* calls"
                )


class TestAC47CheckerControls:
    def test_np_asarray_inside_take_rows_device_flagged(self):
        source = """
def _take_rows_device(stacked, idx):
    return np.asarray(stacked)
"""
        # Must be recognized as D-scoped; embed via the real D constant name.
        violations = check_device_allowlist(source)
        assert any("_take_rows_device" in v for v in violations)

    def test_np_unique_inside_concat_row_bytes_flagged(self):
        source = """
def _concat_row_bytes(blocks):
    return np.unique(blocks)
"""
        violations = check_device_allowlist(source)
        assert any("_concat_row_bytes" in v for v in violations)

    def test_if_inside_device_rows_flagged(self):
        source = """
def _device_rows(leaf, axis, N):
    if axis == 0:
        return leaf
    return leaf
"""
        violations = check_device_allowlist(source)
        assert any("_device_rows" in v and "If" in v for v in violations)

    def test_comprehension_filter_in_mismatch_mask_device_flagged(self):
        source = """
def _mismatch_mask_device(blocks, canon):
    return jnp.stack([b for b in blocks if jnp.any(b)], axis=1)
"""
        violations = check_device_allowlist(source)
        assert any("_mismatch_mask_device" in v for v in violations)

    def test_unaryop_not_in_device_rows_flagged(self):
        source = """
def _device_rows(leaf, axis, N):
    return not jnp.any(leaf)
"""
        violations = check_device_allowlist(source)
        assert any("_device_rows" in v and "Not" in v for v in violations)

    def test_subscript_in_take_rows_device_flagged(self):
        source = """
def _take_rows_device(stacked, idx):
    mask = idx
    return stacked[mask]
"""
        violations = check_device_allowlist(source)
        assert any("_take_rows_device" in v and "Subscript" in v for v in violations)

    def test_call_in_chain_disqualifies_rule_a(self):
        source = """
def _device_bytes_plain(m):
    return jnp.sum(m).item()
"""
        violations = check_device_allowlist(source)
        assert any("_device_bytes_plain" in v for v in violations)

    def test_extra_np_array_site_detected(self):
        source = """
def _validate_batch_leaves(batch_leaves, axis):
    return np.array(batch_leaves)
"""
        sites = collect_np_sites(source)
        assert ("_validate_batch_leaves", "array") in sites

    def test_memoryview_plant_in_sample_stage_detected(self):
        source = """
def _sample_stage(stacked, N, max_sample_rows):
    return memoryview(stacked)
"""
        sites = collect_np_sites(source)
        assert ("_sample_stage", "memoryview") in sites


class TestAC48Guards:
    def test_to_host_rejects_non_jax_array(self):
        from xtrax.tiling.dedup_synthesis import _to_host

        with pytest.raises(TypeError):
            _to_host(np.zeros(3))

    def test_host_leaf_row_bytes_rejects_non_ndarray(self):
        from xtrax.tiling.dedup_synthesis import _host_leaf_row_bytes

        with pytest.raises(TypeError):
            _host_leaf_row_bytes(jnp.zeros((3, 1)), 0, 3, "plain")

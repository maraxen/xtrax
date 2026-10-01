"""Numerical parity: an independent reference vs the IREE-compiled artifact.

Small float differences around 1e-6 are expected and fine, since fusion ordering
differs between XLA and IREE. A large difference means the export changed
semantics, which is what this check exists to catch. Integer and bool values have
no such noise -- an index or a mask is either right or wrong -- so they are
compared exactly: a float tolerance would let ``1_000_009`` pass for
``1_000_000`` at ``rtol=1e-5``.

Every backend compares the same way (:func:`compare_leaves`): output leaf by leaf,
integers and bools exactly, and any dtype change across the export is a failure.

What it bounds, precisely: comparing the compiled artifact against an
independently-computed oracle bounds *lowering* fidelity. Comparing the composed
callable against itself under two backends would bound nothing -- both sides
change identically under a composition error such as wrong nesting, a dropped
boundary, or a mis-shaped carry. Hence ``verify_native_parity`` takes the
expected value as an argument and never re-derives it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.export.compile import run_native_vmfb

__all__ = ["LeafParityResult", "ParityResult", "compare", "compare_leaves", "verify_native_parity"]


@dataclass(frozen=True)
class ParityResult:
    """Outcome of one parity comparison.

    Attributes:
        passed: Whether the arrays matched: within tolerance for floats, exactly
            for integers and bools.
        max_abs_diff: Largest absolute elementwise difference; inf on a shape
            mismatch.
        atol: Absolute tolerance used (0.0 for an exact comparison).
        rtol: Relative tolerance used (0.0 for an exact comparison).
        shape_expected: Shape of the reference value.
        shape_actual: Shape of the artifact's output.
    """

    passed: bool
    max_abs_diff: float
    atol: float
    rtol: float
    shape_expected: tuple[int, ...]
    shape_actual: tuple[int, ...]

    def summary(self) -> str:
        """Render a one-line verdict naming the tolerances or the shape mismatch."""
        verdict = "PASS" if self.passed else "FAIL"
        if self.shape_expected != self.shape_actual:
            return (
                f"{verdict}: shape mismatch expected {self.shape_expected}, got {self.shape_actual}"
            )
        if self.atol == 0.0 and self.rtol == 0.0:
            return f"{verdict}: max|diff| = {self.max_abs_diff:.3e} (exact integer comparison)"
        return (
            f"{verdict}: max|diff| = {self.max_abs_diff:.3e} "
            f"(atol={self.atol:g}, rtol={self.rtol:g})"
        )


def _is_exact_dtype(dtype: np.dtype) -> bool:
    """Integer and bool values are compared exactly, never with a tolerance."""
    return bool(np.issubdtype(dtype, np.integer)) or dtype == np.bool_


def _exact_max_abs_diff(expected: np.ndarray, actual: np.ndarray) -> float:
    """Largest |expected - actual| for an exact comparison, never contradicting it.

    Two integer/bool arrays are differenced as Python ints, so uint64 above 2**63
    neither wraps nor rounds. Anything else (a float artifact output) is differenced
    in float64 without truncation, and a NaN or inf difference reads as inf.
    """
    if _is_exact_dtype(actual.dtype):
        diffs = np.abs(expected.astype(object) - actual.astype(object))
        return float(max(diffs.ravel(), default=0))
    with np.errstate(invalid="ignore", over="ignore"):
        diffs = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    diffs = np.where(np.isfinite(diffs), diffs, np.inf)
    return float(diffs.max())


def _exact(expected: np.ndarray, actual: np.ndarray) -> ParityResult:
    """Exact comparison for integer and bool references.

    The artifact's output must itself be integer or bool: a float output equal in
    value (``1.0`` for ``True``) still FAILS, since the export changed what kind of
    value the program returns.
    """
    shapes_match = expected.shape == actual.shape
    passed = (
        shapes_match and _is_exact_dtype(actual.dtype) and bool(np.array_equal(expected, actual))
    )
    if not shapes_match:
        diff = float("inf")
    elif expected.size:
        diff = _exact_max_abs_diff(expected, actual)
    else:
        diff = 0.0
    return ParityResult(
        passed=passed,
        max_abs_diff=diff,
        atol=0.0,
        rtol=0.0,
        shape_expected=tuple(expected.shape),
        shape_actual=tuple(actual.shape),
    )


def compare(
    expected: object,
    actual: object,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ParityResult:
    """Compare two arrays elementwise.

    Args:
        expected: The reference value.
        actual: The value produced by the compiled artifact.
        atol: Absolute tolerance, for float values.
        rtol: Relative tolerance, for float values.

    Returns:
        A ParityResult. Integer and bool values (judged by ``expected``'s dtype) are
        compared exactly and the tolerances ignored; an integer/bool reference
        against a non-integer, non-bool ``actual`` fails. A shape mismatch
        short-circuits to a failure rather than broadcasting: a silently broadcast
        comparison is how a real regression gets missed.
    """
    exp = np.asarray(jnp.asarray(expected))
    act = np.asarray(actual)
    if _is_exact_dtype(exp.dtype):
        return _exact(exp, act)

    if exp.shape != act.shape:
        return ParityResult(
            passed=False,
            max_abs_diff=float("inf"),
            atol=atol,
            rtol=rtol,
            shape_expected=tuple(exp.shape),
            shape_actual=tuple(act.shape),
        )

    max_diff = float(np.max(np.abs(exp - act))) if exp.size else 0.0
    passed = bool(np.allclose(exp, act, atol=atol, rtol=rtol))
    return ParityResult(
        passed=passed,
        max_abs_diff=max_diff,
        atol=atol,
        rtol=rtol,
        shape_expected=tuple(exp.shape),
        shape_actual=tuple(act.shape),
    )


@dataclass(frozen=True)
class LeafParityResult(ParityResult):
    """A ``ParityResult`` over every output leaf, integers compared exactly.

    The inherited fields describe the first failing leaf, or the first leaf when
    all pass, except ``passed`` (every leaf) and ``max_abs_diff`` (the maximum
    across leaves; inf on a shape or dtype mismatch).

    Attributes:
        leaf_results: One ``ParityResult`` per leaf, in output order.
        dtype_mismatches: ``"leaf <i>: expected <dtype>, got <dtype>"`` for each
            leaf whose dtype changed across the export, or a leaf-count mismatch.
    """

    leaf_results: tuple[ParityResult, ...] = ()
    dtype_mismatches: tuple[str, ...] = ()

    def summary(self) -> str:
        """Render a verdict naming the failing leaf and how it failed."""
        verdict = "PASS" if self.passed else "FAIL"
        if self.dtype_mismatches:
            return f"{verdict}: " + "; ".join(self.dtype_mismatches)
        n = len(self.leaf_results)
        for i, leaf in enumerate(self.leaf_results):
            if not leaf.passed:
                return f"{verdict}: leaf {i} of {n}: {leaf.summary().split(': ', 1)[1]}"
        return f"{verdict}: {n} leaf/leaves, max|diff| = {self.max_abs_diff:.3e}"


def _expected_leaves(expected: Any, n_actual: int) -> list[np.ndarray]:
    """Flatten the oracle the same way for every backend.

    A pytree oracle is flattened to its leaves. But an array-LIKE oracle, e.g. a
    nested Python list for a single (2, 2) output, would flatten to scalar leaves;
    when the artifact has exactly one output and the leaf counts disagree, the
    whole oracle is that one array, as ``compare``'s ``jnp.asarray`` reads it.
    Only a Python list/tuple is read that way: a dict, None, or a ragged list keeps
    its own leaves, and a count mismatch is reported rather than raised. Each leaf
    goes through ``jnp.asarray``, which in a 32-bit process narrows a NumPy oracle's
    float64/int64 to float32/int32, as the traced program would.
    """
    leaves = jax.tree_util.tree_leaves(expected)
    if len(leaves) != n_actual and n_actual == 1 and isinstance(expected, (list, tuple)):
        try:
            leaves = [np.asarray(jnp.asarray(expected))]
        except (TypeError, ValueError):
            pass  # ragged or non-numeric: not one array; report the leaf count
    return [np.asarray(jnp.asarray(x)) for x in leaves]


def _narrower_same_kind(actual: np.dtype, expected: np.dtype) -> bool:
    """``actual`` is the same kind as ``expected`` (int/uint/float) and narrower."""
    return actual.kind == expected.kind and actual.itemsize < expected.itemsize


def _numpy_default_width_leaves(expected: Any, n_actual: int) -> list[bool]:
    """Per leaf: is the oracle a non-JAX value (NumPy array, Python scalar/list)?

    In a 32-bit process ``jnp.asarray`` narrows such a leaf to the program's widths.
    Under ``jax_enable_x64`` it does not, so a NumPy oracle stays at NumPy's 64-bit
    default while an f32/i32 program returns f32/i32. That width difference is the
    oracle's, not the export's: values are still compared (exactly, for integers),
    but it is not a dtype mismatch. A JAX-array oracle's dtype is explicit, and an
    artifact WIDER than the oracle (int32 -> int64) is always a mismatch.
    """
    if not jax.config.jax_enable_x64:
        # 32-bit: jnp.asarray already narrowed the oracle, so every width counts.
        return [False] * max(n_actual, len(jax.tree_util.tree_leaves(expected)))
    leaves = jax.tree_util.tree_leaves(expected)
    if len(leaves) != n_actual and n_actual == 1 and isinstance(expected, (list, tuple)):
        leaves = [expected]
    return [not isinstance(x, jax.Array) for x in leaves]


def compare_leaves(
    expected: Any,
    actual_leaves: Sequence[Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> LeafParityResult:
    """Compare an artifact's output leaves against a reference, leaf by leaf.

    Args:
        expected: The reference value (any pytree, or one array-like).
        actual_leaves: The artifact's outputs, flattened to leaves in order.
        atol: Absolute tolerance for float leaves.
        rtol: Relative tolerance for float leaves.

    Returns:
        A ``LeafParityResult``. It fails on a leaf-count mismatch, on any dtype
        change, on any integer/bool difference, or on a float leaf outside
        tolerance.
    """
    act = [np.asarray(x) for x in actual_leaves]
    exp = _expected_leaves(expected, len(act))
    numpy_wide = _numpy_default_width_leaves(expected, len(act))
    if len(exp) != len(act):
        return LeafParityResult(
            passed=False,
            max_abs_diff=float("inf"),
            atol=atol,
            rtol=rtol,
            shape_expected=(len(exp),),
            shape_actual=(len(act),),
            dtype_mismatches=(f"expected {len(exp)} output leaves, got {len(act)}",),
        )

    results: list[ParityResult] = []
    mismatches: list[str] = []
    for i, (e, a) in enumerate(zip(exp, act, strict=True)):
        if e.dtype != a.dtype and not (numpy_wide[i] and _narrower_same_kind(a.dtype, e.dtype)):
            mismatches.append(f"leaf {i}: expected {e.dtype}, got {a.dtype}")
        results.append(compare(e, a, atol=atol, rtol=rtol))

    passed = not mismatches and all(r.passed for r in results)
    lead = next((r for r in results if not r.passed), results[0] if results else None)
    max_diff = float("inf") if mismatches else max((r.max_abs_diff for r in results), default=0.0)
    return LeafParityResult(
        passed=passed,
        max_abs_diff=max_diff,
        atol=atol,
        rtol=rtol,
        shape_expected=lead.shape_expected if lead else (),
        shape_actual=lead.shape_actual if lead else (),
        leaf_results=tuple(results),
        dtype_mismatches=tuple(mismatches),
    )


def verify_native_parity(
    expected: Any,
    vmfb_path: Path,
    concrete_inputs: Sequence[Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    function: str = "main",
) -> LeafParityResult:
    """Execute a native artifact and compare it against an independent reference.

    Args:
        expected: An independently-computed reference value. This must NOT be
            derived from the callable under test -- passing
            ``jax.jit(build_traceable_callable(...))(inputs)`` compares the
            composed callable against itself and verifies nothing about
            composition. Build it from the model directly, e.g.
            ``jnp.stack([step_fn(x) for x in xs])``.
        vmfb_path: Path to a native vmfb.
        concrete_inputs: Concrete arguments to execute with.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        function: Entry point name within the module.

    Returns:
        A ``LeafParityResult`` from :func:`compare_leaves`: every output leaf, integers
        and bools exactly. The IREE runtime returns a multi-output entry point's
        results as a flat tuple of leaves (a dict output included) and preserves
        int32/bool dtypes (measured 2026-10-01), so the leaves line up with the
        oracle's.
    """
    actual = run_native_vmfb(vmfb_path, *concrete_inputs, function=function)
    leaves = list(actual) if isinstance(actual, (tuple, list)) else [actual]
    return compare_leaves(expected, leaves, atol=atol, rtol=rtol)

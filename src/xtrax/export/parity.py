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
import numpy as np

from xtrax.export.compile import run_native_vmfb

__all__ = [
    "LeafParityResult",
    "ParityResult",
    "compare",
    "compare_leaves",
    "narrow_inputs",
    "to_jax_dtypes",
    "verify_native_parity",
]


@dataclass(frozen=True)
class ParityResult:
    """Outcome of one parity comparison.

    Attributes:
        passed: Whether the arrays matched: within tolerance for floats, exactly
            for integers and bools.
        max_abs_diff: Largest absolute elementwise difference; inf on a shape
            mismatch, or on an exact comparison whose output changed kind.
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
            return f"{verdict}: max|diff| = {self.max_abs_diff:.3e} (exact comparison)"
        return (
            f"{verdict}: max|diff| = {self.max_abs_diff:.3e} "
            f"(atol={self.atol:g}, rtol={self.rtol:g})"
        )


def to_jax_dtypes(value: Any) -> np.ndarray:
    """``value`` as a host NumPy array at the dtype JAX would give it.

    ``jax.dtypes.canonicalize_dtype`` narrows NumPy's float64/int64/uint64 to
    float32/int32/uint32 in a 32-bit process and keeps them under
    ``jax_enable_x64`` -- what ``jnp.asarray`` does, without copying the value to a
    device and back.
    """
    arr = np.asarray(value)
    if arr.dtype == object:
        return arr
    canonical = np.dtype(jax.dtypes.canonicalize_dtype(arr.dtype))
    return arr if arr.dtype == canonical else arr.astype(canonical)


def narrow_inputs(args: Sequence[Any]) -> list[Any]:
    """Narrow every leaf of every concrete input with :func:`to_jax_dtypes`.

    Applied to the inputs of every backend's artifact, so a float64 NumPy input
    feeds an f32 graph or entry point the way it would feed the traced program.
    Pytree structure is kept.
    """
    return [jax.tree_util.tree_map(to_jax_dtypes, arg) for arg in args]


def _is_exact_dtype(dtype: np.dtype) -> bool:
    """Integer and bool values are compared exactly, never with a tolerance."""
    return bool(np.issubdtype(dtype, np.integer)) or dtype == np.bool_


def _same_exact_kind(expected: np.dtype, actual: np.dtype) -> bool:
    """Both bool, or both integer: a bool <-> int change is a change of kind."""
    return (expected == np.bool_) == (actual == np.bool_) and _is_exact_dtype(actual)


def _exact_max_abs_diff(expected: np.ndarray, actual: np.ndarray) -> float:
    """Largest |expected - actual| for a failed exact comparison, never contradicting it.

    Integers up to 32 bits are differenced vectorised in int64, which cannot
    overflow for them. Wider integers (int64/uint64, where int64 differencing is
    exact only modulo 2**64) fall back to Python ints. A non-integer artifact
    output is differenced in float64 without truncation; NaN or inf reads as inf.
    """
    if _is_exact_dtype(actual.dtype):
        if max(expected.dtype.itemsize, actual.dtype.itemsize) <= 4:
            return float(np.max(np.abs(expected.astype(np.int64) - actual.astype(np.int64))))
        diffs = np.abs(expected.astype(object) - actual.astype(object))
        return float(max(diffs.ravel(), default=0))
    with np.errstate(invalid="ignore", over="ignore"):
        diffs = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    diffs = np.where(np.isfinite(diffs), diffs, np.inf)
    return float(diffs.max())


def _exact(expected: np.ndarray, actual: np.ndarray) -> ParityResult:
    """Exact comparison for integer and bool references.

    The artifact's output must be of the same kind -- integer for an integer
    reference, bool for a bool one. A float output equal in value (``1.0`` for
    ``True``), or a bool <-> int swap, FAILS with ``max_abs_diff`` inf: the export
    changed what kind of value the program returns.
    """
    shapes_match = expected.shape == actual.shape
    kind_ok = _same_exact_kind(expected.dtype, actual.dtype)
    passed = shapes_match and kind_ok and bool(np.array_equal(expected, actual))
    if not shapes_match or not kind_ok:
        diff = float("inf")
    elif passed or not expected.size:
        diff = 0.0
    else:
        diff = _exact_max_abs_diff(expected, actual)
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
        compared exactly and the tolerances ignored; ``actual`` must then be of the
        same kind (integer, or bool), or the comparison fails with ``max_abs_diff``
        inf. A shape mismatch short-circuits to a failure rather than
        broadcasting: a silently broadcast comparison is how a real regression
        gets missed.
    """
    exp = to_jax_dtypes(expected)
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


_SCALAR_TYPES = (bool, int, float, complex, np.generic)

# NumPy's default widths and the 32-bit dtypes JAX gives the same values.
_NUMPY_DEFAULT_TO_32 = {
    np.dtype(np.float64): np.dtype(np.float32),
    np.dtype(np.int64): np.dtype(np.int32),
    np.dtype(np.uint64): np.dtype(np.uint32),
}


def _is_scalar_nest(value: Any) -> bool:
    """A Python list/tuple whose leaves are all Python/NumPy scalars, e.g. [[1, 2], [3, 4]]."""
    if not isinstance(value, (list, tuple)):
        return False
    leaves = jax.tree_util.tree_leaves(value)
    return bool(leaves) and all(isinstance(x, _SCALAR_TYPES) for x in leaves)


def _oracle_leaves(expected: Any, n_actual: int) -> tuple[list[np.ndarray], list[bool]]:
    """Flatten the oracle the same way for every backend.

    Returns:
        The oracle's leaves as host arrays at JAX's dtypes (:func:`to_jax_dtypes`),
        and, per leaf, whether a narrower 32-bit artifact dtype is the oracle's own
        width rather than a change across the export (see below).

    A pytree oracle is flattened to its leaves. A nested list of SCALARS for a
    single output (``[[1, 2], [3, 4]]`` for one (2, 2) array) would flatten to
    scalar leaves; when the artifact has exactly one output, it is read as that one
    array. A tuple or list holding arrays is never stacked: it is a multi-output
    oracle, and a count mismatch against one output is reported, so an export that
    collapsed two outputs into one cannot verify. Ragged nests, dicts and None keep
    their leaves too.

    Under ``jax_enable_x64`` a non-JAX oracle leaf stays at NumPy's 64-bit default
    (float64/int64/uint64) while an f32/i32 program returns f32/i32. Exactly that
    pair -- a 64-bit NumPy default against its 32-bit counterpart -- is the oracle's
    width, not the export's; values are still compared (exactly, for integers). Any
    other narrowing, a JAX-array oracle, and everything in a 32-bit process (where
    the oracle was already narrowed) count as dtype changes.
    """
    raw = jax.tree_util.tree_leaves(expected)
    if len(raw) != n_actual and n_actual == 1 and _is_scalar_nest(expected):
        try:
            stacked = to_jax_dtypes(expected)
        except (TypeError, ValueError):
            stacked = None  # ragged: not one array; report the leaf count
        if stacked is not None and stacked.dtype != object:
            return [stacked], [jax.config.jax_enable_x64]
    leaves = [to_jax_dtypes(x) for x in raw]
    numpy_default = [jax.config.jax_enable_x64 and not isinstance(x, jax.Array) for x in raw]
    return leaves, numpy_default


def _is_dtype_change(expected: np.dtype, actual: np.dtype, numpy_default: bool) -> bool:
    if expected == actual:
        return False
    return not (numpy_default and _NUMPY_DEFAULT_TO_32.get(expected) == actual)


def compare_leaves(
    expected: Any,
    actual_leaves: Sequence[Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> LeafParityResult:
    """Compare an artifact's output leaves against a reference, leaf by leaf.

    Args:
        expected: The reference value (any pytree, or one nested list of scalars).
        actual_leaves: The artifact's outputs, flattened to leaves in order.
        atol: Absolute tolerance for float leaves.
        rtol: Relative tolerance for float leaves.

    Returns:
        A ``LeafParityResult``. It fails on a leaf-count mismatch, on any dtype
        change, on any integer/bool difference, or on a float leaf outside
        tolerance.
    """
    act = [np.asarray(x) for x in actual_leaves]
    exp, numpy_default = _oracle_leaves(expected, len(act))
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
    for i, (e, a, wide) in enumerate(zip(exp, act, numpy_default, strict=True)):
        if _is_dtype_change(e.dtype, a.dtype, wide):
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
        concrete_inputs: Concrete arguments to execute with. Each leaf is narrowed
            to JAX's dtypes first (:func:`narrow_inputs`), as for the onnx target.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        function: Entry point name within the module.

    Returns:
        A ``LeafParityResult`` from :func:`compare_leaves`: every output leaf, integers
        and bools exactly. The IREE runtime returns a multi-output entry point's
        results as a flat tuple of leaves (a dict output included) and preserves
        int32/bool dtypes, so the leaves line up with the oracle's
        (``tests/export/test_parity_leaves.py::test_iree_returns_flat_typed_leaves``).
    """
    inputs = narrow_inputs(concrete_inputs)
    actual = run_native_vmfb(vmfb_path, *inputs, function=function)
    leaves = list(actual) if isinstance(actual, (tuple, list)) else [actual]
    return compare_leaves(expected, leaves, atol=atol, rtol=rtol)

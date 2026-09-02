"""Numerical parity: an independent reference vs the IREE-compiled artifact.

Small differences around 1e-6 are expected and fine, since fusion ordering
differs between XLA and IREE. A large difference means the export changed
semantics, which is what this check exists to catch.

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

import jax.numpy as jnp
import numpy as np

from xtrax.export.compile import run_native_vmfb

__all__ = ["ParityResult", "compare", "verify_native_parity"]


@dataclass(frozen=True)
class ParityResult:
    """Outcome of one parity comparison.

    Attributes:
        passed: Whether the arrays matched within tolerance.
        max_abs_diff: Largest absolute elementwise difference; inf on a shape
            mismatch.
        atol: Absolute tolerance used.
        rtol: Relative tolerance used.
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
        return (
            f"{verdict}: max|diff| = {self.max_abs_diff:.3e} "
            f"(atol={self.atol:g}, rtol={self.rtol:g})"
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
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        A ParityResult. A shape mismatch short-circuits to a failure rather than
        broadcasting: a silently broadcast comparison is how a real regression
        gets missed.
    """
    exp = np.asarray(jnp.asarray(expected))
    act = np.asarray(actual)

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


def verify_native_parity(
    expected: Any,
    vmfb_path: Path,
    concrete_inputs: Sequence[Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    function: str = "main",
) -> ParityResult:
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
        A ParityResult comparing ``expected`` against the artifact's output.
    """
    actual = run_native_vmfb(vmfb_path, *concrete_inputs, function=function)
    return compare(expected, actual, atol=atol, rtol=rtol)

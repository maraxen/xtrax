"""Numerical parity: JAX reference vs the IREE-compiled artifact.

Small differences (~1e-6) are expected and fine -- fusion ordering differs between
XLA and IREE. A *large* difference means the export changed semantics, which is the
whole thing this spike exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ParityResult:
    """Outcome of one parity comparison."""

    passed: bool
    max_abs_diff: float
    atol: float
    rtol: float
    shape_expected: tuple[int, ...]
    shape_actual: tuple[int, ...]

    def summary(self) -> str:
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

    Shape mismatch short-circuits to a failure rather than broadcasting -- a
    silently broadcast comparison is how a real regression gets missed.
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


__all__ = ["ParityResult", "compare"]

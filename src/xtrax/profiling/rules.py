"""Shared steady-state rules for the unit probe (spec 261016 X5, rule half).

One definition of the dispatch-ratio rule, used by both ``xtrax profile`` (the
measure path and the ``--check`` path in ``xtrax.cli.profile_verb``) and by the
shipped negative controls in :mod:`xtrax.profiling.controls`. This module is a
leaf: it imports only the standard library and nothing else from xtrax.
"""

# Calibration status of the threshold, shown next to every flag. The orchestrator
# changes this one constant once the calibration result is in (spec X5).
THRESHOLD_STATUS = "provisional, uncalibrated until X5"

# Dispatch-ratio threshold: executions per element at or above this is the
# host-loop signature. Its calibration on real entrypoints is done by
# scripts/prof_calibrate_dispatch_ratio.py (spec 261016 X5); the calibration
# result decides whether the value changes. It is kept at 0.5 until then. The
# spike gap is wide (host loops >= 3.0, batched units <= 0.14).
EXEC_PER_ELEMENT_FLAG = 0.5


def steady_state_flags(
    exec_per_element: float,
    steady_compiles_per_unit: float,
    *,
    threshold: float = EXEC_PER_ELEMENT_FLAG,
) -> list[str]:
    """Return the human-readable flags for one measured unit.

    An empty list means the unit passes the steady-state rules. Both rules are
    applied on every call; the caller decides what a non-empty list means.

    Args:
        exec_per_element: XLA executions per unit divided by the scale-axis
            cardinality. A value at or above ``threshold`` is flagged.
        steady_compiles_per_unit: Backend compiles per steady unit. Any value
            above zero is flagged (the unit rebuilds its executable).
        threshold: Dispatch-ratio threshold. Defaults to ``EXEC_PER_ELEMENT_FLAG``.

    Returns:
        Flag strings in a fixed order: recompiles first, then dispatch ratio.
    """
    flags: list[str] = []
    if steady_compiles_per_unit > 0:
        flags.append(
            f"steady recompiles: n_compilations {steady_compiles_per_unit:.3f} per unit "
            "(expected 0; the unit rebuilds its executable on each call)"
        )
    if exec_per_element >= threshold:
        flags.append(
            f"exec/element {exec_per_element:.3f} >= {threshold} "
            f"({THRESHOLD_STATUS}; host-loop signature)"
        )
    return flags


__all__ = [
    "EXEC_PER_ELEMENT_FLAG",
    "THRESHOLD_STATUS",
    "steady_state_flags",
]

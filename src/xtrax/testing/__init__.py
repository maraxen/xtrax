"""Reference-parity harness: compare a candidate to an independent oracle.

These primitives live in the shipped tree. ``xtrax.devtools`` is the CI audit
surface and is excluded from the wheel; a caller comparing a JAX sampler to a
NumPy or torch reference imports this package instead. Torch stays in the
caller. Every entry point here takes callables and NumPy arrays.

JAX and torch generators are different streams. Feed both sides one
:class:`InjectedSource` rather than a shared integer seed.
"""

from xtrax.testing.guard import SelfParityError, assert_distinct_callables
from xtrax.testing.lanes import (
    CollapsedSamplerResult,
    DistributionalResult,
    KnobCoverage,
    NegativeControl,
    NegativeControlError,
    Scorer,
    StepDistribution,
    TeacherForcedResult,
    TieRecord,
    align_logits,
    collapsed_sampler_lane,
    distributional_lane,
    knob_coverage,
    map_token_ids,
    min_detectable_tv,
    require_negative_control,
    teacher_forced_lane,
)
from xtrax.testing.randomness import InjectedSource, order_from_randn

__all__ = [
    "CollapsedSamplerResult",
    "DistributionalResult",
    "InjectedSource",
    "KnobCoverage",
    "NegativeControl",
    "NegativeControlError",
    "Scorer",
    "SelfParityError",
    "StepDistribution",
    "TeacherForcedResult",
    "TieRecord",
    "align_logits",
    "assert_distinct_callables",
    "collapsed_sampler_lane",
    "distributional_lane",
    "knob_coverage",
    "map_token_ids",
    "min_detectable_tv",
    "order_from_randn",
    "require_negative_control",
    "teacher_forced_lane",
]

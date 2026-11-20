"""Exact categorical co-occurrence counts (xtrax.categorical).

Stage 1 primitive: integer weighted contingency counts over compact
categorical codes. Holds no domain content (no entropy, no MI).

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md
"""

from xtrax.categorical.codes import (
    CategoricalCodes,
    code_dtype,
    encode_codes,
    weight_dtype,
)
from xtrax.categorical.counts import (
    STATUS_CODES_A,
    STATUS_CODES_B,
    STATUS_NEGATIVE_WEIGHT,
    STATUS_NONINTEGRAL_WEIGHT,
    STATUS_PAIRS_RANGE,
    STATUS_TOTAL_OVERFLOW,
    CooccurrenceCounts,
    cooccurrence_counts,
    iter_cooccurrence_row_blocks,
    status_messages,
)

__all__ = [
    "STATUS_CODES_A",
    "STATUS_CODES_B",
    "STATUS_NEGATIVE_WEIGHT",
    "STATUS_NONINTEGRAL_WEIGHT",
    "STATUS_PAIRS_RANGE",
    "STATUS_TOTAL_OVERFLOW",
    "CategoricalCodes",
    "CooccurrenceCounts",
    "code_dtype",
    "cooccurrence_counts",
    "encode_codes",
    "iter_cooccurrence_row_blocks",
    "status_messages",
    "weight_dtype",
]

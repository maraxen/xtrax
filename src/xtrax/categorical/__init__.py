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

__all__ = [
    "CategoricalCodes",
    "code_dtype",
    "encode_codes",
    "weight_dtype",
]

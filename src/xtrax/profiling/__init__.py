"""xtrax.profiling -- stage/scale-stamped JAX profiling measurement primitives.

Upstreamed from prolix ``scripts/profiling`` (branch wt-20260807-132628) on
2026-08-24; see .praxia/docs/specs/
260824_upstream-profiling-probe-tooling-from-prolix.md and, for the original
design rationale behind ProbeRecord and the claim-validity contract, prolix's
260817_jax-profiling-optimization-workflow.md section P1.

This package is a leaf: no imports of prolix, no sibling xtrax submodules,
no relative imports (AST-enforced in tests/profiling/test_claim_contract.py),
and jax is imported lazily only inside ProbeRecord's provenance
default_factories. trace.py's parsers are importable but deliberately NOT
re-exported here -- they are JAX-version-sensitive internals; go through
xtrax.profiling.trace explicitly so upgrades show up in grep.
"""

from xtrax.profiling.claims import (
    CONTRACT_VERSION,
    SCALE_EXTRAPOLATION_LIMIT,
    ClaimClass,
    ClaimValidityError,
    assert_claim_supported,
    paired_configs,
    permitted_claims,
    select_sources,
)
from xtrax.profiling.record import ProbeRecord

__all__ = [
    "CONTRACT_VERSION",
    "SCALE_EXTRAPOLATION_LIMIT",
    "ClaimClass",
    "ClaimValidityError",
    "ProbeRecord",
    "assert_claim_supported",
    "paired_configs",
    "permitted_claims",
    "select_sources",
]

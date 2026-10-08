"""xtrax.profiling -- stage/scale-stamped JAX profiling measurement primitives.

Upstreamed from prolix ``scripts/profiling`` (branch wt-20260807-132628) on
2026-08-24; see .praxia/docs/specs/
260824_upstream-profiling-probe-tooling-from-prolix.md and, for the original
design rationale behind ProbeRecord and the claim-validity contract, prolix's
260817_jax-profiling-optimization-workflow.md section P1.

This package is a leaf: no imports of prolix, no sibling xtrax submodules,
no relative imports (AST-enforced in tests/profiling/test_claim_contract.py),
and jax is imported lazily inside the functions that need it, not at package
import. trace.py's parsers (``parse_scopes``, ``scope_map_from_hlo_text``, ...)
are importable but deliberately NOT re-exported here -- they are
JAX-version-sensitive internals; go through ``xtrax.profiling.trace`` explicitly
so upgrades show up in grep. The trace instruments ``load_trace_events`` and
``hlo_text_for``, and the recompile guard ``count_backend_compiles`` /
``assert_no_recompile_after``, are public.
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
from xtrax.profiling.compile_count import (
    assert_no_recompile_after,
    count_backend_compiles,
)
from xtrax.profiling.jaxpr import iter_jaxpr_eqns, sub_jaxprs
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.trace import hlo_text_for, load_trace_events

__all__ = [
    "CONTRACT_VERSION",
    "SCALE_EXTRAPOLATION_LIMIT",
    "ClaimClass",
    "ClaimValidityError",
    "ProbeRecord",
    "assert_claim_supported",
    "assert_no_recompile_after",
    "count_backend_compiles",
    "hlo_text_for",
    "iter_jaxpr_eqns",
    "load_trace_events",
    "paired_configs",
    "permitted_claims",
    "select_sources",
    "sub_jaxprs",
]

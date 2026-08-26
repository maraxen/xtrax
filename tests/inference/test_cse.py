"""Tests for xtrax.inference.cse — spec 260825 §4.1 (AC1, AC2 + meta)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.inference import (
    CseDuplicateClass,
    CseReport,
    CseTraceError,
    analyze_cse,
)
from xtrax.inference import cse as cse_module

ABS = [jax.ShapeDtypeStruct((8,), jnp.float32)]


def _duplicated_fn(x):
    y = jnp.sin(x) * 2.0
    z = jnp.sin(x) * 2.0
    return y + z + jnp.exp(y)


class TestAnalyzeCse:
    def test_ac1_duplicated_fn_reports_two_classes(self):
        """AC1: duplicated sin(x)*2 feeding separate consumers -> TWO classes."""
        report = analyze_cse(_duplicated_fn, ABS)
        by_prim = {c.primitive: c.eqn_count for c in report.duplicates}
        assert by_prim == {"sin": 2, "mul": 2}
        assert all(c.eqn_count == 2 for c in report.duplicates)
        assert report.total_eqns == 7
        assert report.duplicate_eqns == 4

    def test_ac2_clean_fn_reports_no_duplicates(self):
        """AC2: zero duplicates -> empty tuple, sane totals (sin+cos+add = 3 eqns)."""
        report = analyze_cse(lambda x: jnp.sin(x) + jnp.cos(x), ABS)
        assert report.duplicates == ()
        assert report.duplicate_eqns == 0
        assert report.total_eqns == 3

    def test_report_sorted_by_wasted_bytes_desc(self):
        report = analyze_cse(_duplicated_fn, ABS)
        sizes = [c.est_wasted_bytes for c in report.duplicates]
        assert sizes == sorted(sizes, reverse=True)

    def test_literal_value_canonicalization(self):
        """Distinct Literal objects with equal values merge (OBJ-R2-01)."""

        # Two muls with distinct-but-equal scalar literals via different consts.
        def fn(x):
            a = x * jnp.asarray(2.0, jnp.float32)
            b = x * jnp.asarray(2.0, jnp.float32)
            return a + b

        report = analyze_cse(fn, ABS)
        by_prim = {c.primitive: c.eqn_count for c in report.duplicates}
        assert by_prim.get("mul", 0) == 2

    def test_no_false_merge_on_different_values(self):
        """Different literal values must NOT merge."""

        def fn(x):
            a = x * 2.0
            b = x * 3.0
            return a + b

        report = analyze_cse(fn, ABS)
        assert report.duplicates == ()

    def test_trace_cache_hit_flag(self):
        """Same fn object analyzed twice -> second reports trace_cache_hit."""
        # Isolate the module-level memo: prior tests may have analyzed fns.
        cse_module._TRACE_MEMO.clear()

        def fn(x):
            y = jnp.sin(x) * 2.0
            z = jnp.sin(x) * 2.0
            return y + z + jnp.exp(y)

        r1 = analyze_cse(fn, ABS)
        assert r1.trace_cache_hit is False
        r2 = analyze_cse(fn, ABS)
        assert r2.trace_cache_hit is True

    def test_fresh_callable_no_cache_hit(self):
        """Fresh callable with identical body -> fresh trace, no hit."""

        def other(x):
            y = jnp.sin(x) * 2.0
            z = jnp.sin(x) * 2.0
            return y + z + jnp.exp(y)

        assert analyze_cse(other, ABS).trace_cache_hit is False

    def test_arity_mismatch_raises_cse_trace_error(self):
        with pytest.raises(CseTraceError):
            analyze_cse(lambda a, b: a + b, ABS)

    def test_report_is_frozen_dataclasses(self):
        report = analyze_cse(_duplicated_fn, ABS)
        assert isinstance(report, CseReport)
        for dup in report.duplicates:
            assert isinstance(dup, CseDuplicateClass)
            with pytest.raises(Exception):
                dup.eqn_count = 99  # type: ignore[misc]

    def test_note_documents_xla_correspondence(self):
        report = analyze_cse(_duplicated_fn, ABS)
        assert "XLA" in report.note

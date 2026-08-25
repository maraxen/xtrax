"""Pins for review-swarm findings (jury audit of 716b156..HEAD).

Each test corresponds to a finding from the three-juror review of the
profiling upstream; the finding id in the docstring matches the triage
list. If one of these fails, a contract hole the jury found has re-opened.
"""

from __future__ import annotations

import json

import pytest

from xtrax.profiling import ClaimClass, ClaimValidityError, ProbeRecord
from xtrax.profiling.bench import (
    bench_metrics_from_stats,
    check_probe_id_collision,
    parse_bench_extra_info,
    sanitize_bench_fullname,
)
from xtrax.profiling.claims import assert_claim_supported
from xtrax.profiling.report import discover_records


def _base(**overrides):
    kwargs = dict(
        probe_id="t",
        stage=1,
        n_atoms=32,
        platform="cpu",
        metrics={"m": 1.0},
        git_sha="a" * 40,
        jax_version="0.10.2",
        jaxlib_version="0.10.2",
    )
    kwargs.update(overrides)
    return ProbeRecord(**kwargs)


class TestScopeValueValidation:
    """Finding 1: NaN/inf/wrong-arity/string scope entries via from_json."""

    def test_nan_scope_seconds_rejected_on_read(self, tmp_path):
        raw = json.loads(_base().to_json())
        raw["scopes"] = {"a": [float("nan"), 3]}
        path = tmp_path / "r.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ClaimValidityError, match="exclusive_seconds"):
            ProbeRecord.read(path)

    def test_nan_inf_negative_and_zero_count_rejected_as_claim_error(self):
        # These pass beartype's structural hint but violate the contract's
        # value rules -- __post_init__ must catch them as ClaimValidityError.
        for bad in (
            (float("nan"), 3),
            (float("inf"), 2),
            (-1.0, 2),
            (1.0, True),
            (1.0, 0),
        ):
            with pytest.raises(ClaimValidityError):
                _base(scopes={"a": bad}, attribution_method={"a": "named_scope"})

    def test_type_arity_garbage_rejected_at_boundary(self):
        # Wrong arity/types are refused by the beartype-checked annotation
        # itself; any exception is fine, silence is not.
        for bad in ([1.0, 2, 99], ["0.5", 3], "nope", 5):
            with pytest.raises(Exception):
                _base(scopes={"a": bad}, attribution_method={"a": "named_scope"})

    def test_negative_seconds_rejected(self):
        with pytest.raises(ClaimValidityError, match="non-negative"):
            _base(scopes={"a": (-1.0, 2)})

    def test_valid_mixed_none_scopes_still_accepted(self):
        rec = _base(
            scopes={"a": (0.5, 2), "b": None},
            attribution_method={"a": "named_scope"},
        )
        assert rec.scopes["b"] is None

    def test_attribution_for_null_label_allowed_fixture_style(self):
        # Committed prolix fixtures attribute null labels too (the label is
        # KNOWN named_scope even though it never appeared in the trace).
        rec = _base(
            scopes={"a": None, "b": (1.0, 1)},
            attribution_method={"a": "named_scope", "b": "named_scope"},
        )
        assert rec.scopes["a"] is None


class TestAttributionKeyEquality:
    """Finding 7: attribution keys must exactly match present scope labels."""

    def test_empty_attribution_with_present_label_rejected(self):
        with pytest.raises(ClaimValidityError, match="disagree"):
            _base(scopes={"a": (1.0, 1)}, attribution_method={})

    def test_extra_ghost_key_rejected(self):
        with pytest.raises(ClaimValidityError, match="disagree"):
            _base(
                scopes={"a": None},
                attribution_method={"ghost": "named_scope"},
            )


class TestDispatchCountStageFloor:
    """Finding 2: assert_claim_supported must enforce stage>=1 too."""

    def test_stage0_dispatch_metrics_fail_assert(self):
        rec = _base(
            stage=0,
            metrics={
                "n_executions": 1.0,
                "n_compilations": 1.0,
                "n_jit_traces": 1.0,
            },
        )
        with pytest.raises(ClaimValidityError, match="DISPATCH_COUNT requires stage>=1"):
            assert_claim_supported([rec], ClaimClass.DISPATCH_COUNT)


class TestGitShaAndTarget:
    """Findings 6 + 9: empty sha and non-positive END_TO_END target."""

    def test_empty_git_sha_rejected_for_term_ranking(self):
        recs = [
            _base(
                probe_id=f"p{i}",
                stage=2,
                platform="gpu",
                device_kind="h200",
                git_sha="",
                metrics={"total_step_seconds": 1.0},
                scopes={"s1": (1.0, 1), "s2": (0.5, 1)},
                attribution_method={"s1": "named_scope", "s2": "named_scope"},
            )
            for i in range(2)
        ]
        with pytest.raises(ClaimValidityError, match="unverifiable"):
            assert_claim_supported(recs, ClaimClass.TERM_RANKING)

    def test_end_to_end_zero_target_rejected(self):
        recs = [
            _base(
                probe_id=f"p{i}",
                stage=2,
                platform="gpu",
                device_kind="h200",
                metrics={"total_step_seconds": 1.0},
            )
            for i in range(2)
        ]
        with pytest.raises(ClaimValidityError, match="positive atom count"):
            assert_claim_supported(recs, ClaimClass.END_TO_END, target_n_atoms=0)


class TestMetricCoercion:
    """Findings 4 + bool smuggling: overflow wraps as ClaimValidityError."""

    def test_overflow_error_wrapped(self):
        with pytest.raises(ClaimValidityError, match="not coercible"):
            _base(metrics={"big": 10**400})

    def test_bool_metric_rejected(self):
        with pytest.raises(ClaimValidityError, match="boolean"):
            _base(metrics={"flag": True})


class TestBenchHardening:
    """Findings on the bridge itself."""

    def test_bool_declaration_rejected(self):
        with pytest.raises(ClaimValidityError, match="boolean"):
            parse_bench_extra_info({"xtrax_stage": True, "xtrax_n_atoms": 32})

    def test_non_coercible_stat_wrapped(self):
        with pytest.raises(ClaimValidityError, match="refusing to record"):
            bench_metrics_from_stats({"mean": "not-a-number"})

    def test_sanitize_collision_exists_and_is_detected(self):
        # The collision is real; detection makes it loud instead of an
        # overwrite.
        a = sanitize_bench_fullname("benchmarks/f.py::t[1__2]")
        b = sanitize_bench_fullname("benchmarks/f.py::t[1_2]")
        assert a == b
        seen = {a: "benchmarks/f.py::t[1_2]"}
        assert check_probe_id_collision(a, seen) == "benchmarks/f.py::t[1_2]"
        assert check_probe_id_collision("other", seen) is None


class TestReportLoudSkips:
    """Finding 3: unreadable records must warn, not vanish."""

    def test_unreadable_file_warns_but_others_load(self, tmp_path, capsys):
        good = tmp_path / "good.json"
        _base(probe_id="good").write(good)
        bad = tmp_path / "bad.json"
        bad.write_text("{corrupt")
        records = discover_records(paths=[good, bad])
        assert [r.probe_id for r in records] == ["good"]
        err = capsys.readouterr().err
        assert "INCOMPLETE" in err
        assert "bad.json" in err

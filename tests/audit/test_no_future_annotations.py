"""Foundation gate N0.2: ratchet against new `from __future__ import annotations` in xtrax."""

from pathlib import Path

FORBIDDEN = "from __future__ import annotations"

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"

# Legacy allowlist — burndown tracked in audit type-hardening dimension (N2+).
LEGACY_ALLOWLIST = {
    "src/xtrax/cli/__init__.py",
    "src/xtrax/cli/emit.py",
    "src/xtrax/cli/entrypoint.py",
    "src/xtrax/cli/errors.py",
    "src/xtrax/cli/explain.py",
    "src/xtrax/cli/export.py",
    "src/xtrax/cli/loader.py",
    "src/xtrax/cli/plan.py",
    "src/xtrax/cli/registry.py",
    "src/xtrax/cli/shapes.py",
    "src/xtrax/cli/sweep_verb.py",
    "src/xtrax/devtools/baseline.py",
    "src/xtrax/devtools/bootstrap.py",
    "src/xtrax/devtools/docs_judgment.py",
    "src/xtrax/findings.py",
    "src/xtrax/tombstone.py",
    "src/xtrax/devtools/empirical_oracle.py",
    "src/xtrax/devtools/gates/_interrogate.py",
    "src/xtrax/jaxlint_runner.py",
    "src/xtrax/devtools/gates/_performance_probes.py",
    "src/xtrax/devtools/gates/_trace_probe.py",
    "src/xtrax/devtools/gates/added_types_diff.py",
    "src/xtrax/devtools/gates/api_ergonomics.py",
    "src/xtrax/devtools/gates/correctness.py",
    "src/xtrax/devtools/gates/documentation.py",
    "src/xtrax/devtools/gates/jax_purity.py",
    "src/xtrax/devtools/gates/performance.py",
    "src/xtrax/devtools/gates/structure_complexity.py",
    "src/xtrax/devtools/gates/test_rigor.py",
    "src/xtrax/devtools/gates/type_hardening.py",
    "src/xtrax/devtools/judgment.py",
    "src/xtrax/devtools/judgment_scheduled.py",
    "src/xtrax/devtools/refute_promote.py",
    "src/xtrax/devtools/routing.py",
    "src/xtrax/devtools/rubrics.py",
    "src/xtrax/devtools/ruff_schedule.py",
    "src/xtrax/eda/explain.py",
    "src/xtrax/inference/__init__.py",
    "src/xtrax/inference/abstract.py",
    "src/xtrax/inference/api.py",
    "src/xtrax/inference/axes.py",
    "src/xtrax/inference/cse.py",
    "src/xtrax/inference/config.py",
    "src/xtrax/inference/memo.py",
    "src/xtrax/inference/errors.py",
    "src/xtrax/inference/schema.py",
    "src/xtrax/inference/verify.py",
    "src/xtrax/run/resolver.py",
    "src/xtrax/run/sink.py",
    "src/xtrax/run/spec.py",
    "src/xtrax/sparse/config.py",
    "src/xtrax/sparse/inference.py",
    "src/xtrax/sparse/manager.py",
    "src/xtrax/sparse/policy.py",
    "src/xtrax/stages/boundaries.py",
    "src/xtrax/tiling/bucket.py",
    "src/xtrax/tiling/carry.py",
    "src/xtrax/tiling/carry_shape.py",
    "src/xtrax/tiling/dedup.py",
    "src/xtrax/tiling/dispatch.py",
    "src/xtrax/tiling/iterator.py",
    "src/xtrax/tiling/plan.py",
    "src/xtrax/tiling/roles.py",
    "src/xtrax/tiling/strategy.py",
}


def test_no_new_future_annotations_imports() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN not in text:
            continue
        if rel not in LEGACY_ALLOWLIST:
            offenders.append(rel)
    assert offenders == [], f"New forbidden import in: {offenders}"


def test_legacy_allowlist_matches_actual_offenders() -> None:
    """Allowlist must equal current offenders — shrink-only ratchet."""
    actual = {
        str(p.relative_to(ROOT))
        for p in SRC.rglob("*.py")
        if FORBIDDEN in p.read_text(encoding="utf-8")
    }
    assert LEGACY_ALLOWLIST == actual, (
        f"Update LEGACY_ALLOWLIST after burndown.\n"
        f"extra in allowlist: {LEGACY_ALLOWLIST - actual}\n"
        f"missing from allowlist: {actual - LEGACY_ALLOWLIST}"
    )

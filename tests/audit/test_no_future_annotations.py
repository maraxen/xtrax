"""Foundation gate N0.2: ratchet against new `from __future__ import annotations` in xtrax."""

from pathlib import Path

FORBIDDEN = "from __future__ import annotations"

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"

# Legacy allowlist — burndown tracked in audit type-hardening dimension (N2+).
LEGACY_ALLOWLIST = {
    "src/xtrax/eda/explain.py",
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

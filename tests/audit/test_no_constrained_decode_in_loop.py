"""T1-12 fast/loud gate (#3067): constrained decoding (Outlines) stays a smoke demo only.

Outlines' constrained decoding (`scripts/smoke_outlines_constrained_decode.py`) is not wired
into the generate-then-validate authoring loop this sprint -- the AC requires "a test asserts
no loop code imports the constrained path as a hard dependency." `src/xtrax/` is that loop
code; the smoke script deliberately lives outside it and outside `REGISTRY`.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"

FORBIDDEN_SUBSTRINGS = ("import outlines", "from outlines")


def test_no_outlines_import_in_src_xtrax() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if any(forbidden in text for forbidden in FORBIDDEN_SUBSTRINGS):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"outlines import found in loop code: {offenders}"

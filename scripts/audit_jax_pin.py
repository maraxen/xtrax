#!/usr/bin/env python3
"""T1-03 AC-shim-import-assert (#3007): CI enforcement of the io_callback jax pin.

Imports xtrax.stages._callback -- which asserts the resolved jax's version and
io_callback signature at import time -- and reports PASS/FAIL. The shim's own
checks are the single source of truth; this script only wraps them for
release-readiness visibility, matching the other audit_*.py scripts.
"""

from __future__ import annotations

import sys


def audit_jax_pin() -> tuple[bool, str]:
    try:
        import xtrax.stages._callback  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, ""


def main(argv: list[str] | None = None) -> int:
    del argv
    passed, message = audit_jax_pin()
    if passed:
        print("PASS: jax pin gate (xtrax.stages._callback imported cleanly)")
        return 0

    print("FAIL: jax pin gate", file=sys.stderr)
    print(f"  {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

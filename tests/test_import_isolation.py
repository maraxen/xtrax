"""Lesson #145 guard: every top-level subpackage imports standalone.

Each subpackage under test is imported in a FRESH subprocess so there is no
shared module cache that could mask a cycle.  A returncode != 0 means the
import itself raised (ImportError, circular import, etc.).

Add new entries here whenever a new top-level subpackage is introduced.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Packages to verify — extend this list as the codebase grows.
_PACKAGES = [
    "xtrax.tiling.plan",
    "xtrax.tiling.roles",
    "xtrax.inference",
    "xtrax.inference.errors",
    "xtrax.run.spec",
    "xtrax.stages.bundle",
    "xtrax.loop",
    "xtrax.loop.admission",
]


@pytest.mark.parametrize("pkg", _PACKAGES)
def test_standalone_import(pkg: str) -> None:
    """Assert that *pkg* can be imported in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {pkg}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Standalone import of '{pkg}' failed (returncode={result.returncode}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_tiling_does_not_import_inference() -> None:
    """Assert that importing xtrax.tiling.plan does NOT pull in xtrax.inference."""
    code = (
        "import sys; "
        "import xtrax.tiling.plan; "
        "bad = [m for m in sys.modules if m.startswith('xtrax.inference')]; "
        "assert not bad, f'tiling imported inference modules: {bad}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"tiling.plan pulled in xtrax.inference modules.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

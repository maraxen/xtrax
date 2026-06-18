"""Tests for scoped beartype import hook (N2.3 / #1583)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import install_import_hook

from tests.conftest import XTRAX_BEARTYPE_PACKAGES

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PROBE = ROOT / "tests" / "fixtures" / "beartype_probe.py"


def test_beartype_packages_exclude_lazy_root() -> None:
    assert "xtrax" not in XTRAX_BEARTYPE_PACKAGES
    assert all(pkg.startswith("xtrax.") for pkg in XTRAX_BEARTYPE_PACKAGES)


def test_conftest_hook_enforces_xtrax_submodule_types() -> None:
    if os.environ.get("XTRAX_DISABLE_BEARTYPE") == "1":
        pytest.skip("beartype hook disabled via XTRAX_DISABLE_BEARTYPE=1")

    probe = importlib.import_module("xtrax.devtools._beartype_probe")
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        probe.bad_call()


def test_install_import_hook_smoke_on_fixture_module(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "probe_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "bad.py").write_text(
        FIXTURE_PROBE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        hook = install_import_hook("probe_pkg", "beartype.beartype")
        bad = importlib.import_module("probe_pkg.bad")
        importlib.reload(bad)
        with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
            bad.bad_call()
        hook.uninstall()
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "probe_pkg" or name.startswith("probe_pkg."):
                del sys.modules[name]


def test_valid_shape_passes_under_hook(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "good_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "good.py").write_text(
        """
import jax
import jax.numpy as jnp
from jaxtyping import Float

def strict_vec(x: Float[jax.Array, "3"]) -> Float[jax.Array, "3"]:
    return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        hook = install_import_hook("good_pkg", "beartype.beartype")
        good = importlib.import_module("good_pkg.good")
        importlib.reload(good)
        out = good.strict_vec(jnp.ones(3))
        assert out.shape == (3,)
        hook.uninstall()
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "good_pkg" or name.startswith("good_pkg."):
                del sys.modules[name]

"""Pytest session hooks for xtrax."""

from __future__ import annotations

import os

import pytest

# Scoped to xtrax subpackages — NOT the lazy ``xtrax`` root (pre-mortem #7).
XTRAX_BEARTYPE_PACKAGES = [
    "xtrax.checkpoint",
    "xtrax.data",
    "xtrax.devtools",
    "xtrax.distributed",
    "xtrax.eda",
    "xtrax.engine",
    "xtrax.export",
    "xtrax.io",
    "xtrax.profiling",
    "xtrax.run",
    "xtrax.safety",
    "xtrax.sparse",
    "xtrax.stages",
    "xtrax.tiling",
    "xtrax.training",
    "xtrax.transforms",
]


def pytest_configure(config: pytest.Config) -> None:
    del config
    if os.environ.get("XTRAX_DISABLE_BEARTYPE") == "1":
        return
    from jaxtyping import install_import_hook

    install_import_hook(XTRAX_BEARTYPE_PACKAGES, "beartype.beartype")

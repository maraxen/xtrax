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


# Process-global JAX numerics flags a test can flip with `jax.config.update` (#5681).
# Read once in pytest_configure, before any test (port/tests included) runs, as the
# values every test should start and end with.
_GUARDED_JAX_FLAGS = (
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_numpy_dtype_promotion",
    "jax_threefry_partitionable",
    "jax_default_prng_impl",
)


def _jax_flags() -> dict[str, object]:
    import jax

    return {name: getattr(jax.config, name) for name in _GUARDED_JAX_FLAGS}


_SESSION_JAX_FLAGS: dict[str, object] | None = None


def _restore_and_describe(found: dict[str, object], expected: dict[str, object]) -> str:
    import jax

    changed = {k: (expected[k], v) for k, v in found.items() if v != expected[k]}
    for name, (value, _) in changed.items():
        jax.config.update(name, value)
    return ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in changed.items())


@pytest.fixture(autouse=True)
def _jax_global_flags_unchanged():
    """Fail any test that leaves a global JAX flag changed, and restore it so the leak
    cannot fail unrelated later tests. Scope changes with `jax.enable_x64(...)` /
    `jax.default_matmul_precision(...)`, or restore in a `finally`."""
    expected = _SESSION_JAX_FLAGS
    assert expected is not None, "pytest_configure did not snapshot the JAX flags"
    before = _jax_flags()
    if before != expected:
        diff = _restore_and_describe(before, expected)
        pytest.fail(
            f"global JAX flags changed before this test started ({diff}): leaked by code "
            "outside tests/ (e.g. port/tests, which its conftest runs first) or at import"
        )
    yield
    after = _jax_flags()
    if after != expected:
        diff = _restore_and_describe(after, expected)
        pytest.fail(f"this test left global JAX flags changed ({diff}); restored")


def pytest_configure(config: pytest.Config) -> None:
    del config
    global _SESSION_JAX_FLAGS
    _SESSION_JAX_FLAGS = _jax_flags()
    if os.environ.get("XTRAX_DISABLE_BEARTYPE") == "1":
        return
    from jaxtyping import install_import_hook

    install_import_hook(XTRAX_BEARTYPE_PACKAGES, "beartype.beartype")

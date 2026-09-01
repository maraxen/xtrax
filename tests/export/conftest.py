"""Fakes letting the export suite run with no IREE toolchain installed.

Every fixture here is function-scoped and installs its fake with
``monkeypatch.setitem``, which auto-reverts. A session-scoped ``sys.modules``
mutation would leak a fake into a real-toolchain test running later in the same
pytest session -- exactly the job the CI toolchain job exists to do -- and the
failure would look like a bug in the artifact rather than in the fixture.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.plan import AxisSpec, BatchPlanner

# A recognisable stand-in for real vmfb bytes. Not valid flatbuffer; nothing in
# the fake path parses it.
FAKE_VMFB = b"FAKE-VMFB-CONTENTS"


class TinyMLP(eqx.Module):
    """The smallest model that still exercises a matmul and a nonlinearity."""

    w1: jax.Array
    w2: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the two layers with a tanh between them."""
        return jnp.tanh(x @ self.w1) @ self.w2


@pytest.fixture
def model() -> TinyMLP:
    """A deterministic TinyMLP with 8 -> 16 -> 4 shape."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    return TinyMLP(
        w1=jax.random.normal(k1, (8, 16), dtype=jnp.float32) * 0.1,
        w2=jax.random.normal(k2, (16, 4), dtype=jnp.float32) * 0.1,
    )


@pytest.fixture
def plan() -> Any:
    """A single-axis plan over 32 elements, batch size 8 (resolves to SafeMap)."""
    return BatchPlanner().plan([AxisSpec(name="batch", cardinality=32, default_batch_size=8)])


@pytest.fixture
def xs() -> jax.Array:
    """Concrete inputs matching ``plan``."""
    return jnp.arange(32 * 8, dtype=jnp.float32).reshape(32, 8) / 256.0


@pytest.fixture
def abstract_inputs(xs: jax.Array) -> list[jax.ShapeDtypeStruct]:
    """Abstract inputs matching ``xs``."""
    return [jax.ShapeDtypeStruct(xs.shape, xs.dtype)]


@pytest.fixture
def reference_fn(model: TinyMLP) -> Any:
    """An independent oracle built from the model, never from the composed callable.

    This is the shape a real caller must supply: applying the model directly,
    element by element. Passing ``jax.jit(build_traceable_callable(...))`` here
    would compare the callable under test against itself.
    """

    def _reference(inputs: Any) -> jax.Array:
        (arr,) = inputs
        return jnp.stack([model(arr[i]) for i in range(arr.shape[0])])

    return _reference


@dataclass
class FakeCompilerTools:
    """Stands in for ``iree.compiler.tools``, recording every invocation."""

    calls: list[dict[str, Any]]
    fail_first: bool = False
    payload: bytes = FAKE_VMFB

    def compile_str(self, source: Any, *, input_type: str, extra_args: list[str]) -> bytes:
        """Record the call and return fake vmfb bytes, or fail once if asked."""
        self.calls.append(
            {"source": source, "input_type": input_type, "extra_args": list(extra_args)}
        )
        if self.fail_first and len(self.calls) == 1:
            msg = "fake iree-compile rejected the current-version StableHLO"
            raise RuntimeError(msg)
        return self.payload


@pytest.fixture
def fake_tools() -> FakeCompilerTools:
    """A fake compiler that always succeeds."""
    return FakeCompilerTools(calls=[])


@pytest.fixture
def fake_compiler(monkeypatch: pytest.MonkeyPatch, fake_tools: FakeCompilerTools):
    """Install ``fake_tools`` as the compiler ``compile_for_target`` resolves."""
    from xtrax.export import compile as compile_mod

    monkeypatch.setattr(compile_mod, "_require_compiler", lambda: fake_tools)
    return fake_tools


class _FakeVmModule:
    name = "jit_fake_module"

    @classmethod
    def mmap(cls, _instance: Any, _path: str) -> _FakeVmModule:
        """Ignore the path; a fake artifact has nothing to map."""
        return cls()


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``iree.runtime`` whose entry point returns a fixed array.

    Returns the mutable holder so a test can set the array the fake "executes"
    to, which is how parity is driven without a real artifact.
    """
    holder: dict[str, Any] = {"result": np.zeros((32, 4), dtype=np.float32), "calls": []}

    class _FakeEntry:
        def __call__(self, *args: Any) -> Any:
            holder["calls"].append(args)
            return holder["result"]

    class _FakeLoaded:
        def __getitem__(self, key: str) -> Any:
            if key != "main":
                raise KeyError(key)
            return _FakeEntry()

    class _FakeContext:
        instance = object()

        def __init__(self, config: Any) -> None:
            del config
            self.modules = {"jit_fake_module": _FakeLoaded()}

        def add_vm_module(self, module: Any) -> None:
            """No-op: the fake context already holds its module."""

    fake = types.ModuleType("iree.runtime")
    fake.Config = lambda name: name  # type: ignore[attr-defined]
    fake.SystemContext = lambda config: _FakeContext(config)  # type: ignore[attr-defined]
    fake.VmModule = _FakeVmModule  # type: ignore[attr-defined]

    # The parent package must exist too: `import iree.runtime` imports `iree`
    # first, so injecting only the submodule works when iree happens to be
    # installed and fails when it is not -- which is the configuration the
    # no-toolchain job actually runs in.
    parent = sys.modules.get("iree")
    if parent is None:
        parent = types.ModuleType("iree")
        monkeypatch.setitem(sys.modules, "iree", parent)
    monkeypatch.setattr(parent, "runtime", fake, raising=False)
    monkeypatch.setitem(sys.modules, "iree.runtime", fake)
    return holder


@pytest.fixture
def no_toolchain(monkeypatch: pytest.MonkeyPatch):
    """Make every IREE import fail, as it would without the export extra."""
    for name in ("iree", "iree.compiler", "iree.runtime"):
        monkeypatch.setitem(sys.modules, name, None)

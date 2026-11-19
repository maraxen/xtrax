"""AST sweep: nothing under ``src/xtrax`` imports equinox (equinox decoupling, step 5).

Spec ``.praxia/docs/specs/261023_equinox-decoupling.md`` section 6. equinox is a dev-only
dependency, so a stray import would make ``import xtrax`` fail for every user without it. The
sweep reads every ``.py`` under the package and flags ``import equinox``, ``from equinox ...``
and ``importlib.import_module("equinox")`` / ``__import__("equinox")`` at ANY depth, including
inside functions, ``try`` blocks and ``if TYPE_CHECKING`` blocks. There is no allow-list.
Looking equinox up in ``sys.modules`` without importing it (``_tree.inference_mode``) is not an
import and is deliberately not flagged. The runtime counterpart is
``tests/test_no_equinox_training.py``, which blocks the import in a subprocess.
"""

import ast
import pathlib

import pytest

import xtrax

_SRC = pathlib.Path(xtrax.__file__).parent
_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _is_equinox(name):
    return isinstance(name, str) and name.split(".")[0] == "equinox"


def equinox_references(source):
    """Return every static or literal-string dynamic equinox import in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if _is_equinox(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and _is_equinox(node.module):
                found.append(node.module)
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if name in _DYNAMIC_IMPORTERS and isinstance(first, ast.Constant):
                if _is_equinox(first.value):
                    found.append(first.value)
    return found


def _sources():
    return sorted(_SRC.rglob("*.py"))


def test_the_sweep_covers_the_whole_package():
    assert len(_sources()) > 150


def test_no_module_under_src_imports_equinox():
    offenders = {}
    for path in _sources():
        refs = equinox_references(path.read_text(encoding="utf-8"))
        if refs:
            offenders[path.relative_to(_SRC).as_posix()] = refs
    assert offenders == {}


# --- controls: prove the checker goes red on each shape of reference -----------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "import equinox",
        "import equinox as eqx",
        "import equinox.nn",
        "from equinox import nn",
        "from equinox.nn import Linear",
        "import jax\n\ndef f():\n    import equinox as eqx\n    return eqx.is_array",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import equinox",
        "try:\n    import equinox\nexcept ImportError:\n    equinox = None",
        "class C:\n    def m(self):\n        from equinox import filter_jit",
        "import importlib\nm = importlib.import_module('equinox.nn')",
        "m = __import__('equinox')",
    ],
    ids=[
        "plain",
        "aliased",
        "submodule",
        "from",
        "from-submodule",
        "in-function",
        "type-checking",
        "guarded",
        "in-method",
        "import-module",
        "dunder-import",
    ],
)
def test_control_the_checker_flags_an_equinox_reference(bad):
    assert equinox_references(bad) != []


@pytest.mark.parametrize(
    "good",
    [
        "import jax\nfrom jax import numpy as jnp",
        "from . import equinox_like",
        "# import equinox\nx = 'import equinox'",
        '"""Equinox models are plain pytrees."""',
        "import sys\neqx = sys.modules.get('equinox')\nok = eqx is not None",
        "import importlib\nm = importlib.import_module('jax.numpy')",
    ],
    ids=[
        "jax",
        "relative",
        "comment-and-string",
        "docstring",
        "sys-modules-lookup",
        "other-module",
    ],
)
def test_control_the_checker_does_not_flag_prose_or_lookalikes(good):
    assert equinox_references(good) == []


def test_control_a_real_file_with_an_appended_import_goes_red():
    # the sweep's own loop, pointed at a real module with one import appended
    real = (_SRC / "_tree.py").read_text(encoding="utf-8")
    assert equinox_references(real) == []
    assert equinox_references(real + "\nimport equinox\n") == ["equinox"]

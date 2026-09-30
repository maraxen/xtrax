"""B5 direction 3 (#5036): a module the wheel ships must not import a dev-only package
at module level, since `pip install xtrax` would then fail to import it.

Direction 2 accepts an import resolving to ANY declared name (dev extra included), so
it could not see this. The acceptance criteria from #5036 are the first four tests.
"""

from __future__ import annotations

import importlib.metadata
import textwrap
import tomllib
from pathlib import Path

import pytest

from scripts.audit_project_hygiene import (
    _module_level_unguarded_imports,
    check_shipped_imports_reach_consumers,
)

ROOT = Path(__file__).resolve().parents[2]

# Synthetic project: runtime depends on equinox (which requires jaxtyping); pytest is
# dev-only; zarr is a user-facing extra. The wheel ships src/xtrax minus devtools.
PYPROJECT = {
    "project": {
        "dependencies": ["equinox>=0.13"],
        "optional-dependencies": {"dev": ["pytest>=8"], "io": ["zarr>=3"]},
    },
    "tool": {
        "hatch": {
            "build": {
                "targets": {"wheel": {"packages": ["src/xtrax"], "exclude": ["src/xtrax/devtools"]}}
            }
        }
    },
}


def _check(tmp_path: Path, rel: str, source: str) -> list[str]:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return check_shipped_imports_reach_consumers(
        tmp_path, PYPROJECT, importlib.metadata.packages_distributions()
    )


def test_shipped_module_importing_a_dev_only_package_fails(tmp_path: Path) -> None:
    failures = _check(tmp_path, "src/xtrax/shipped.py", "import pytest\n")
    assert len(failures) == 1
    assert "'pytest'" in failures[0] and "src/xtrax/shipped.py:1" in failures[0]


def test_shipped_module_importing_a_transitive_runtime_dep_passes(tmp_path: Path) -> None:
    """jaxtyping is undeclared but equinox requires it, so a plain install has it."""
    assert _check(tmp_path, "src/xtrax/shipped.py", "from jaxtyping import Array\n") == []


def test_non_shipped_module_importing_a_dev_only_package_passes(tmp_path: Path) -> None:
    assert _check(tmp_path, "src/xtrax/devtools/tool.py", "import pytest\n") == []


def test_shipped_split_is_read_from_pyproject(tmp_path: Path) -> None:
    """Move the exclusion and the same file flips: the split is data, not hand-listed."""
    shipped_everywhere = {
        **PYPROJECT,
        "tool": {"hatch": {"build": {"targets": {"wheel": {"packages": ["src/xtrax"]}}}}},
    }
    path = tmp_path / "src/xtrax/devtools/tool.py"
    path.parent.mkdir(parents=True)
    path.write_text("import pytest\n")
    env = importlib.metadata.packages_distributions()
    assert check_shipped_imports_reach_consumers(tmp_path, PYPROJECT, env) == []
    assert len(check_shipped_imports_reach_consumers(tmp_path, shipped_everywhere, env)) == 1


def test_user_facing_extra_import_passes(tmp_path: Path) -> None:
    """An optional subpackage importing its own extra's package is by design."""
    pytest.importorskip("zarr")
    assert _check(tmp_path, "src/xtrax/shipped.py", "import zarr\n") == []


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    import pytest\n",  # lazy: runs only when called
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pytest\n",
        "try:\n    import pytest\nexcept ImportError:\n    pytest = None\n",
        "class C:\n    def m(self):\n        import pytest\n",
    ],
    ids=["function", "type-checking", "import-error-guard", "method"],
)
def test_imports_that_do_not_run_on_module_import_are_exempt(tmp_path: Path, source: str) -> None:
    assert _check(tmp_path, "src/xtrax/shipped.py", source) == []


def test_module_level_if_and_unguarded_try_still_count() -> None:
    import ast

    tree = ast.parse(
        "import sys\nif sys.version_info > (3,):\n    import pytest\n"
        "try:\n    import hypothesis\nexcept ValueError:\n    pass\n"
    )
    names = {name for name, _ in _module_level_unguarded_imports(tree)}
    assert {"pytest", "hypothesis"} <= names


def test_real_repository_passes() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert (
        check_shipped_imports_reach_consumers(
            ROOT, pyproject, importlib.metadata.packages_distributions()
        )
        == []
    )

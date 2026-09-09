"""Tests for the declared-vs-imported dependency contract (#4969, B5).

Two directions, both checked against the CURRENT install environment via
`importlib.metadata.packages_distributions()` rather than a hand-written
distribution/import name table -- a static table would just be a second copy of
pyproject.toml's names with nothing keeping it in sync, exactly the divergence
class #4969 exists to kill.

Direction 1 -- every name in `[project].dependencies` is imported somewhere under
`src/`. Must walk `ast.Import`/`ast.ImportFrom` nodes at ANY depth (`ast.walk`),
not just `tree.body`: `jaxlib`'s only appearance in the tree is `import jaxlib`
inside a function body (`src/xtrax/profiling/record.py:105`).

Direction 2 -- every third-party top-level import under `src/` resolves to a
declared dependency or extra, via (1) the environment map, (2) name-equality
against a declared name, (3) an explicit `[import_name_overrides]` entry for the
un-installable residue (e.g. `iree`, gated behind an unsynced extra). An override
that becomes resolvable via (1) or (2) is stale and must fail.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from scripts.audit_project_hygiene import (
    check_dependencies_are_imported,
    check_imports_are_declared,
    load_project_hygiene_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "project_hygiene.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _env_map() -> dict[str, list[str]]:
    import importlib.metadata

    return importlib.metadata.packages_distributions()


# --- Direction 1: declared dependency must be imported under src/ -----------------


def test_check_dependencies_are_imported_names_the_unused_dependency(tmp_path: Path) -> None:
    """A declared dependency with zero imports anywhere under src/ is flagged BY
    NAME, while a properly-imported sibling in the same `dependencies` list is
    not -- the checker must distinguish between them, not just detect "something
    somewhere is unused".

    Synthetic fixture, not the real pyproject.toml: the real `grain`/
    `pytest-asyncio` case this modeled was fixed by #4969 (grain moved to a
    `data` extra, pytest-asyncio dropped from runtime deps entirely), so a test
    reading the real file would rot to "passes" the moment the fix landed --
    which it did. The checker's ability to flag an unused dependency is the
    durable contract this fixture keeps exercising.
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text("import widgets\n", encoding="utf-8")

    pyproject = {"project": {"dependencies": ["widgets>=1", "unused-thing>=2"]}}
    env_map = {"widgets": ["widgets"], "unused_thing": ["unused-thing"]}

    failures = check_dependencies_are_imported(tmp_path, pyproject, env_map)

    assert len(failures) == 1, failures
    assert "'unused-thing'" in failures[0]
    assert "never imported anywhere under src/" in failures[0]
    assert not any("'widgets'" in item for item in failures), failures


def test_check_dependencies_are_imported_flags_each_unused_dependency_independently(
    tmp_path: Path,
) -> None:
    """Multiple declared-but-unused dependencies in the same pyproject.toml (the
    real `grain` + `pytest-asyncio` shape, before the #4969 fix) each get their
    own, independently-identifiable failure -- not one combined failure.
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

    pyproject = {"project": {"dependencies": ["grain>=0.2.0", "pytest-asyncio>=0.23"]}}
    env_map = {"grain": ["grain"], "pytest_asyncio": ["pytest-asyncio"]}

    failures = check_dependencies_are_imported(tmp_path, pyproject, env_map)

    assert len(failures) == 2, failures
    assert any("'grain'" in item for item in failures), failures
    assert any("'pytest-asyncio'" in item for item in failures), failures


def test_jaxlib_import_inside_a_function_body_counts(tmp_path: Path) -> None:
    """Regression guard for the ast.walk-not-tree.body requirement: an import
    nested inside a function must be found, or a genuinely-used dependency is
    falsely flagged as unused.
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "record.py").write_text(
        "def _capture_jaxlib_version():\n    import jaxlib\n\n    return jaxlib.__version__\n",
        encoding="utf-8",
    )

    pyproject = {"project": {"dependencies": ["jaxlib>=0.10.2,<0.12"]}}
    env_map = {"jaxlib": ["jaxlib"]}

    failures = check_dependencies_are_imported(tmp_path, pyproject, env_map)

    assert failures == [], failures


def test_dependency_with_no_imports_anywhere_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

    pyproject = {"project": {"dependencies": ["grain>=0.2.0"]}}
    env_map = {"grain": ["grain"]}

    failures = check_dependencies_are_imported(tmp_path, pyproject, env_map)

    assert len(failures) == 1
    assert "'grain'" in failures[0]
    assert "never imported anywhere under src/" in failures[0]


# --- Direction 2: import under src/ must resolve to a declared name ---------------


def test_iree_import_resolves_via_the_override_table() -> None:
    """`iree` is not installed in the dev+io audit environment, so
    packages_distributions() can't see it, and its import name doesn't
    name-equal any declared dependency/extra -- the [import_name_overrides]
    entry is required for this to pass.
    """
    config = load_project_hygiene_config(CONFIG_PATH)
    failures = check_imports_are_declared(
        ROOT, _pyproject(), _env_map(), config.import_name_overrides
    )

    assert not any(item.startswith("import 'iree'") for item in failures), failures


def test_matplotlib_pandas_seaborn_resolve_without_an_override() -> None:
    """eda's extras happen to already be present in the dev+io environment here
    (as transitive deps), so they resolve via the environment map/name-equality
    step -- no override table entry should be needed for any of them.
    """
    config = load_project_hygiene_config(CONFIG_PATH)
    failures = check_imports_are_declared(
        ROOT, _pyproject(), _env_map(), config.import_name_overrides
    )

    for name in ("matplotlib", "pandas", "seaborn"):
        assert not any(item.startswith(f"import {name!r}") for item in failures), (
            name,
            failures,
        )


def test_zarr_lazy_import_resolves_via_the_io_extra() -> None:
    """A lazy/optional import guarded behind an extra is EXPECTED to resolve via
    a declared extra rather than `dependencies` -- that is success, not failure.
    `zarr` (src/xtrax/run/zarr_integrity.py) is the model case.
    """
    config = load_project_hygiene_config(CONFIG_PATH)
    failures = check_imports_are_declared(
        ROOT, _pyproject(), _env_map(), config.import_name_overrides
    )

    assert not any(item.startswith("import 'zarr'") for item in failures), failures


def test_guarded_optional_imports_are_exempted_on_repo() -> None:
    """`cisternal` (src/xtrax/telemetry/record.py:347-355) and `zstandard`
    (src/xtrax/telemetry/store.py:65-76) are deliberately never declared ANYWHERE
    -- not in dependencies, not in any extra, not in the override table -- but
    both are guarded: `try: import ... except ImportError: ...`. Such an import
    cannot raise ModuleNotFoundError at a consumer's install, which is direction
    2's whole rationale for existing, so the guarded-import exemption (applied
    last, after all three resolution steps miss) exempts both.
    """
    config = load_project_hygiene_config(CONFIG_PATH)
    failures = check_imports_are_declared(
        ROOT, _pyproject(), _env_map(), config.import_name_overrides
    )

    assert not any(item.startswith("import 'cisternal'") for item in failures), failures
    assert not any(item.startswith("import 'zstandard'") for item in failures), failures


def test_override_visible_in_the_environment_is_NOT_stale(tmp_path: Path) -> None:
    """Staleness is judged on DECLARATION, never on what happens to be installed.

    Regression guard for a gate that could not be satisfied in both environments at
    once. `iree` is invisible to packages_distributions() under the audit's own
    dev+io sync, so the override is required; under `--extra export` it becomes
    visible, and an environment-keyed staleness check would then demand the override
    be removed -- which breaks dev+io again. Whether a name is declared is a fact
    about pyproject.toml; whether it is installed is a fact about which extras
    someone happened to sync, and only the former may drive this check.

    Here `widgets` is installed and maps to `widgets-pkg`, but no declared
    requirement is NAMED `widgets`, so the override is still doing real work.
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text("import widgets\n", encoding="utf-8")

    pyproject = {"project": {"dependencies": ["widgets-pkg>=1"]}}
    env_map = {"widgets": ["widgets-pkg"]}
    overrides = {"widgets": ("widgets-pkg",)}

    failures = check_imports_are_declared(tmp_path, pyproject, env_map, overrides)

    assert not any("is stale" in item for item in failures), failures


def test_override_import_that_resolves_via_name_equality_is_stale(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text("import widgets\n", encoding="utf-8")

    pyproject = {"project": {"dependencies": ["widgets>=1"]}}
    env_map: dict[str, list[str]] = {}
    overrides = {"widgets": ("something-else",)}

    failures = check_imports_are_declared(tmp_path, pyproject, env_map, overrides)

    assert any("is stale" in item and "name-equality" in item for item in failures), failures


def test_import_with_no_resolution_path_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text("import totally_unknown_thing\n", encoding="utf-8")

    pyproject = {"project": {"dependencies": []}}
    env_map: dict[str, list[str]] = {}
    overrides: dict[str, tuple[str, ...]] = {}

    failures = check_imports_are_declared(tmp_path, pyproject, env_map, overrides)

    assert len(failures) == 1
    assert "totally_unknown_thing" in failures[0]
    assert "not in the installed environment" in failures[0]


def test_stdlib_and_self_imports_are_never_flagged(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text(
        "import os\nimport sys\nimport xtrax\nfrom xtrax.run import stages\n",
        encoding="utf-8",
    )

    pyproject = {"project": {"dependencies": []}}
    failures = check_imports_are_declared(tmp_path, pyproject, {}, {})

    assert failures == [], failures


# --- Wiring into the main audit entry point ----------------------------------------


def test_check_dependencies_are_imported_passes_on_repo() -> None:
    """Direction 1 (declared dependency -> imported under src/) contributes zero
    failures on the real repo: `grain` moved to a `data` extra and
    `pytest-asyncio` was dropped from `[project].dependencies` entirely (#4969),
    so every remaining runtime dependency is genuinely imported somewhere under
    src/.

    This checks `check_dependencies_are_imported` directly rather than going
    through `audit_project_hygiene()` -- the overall-pass assertion for the whole
    gate already lives in `tests/distribution/test_project_hygiene.py`
    (`test_audit_project_hygiene_passes_on_repo`); duplicating it here would add
    nothing. This test isolates direction 1's own zero-failures state from the
    other checks the gate also runs.
    """
    failures = check_dependencies_are_imported(ROOT, _pyproject(), _env_map())

    assert failures == [], failures


def test_check_imports_are_declared_passes_on_repo() -> None:
    """Direction 2 (import -> declared name) contributes zero failures on the real
    repo: `iree`/`matplotlib`/`pandas`/`seaborn` all resolve via steps 1-3, and
    `cisternal`/`zstandard` are exempted by the guarded-import rule. Same
    non-duplication rationale as `test_check_dependencies_are_imported_passes_on_repo`
    above -- this isolates direction 2 specifically, rather than repeating
    `test_project_hygiene.py`'s whole-gate assertion.
    """
    config = load_project_hygiene_config(CONFIG_PATH)
    failures = check_imports_are_declared(
        ROOT, _pyproject(), _env_map(), config.import_name_overrides
    )

    assert failures == [], failures


def test_ast_walk_finds_nested_module_level_import_from(tmp_path: Path) -> None:
    """`ImportFrom` nested inside a function body must be found too, not just plain
    `Import` -- e.g. `from cisternal.provenance import capture_git_state`, which
    lives inside a function body in the real repo (guard-exemption behavior for
    that specific pattern is covered separately, below -- this fixture is
    deliberately UNGUARDED so it isolates AST-walk depth from the guard
    exemption).
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "mod.py").write_text(
        "def f():\n    from widgets.sub import thing\n    return thing\n",
        encoding="utf-8",
    )

    pyproject = {"project": {"dependencies": []}}
    failures = check_imports_are_declared(tmp_path, pyproject, {}, {})

    assert any("widgets" in item for item in failures), failures


@pytest.mark.parametrize(
    "relative_import_source", ["from . import sibling\n", "from .mod import x\n"]
)
def test_relative_imports_are_never_flagged(tmp_path: Path, relative_import_source: str) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "sibling.py").write_text("VALUE = 1\n", encoding="utf-8")
    (src / "mod2.py").write_text(relative_import_source, encoding="utf-8")

    pyproject = {"project": {"dependencies": []}}
    failures = check_imports_are_declared(tmp_path, pyproject, {}, {})

    assert failures == [], failures


# --- Guarded-import exemption -------------------------------------------------------
#
# Applied LAST, only after all three resolution steps above miss. Models the real
# `cisternal`/`zstandard` sites: `try: import X ... except ImportError: ...` cannot
# raise ModuleNotFoundError at a consumer's install, which is direction 2's whole
# reason for existing, so such an import is exempt. An UNGUARDED import that
# resolves nowhere must still fail exactly as before -- the exemption must never
# weaken that failure mode.


def _write_guarded_import_module(tmp_path: Path, body: str) -> Path:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    mod = src / "mod.py"
    mod.write_text(body, encoding="utf-8")
    return mod


def test_guarded_import_is_exempted(tmp_path: Path) -> None:
    """1. A guarded, undeclared import passes."""
    _write_guarded_import_module(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        import widgets\n"
        "    except ImportError:\n"
        "        return None\n"
        "    return widgets\n",
    )

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert failures == [], failures


def test_unguarded_import_is_not_exempted_and_still_fails(tmp_path: Path) -> None:
    """2. An UNGUARDED undeclared import still fails.

    This is the regression guard on the exemption itself, and the most important
    of these four: a bug that exempted everything (not just genuinely-guarded
    imports) would silently reopen the ModuleNotFoundError-at-install hole
    direction 2 exists to catch.
    """
    _write_guarded_import_module(tmp_path, "import widgets\n")

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert len(failures) == 1
    assert "'widgets'" in failures[0]
    assert "does not resolve to any declared dependency or extra" in failures[0]


def test_import_guarded_by_a_reraising_handler_still_fails(tmp_path: Path) -> None:
    """A handler that catches ImportError only to re-raise earns NO exemption.

    `except ImportError as e: raise RuntimeError(...) from e` is a common way to
    give a friendlier "pip install foo" message, but it still fails at the same
    point in a consumer's install -- only the exception type differs. Exempting it
    would let a genuinely-missing dependency past the gate, which is precisely the
    failure direction 2 exists to catch.
    """
    _write_guarded_import_module(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        import widgets\n"
        "    except ImportError as exc:\n"
        '        msg = "install widgets"\n'
        "        raise RuntimeError(msg) from exc\n"
        "    return widgets\n",
    )

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert len(failures) == 1
    assert "'widgets'" in failures[0]


def test_import_guarded_by_a_handler_that_reraises_conditionally_still_fails(
    tmp_path: Path,
) -> None:
    """A `raise` anywhere in the handler body, even nested, blocks the exemption.

    The check is deliberately conservative: it walks the whole handler body rather
    than only its top level. A false positive here costs a dependency declaration;
    a false negative costs a consumer a ModuleNotFoundError at install.
    """
    _write_guarded_import_module(
        tmp_path,
        "STRICT = True\n"
        "\n"
        "def f():\n"
        "    try:\n"
        "        import widgets\n"
        "    except ImportError:\n"
        "        if STRICT:\n"
        "            raise\n"
        "        return None\n"
        "    return widgets\n",
    )

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert len(failures) == 1
    assert "'widgets'" in failures[0]


def test_import_guarded_by_unrelated_exception_still_fails(tmp_path: Path) -> None:
    """3. A `try` whose handlers catch only something unrelated (e.g. `OSError`)
    must NOT exempt the import.
    """
    _write_guarded_import_module(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        import widgets\n"
        "    except OSError:\n"
        "        return None\n"
        "    return widgets\n",
    )

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert len(failures) == 1
    assert "'widgets'" in failures[0]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except ImportError:\n"
            "        pass\n"
            "    else:\n"
            "        import widgets\n"
            "    return None\n",
            id="orelse",
        ),
        pytest.param(
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except ImportError:\n"
            "        pass\n"
            "    finally:\n"
            "        import widgets\n",
            id="finally",
        ),
    ],
)
def test_import_in_orelse_or_finally_is_not_protected(tmp_path: Path, body: str) -> None:
    """4. An import in a Try's `orelse`/`finally` is NOT protected by that try's own
    handler -- only `try.body` is guarded.
    """
    _write_guarded_import_module(tmp_path, body)

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert len(failures) == 1
    assert "'widgets'" in failures[0]


def test_nested_try_composes_with_an_outer_handler(tmp_path: Path) -> None:
    """Bonus: nested try blocks work naturally -- an inner try whose OWN handler
    doesn't match is still exempted if an outer try's handler (still active on the
    guard stack) catches ImportError, since the import is lexically within the
    outer try's body either way.
    """
    _write_guarded_import_module(
        tmp_path,
        "def f():\n"
        "    try:\n"
        "        try:\n"
        "            import widgets\n"
        "        except OSError:\n"
        "            pass\n"
        "    except ImportError:\n"
        "        return None\n"
        "    return widgets\n",
    )

    failures = check_imports_are_declared(tmp_path, {"project": {"dependencies": []}}, {}, {})

    assert failures == [], failures

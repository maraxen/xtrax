"""#3644: SafeMap -> ChunkedMap. The old names were deprecated aliases for one release
(0.4.0a11) and are removed (#5680). Name-based strategy matching still accepts the legacy
"SafeMap" for consumers' own duck-typed classes (aminx debt #2371)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODEMOD = ROOT / "codemods" / "safemap-to-chunkedmap"


OLD_NAMES = [
    ("xtrax", "SafeMap"),
    ("xtrax", "safe_map"),
    ("xtrax.tiling", "SafeMap"),
    ("xtrax.tiling", "SafeMapIterator"),
    ("xtrax.tiling.strategy", "SafeMap"),
    ("xtrax.tiling.iterator", "SafeMapIterator"),
    ("xtrax.transforms", "safe_map"),
    ("xtrax.transforms.map", "safe_map"),
]


@pytest.mark.parametrize(("module", "old"), OLD_NAMES)
def test_old_names_are_removed(module: str, old: str):
    """The one-release deprecation window (0.4.0a11) is over (#5680)."""
    import importlib

    mod = importlib.import_module(module)
    assert not hasattr(mod, old)


@pytest.mark.parametrize(
    ("module", "new"),
    [
        ("xtrax", "ChunkedMap"),
        ("xtrax", "chunked_map"),
        ("xtrax.tiling", "ChunkedMap"),
        ("xtrax.tiling", "ChunkedMapIterator"),
        ("xtrax.tiling.strategy", "ChunkedMap"),
        ("xtrax.tiling.iterator", "ChunkedMapIterator"),
        ("xtrax.transforms", "chunked_map"),
        ("xtrax.transforms.map", "chunked_map"),
    ],
)
def test_new_names_resolve_in_every_module_the_old_ones_did(module: str, new: str):
    """Control for the removal test: the same lookups succeed under the new names."""
    import importlib

    assert hasattr(importlib.import_module(module), new)


def test_from_import_of_an_old_name_fails():
    with pytest.raises(ImportError, match="SafeMap"):
        from xtrax.tiling import SafeMap  # noqa: F401


def test_unknown_names_still_raise_attribute_error():
    import xtrax
    import xtrax.tiling

    with pytest.raises(AttributeError, match="NoSuchThing"):
        xtrax.tiling.NoSuchThing  # noqa: B018
    with pytest.raises(AttributeError, match="NoSuchThing"):
        xtrax.NoSuchThing  # noqa: B018


def test_new_names_do_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from xtrax import ChunkedMap, chunked_map  # noqa: F401
        from xtrax.tiling import ChunkedMapIterator  # noqa: F401


def test_name_matching_accepts_a_consumers_duck_typed_safemap_class():
    """aminx defines its own class named `SafeMap`; name-based matching must still
    recognise it until aminx deprecates it (aminx debt #2371; xtrax #5680 step 4)."""
    from xtrax._renamed import CHUNKED_MAP_NAMES
    from xtrax.stages.topology import _EXPORTABLE_STRATEGIES

    assert "SafeMap" in CHUNKED_MAP_NAMES and "ChunkedMap" in CHUNKED_MAP_NAMES
    assert {"SafeMap", "ChunkedMap"} <= set(_EXPORTABLE_STRATEGIES)


def test_chunked_map_behaves_like_the_old_function():
    from xtrax.transforms import chunked_map

    xs = jnp.arange(10.0).reshape(10, 1)
    assert jnp.array_equal(
        chunked_map(lambda r: r * 2, xs, batch_size=3), jax.vmap(lambda r: r * 2)(xs)
    )


@pytest.mark.skipif(shutil.which("ast-grep") is None, reason="ast-grep not installed")
def test_codemod_rewrites_the_fixture_exactly_and_is_idempotent(tmp_path: Path):
    target = tmp_path / "fixture.py"
    target.write_text((CODEMOD / "fixture_before.py").read_text())
    cmd = ["ast-grep", "scan", "--rule", str(CODEMOD / "rules.yml"), "--update-all", str(target)]
    subprocess.run(cmd, check=True, capture_output=True)
    assert target.read_text() == (CODEMOD / "fixture_after.py").read_text()
    again = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "Applied" not in again.stdout + again.stderr


def _rename_markdown():  # noqa: ANN202
    import importlib.util

    spec = importlib.util.spec_from_file_location("rename_markdown", CODEMOD / "rename_markdown.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rename_markdown_whole_words_and_jax_exclusion():
    rename = _rename_markdown().rename_text
    text = (
        "Use `SafeMap` or `SafeMapIterator`; call safe_map(fn, xs).\n"
        "JAX's own safe_map is unrelated; so is util.safe_map.\n"
        "safe_map_count and MySafeMapper stay.\n"
    )
    assert rename(text) == (
        "Use `ChunkedMap` or `ChunkedMapIterator`; call chunked_map(fn, xs).\n"
        "JAX's own safe_map is unrelated; so is util.safe_map.\n"
        "safe_map_count and MySafeMapper stay.\n"
    )
    assert rename(rename(text)) == rename(text)  # idempotent


def test_rename_markdown_check_mode_reports_without_writing(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text("SafeMap\n")
    assert _rename_markdown().main(["--check", str(tmp_path)]) == 1
    assert doc.read_text() == "SafeMap\n"
    assert _rename_markdown().main([str(tmp_path)]) == 0
    assert doc.read_text() == "ChunkedMap\n"

"""#3644: SafeMap -> ChunkedMap. The old names stay importable for one release as
deprecated aliases, and name-based strategy matching accepts both names."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODEMOD = ROOT / "codemods" / "safemap-to-chunkedmap"


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("xtrax", "SafeMap", "ChunkedMap"),
        ("xtrax", "safe_map", "chunked_map"),
        ("xtrax.tiling", "SafeMap", "ChunkedMap"),
        ("xtrax.tiling", "SafeMapIterator", "ChunkedMapIterator"),
        ("xtrax.tiling.strategy", "SafeMap", "ChunkedMap"),
        ("xtrax.tiling.iterator", "SafeMapIterator", "ChunkedMapIterator"),
        ("xtrax.transforms", "safe_map", "chunked_map"),
        ("xtrax.transforms.map", "safe_map", "chunked_map"),
    ],
)
def test_old_name_is_a_warning_alias_for_the_new_object(module: str, old: str, new: str):
    import importlib

    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=rf"{old} was renamed to {new}.*#3644") as rec:
        aliased = getattr(mod, old)
    assert aliased is getattr(mod, new)
    # Attributed to the caller (this file), or Python's default filter would hide it.
    assert {w.filename for w in rec} == {__file__}


def test_from_import_of_an_old_name_warns():
    with pytest.warns(DeprecationWarning, match="SafeMap was renamed"):
        from xtrax.tiling import SafeMap  # noqa: F401


def test_unknown_names_still_raise_attribute_error():
    import xtrax.tiling

    with pytest.raises(AttributeError, match="NoSuchThing"):
        xtrax.tiling.NoSuchThing  # noqa: B018


def test_the_alias_constructs_the_new_class():
    """So code dispatching on type(...).__name__ sees the new name (documented)."""
    import xtrax.tiling

    with pytest.warns(DeprecationWarning):
        strategy = xtrax.tiling.SafeMap(batch_size=4)
    assert type(strategy).__name__ == "ChunkedMap"


def test_new_names_do_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from xtrax import ChunkedMap, chunked_map  # noqa: F401
        from xtrax.tiling import ChunkedMapIterator  # noqa: F401


def test_name_matching_accepts_a_consumers_duck_typed_safemap_class():
    """aminx defines its own class named `SafeMap`; name-based matching must still
    recognise it for this release."""
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

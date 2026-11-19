"""Executing-distribution resolver (debt #2947).

Each test builds a throwaway ``site`` directory holding a ``.dist-info`` and the
package it installed, then puts it on ``sys.path``, so the resolver reads real
metadata files rather than a mock of ``importlib.metadata``.
"""

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from xtrax.provenance import DISTRIBUTION_FIELDS, UNKNOWN, resolve_distribution

_NAME = "xtrax-fakedist"
_MODULE = "xtrax_fakedist"
_SHA = "a" * 64


def _install(
    site: Path,
    *,
    direct_url: "dict[str, object] | str | bytes | None" = None,
    package_dir: "Path | None" = None,
    record_eol: str = "\n",
    top_level: "str | None" = None,
) -> Path:
    """Write a dist-info (and, unless ``package_dir`` is given, its package) into ``site``."""
    site.mkdir(parents=True, exist_ok=True)
    info = site / f"{_MODULE}-1.2.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {_NAME}\nVersion: 1.2.3\n", encoding="utf-8"
    )
    record_lines = [f"{info.name}/METADATA,,", f"{info.name}/RECORD,,"]
    if package_dir is None:
        pkg = site / _MODULE
        pkg.mkdir()
        (pkg / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        record_lines.append(f"{_MODULE}/__init__.py,,")
    else:
        (site / f"_editable_impl_{_MODULE}.pth").write_text(str(package_dir), encoding="utf-8")
        record_lines.append(f"_editable_impl_{_MODULE}.pth,,")
    if direct_url is not None:
        if isinstance(direct_url, bytes):
            (info / "direct_url.json").write_bytes(direct_url)
        else:
            text = direct_url if isinstance(direct_url, str) else json.dumps(direct_url)
            (info / "direct_url.json").write_text(text, encoding="utf-8")
        record_lines.append(f"{info.name}/direct_url.json,,")
    if top_level is not None:
        (info / "top_level.txt").write_text(top_level, encoding="utf-8")
        record_lines.append(f"{info.name}/top_level.txt,,")
    (info / "RECORD").write_bytes((record_eol.join(record_lines) + record_eol).encode("utf-8"))
    return info


@pytest.fixture
def on_path(monkeypatch):
    """Prepend directories to sys.path and forget any cached import of the fake module."""
    monkeypatch.delitem(sys.modules, _MODULE, raising=False)

    def add(*dirs: Path) -> None:
        for directory in reversed(dirs):
            monkeypatch.syspath_prepend(str(directory))
        importlib.invalidate_caches()

    return add


def test_index_install_reports_version_record_and_module(tmp_path, on_path):
    site = tmp_path / "site"
    info = _install(site)
    on_path(site)

    record = resolve_distribution(_NAME)

    assert tuple(record) == DISTRIBUTION_FIELDS
    assert all(isinstance(value, str) for value in record.values())
    assert record["name"] == _NAME
    assert record["version"] == "1.2.3"
    assert record["install_kind"] == "index"
    assert record["wheel_sha256"] == UNKNOWN
    assert record["vcs_commit"] == UNKNOWN
    assert record["source_url"] == UNKNOWN
    expected = hashlib.sha256((info / "RECORD").read_bytes()).hexdigest()
    assert record["record_sha256"] == expected
    assert record["module"] == _MODULE
    assert Path(record["module_file"]) == site / _MODULE / "__init__.py"
    assert record["module_in_dist"] == "true"


def test_vcs_install_reports_commit_and_strips_credentials(tmp_path, on_path):
    site = tmp_path / "site"
    _install(
        site,
        direct_url={
            "url": "https://user:token@example.com/org/repo.git",
            "vcs_info": {"vcs": "git", "commit_id": "0123abcd" * 5},
        },
    )
    on_path(site)

    record = resolve_distribution(_NAME)

    assert record["install_kind"] == "vcs"
    assert record["vcs_commit"] == "0123abcd" * 5
    assert record["source_url"] == "https://example.com/org/repo.git"
    assert "token" not in json.dumps(record)


@pytest.mark.parametrize(
    ("archive_info", "url", "expected"),
    [
        ({"hashes": {"sha256": _SHA}}, "file:///w/pkg-1.2.3-py3-none-any.whl", _SHA),
        ({"hash": f"sha256={_SHA}"}, "https://h/pkg-1.2.3-py3-none-any.whl", _SHA),
        ({"hashes": {"sha256": _SHA}}, "https://h/pkg-1.2.3.tar.gz", UNKNOWN),
        ({}, "file:///w/pkg-1.2.3-py3-none-any.whl", UNKNOWN),
        ({"hashes": {"sha256": "not-a-digest"}}, "file:///w/pkg-1.2.3-py3-none-any.whl", UNKNOWN),
    ],
)
def test_wheel_sha256_is_reported_only_when_recorded_for_a_wheel(
    tmp_path, on_path, archive_info, url, expected
):
    site = tmp_path / "site"
    _install(site, direct_url={"url": url, "archive_info": archive_info})
    on_path(site)

    record = resolve_distribution(_NAME)

    assert record["install_kind"] == "archive"
    assert record["wheel_sha256"] == expected


def test_editable_install_counts_its_source_tree_as_the_distribution(tmp_path, on_path):
    source = tmp_path / "checkout" / "src"
    (source / _MODULE).mkdir(parents=True)
    (source / _MODULE / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    site = tmp_path / "site"
    _install(
        site,
        direct_url={"url": (tmp_path / "checkout").as_uri(), "dir_info": {"editable": True}},
        package_dir=source,
    )
    on_path(site, source)

    record = resolve_distribution(_NAME, module=_MODULE)

    assert record["install_kind"] == "editable"
    assert record["module_in_dist"] == "true"


def test_editable_install_pointing_elsewhere_is_not_the_running_module(tmp_path, on_path):
    source = tmp_path / "other" / _MODULE
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VALUE = 3\n", encoding="utf-8")
    site = tmp_path / "site"
    _install(
        site,
        direct_url={"url": (tmp_path / "checkout").as_uri(), "dir_info": {"editable": True}},
        package_dir=source.parent,
    )
    on_path(site, source.parent)

    assert resolve_distribution(_NAME, module=_MODULE)["module_in_dist"] == "false"


def test_a_shadowing_copy_earlier_on_sys_path_is_reported(tmp_path, on_path):
    """The case importlib.metadata alone cannot see: right version, wrong code."""
    site = tmp_path / "site"
    _install(site)
    vendored = tmp_path / "vendored"
    (vendored / _MODULE).mkdir(parents=True)
    (vendored / _MODULE / "__init__.py").write_text("VALUE = 99\n", encoding="utf-8")
    on_path(vendored, site)

    record = resolve_distribution(_NAME)

    assert record["version"] == "1.2.3"
    assert Path(record["module_file"]) == vendored / _MODULE / "__init__.py"
    assert record["module_in_dist"] == "false"


def test_an_already_imported_module_is_located_by_its_file(tmp_path, on_path, monkeypatch):
    site = tmp_path / "site"
    _install(site)
    on_path(site)
    loaded = types.ModuleType(_MODULE)
    loaded.__file__ = str(tmp_path / "elsewhere" / "__init__.py")
    monkeypatch.setitem(sys.modules, _MODULE, loaded)

    record = resolve_distribution(_NAME)

    assert record["module_file"] == loaded.__file__
    assert record["module_in_dist"] == "false"


def test_missing_distribution_reports_not_installed_and_unknowns(tmp_path, on_path):
    on_path(tmp_path)

    record = resolve_distribution("xtrax-no-such-distribution")

    assert record["name"] == "xtrax-no-such-distribution"
    assert record["install_kind"] == "not_installed"
    others = {key: value for key, value in record.items() if key not in {"name", "install_kind"}}
    assert set(others.values()) == {UNKNOWN}


def test_missing_distribution_with_an_importable_module_is_not_in_dist(tmp_path, on_path):
    (tmp_path / _MODULE).mkdir()
    (tmp_path / _MODULE / "__init__.py").write_text("", encoding="utf-8")
    on_path(tmp_path)

    record = resolve_distribution("xtrax-no-such-distribution", module=_MODULE)

    assert record["module_file"] != UNKNOWN
    assert record["module_in_dist"] == "false"


def test_malformed_direct_url_is_unknown_not_index(tmp_path, on_path):
    site = tmp_path / "site"
    _install(site, direct_url="{not json")
    on_path(site)

    assert resolve_distribution(_NAME)["install_kind"] == UNKNOWN


def test_unlocatable_module_is_unknown(tmp_path, on_path):
    site = tmp_path / "site"
    _install(site)
    on_path(site)

    record = resolve_distribution(_NAME, module="xtrax_fakedist_absent")

    assert record["module_file"] == UNKNOWN
    assert record["module_in_dist"] == UNKNOWN


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://u:p@ss@example.com/r.git", "https://example.com/r.git"),
        ("git+ssh://git@github.com/o/r.git", "git+ssh://github.com/o/r.git"),
        ("https://example.com/r.git", "https://example.com/r.git"),
        ("https://example.com/a@b/r.git", "https://example.com/a@b/r.git"),
        ("file:///w/pkg.whl", "file:///w/pkg.whl"),
    ],
)
def test_source_url_userinfo_is_removed(tmp_path, on_path, url, expected):
    site = tmp_path / "site"
    _install(site, direct_url={"url": url, "vcs_info": {"vcs": "git", "commit_id": "c"}})
    on_path(site)

    assert resolve_distribution(_NAME)["source_url"] == expected


def test_record_sha256_is_over_the_file_bytes_including_crlf(tmp_path, on_path):
    """pip writes CRLF RECORDs; a text-mode read would hash a different byte string."""
    site = tmp_path / "site"
    info = _install(site, record_eol="\r\n")
    on_path(site)

    raw = (info / "RECORD").read_bytes()
    assert b"\r\n" in raw
    assert resolve_distribution(_NAME)["record_sha256"] == hashlib.sha256(raw).hexdigest()


def test_editable_copy_inside_the_checkouts_own_venv_is_not_the_source(tmp_path, on_path):
    """A stale installed copy under <checkout>/.venv/.../site-packages shadows the source."""
    checkout = tmp_path / "checkout"
    (checkout / "src" / _MODULE).mkdir(parents=True)
    (checkout / "src" / _MODULE / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    stale = checkout / ".venv" / "lib" / "python3.13" / "site-packages"
    (stale / _MODULE).mkdir(parents=True)
    (stale / _MODULE / "__init__.py").write_text("VALUE = 0\n", encoding="utf-8")
    site = tmp_path / "site"
    _install(
        site,
        direct_url={"url": checkout.as_uri(), "dir_info": {"editable": True}},
        package_dir=checkout / "src",
    )
    on_path(site, stale)

    record = resolve_distribution(_NAME, module=_MODULE)

    assert Path(record["module_file"]) == stale / _MODULE / "__init__.py"
    assert record["module_in_dist"] == "false"


def test_editable_default_module_comes_from_the_dist_files(tmp_path, on_path):
    """uv/hatch editable dists list only a .pth plus package data; no top_level.txt."""
    source = tmp_path / "checkout" / "src"
    (source / _MODULE).mkdir(parents=True)
    (source / _MODULE / "__init__.py").write_text("", encoding="utf-8")
    site = tmp_path / "site"
    _install(
        site,
        direct_url={"url": (tmp_path / "checkout").as_uri(), "dir_info": {"editable": True}},
        package_dir=source,
        top_level=f"{_MODULE}\n",
    )
    on_path(site, source)

    record = resolve_distribution(_NAME)

    assert record["module"] == _MODULE
    assert record["module_in_dist"] == "true"


def test_default_module_does_not_scan_every_installed_distribution(tmp_path, on_path, monkeypatch):
    site = tmp_path / "site"
    _install(site)
    on_path(site)

    def _refuse() -> None:
        raise AssertionError("packages_distributions scans the whole environment")

    monkeypatch.setattr(importlib.metadata, "packages_distributions", _refuse)

    assert resolve_distribution(_NAME)["module"] == _MODULE


def test_a_dotted_module_is_located_without_importing_its_parent(tmp_path, on_path):
    site = tmp_path / "site"
    _install(site)
    pkg = site / _MODULE
    (pkg / "__init__.py").write_text("raise RuntimeError('imported')\n", encoding="utf-8")
    (pkg / "sub.py").write_text("", encoding="utf-8")
    on_path(site)

    record = resolve_distribution(_NAME, module=f"{_MODULE}.sub")

    assert _MODULE not in sys.modules
    assert Path(record["module_file"]) == pkg / "sub.py"
    assert record["module_in_dist"] == "false"  # sub.py is not in RECORD


def test_undecodable_direct_url_is_unknown_and_keeps_the_rest(tmp_path, on_path):
    site = tmp_path / "site"
    _install(site, direct_url=b"\xff\xfe{not utf-8")
    on_path(site)

    record = resolve_distribution(_NAME)

    assert record["install_kind"] == UNKNOWN
    assert record["version"] == "1.2.3"
    assert record["record_sha256"] != UNKNOWN


def test_blank_name_is_rejected():
    with pytest.raises(ValueError, match="distribution name is required"):
        resolve_distribution("  ")


def test_resolves_a_real_installed_distribution():
    record = resolve_distribution("jax")
    assert record["version"] != UNKNOWN
    assert record["module"] == "jax"
    assert record["module_in_dist"] == "true"

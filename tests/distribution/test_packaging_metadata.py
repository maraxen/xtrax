"""Tests for distribution N2 packaging metadata + py.typed wheel gate (#1455)."""

from __future__ import annotations

import subprocess
import textwrap
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest

from scripts.audit_packaging_metadata import (
    audit_packaging_metadata,
    license_file_contains_apache,
    load_packaging_contract,
    read_hatch_force_include_map,
    read_pyproject_classifiers,
    read_pyproject_license,
    read_wheel_metadata_license_fields,
    wheel_contains_path,
    wheel_metadata_mentions_apache,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "packaging_metadata.toml"


def _write_contract(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "packaging_metadata.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_minimal_pyproject(
    repo_root: Path,
    *,
    license_spdx: str = "Apache-2.0",
    classifiers: list[str] | None = None,
    force_include: dict[str, str] | None = None,
) -> Path:
    if classifiers is None:
        classifiers = [
            "License :: OSI Approved :: Apache Software License",
            "Typing :: Typed",
        ]
    if force_include is None:
        force_include = {"src/xtrax/py.typed": "xtrax/py.typed"}

    classifier_lines = "\n".join(f'  "{item}",' for item in classifiers)
    force_include_lines = "\n".join(
        f'"{source}" = "{dest}"' for source, dest in force_include.items()
    )
    pyproject = repo_root / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "xtrax"
            dynamic = ["version"]
            license = "{license_spdx}"
            classifiers = [
            {classifier_lines}
            ]

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [tool.hatch.version]
            path = "src/xtrax/__init__.py"

            [tool.hatch.build.targets.wheel]
            packages = ["src/xtrax"]

            [tool.hatch.build.targets.wheel.force-include]
            {force_include_lines}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return pyproject


def _write_license(repo_root: Path, *, text: str | None = None) -> Path:
    license_path = repo_root / "LICENSE"
    license_path.write_text(
        text if text is not None else "                                 Apache License\n",
        encoding="utf-8",
    )
    return license_path


def _write_py_typed(repo_root: Path, *, relative: str = "src/xtrax/py.typed") -> Path:
    py_typed_path = repo_root / relative
    py_typed_path.parent.mkdir(parents=True, exist_ok=True)
    py_typed_path.write_text("", encoding="utf-8")
    return py_typed_path


def _make_wheel(
    tmp_path: Path,
    *,
    py_typed_in_wheel: bool = True,
    license_expression: str | None = "Apache-2.0",
    license_field: str | None = None,
) -> Path:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = dist_dir / "xtrax-0.3.0-py3-none-any.whl"
    metadata = EmailMessage()
    metadata["Metadata-Version"] = "2.1"
    metadata["Name"] = "xtrax"
    metadata["Version"] = "0.3.0"
    if license_expression is not None:
        metadata["License-Expression"] = license_expression
    if license_field is not None:
        metadata["License"] = license_field
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("xtrax-0.3.0.dist-info/METADATA", metadata.as_bytes())
        if py_typed_in_wheel:
            archive.writestr("xtrax/py.typed", b"")
    return wheel_path


def test_load_packaging_contract_reads_committed_toml() -> None:
    contract = load_packaging_contract(CONFIG_PATH)
    assert contract.version == "0.1.0"
    assert contract.license_spdx == "Apache-2.0"
    assert contract.license_file == "LICENSE"
    assert contract.py_typed_source == "src/xtrax/py.typed"
    assert contract.wheel_py_typed_path == "xtrax/py.typed"
    assert "Typing :: Typed" in contract.required_classifiers


def test_license_file_contains_apache(tmp_path: Path) -> None:
    good = tmp_path / "LICENSE"
    good.write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    bad = tmp_path / "MIT"
    bad.write_text("MIT License\n", encoding="utf-8")
    assert license_file_contains_apache(good) is True
    assert license_file_contains_apache(bad) is False
    assert license_file_contains_apache(tmp_path / "missing") is False


def test_read_pyproject_license_and_classifiers(tmp_path: Path) -> None:
    _write_minimal_pyproject(tmp_path)
    pyproject_path = tmp_path / "pyproject.toml"
    assert read_pyproject_license(pyproject_path) == "Apache-2.0"
    classifiers = read_pyproject_classifiers(pyproject_path)
    assert "Typing :: Typed" in classifiers


def test_read_hatch_force_include_map(tmp_path: Path) -> None:
    _write_minimal_pyproject(tmp_path)
    pyproject_path = tmp_path / "pyproject.toml"
    mapping = read_hatch_force_include_map(pyproject_path)
    assert mapping["src/xtrax/py.typed"] == "xtrax/py.typed"


def test_wheel_contains_path(tmp_path: Path) -> None:
    wheel_path = _make_wheel(tmp_path)
    assert wheel_contains_path(wheel_path, "xtrax/py.typed") is True
    assert wheel_contains_path(wheel_path, "missing/py.typed") is False


def test_wheel_metadata_mentions_apache(tmp_path: Path) -> None:
    expr_wheel = _make_wheel(tmp_path, license_expression="Apache-2.0")
    license_wheel = _make_wheel(
        tmp_path / "license",
        license_expression=None,
        license_field="Apache License 2.0",
    )
    missing_wheel = _make_wheel(
        tmp_path / "missing",
        license_expression=None,
        license_field="MIT",
    )
    assert wheel_metadata_mentions_apache(expr_wheel) is True
    assert wheel_metadata_mentions_apache(license_wheel) is True
    assert wheel_metadata_mentions_apache(missing_wheel) is False


def test_read_wheel_metadata_license_fields(tmp_path: Path) -> None:
    wheel_path = _make_wheel(tmp_path)
    expression, license_field = read_wheel_metadata_license_fields(wheel_path)
    assert expression == "Apache-2.0"
    assert license_field is None


def test_audit_packaging_metadata_source_only_passes(tmp_path: Path) -> None:
    _write_license(tmp_path)
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(tmp_path)
    config_path = _write_contract(tmp_path)

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=True,
    )
    assert passed is True
    assert failures == []


def test_audit_packaging_metadata_source_only_fails_missing_license(
    tmp_path: Path,
) -> None:
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(tmp_path)
    config_path = _write_contract(tmp_path)

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=True,
    )
    assert passed is False
    assert any("license file missing" in item for item in failures)


def test_audit_packaging_metadata_source_only_fails_license_spdx_mismatch(
    tmp_path: Path,
) -> None:
    _write_license(tmp_path)
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(tmp_path, license_spdx="MIT")
    config_path = _write_contract(tmp_path)

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=True,
    )
    assert passed is False
    assert any("pyproject license mismatch" in item for item in failures)


def test_audit_packaging_metadata_source_only_fails_missing_classifier(
    tmp_path: Path,
) -> None:
    _write_license(tmp_path)
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(
        tmp_path,
        classifiers=["License :: OSI Approved :: Apache Software License"],
    )
    config_path = _write_contract(tmp_path)

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=True,
    )
    assert passed is False
    assert any("missing classifier: 'Typing :: Typed'" in item for item in failures)


def test_audit_packaging_metadata_source_only_fails_force_include_mismatch(
    tmp_path: Path,
) -> None:
    _write_license(tmp_path)
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(
        tmp_path,
        force_include={"src/xtrax/py.typed": "wrong/py.typed"},
    )
    config_path = _write_contract(tmp_path)

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=True,
    )
    assert passed is False
    assert any("hatch force-include py.typed mismatch" in item for item in failures)


def test_audit_packaging_metadata_fails_when_wheel_missing_py_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_license(tmp_path)
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(tmp_path)
    config_path = _write_contract(tmp_path)
    _make_wheel(tmp_path, py_typed_in_wheel=False)

    monkeypatch.setattr(
        "scripts.audit_packaging_metadata.run_uv_build",
        lambda root: (True, ""),
    )

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=False,
    )
    assert passed is False
    assert any("wheel missing py.typed" in item for item in failures)


def test_audit_packaging_metadata_fails_when_wheel_metadata_missing_apache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_license(tmp_path)
    _write_py_typed(tmp_path)
    _write_minimal_pyproject(tmp_path)
    config_path = _write_contract(tmp_path)
    _make_wheel(
        tmp_path,
        license_expression=None,
        license_field="MIT",
    )

    monkeypatch.setattr(
        "scripts.audit_packaging_metadata.run_uv_build",
        lambda root: (True, ""),
    )

    passed, failures = audit_packaging_metadata(
        root=tmp_path,
        config_path=config_path,
        source_only=False,
    )
    assert passed is False
    assert any("wheel METADATA missing Apache license" in item for item in failures)


def test_script_source_only_subprocess_exits_zero() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_packaging_metadata.py",
            "--source-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS: packaging metadata" in result.stdout

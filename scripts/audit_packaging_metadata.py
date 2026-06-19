#!/usr/bin/env python3
"""Distribution N2 packaging metadata + py.typed-in-wheel gate (#1455)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from email import message_from_bytes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "packaging_metadata.toml"


@dataclass(frozen=True)
class PackagingContract:
    version: str
    license_spdx: str
    license_file: str
    py_typed_source: str
    wheel_py_typed_path: str
    required_classifiers: tuple[str, ...]


def load_packaging_contract(config_path: Path) -> PackagingContract:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    packaging = data.get("packaging")
    if not isinstance(packaging, dict):
        raise ValueError(f"missing [packaging] section in {config_path}")

    version = packaging.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("packaging.version must be a non-empty string")

    license_spdx = packaging.get("license_spdx")
    if not isinstance(license_spdx, str) or not license_spdx:
        raise ValueError("packaging.license_spdx must be a non-empty string")

    license_file = packaging.get("license_file")
    if not isinstance(license_file, str) or not license_file:
        raise ValueError("packaging.license_file must be a non-empty string")

    py_typed_source = packaging.get("py_typed_source")
    if not isinstance(py_typed_source, str) or not py_typed_source:
        raise ValueError("packaging.py_typed_source must be a non-empty string")

    wheel_py_typed_path = packaging.get("wheel_py_typed_path")
    if not isinstance(wheel_py_typed_path, str) or not wheel_py_typed_path:
        raise ValueError("packaging.wheel_py_typed_path must be a non-empty string")

    required = packaging.get("required_classifiers")
    if not isinstance(required, list) or not required:
        raise ValueError("packaging.required_classifiers must be a non-empty list")
    if not all(isinstance(item, str) and item for item in required):
        raise ValueError("packaging.required_classifiers must contain strings")

    return PackagingContract(
        version=version,
        license_spdx=license_spdx,
        license_file=license_file,
        py_typed_source=py_typed_source,
        wheel_py_typed_path=wheel_py_typed_path,
        required_classifiers=tuple(required),
    )


def read_pyproject_license(pyproject_path: Path) -> str | None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    license_value = data.get("project", {}).get("license")
    if isinstance(license_value, str):
        return license_value
    return None


def read_pyproject_classifiers(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    classifiers = data.get("project", {}).get("classifiers")
    if not isinstance(classifiers, list):
        return []
    return [item for item in classifiers if isinstance(item, str)]


def read_hatch_force_include_map(pyproject_path: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    force_include = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("force-include")
    )
    if not isinstance(force_include, dict):
        return {}
    return {
        key: value
        for key, value in force_include.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def license_file_contains_apache(license_path: Path) -> bool:
    if not license_path.is_file():
        return False
    text = license_path.read_text(encoding="utf-8")
    return "Apache License" in text


def run_uv_build(root: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["uv", "build"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "uv build failed"
        return False, stderr
    return True, ""


def find_built_wheel(dist_dir: Path) -> Path | None:
    if not dist_dir.is_dir():
        return None
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        return None
    return wheels[-1]


def wheel_contains_path(wheel_path: Path, member_path: str) -> bool:
    with zipfile.ZipFile(wheel_path) as archive:
        return member_path in archive.namelist()


def read_wheel_metadata_license_fields(
    wheel_path: Path,
) -> tuple[str | None, str | None]:
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith("/METADATA")
        ]
        if not metadata_names:
            raise ValueError(f"no METADATA file found in {wheel_path}")
        raw = archive.read(metadata_names[0])
    message = message_from_bytes(raw)
    return message.get("License-Expression"), message.get("License")


def wheel_metadata_mentions_apache(wheel_path: Path) -> bool:
    license_expression, license_field = read_wheel_metadata_license_fields(wheel_path)
    for value in (license_expression, license_field):
        if value and "Apache" in value:
            return True
    return False


def audit_packaging_metadata(
    *,
    root: Path,
    config_path: Path,
    source_only: bool = False,
) -> tuple[bool, list[str]]:
    contract = load_packaging_contract(config_path)
    pyproject_path = root / "pyproject.toml"
    license_path = root / contract.license_file
    py_typed_path = root / contract.py_typed_source

    failures: list[str] = []

    if not license_path.is_file():
        failures.append(f"license file missing: {contract.license_file}")
    elif not license_file_contains_apache(license_path):
        failures.append(
            f"license file {contract.license_file!r} missing 'Apache License' marker"
        )

    if not py_typed_path.is_file():
        failures.append(f"py.typed source missing: {contract.py_typed_source}")

    if not pyproject_path.is_file():
        failures.append("pyproject.toml missing")
        return False, failures

    project_license = read_pyproject_license(pyproject_path)
    if project_license != contract.license_spdx:
        failures.append(
            "pyproject license mismatch: "
            f"expected {contract.license_spdx!r}, got {project_license!r}"
        )

    classifiers = read_pyproject_classifiers(pyproject_path)
    for required in contract.required_classifiers:
        if required not in classifiers:
            failures.append(f"pyproject missing classifier: {required!r}")

    force_include = read_hatch_force_include_map(pyproject_path)
    expected_wheel_path = force_include.get(contract.py_typed_source)
    if expected_wheel_path != contract.wheel_py_typed_path:
        failures.append(
            "hatch force-include py.typed mismatch: "
            f"expected {contract.py_typed_source!r} -> "
            f"{contract.wheel_py_typed_path!r}, "
            f"got {expected_wheel_path!r}"
        )

    if source_only:
        return len(failures) == 0, failures

    build_ok, build_error = run_uv_build(root)
    if not build_ok:
        failures.append(f"uv build failed: {build_error}")
        return False, failures

    dist_dir = root / "dist"
    wheel_path = find_built_wheel(dist_dir)
    if wheel_path is None:
        failures.append(f"no wheel found under {dist_dir}")
        return False, failures

    if not wheel_contains_path(wheel_path, contract.wheel_py_typed_path):
        failures.append(
            f"wheel missing py.typed at {contract.wheel_py_typed_path!r}"
        )

    try:
        apache_in_metadata = wheel_metadata_mentions_apache(wheel_path)
    except ValueError as exc:
        failures.append(str(exc))
        apache_in_metadata = False

    if not apache_in_metadata:
        failures.append(
            "wheel METADATA missing Apache license "
            "(expected License-Expression or License to mention Apache)"
        )

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to packaging_metadata.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Skip wheel build; verify LICENSE, py.typed source, and pyproject only",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_packaging_metadata(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        source_only=args.source_only,
    )
    if passed:
        if args.source_only:
            print(
                "PASS: packaging metadata — LICENSE, py.typed source, "
                "pyproject contract ok (source-only)"
            )
        else:
            print(
                "PASS: packaging metadata — LICENSE, pyproject, "
                "py.typed in wheel, Apache in METADATA"
            )
        return 0

    print("FAIL: packaging metadata", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

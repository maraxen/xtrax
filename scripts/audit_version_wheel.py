#!/usr/bin/env python3
"""Distribution N1 version contract + wheel verification gate (#1452)."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from email import message_from_bytes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "version_contract.toml"


@dataclass(frozen=True)
class VersionContract:
    version: str
    version_source: str
    attribute: str
    hatch_path_key: str


def load_version_contract(config_path: Path) -> VersionContract:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    version_section = data.get("version")
    if not isinstance(version_section, dict):
        raise ValueError(f"missing [version] section in {config_path}")

    version = version_section.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("version.version must be a non-empty string")

    version_source = version_section.get("version_source")
    if not isinstance(version_source, str) or not version_source:
        raise ValueError("version.version_source must be a non-empty string")

    attribute = version_section.get("attribute")
    if not isinstance(attribute, str) or not attribute:
        raise ValueError("version.attribute must be a non-empty string")

    hatch_path_key = version_section.get("hatch_path_key")
    if not isinstance(hatch_path_key, str) or not hatch_path_key:
        raise ValueError("version.hatch_path_key must be a non-empty string")

    return VersionContract(
        version=version,
        version_source=version_source,
        attribute=attribute,
        hatch_path_key=hatch_path_key,
    )


def parse_init_version(init_path: Path, *, attribute: str = "__version__") -> str:
    """Parse a module-level string assignment without importing the package."""
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == attribute:
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                raise ValueError(f"{init_path}:{node.lineno}: {attribute} must be a string literal")
    raise ValueError(f"{attribute} assignment not found in {init_path}")


def read_hatch_version_path(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    hatch_version = data.get("tool", {}).get("hatch", {}).get("version")
    if not isinstance(hatch_version, dict):
        raise ValueError(f"missing [tool.hatch.version] in {pyproject_path}")
    path = hatch_version.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("[tool.hatch.version].path must be a non-empty string")
    return path


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


def read_wheel_metadata_version(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith("/METADATA")]
        if not metadata_names:
            raise ValueError(f"no METADATA file found in {wheel_path}")
        raw = archive.read(metadata_names[0])
    message = message_from_bytes(raw)
    version = message.get("Version")
    if not version:
        raise ValueError(f"METADATA in {wheel_path} missing Version field")
    return version


def twine_is_available() -> bool:
    return importlib.util.find_spec("twine") is not None


def run_twine_check(dist_dir: Path) -> tuple[bool, str]:
    artifacts = sorted(dist_dir.glob("*"))
    if not artifacts:
        return False, f"no artifacts in {dist_dir}"
    result = subprocess.run(
        ["twine", "check", *[str(path) for path in artifacts]],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "twine check failed"
        return False, stderr
    return True, result.stdout.strip()


def audit_version_wheel(
    *,
    root: Path,
    config_path: Path,
    metadata_only: bool = False,
) -> tuple[bool, list[str], str | None]:
    contract = load_version_contract(config_path)
    init_path = root / contract.version_source
    pyproject_path = root / "pyproject.toml"

    failures: list[str] = []
    resolved_version: str | None = None

    if not init_path.is_file():
        failures.append(f"version source missing: {contract.version_source}")
        return False, failures, None

    try:
        resolved_version = parse_init_version(init_path, attribute=contract.attribute)
    except ValueError as exc:
        failures.append(str(exc))
        return False, failures, None

    try:
        hatch_path = read_hatch_version_path(pyproject_path)
    except ValueError as exc:
        failures.append(str(exc))
        return False, failures, resolved_version

    if hatch_path != contract.version_source:
        failures.append(
            "pyproject hatch version path mismatch: "
            f"expected {contract.version_source!r}, got {hatch_path!r}"
        )

    if metadata_only:
        return len(failures) == 0, failures, resolved_version

    build_ok, build_error = run_uv_build(root)
    if not build_ok:
        failures.append(f"uv build failed: {build_error}")
        return False, failures, resolved_version

    dist_dir = root / "dist"
    wheel_path = find_built_wheel(dist_dir)
    if wheel_path is None:
        failures.append(f"no wheel found under {dist_dir}")
        return False, failures, resolved_version

    try:
        wheel_version = read_wheel_metadata_version(wheel_path)
    except ValueError as exc:
        failures.append(str(exc))
        return False, failures, resolved_version

    if wheel_version != resolved_version:
        failures.append(
            f"wheel METADATA Version mismatch: expected {resolved_version!r}, got {wheel_version!r}"
        )

    if twine_is_available():
        twine_ok, twine_error = run_twine_check(dist_dir)
        if not twine_ok:
            failures.append(f"twine check failed: {twine_error}")
    else:
        print("NOTE: twine not installed; skipping twine check")

    return len(failures) == 0, failures, resolved_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to version_contract.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip wheel build; verify contract + pyproject path only",
    )
    args = parser.parse_args(argv)

    passed, failures, version = audit_version_wheel(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        metadata_only=args.metadata_only,
    )
    if passed:
        label = version or "unknown"
        if args.metadata_only:
            print(
                f"PASS: version contract — __version__={label!r}; "
                "hatch path matches (metadata-only)"
            )
        else:
            print(f"PASS: version wheel — __version__={label!r}; wheel METADATA matches")
        return 0

    print("FAIL: version wheel", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

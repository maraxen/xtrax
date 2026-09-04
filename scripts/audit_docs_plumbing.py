#!/usr/bin/env python3
"""Distribution N4a docs plumbing gate — Sphinx -W build + RTD/CI contract (#1457)."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "docs_plumbing.toml"


@dataclass(frozen=True)
class DocsPlumbingConfig:
    version: str
    sphinx_conf: str
    readthedocs_config: str
    workflow: str
    source_dir: str
    build_dir: str
    sphinx_builder: str
    sphinx_warn_is_error: bool
    sphinx_nitpicky: bool
    install_groups: tuple[str, ...]
    install_extras: tuple[str, ...]
    required_extensions: tuple[str, ...]
    required_conf_keys: tuple[str, ...]


def load_docs_plumbing_config(config_path: Path) -> DocsPlumbingConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    docs = data.get("docs")
    if not isinstance(docs, dict):
        raise ValueError(f"missing [docs] section in {config_path}")

    version = docs.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("docs.version must be a non-empty string")

    def _req_str(key: str) -> str:
        value = docs.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"docs.{key} must be a non-empty string")
        return value

    install_groups = docs.get("install_groups")
    if not isinstance(install_groups, list) or not install_groups:
        raise ValueError("docs.install_groups must be a non-empty list")
    install_extras = docs.get("install_extras", [])
    if not isinstance(install_extras, list):
        raise ValueError("docs.install_extras must be a list")

    extensions_section = docs.get("required_extensions", {})
    if not isinstance(extensions_section, dict):
        raise ValueError("docs.required_extensions must be a table")
    ext_names = extensions_section.get("names")
    if not isinstance(ext_names, list) or not ext_names:
        raise ValueError("docs.required_extensions.names must be a non-empty list")

    conf_keys_section = docs.get("required_conf_keys", {})
    if not isinstance(conf_keys_section, dict):
        raise ValueError("docs.required_conf_keys must be a table")
    conf_keys = conf_keys_section.get("keys")
    if not isinstance(conf_keys, list) or not conf_keys:
        raise ValueError("docs.required_conf_keys.keys must be a non-empty list")

    warn_is_error = docs.get("sphinx_warn_is_error")
    nitpicky = docs.get("sphinx_nitpicky")
    if not isinstance(warn_is_error, bool) or not isinstance(nitpicky, bool):
        raise ValueError("docs.sphinx_warn_is_error and sphinx_nitpicky must be booleans")

    return DocsPlumbingConfig(
        version=version,
        sphinx_conf=_req_str("sphinx_conf"),
        readthedocs_config=_req_str("readthedocs_config"),
        workflow=_req_str("workflow"),
        source_dir=_req_str("source_dir"),
        build_dir=_req_str("build_dir"),
        sphinx_builder=_req_str("sphinx_builder"),
        sphinx_warn_is_error=warn_is_error,
        sphinx_nitpicky=nitpicky,
        install_groups=tuple(str(item) for item in install_groups),
        install_extras=tuple(str(item) for item in install_extras),
        required_extensions=tuple(str(item) for item in ext_names),
        required_conf_keys=tuple(str(item) for item in conf_keys),
    )


def _parse_conf_assignments(conf_path: Path) -> dict[str, ast.expr]:
    tree = ast.parse(conf_path.read_text(encoding="utf-8"), filename=str(conf_path))
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    return assignments


def check_sphinx_conf(config: DocsPlumbingConfig, root: Path) -> list[str]:
    failures: list[str] = []
    conf_path = root / config.sphinx_conf
    if not conf_path.is_file():
        return [f"missing sphinx conf: {config.sphinx_conf}"]

    assignments = _parse_conf_assignments(conf_path)

    extensions = assignments.get("extensions")
    if not isinstance(extensions, ast.List):
        failures.append("docs/conf.py must define extensions = [...]")
    else:
        present = {
            elt.value
            for elt in extensions.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        missing = [name for name in config.required_extensions if name not in present]
        if missing:
            failures.append(f"sphinx extensions missing: {', '.join(missing)}")

    for key in config.required_conf_keys:
        if key not in assignments:
            failures.append(f"docs/conf.py missing required setting: {key}")

    return failures


def check_wiring_files(config: DocsPlumbingConfig, root: Path) -> list[str]:
    failures: list[str] = []
    for rel in (config.readthedocs_config, config.workflow):
        if not (root / rel).is_file():
            failures.append(f"missing docs wiring file: {rel}")
    return failures


def run_uv_sync(root: Path, config: DocsPlumbingConfig) -> tuple[bool, str]:
    cmd = ["uv", "sync"]
    for group in config.install_groups:
        cmd.append(f"--group={group}")
    for extra in config.install_extras:
        cmd.append(f"--extra={extra}")
    result = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "uv sync failed"
        return False, stderr
    return True, ""


def run_sphinx_build(root: Path, config: DocsPlumbingConfig) -> tuple[bool, str]:
    source = root / config.source_dir
    build_dir = root / config.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    # The groups/extras go on `uv run`, not through a separate `uv sync`. `uv run` resolves
    # additively and leaves the environment alone; `uv sync` REPLACES it, which is how this
    # gate used to strip beartype out of the shared venv and break every audit sequenced
    # after it. Self-sufficient either way -- the caller need not pre-install anything.
    cmd = ["uv", "run"]
    for group in config.install_groups:
        cmd.extend(["--group", group])
    for extra in config.install_extras:
        cmd.extend(["--extra", extra])
    cmd += [
        "sphinx-build",
        "-b",
        config.sphinx_builder,
        str(source),
        str(build_dir),
    ]
    if config.sphinx_warn_is_error:
        cmd.append("-W")
    if config.sphinx_nitpicky:
        cmd.append("-n")
    result = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        return False, combined.strip() or "sphinx-build failed"
    return True, combined.strip()


def audit_docs_plumbing(
    root: Path,
    config_path: Path,
    *,
    skip_build: bool = False,
) -> tuple[bool, list[str]]:
    config = load_docs_plumbing_config(config_path)
    failures = check_wiring_files(config, root)
    failures.extend(check_sphinx_conf(config, root))

    if skip_build:
        return len(failures) == 0, failures

    # No `uv sync` here. run_sphinx_build carries the groups/extras on its own `uv run`,
    # which is additive. Syncing first replaced the shared venv and silently dropped
    # beartype, so every gate sequenced after this one failed on an unrelated ImportError
    # -- and because this audit still exited 0, the corruption was invisible at the point
    # it happened. run_uv_sync is kept only for the tests that assert the old failure path.
    build_ok, build_output = run_sphinx_build(root, config)
    if not build_ok:
        failures.append("sphinx-build -W -n failed")
        if build_output:
            failures.append(build_output)
    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to docs_plumbing.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate manifest + conf wiring without running sphinx-build",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_docs_plumbing(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        skip_build=args.check_only,
    )
    if passed:
        print("PASS: docs plumbing gate")
        return 0

    print("FAIL: docs plumbing gate", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

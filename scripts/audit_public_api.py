#!/usr/bin/env python3
"""Distribution N3 public API contract + lazy export verification gate (#1453)."""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "public_api.toml"

_ALLOWED_ROOT_TOP_LEVEL_NAMES = frozenset(
    {"__version__", "__all__", "_LAZY", "__getattr__", "__dir__"}
)


@dataclass(frozen=True)
class PublicApiContract:
    version: str
    lazy_root: bool
    root_module: str
    forbid_eager_imports_at_root: bool
    required_subpackage_inits: tuple[str, ...]
    tier1_exports: tuple[str, ...]
    smoke_exports: tuple[str, ...]


def load_public_api_contract(config_path: Path) -> PublicApiContract:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    api = data.get("api")
    if not isinstance(api, dict):
        raise ValueError(f"missing [api] section in {config_path}")

    version = api.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("api.version must be a non-empty string")

    lazy_root = api.get("lazy_root")
    if not isinstance(lazy_root, bool):
        raise ValueError("api.lazy_root must be a boolean")

    root_module = api.get("root_module")
    if not isinstance(root_module, str) or not root_module:
        raise ValueError("api.root_module must be a non-empty string")

    forbid_eager = api.get("forbid_eager_imports_at_root")
    if not isinstance(forbid_eager, bool):
        raise ValueError("api.forbid_eager_imports_at_root must be a boolean")

    required = api.get("required_subpackage_inits")
    if not isinstance(required, list) or not required:
        raise ValueError("api.required_subpackage_inits must be a non-empty list")
    if not all(isinstance(item, str) and item for item in required):
        raise ValueError("api.required_subpackage_inits must contain strings")

    tier1 = api.get("tier1_exports")
    if not isinstance(tier1, list) or not tier1:
        raise ValueError("api.tier1_exports must be a non-empty list")
    if not all(isinstance(item, str) and item for item in tier1):
        raise ValueError("api.tier1_exports must contain strings")

    smoke = api.get("smoke_exports")
    if smoke is None:
        smoke_exports = tuple(tier1)
    elif not isinstance(smoke, list) or not smoke:
        raise ValueError("api.smoke_exports must be a non-empty list when present")
    elif not all(isinstance(item, str) and item for item in smoke):
        raise ValueError("api.smoke_exports must contain strings")
    else:
        smoke_exports = tuple(smoke)

    return PublicApiContract(
        version=version,
        lazy_root=lazy_root,
        root_module=root_module,
        forbid_eager_imports_at_root=forbid_eager,
        required_subpackage_inits=tuple(required),
        tier1_exports=tuple(tier1),
        smoke_exports=smoke_exports,
    )


def _string_list_from_assign(node: ast.Assign) -> list[str]:
    if not isinstance(node.value, (ast.List, ast.Tuple)):
        raise ValueError("expected list or tuple literal")
    names: list[str] = []
    for elt in node.value.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
            raise ValueError("expected string literals in export list")
        names.append(elt.value)
    return names


def parse_init_all(init_path: Path) -> list[str]:
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return _string_list_from_assign(node)
    raise ValueError(f"__all__ assignment not found in {init_path}")


def parse_init_lazy_keys(init_path: Path) -> list[str]:
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_LAZY":
                break
        else:
            continue
        if not isinstance(node.value, ast.Dict):
            raise ValueError(f"{init_path}: _LAZY must be a dict literal")
        keys: list[str] = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError(f"{init_path}: _LAZY keys must be string literals")
            keys.append(key.value)
        return keys
    raise ValueError(f"_LAZY assignment not found in {init_path}")


def check_forbid_eager_imports_at_root(init_path: Path) -> list[str]:
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))
    failures: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                allowed = _ALLOWED_ROOT_TOP_LEVEL_NAMES
                if isinstance(target, ast.Name) and target.id in allowed:
                    break
            else:
                failures.append(f"{init_path}:{node.lineno}: disallowed top-level assignment")
            continue
        allowed_funcs = _ALLOWED_ROOT_TOP_LEVEL_NAMES
        if isinstance(node, ast.FunctionDef) and node.name in allowed_funcs:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "xtrax" or module.startswith("xtrax."):
                    failures.append(f"{init_path}:{node.lineno}: forbidden eager import from xtrax")
            for alias in node.names:
                if isinstance(node, ast.Import) and alias.name.startswith("xtrax"):
                    failures.append(f"{init_path}:{node.lineno}: forbidden eager import of xtrax")
            continue
        failures.append(f"{init_path}:{node.lineno}: disallowed top-level statement")
    return failures


def check_subpackage_all(
    root: Path,
    *,
    root_module: str,
    subpackage: str,
) -> tuple[bool, str | None]:
    init_path = root / "src" / root_module / subpackage / "__init__.py"
    if not init_path.is_file():
        return False, f"missing subpackage init: {init_path.relative_to(root)}"
    try:
        names = parse_init_all(init_path)
    except ValueError as exc:
        return False, str(exc)
    if not names:
        return False, f"{init_path.relative_to(root)}: __all__ must be non-empty"
    return True, None


def smoke_resolve_exports(
    root_module: str,
    exports: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    module = importlib.import_module(root_module)
    for name in exports:
        try:
            obj = getattr(module, name)
        except AttributeError:
            failures.append(f"getattr({root_module!r}, {name!r}) raised AttributeError")
            continue
        if obj is None:
            failures.append(f"getattr({root_module!r}, {name!r}) returned None")
    return failures


def audit_public_api(
    *,
    root: Path,
    config_path: Path,
    smoke_imports: bool = True,
) -> tuple[bool, list[str]]:
    contract = load_public_api_contract(config_path)
    failures: list[str] = []

    init_rel = f"src/{contract.root_module}/__init__.py"
    init_path = root / init_rel
    if not init_path.is_file():
        failures.append(f"root init missing: {init_rel}")
        return False, failures

    if contract.forbid_eager_imports_at_root:
        failures.extend(check_forbid_eager_imports_at_root(init_path))

    try:
        init_all = parse_init_all(init_path)
        lazy_keys = parse_init_lazy_keys(init_path)
    except ValueError as exc:
        failures.append(str(exc))
        return False, failures

    if set(init_all) != set(lazy_keys):
        failures.append(
            "__all__ / _LAZY mismatch: "
            f"only in __all__={sorted(set(init_all) - set(lazy_keys))!r}; "
            f"only in _LAZY={sorted(set(lazy_keys) - set(init_all))!r}"
        )

    if set(contract.tier1_exports) != set(init_all):
        only_contract = sorted(set(contract.tier1_exports) - set(init_all))
        only_init = sorted(set(init_all) - set(contract.tier1_exports))
        failures.append(
            "tier1_exports / __all__ mismatch: "
            f"only in contract={only_contract!r}; only in __all__={only_init!r}"
        )

    unknown_smoke = sorted(set(contract.smoke_exports) - set(init_all))
    if unknown_smoke:
        failures.append(f"smoke_exports not in __all__: {unknown_smoke!r}")
    if "DedupGather" not in contract.smoke_exports:
        failures.append("smoke_exports must include DedupGather")

    for subpackage in contract.required_subpackage_inits:
        ok, error = check_subpackage_all(
            root,
            root_module=contract.root_module,
            subpackage=subpackage,
        )
        if not ok and error:
            failures.append(error)

    if smoke_imports and not failures:
        failures.extend(
            smoke_resolve_exports(
                contract.root_module,
                contract.smoke_exports,
            )
        )

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to public_api.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--no-smoke-imports",
        action="store_true",
        help="Skip lazy resolve smoke tests (slower)",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_public_api(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        smoke_imports=not args.no_smoke_imports,
    )
    if passed:
        contract = load_public_api_contract(args.config.resolve())
        print(
            "PASS: public API contract — "
            f"{len(contract.tier1_exports)} tier-1 exports; "
            f"{len(contract.smoke_exports)} smoke-checked"
        )
        return 0

    print("FAIL: public API contract", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

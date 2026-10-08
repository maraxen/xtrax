#!/usr/bin/env python3
"""Fail when a module known to be ported has no ``__provenance__`` declaration.

A module is known to be ported when either condition holds:

* its repo-relative path is a ``module_path`` on a ``[[kernels]]`` entry in
  any ``port/manifests/*.toml``, or
* its module docstring matches ``ported|vendored|adapted|upstreamed from``
  (case-insensitive).

"Vendored" without "from", and "imported from" / "re-exported from", do not
count. The check parses source and does not import the modules.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

from xtrax.provenance import read_provenance

ROOT = Path(__file__).resolve().parents[1]
_ORIGIN_RE = re.compile(r"(?i)\b(?:ported|vendored|adapted|upstreamed)\s+from\b")
_DOC_REASON = "docstring says ported/vendored/adapted/upstreamed from"


def known_ported(repo_root: Path) -> list[tuple[Path, str]]:
    """Return ``(path, reason)`` for every module this gate treats as ported."""
    reasons: dict[Path, list[str]] = {}
    src = repo_root / "src" / "xtrax"
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        doc = ast.get_docstring(tree)
        if doc and _ORIGIN_RE.search(doc):
            reasons.setdefault(path.resolve(), []).append(_DOC_REASON)
    manifests = repo_root / "port" / "manifests"
    for manifest in sorted(manifests.glob("*.toml")):
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        for kernel in data.get("kernels", []):
            module_path = kernel["module_path"]
            path = (repo_root / module_path).resolve()
            reason = f"module_path in port/manifests/{manifest.name}"
            reasons.setdefault(path, []).append(reason)
    ordered = sorted(reasons.items(), key=lambda item: item[0].as_posix())
    return [(path, "; ".join(why)) for path, why in ordered]


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def audit_port_provenance(repo_root: Path) -> list[str]:
    """Return failure lines. An empty list means every known ported module declares."""
    failures: list[str] = []
    for path, why in known_ported(repo_root):
        rel = _relative(repo_root, path)
        if not path.is_file():
            failures.append(f"{rel}: known ported module is missing ({why})")
            continue
        try:
            record = read_provenance(path)
        except ValueError as exc:
            failures.append(f"{rel}: invalid __provenance__: {exc}")
            continue
        if record is None:
            failures.append(f"{rel}: known ported module lacks __provenance__ ({why})")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    failures = audit_port_provenance(root)
    if failures:
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    print(f"port provenance ok: {len(known_ported(root))} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

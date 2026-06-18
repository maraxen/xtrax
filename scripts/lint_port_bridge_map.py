#!/usr/bin/env python3
"""Lint port/bridge/composition_map.toml qualnames against live src/xtrax symbols."""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "port" / "bridge" / "composition_map.toml"


def import_qualname(qualname: str) -> Any:
    module_path, _, attr = qualname.rpartition(".")
    if not module_path or not attr:
        raise ImportError(f"invalid qualname: {qualname!r}")
    if not qualname.startswith("xtrax."):
        raise ImportError(f"qualname must start with 'xtrax.': {qualname!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def load_symbols(map_path: Path) -> dict[str, str]:
    data = tomllib.loads(map_path.read_text(encoding="utf-8"))
    symbols = data.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("[symbols] must be a table mapping qualname strings to node_id strings")
    result: dict[str, str] = {}
    for qualname, node_id in symbols.items():
        if not isinstance(qualname, str) or not qualname:
            raise ValueError(f"invalid symbol qualname key: {qualname!r}")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"invalid node_id for {qualname!r}: {node_id!r}")
        result[qualname] = node_id
    return result


def lint_composition_map(map_path: Path) -> tuple[list[str], int]:
    if not map_path.is_file():
        return [], 0

    symbols = load_symbols(map_path)
    if not symbols:
        return [], 0

    failures: list[str] = []
    for qualname in sorted(symbols):
        try:
            import_qualname(qualname)
        except Exception as exc:  # noqa: BLE001 — aggregate per-symbol failures
            failures.append(f"{qualname}: {exc}")
    return failures, (1 if failures else 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP,
        help="Path to composition_map.toml (default: port/bridge/composition_map.toml)",
    )
    args = parser.parse_args(argv)

    map_path = args.map.resolve()
    try:
        failures, exit_code = lint_composition_map(map_path)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for message in failures:
        print(f"error: {message}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

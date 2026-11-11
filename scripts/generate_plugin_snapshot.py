#!/usr/bin/env python3
"""Generate or verify the packaged src/xtrax/agent_plugin.json snapshot.

Usage:
    uv run python scripts/generate_plugin_snapshot.py
    uv run python scripts/generate_plugin_snapshot.py --check
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("generate_plugin_snapshot")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / ".praxia" / "manifest.toml"
DEFAULT_OUT = REPO_ROOT / "src" / "xtrax" / "agent_plugin.json"


def generate_snapshot(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_path: Path = DEFAULT_OUT,
    check_only: bool = False,
) -> int:
    """Generate or check the plugin snapshot using cisternal assets snapshot.

    Note: cisternal.assets.snapshot.dumps_snapshot() natively formats the bundle
    deterministically (indent=2, sort_keys=True, ensure_ascii=False) with a trailing
    newline. Do NOT re-serialize with standard json.dumps(..., ensure_ascii=True)
    which escapes Unicode and breaks byte-exact snapshot freshness verification.
    """
    if not manifest_path.is_file():
        logger.error("Manifest not found at %s", manifest_path)
        return 1

    try:
        from cisternal.cli import app as cisternal_app
    except ImportError as exc:
        logger.error(
            "cisternal is not installed. Install it via `pip install .[plugin]` "
            "or `uv sync --extra plugin`: %s",
            exc,
        )
        return 1

    cmd_args = [
        "assets",
        "snapshot",
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_path),
    ]
    if check_only:
        cmd_args.append("--check")

    try:
        cisternal_app(cmd_args)
        logger.info("Plugin snapshot successfully processed at %s", out_path)
        return 0
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        if code != 0:
            logger.error("cisternal snapshot returned exit code %s", code)
        else:
            logger.info("Plugin snapshot successfully processed at %s", out_path)
        return code
    except Exception as exc:
        logger.exception("cisternal snapshot execution failed: %s", exc)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to .praxia/manifest.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to src/xtrax/agent_plugin.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check snapshot freshness without modifying file",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    return generate_snapshot(
        manifest_path=args.manifest,
        out_path=args.out,
        check_only=args.check,
    )


if __name__ == "__main__":
    sys.exit(main())

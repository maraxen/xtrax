#!/usr/bin/env python3
"""N5.2 scheduled judgment CLI — validate, self-test, record staleness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xtrax.devtools.judgment import DEFAULT_DISPATCH_PATH
from xtrax.devtools.judgment_scheduled import run_scheduled_judgment
from xtrax.devtools.routing import DEFAULT_ROUTING_PATH
from xtrax.devtools.rubrics import DEFAULT_RUBRICS_DIR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audits-path",
        type=Path,
        default=DEFAULT_AUDITS_PATH,
        help="JSONL path for emitted judgment findings",
    )
    parser.add_argument(
        "--dispatch-path",
        type=Path,
        default=ROOT / DEFAULT_DISPATCH_PATH,
        help="judgment_dispatch.toml roster path",
    )
    parser.add_argument(
        "--rubrics-dir",
        type=Path,
        default=ROOT / DEFAULT_RUBRICS_DIR,
        help="Directory containing dimension rubric TOML files",
    )
    parser.add_argument(
        "--routing-path",
        type=Path,
        default=ROOT / DEFAULT_ROUTING_PATH,
        help="CC5 routing matrix path",
    )
    parser.add_argument(
        "--no-emit",
        action="store_true",
        help="Validate wiring and self-tests only; skip dispatch observation emit",
    )
    args = parser.parse_args(argv)

    try:
        result = run_scheduled_judgment(
            args.audits_path,
            emit_dispatch=not args.no_emit,
            root=ROOT,
            dispatch_path=args.dispatch_path,
            rubrics_dir=args.rubrics_dir,
            routing_path=args.routing_path,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not result.passed:
        print(
            f"FAIL: scheduled judgment "
            f"refute_promote_ok={result.refute_promote_ok} "
            f"docs_judgment_ok={result.docs_judgment_ok}",
            file=sys.stderr,
        )
        return 1

    mode = "emit" if result.dispatch_emitted else "no-emit"
    print(
        f"PASS: scheduled judgment {mode} "
        f"run_id={result.run_id} "
        f"staleness_days={result.staleness_days:.4f} "
        f"staleness_passed={result.staleness_passed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

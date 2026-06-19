#!/usr/bin/env python3
"""N4.1 judgment dispatch CLI — validate wiring, optional observation emit."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.judgment import run_judgment_dispatch, validate_judgment_wiring

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
        default=ROOT / "audit" / "judgment_dispatch.toml",
        help="judgment_dispatch.toml roster path",
    )
    parser.add_argument(
        "--rubrics-dir",
        type=Path,
        default=ROOT / "audit" / "rubrics",
        help="Directory containing dimension rubric TOML files",
    )
    parser.add_argument(
        "--routing-path",
        type=Path,
        default=ROOT / "audit" / "routing.toml",
        help="CC5 routing matrix path",
    )
    parser.add_argument(
        "--no-emit",
        action="store_true",
        help="Validate wiring only; do not append observation findings",
    )
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Append info-level armed observations to audits JSONL",
    )
    args = parser.parse_args(argv)

    emit_observations = args.emit and not args.no_emit
    if args.no_emit:
        try:
            validate_judgment_wiring(
                dispatch_path=args.dispatch_path,
                rubrics_dir=args.rubrics_dir,
                routing_path=args.routing_path,
            )
        except ValueError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print("PASS: judgment dispatch wiring validated (no emit)")
        return 0

    result = run_judgment_dispatch(
        audits_path=args.audits_path,
        emit_observations=emit_observations,
        run_id=str(uuid.uuid4()),
        dispatch_path=args.dispatch_path,
        rubrics_dir=args.rubrics_dir,
        routing_path=args.routing_path,
    )
    mode = "emit" if emit_observations else "validate-only"
    print(
        f"PASS: judgment dispatch {mode} "
        f"dimensions={len(result.entries)} findings_emitted={result.findings_emitted}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

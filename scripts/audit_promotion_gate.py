#!/usr/bin/env python3
"""T2-30 promotion_gate CI check — verify human approval attestation freshness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xtrax.loop.promotion_gate import (
    DEFAULT_GATES_TOML,
    ApprovalExpiredError,
    NoMatchingApprovalError,
    PromotionNotApprovedError,
    assert_promotion_approved,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit-sha",
        type=str,
        required=True,
        help="Commit SHA being proposed for promotion",
    )
    parser.add_argument(
        "--toml-path",
        type=Path,
        default=DEFAULT_GATES_TOML,
        help="Path to .praxia/loop_human_gates.toml",
    )
    args = parser.parse_args(argv)

    try:
        attestation = assert_promotion_approved(args.commit_sha, toml_path=args.toml_path)
        print(
            f"✓ Promotion approved: SHA {args.commit_sha} "
            f"attested by {attestation.attested_by} on {attestation.attested_at}"
        )
        if attestation.note:
            print(f"  Note: {attestation.note}")
        return 0
    except NoMatchingApprovalError as exc:
        print(f"✗ No matching approval found: {exc}", file=sys.stderr)
        return 1
    except ApprovalExpiredError as exc:
        print(f"✗ Approval expired or not fresh: {exc}", file=sys.stderr)
        return 1
    except PromotionNotApprovedError as exc:
        print(f"✗ Promotion gate check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

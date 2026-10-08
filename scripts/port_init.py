#!/usr/bin/env python3
"""Emit a ``__provenance__`` declaration for a new ported module.

The porter supplies upstream, revision, and licence as arguments. Omitting a
revision or a licence is a waiver and requires ``--waiver-reason``. This
command does not use the network; revision is whatever the porter passes,
one module at a time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xtrax.provenance import render_declaration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="Upstream repository URL or name")
    parser.add_argument(
        "--relationship",
        required=True,
        choices=("vendored", "ported", "derived", "inspired"),
    )
    parser.add_argument("--revision", default=None, help="Commit sha or tag for this module")
    parser.add_argument("--licence", default=None, help="SPDX licence identifier")
    parser.add_argument(
        "--waiver-reason",
        default=None,
        help="Required when revision or licence is omitted",
    )
    parser.add_argument(
        "--form",
        choices=("call", "dict"),
        default="call",
        help="dict emits a literal for modules that cannot import xtrax.provenance",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write the declaration here")
    args = parser.parse_args(argv)
    try:
        text = render_declaration(
            upstream=args.upstream,
            relationship=args.relationship,
            revision=args.revision,
            licence=args.licence,
            waiver_reason=args.waiver_reason,
            form=args.form,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

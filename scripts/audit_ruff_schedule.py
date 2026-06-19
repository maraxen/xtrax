#!/usr/bin/env python3
"""N3.2 ruff enablement schedule CLI — validate pyproject sync, print active + next."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xtrax.devtools.ruff_schedule import (
    DEFAULT_PYPROJECT_PATH,
    DEFAULT_SCHEDULE_PATH,
    load_ruff_schedule,
    next_pending_wave,
    validate_ruff_schedule_sync,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule-path",
        type=Path,
        default=ROOT / DEFAULT_SCHEDULE_PATH,
        help="ruff_enablement_schedule.toml path",
    )
    parser.add_argument(
        "--pyproject-path",
        type=Path,
        default=ROOT / DEFAULT_PYPROJECT_PATH,
        help="pyproject.toml path",
    )
    args = parser.parse_args(argv)

    try:
        validate_ruff_schedule_sync(
            schedule_path=args.schedule_path,
            pyproject_path=args.pyproject_path,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    schedule = load_ruff_schedule(args.schedule_path)
    active = next(wave for wave in schedule.waves if wave.id == schedule.active_wave_id)
    pending = next_pending_wave(args.schedule_path)

    print(f"PASS: ruff schedule sync active={active.id} select={sorted(active.select)}")
    if pending is not None:
        print(f"next pending: {pending.id} select={sorted(pending.select)} notes={pending.notes!r}")
    else:
        print("next pending: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())

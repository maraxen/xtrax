"""Install xtrax agent skills to ~/.claude/skills.

Usage:
    uv run python scripts/install_skills.py               # install all
    uv run python scripts/install_skills.py --dry-run     # preview only
    uv run python scripts/install_skills.py --skill using-xtrax  # single skill
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def install_skill(source: Path, target: Path, *, dry_run: bool) -> str:
    action = "update" if target.exists() else "install"
    if not dry_run:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--skill", metavar="NAME", help="Install only this skill subdirectory")
    parser.add_argument("--source", type=Path, default=None, help="Override source dir (default: agent_assets/skills)")
    parser.add_argument("--target", type=Path, default=Path.home() / ".claude" / "skills", help="Override target dir")
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)
    source_dir = args.source or project_root / "agent_assets" / "skills"

    if not source_dir.exists():
        print(f"ERROR: source directory not found: {source_dir}", file=sys.stderr)
        return 1

    if args.skill:
        candidates = [source_dir / args.skill]
        if not candidates[0].exists():
            print(f"ERROR: skill '{args.skill}' not found in {source_dir}", file=sys.stderr)
            return 1
    else:
        candidates = sorted(d for d in source_dir.iterdir() if d.is_dir())

    if not candidates:
        print(f"No skill directories found in {source_dir}")
        return 0

    args.target.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"install": 0, "update": 0}
    for skill_dir in candidates:
        target = args.target / skill_dir.name
        action = install_skill(skill_dir, target, dry_run=args.dry_run)
        counts[action] += 1
        tag = "dry" if args.dry_run else action
        print(f"  [{tag}] {skill_dir.name}  →  {target}")

    labels = {"install": "installed", "update": "updated"}
    summary = ", ".join(f"{n} {labels[k]}" for k, n in counts.items() if n)
    prefix = "Would: " if args.dry_run else "Done: "
    print(f"\n{prefix}{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

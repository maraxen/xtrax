"""Install xtrax agent skills to ~/.claude/skills.

Usage:
    uv run python scripts/install_skills.py               # install all
    uv run python scripts/install_skills.py --dry-run     # preview only
    uv run python scripts/install_skills.py --skill using-xtrax  # single skill
    uv run python scripts/install_skills.py --check       # compare installed copies

``--check`` reads each skill under the source tree and the matching copy in
the target directory (default ``~/.claude/skills``). Override the target with
``--target`` or the ``XTRAX_SKILLS_TARGET`` environment variable (``--target``
wins). A skill with no installed ``SKILL.md`` is MISSING. An installed copy
whose ``xtrax_version`` frontmatter differs from the repo copy is STALE. A
repo skill that declares no ``xtrax_version`` has nothing to go stale against.
Exit status is 1 when any skill is missing or stale, and 0 with no output
when every installed copy matches. ``--check`` does not write the target.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def resolve_skills_target(explicit: Path | None) -> Path:
    """Target directory: ``--target``, else ``XTRAX_SKILLS_TARGET``, else ~/.claude/skills."""
    if explicit is not None:
        return explicit
    override = os.environ.get("XTRAX_SKILLS_TARGET")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude" / "skills"


def read_xtrax_version(skill_md: Path) -> str | None:
    """Return ``xtrax_version`` from a SKILL.md frontmatter block, if declared."""
    if not skill_md.is_file():
        return None
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    frontmatter = text if end == -1 else text[:end]
    match = re.search(r"^xtrax_version:\s*['\"]?([^'\"\n]+)", frontmatter, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def check_installed_skills(skill_dirs: list[Path], target_dir: Path) -> list[str]:
    """Report MISSING and STALE installed copies. Empty when every copy matches."""
    issues: list[str] = []
    for skill_dir in skill_dirs:
        installed = target_dir / skill_dir.name / "SKILL.md"
        if not installed.is_file():
            issues.append(f"MISSING {skill_dir.name}")
            continue
        repo_version = read_xtrax_version(skill_dir / "SKILL.md")
        if repo_version is None:
            continue
        installed_version = read_xtrax_version(installed)
        if installed_version == repo_version:
            continue
        shown = installed_version if installed_version is not None else "<missing>"
        issues.append(f"STALE {skill_dir.name} installed={shown} repo={repo_version}")
    return issues


def install_skill(source: Path, target: Path, *, dry_run: bool) -> str:
    action = "update" if target.exists() else "install"
    if not dry_run:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report MISSING and STALE installed copies; write nothing",
    )
    parser.add_argument("--skill", metavar="NAME", help="Install only this skill subdirectory")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Override source dir (default: agent_assets/skills)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Override target dir (else XTRAX_SKILLS_TARGET, else ~/.claude/skills)",
    )
    args = parser.parse_args()
    target_dir = resolve_skills_target(args.target)

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

    if args.check:
        issues = check_installed_skills(candidates, target_dir)
        for issue in issues:
            print(issue)
        return 1 if issues else 0

    target_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"install": 0, "update": 0}
    for skill_dir in candidates:
        target = target_dir / skill_dir.name
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

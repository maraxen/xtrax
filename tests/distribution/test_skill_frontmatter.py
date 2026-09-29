"""Every shipped SKILL.md frontmatter must be valid YAML with a name and description.

The version-marker gate (``scripts/audit_project_hygiene.py``) reads ``xtrax_version``
with a regex, so it passes a frontmatter that no YAML loader can parse -- an unquoted
``Covers: `` inside ``description`` shipped that way through 0.4.0a10.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
SKILLS = sorted((ROOT / "agent_assets" / "skills").glob("*/SKILL.md"))


def test_skills_are_discovered() -> None:
    assert SKILLS, "no agent_assets/skills/*/SKILL.md found -- the glob is wrong"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_skill_frontmatter_parses(skill: Path) -> None:
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{skill.parent.name}: no frontmatter block"
    end = text.find("\n---", 4)
    assert end != -1, f"{skill.parent.name}: unterminated frontmatter block"
    data = yaml.safe_load(text[4:end])
    assert isinstance(data, dict)
    assert data.get("name") == skill.parent.name
    assert isinstance(data.get("description"), str)
    assert data["description"].strip()

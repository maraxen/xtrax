# xtrax Justfile

# Foundation audit gates (N0)
audit-imports:
    uv run lint-imports

audit-no-future-annotations:
    uv run pytest tests/audit/test_no_future_annotations.py -v

audit-foundation: audit-imports audit-no-future-annotations
    uv run pytest tests/audit/ -v

# Install all skills from agent_assets/skills/ to ~/.claude/skills/
install-skills *args:
    uv run python scripts/install_skills.py {{args}}

# Preview what install-skills would do without writing
dry-run-skills:
    uv run python scripts/install_skills.py --dry-run

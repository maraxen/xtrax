# xtrax Justfile

# Foundation audit gates (N0)
audit-imports:
    uv run lint-imports

audit-no-future-annotations:
    uv run pytest tests/audit/test_no_future_annotations.py -v

audit-jaxlint:
    uv run python scripts/audit_jaxlint_json.py --performance-only

audit-emit-contract:
    uv run pytest tests/audit/test_emit_contract.py -v

validate-capability-registry:
    uv run python scripts/load_capability_registry.py

validate-node-metadata-schema:
    uv run python -c "import sys; sys.path.insert(0, 'scripts'); from load_capability_registry import load_node_metadata_schema; s=load_node_metadata_schema(); print(f'node metadata schema v{s.version} ({len(s.slots)} slots)')"

audit-foundation: audit-imports audit-no-future-annotations audit-jaxlint
    uv run pytest tests/audit/ -v

# Install all skills from agent_assets/skills/ to ~/.claude/skills/
install-skills *args:
    uv run python scripts/install_skills.py {{args}}

# Preview what install-skills would do without writing
dry-run-skills:
    uv run python scripts/install_skills.py --dry-run

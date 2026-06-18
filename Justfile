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

audit-baseline-contract:
    uv run pytest tests/audit/test_baseline_engine.py -v

audit-routing-contract:
    uv run ruff check src/xtrax/devtools/routing.py tests/audit/test_routing_toml.py tests/audit/test_routing_engine.py
    uv run pytest tests/audit/test_routing_toml.py tests/audit/test_routing_engine.py -v

audit-tombstone-contract:
    uv run ruff check src/xtrax/devtools/tombstone.py tests/audit/test_tombstone_ledger.py src/xtrax/devtools/emit.py
    uv run pytest tests/audit/test_tombstone_ledger.py tests/audit/test_emit_contract.py -v

audit-rubrics-contract:
    uv run ruff check src/xtrax/devtools/rubrics.py tests/audit/test_rubric_tables.py
    uv run pytest tests/audit/test_rubric_tables.py -v

audit-correctness-gate:
    uv run ruff check src/xtrax/devtools/gates/correctness.py scripts/audit_correctness_gate.py tests/audit/test_correctness_gate.py
    uv run pytest tests/audit/test_correctness_gate.py -v
    uv run python scripts/audit_correctness_gate.py --no-write-baseline

audit-jax-purity-gate:
    uv run ruff check src/xtrax/devtools/gates/_jaxlint.py src/xtrax/devtools/gates/jax_purity.py scripts/audit_jax_purity_gate.py tests/audit/test_jax_purity_gate.py
    uv run pytest tests/audit/test_jax_purity_gate.py tests/audit/test_correctness_gate.py -v
    uv run python scripts/audit_jax_purity_gate.py --no-write-baseline

audit-type-hardening-gate:
    uv run ruff check src/xtrax/devtools/gates/type_hardening.py scripts/audit_type_hardening_gate.py tests/audit/test_type_hardening_gate.py tests/audit/test_beartype_hook.py tests/conftest.py src/xtrax/devtools/_beartype_probe.py
    uv run pytest tests/audit/test_type_hardening_gate.py tests/audit/test_beartype_hook.py -v
    uv run python scripts/audit_type_hardening_gate.py --no-write-baseline

audit-performance-gate:
    uv run ruff check src/xtrax/devtools/gates/_trace_probe.py src/xtrax/devtools/gates/_performance_probes.py src/xtrax/devtools/gates/performance.py scripts/audit_performance_gate.py tests/audit/test_performance_gate.py
    uv run pytest tests/audit/test_performance_gate.py -v
    uv run python scripts/audit_performance_gate.py --no-write-baseline

audit-documentation-gate:
    uv run ruff check src/xtrax/devtools/gates/_interrogate.py src/xtrax/devtools/gates/documentation.py scripts/audit_documentation_gate.py tests/audit/test_documentation_gate.py
    uv run pytest tests/audit/test_documentation_gate.py -v
    uv run python scripts/audit_documentation_gate.py --no-write-baseline

validate-capability-registry:
    uv run python scripts/load_capability_registry.py

validate-node-metadata-schema:
    uv run python -c "import sys; sys.path.insert(0, 'scripts'); from load_capability_registry import load_node_metadata_schema; s=load_node_metadata_schema(); print(f'node metadata schema v{s.version} ({len(s.slots)} slots)')"

audit-foundation: audit-imports audit-no-future-annotations audit-jaxlint
    uv run pytest tests/audit/ -v

# Port validation gates (Epic #2180 Wave 1)
audit-port: audit-port-oracle-seal audit-port-static audit-port-parity audit-port-emit-contract

audit-port-oracle-seal:
    uv run python scripts/audit_port_oracle_seal.py --target port/port_target.toml

audit-port-static:
    uv run python scripts/audit_jaxlint_json.py --paths-from port/port_target.toml
    uv run python scripts/audit_port_trace_count.py

audit-port-parity:
    uv run pytest port/tests/ -v --tb=short

audit-port-emit-contract:
    uv run pytest tests/contract/test_port_emit_schema.py -v

audit-port-bridge:
    uv run python scripts/lint_port_bridge_map.py

# Install all skills from agent_assets/skills/ to ~/.claude/skills/
install-skills *args:
    uv run python scripts/install_skills.py {{args}}

# Preview what install-skills would do without writing
dry-run-skills:
    uv run python scripts/install_skills.py --dry-run

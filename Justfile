# xtrax Justfile

# Foundation audit gates (N0)
audit-imports:
    uv run --extra dev lint-imports

audit-no-future-annotations:
    uv run pytest tests/audit/test_no_future_annotations.py -v

audit-jaxlint:
    uv run python scripts/audit_jaxlint_json.py --performance-only

audit-substrate-lock:
    uv run ruff check scripts/audit_substrate_lock.py tests/audit/test_substrate_lock.py
    uv run pytest tests/audit/test_substrate_lock.py -v
    uv run python scripts/audit_substrate_lock.py

audit-wave1-load-bearing:
    uv run ruff check scripts/audit_wave1_load_bearing.py tests/audit/test_wave1_load_bearing.py
    uv run pytest tests/audit/test_wave1_load_bearing.py -v
    uv run python scripts/audit_wave1_load_bearing.py

audit-jax-pin:
    uv run ruff check scripts/audit_jax_pin.py src/xtrax/stages/_callback.py tests/stages/test_callback.py
    uv run pytest tests/stages/test_callback.py -v
    uv run python scripts/audit_jax_pin.py

# Local-only verification (praxia/bathos CLIs not installed in CI; not wired into release_readiness.toml)
audit-plugin-manifest:
    uv run ruff check tests/composition/test_plugin_manifest.py
    uv run pytest tests/composition/test_plugin_manifest.py -v

probe-mcp-reachability:
    uv run ruff check scripts/probe_mcp_reachability.py tests/audit/test_mcp_reachability_probe.py
    uv run pytest tests/audit/test_mcp_reachability_probe.py -v
    uv run python scripts/probe_mcp_reachability.py

# T3-03 (#3041, AC-X3/AC-X3b): needs a real `praxia plugin install` export to diff against;
# praxia CLI not installed in CI, so this stays local-only until T3-01's install step is wired
# into a CI job. Unit/fixture tests (tests/audit/test_drift_check.py) run everywhere.
audit-drift-check:
    uv run --extra dev ruff check scripts/audit_drift_check.py tests/audit/test_drift_check.py
    uv run --extra dev pytest tests/audit/test_drift_check.py -v
    uv run --extra dev python scripts/audit_drift_check.py --check

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

audit-added-types-diff:
    uv run ruff check src/xtrax/devtools/gates/added_types_diff.py scripts/audit_added_types_diff.py tests/audit/test_added_types_diff.py
    uv run pytest tests/audit/test_added_types_diff.py -v
    uv run python scripts/audit_added_types_diff.py --no-emit

audit-performance-gate:
    uv run ruff check src/xtrax/devtools/gates/_trace_probe.py src/xtrax/devtools/gates/_performance_probes.py src/xtrax/devtools/gates/performance.py scripts/audit_performance_gate.py tests/audit/test_performance_gate.py
    uv run pytest tests/audit/test_performance_gate.py -v
    uv run python scripts/audit_performance_gate.py --no-write-baseline

audit-documentation-gate:
    uv run ruff check src/xtrax/devtools/gates/_interrogate.py src/xtrax/devtools/gates/documentation.py scripts/audit_documentation_gate.py tests/audit/test_documentation_gate.py
    uv run pytest tests/audit/test_documentation_gate.py -v
    uv run python scripts/audit_documentation_gate.py --no-write-baseline

audit-api-ergonomics-gate:
    uv run ruff check src/xtrax/devtools/gates/api_ergonomics.py scripts/audit_api_ergonomics_gate.py tests/audit/test_api_ergonomics_gate.py
    uv run pytest tests/audit/test_api_ergonomics_gate.py -v
    uv run python scripts/audit_api_ergonomics_gate.py --no-write-baseline

audit-test-rigor-gate:
    uv run ruff check src/xtrax/devtools/gates/test_rigor.py scripts/audit_test_rigor_gate.py tests/audit/test_test_rigor_gate.py
    uv run pytest tests/audit/test_test_rigor_gate.py -v
    uv run python scripts/audit_test_rigor_gate.py --no-write-baseline

audit-test-rigor-gate-quick:
    uv run ruff check src/xtrax/devtools/gates/test_rigor.py scripts/audit_test_rigor_gate.py tests/audit/test_test_rigor_gate.py
    uv run pytest tests/audit/test_test_rigor_gate.py -v
    uv run python scripts/audit_test_rigor_gate.py --quick --no-write-baseline

audit-structure-complexity-gate:
    uv run ruff check src/xtrax/devtools/gates/structure_complexity.py scripts/audit_structure_complexity_gate.py tests/audit/test_structure_complexity_gate.py
    uv run pytest tests/audit/test_structure_complexity_gate.py -v
    uv run python scripts/audit_structure_complexity_gate.py --no-write-baseline

audit-bootstrap:
    uv run ruff check src/xtrax/devtools/bootstrap.py scripts/audit_bootstrap.py tests/audit/test_bootstrap.py
    uv run pytest tests/audit/test_bootstrap.py -v
    uv run python scripts/audit_bootstrap.py

audit-bootstrap-dry:
    uv run ruff check src/xtrax/devtools/bootstrap.py scripts/audit_bootstrap.py tests/audit/test_bootstrap.py
    uv run pytest tests/audit/test_bootstrap.py -v

audit-judgment:
    uv run ruff check src/xtrax/devtools/judgment.py scripts/audit_judgment.py tests/audit/test_judgment_dispatch.py
    uv run pytest tests/audit/test_judgment_dispatch.py -v
    uv run python scripts/audit_judgment.py --no-emit

audit-judgment-scheduled:
    uv run ruff check src/xtrax/devtools/judgment_scheduled.py scripts/audit_judgment_scheduled.py tests/audit/test_judgment_scheduled.py
    uv run pytest tests/audit/test_judgment_scheduled.py -v
    uv run python scripts/audit_judgment_scheduled.py --no-emit

audit-refute-promote:
    uv run ruff check src/xtrax/devtools/refute_promote.py scripts/audit_refute_promote.py tests/audit/test_refute_promote.py
    uv run pytest tests/audit/test_refute_promote.py -v
    uv run python scripts/audit_refute_promote.py --self-test

audit-empirical-oracle:
    uv run ruff check src/xtrax/devtools/empirical_oracle.py scripts/audit_empirical_oracle.py tests/audit/test_empirical_oracle.py
    uv run pytest tests/audit/test_empirical_oracle.py -v
    uv run python scripts/audit_empirical_oracle.py --self-test

audit-docs-judgment:
    uv run ruff check src/xtrax/devtools/docs_judgment.py scripts/audit_docs_judgment.py tests/audit/test_docs_judgment.py
    uv run pytest tests/audit/test_docs_judgment.py -v
    uv run python scripts/audit_docs_judgment.py --self-test --no-emit

audit-ruff-schedule:
    uv run ruff check src/xtrax/devtools/ruff_schedule.py scripts/audit_ruff_schedule.py tests/audit/test_ruff_schedule.py
    uv run pytest tests/audit/test_ruff_schedule.py -v
    uv run python scripts/audit_ruff_schedule.py

validate-capability-registry:
    uv run python scripts/load_capability_registry.py

validate-node-metadata-schema:
    uv run python -c "import sys; sys.path.insert(0, 'scripts'); from load_capability_registry import load_node_metadata_schema; s=load_node_metadata_schema(); print(f'node metadata schema v{s.version} ({len(s.slots)} slots)')"

validate-episodic-memory-contract:
    uv run python scripts/load_episodic_memory_contract.py

audit-graph-auditor:
    uv run ruff check scripts/graph_auditor.py tests/composition/conftest.py tests/composition/test_graph_auditor.py
    uv run pytest tests/composition/test_graph_auditor.py -v
    uv run python scripts/graph_auditor.py .praxia/composition/samples/valid_graph.json

audit-compiler-boundary:
    uv run ruff check scripts/audit_compiler_boundary.py tests/audit/test_compiler_boundary_gate.py
    uv run pytest tests/audit/test_compiler_boundary_gate.py -v
    uv run python scripts/audit_compiler_boundary.py

audit-bathos-independence:
    uv run ruff check scripts/audit_bathos_independence.py tests/audit/test_bathos_independence_gate.py
    uv run pytest tests/audit/test_bathos_independence_gate.py -v
    uv run python scripts/audit_bathos_independence.py

audit-dispatch-independence:
    uv run ruff check scripts/audit_dispatch_independence.py tests/audit/test_dispatch_independence_gate.py
    uv run pytest tests/audit/test_dispatch_independence_gate.py -v
    uv run python scripts/audit_dispatch_independence.py

audit-foundation: audit-imports audit-no-future-annotations audit-jaxlint
    uv run pytest tests/audit/ -v

audit-coverage-hygiene:
    uv run python scripts/audit_coverage_hygiene.py

audit-version-wheel:
    uv run python scripts/audit_version_wheel.py

audit-packaging-metadata:
    uv run python scripts/audit_packaging_metadata.py

audit-public-api:
    uv run python scripts/audit_public_api.py

audit-docs-build:
    uv run ruff check scripts/audit_docs_plumbing.py tests/distribution/test_docs_plumbing.py
    uv run pytest tests/distribution/test_docs_plumbing.py -v
    uv run python scripts/audit_docs_plumbing.py

audit-project-hygiene:
    uv run ruff check scripts/audit_project_hygiene.py tests/distribution/test_project_hygiene.py
    uv run pytest tests/distribution/test_project_hygiene.py -v
    uv run python scripts/audit_project_hygiene.py

audit-narrative-docs:
    uv run ruff check scripts/audit_narrative_docs.py tests/distribution/test_narrative_docs.py
    uv run pytest tests/distribution/test_narrative_docs.py -v
    uv run python scripts/audit_narrative_docs.py

audit-output-sink-docs:
    uv run ruff check scripts/audit_output_sink_docs.py tests/distribution/test_output_sink_docs.py
    uv run pytest tests/distribution/test_output_sink_docs.py -v
    uv run python scripts/audit_output_sink_docs.py

audit-publish-oidc:
    uv run ruff check scripts/audit_publish_oidc.py tests/distribution/test_publish_oidc.py
    uv run pytest tests/distribution/test_publish_oidc.py -v
    uv run python scripts/audit_publish_oidc.py

audit-release-readiness:
    uv run ruff check scripts/audit_release_readiness.py tests/distribution/test_release_readiness.py
    uv run pytest tests/distribution/test_release_readiness.py -v
    uv run python scripts/audit_release_readiness.py

audit-coverage-dag:
    uv run python scripts/audit_coverage_dag.py

audit-coverage-dag-all:
    uv run python scripts/audit_coverage_dag.py --all-tiers

audit-coverage-tier1:
    uv run python scripts/audit_coverage_dag.py --tier tier1_core --enforce tier1_core

audit-coverage-tier2:
    uv run python scripts/audit_coverage_dag.py --tier tier2_eda --enforce tier2_eda

# CI-safe deterministic track (N5.1): foundation gates + contract tests, no live judgment gates.
audit-deterministic: audit-imports audit-no-future-annotations audit-jaxlint audit-substrate-lock audit-wave1-load-bearing audit-jax-pin audit-coverage-hygiene audit-version-wheel audit-packaging-metadata audit-public-api audit-project-hygiene audit-narrative-docs audit-output-sink-docs audit-publish-oidc audit-added-types-diff
    uv run pytest tests/audit/ -v
    just audit-coverage-dag
    just audit-bootstrap-dry
    just audit-ruff-schedule
    just validate-capability-registry
    just validate-episodic-memory-contract
    just audit-graph-auditor
    just audit-compiler-boundary
    just audit-dispatch-independence
    just audit-bathos-independence
    just audit-port-emit-contract

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

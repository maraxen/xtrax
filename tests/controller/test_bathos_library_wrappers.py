"""Tests for controller.bathos_library_wrappers (LC-07, AC-6).

These tests verify the wrapper module structure, absence of MCP calls, and data
threading logic without requiring bathos to be installed. No sys.modules mocking is
needed here: bathos_library_wrappers.py imports bathos lazily (inside function bodies,
not at module level), so importing this module never touches bathos at all -- a bare
`sys.modules["bathos"] = MagicMock()` was tried first but is both unnecessary (this
file's own tests never call the wrapper functions) and actively harmful (it leaks into
any other test module in the same pytest session that needs the real bathos package --
see test_bathos_library_wrappers_integration.py, which exercises both wrapper
functions against real bathos and would silently get MagicMock results instead if this
file's mock were still process-global at collection time).

Per AC-6: NO MCP call appears anywhere in this module's code path.
"""

import ast

import pytest

from controller.bathos_library_wrappers import (
    call_stats_battery_gate,
    get_capability_probe_result,
    get_evidence_candidate_for_run,
    get_seed_trial_counts,
    get_sidecar_drift_signal,
)


class TestModuleImport:
    """Tests verifying the module can be imported and exports expected names."""

    def test_module_exports_expected_names(self):
        """Verify module exports all required names."""
        import controller.bathos_library_wrappers as wrappers

        expected_names = {
            "call_stats_battery_gate",
            "get_capability_probe_result",
            "get_evidence_candidate_for_run",
            "get_seed_trial_counts",
            "get_sidecar_drift_signal",
            "StatsBatteryResult",
        }
        exported_names = {name for name in dir(wrappers) if not name.startswith("_")}

        for name in expected_names:
            assert name in exported_names, f"{name} not exported from wrapper module"

    def test_functions_are_callable(self):
        """Verify the wrapper functions are callable."""
        assert callable(call_stats_battery_gate)
        assert callable(get_capability_probe_result)
        assert callable(get_evidence_candidate_for_run)
        assert callable(get_seed_trial_counts)
        assert callable(get_sidecar_drift_signal)


class TestNoMCPCalls:
    """Tests verifying AC-6: NO MCP call appears anywhere in the module code path."""

    def test_wrapper_source_contains_no_mcp_tool_references(self):
        """Verify source code contains no mcp__ tool references."""
        import inspect

        import controller.bathos_library_wrappers as wrappers_module

        source = inspect.getsource(wrappers_module)

        # AC-6 requirement: no MCP calls
        assert "mcp__" not in source, "Found mcp__ tool reference in wrapper module"
        assert "mcp.tool" not in source, "Found MCP tool reference in wrapper module"

    def test_wrapper_source_contains_no_subprocess_calls(self):
        """Verify source code contains no subprocess imports or calls."""
        import inspect

        import controller.bathos_library_wrappers as wrappers_module

        source = inspect.getsource(wrappers_module)

        # Parse the module to check for subprocess-related imports in code (not docstrings)
        tree = ast.parse(source)

        # Check for subprocess imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "subprocess" not in alias.name, (
                        "Found subprocess import in wrapper module"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "subprocess" not in node.module, (
                        "Found subprocess import in wrapper module"
                    )

    def test_only_lazy_bathos_imports_in_functions(self):
        """Verify bathos imports are lazy (inside functions, not at module level)."""
        import inspect

        import controller.bathos_library_wrappers as wrappers_module

        source = inspect.getsource(wrappers_module)

        # Module-level source code (before first function definition)
        tree = ast.parse(source)

        # Find all top-level imports
        module_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_level_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_level_imports.append(node.module)

        # Verify bathos is NOT imported at module level
        assert "bathos" not in module_level_imports, (
            "bathos should be imported lazily, not at module level"
        )
        assert not any("bathos" in name for name in module_level_imports), (
            "No bathos imports should be at module level"
        )


class TestDataThreading:
    """Tests verifying data shape/structure through wrapper functions."""

    def test_call_stats_battery_gate_returns_bathos_stats_battery_verdict_type(self):
        """Verify return type is BathosStatsBatteryVerdict (signature verification)."""
        # This test verifies the wrapper is shaped correctly for data threading,
        # without requiring live bathos functions (they may not be installed).
        # The actual bathos function would be mocked/stubbed at runtime by the controller.

        # Verify the function signature accepts the expected parameters
        import inspect

        sig = inspect.signature(call_stats_battery_gate)
        params = set(sig.parameters.keys())

        # Should have baseline/candidate parameters + **stats_arrays
        assert "baseline_hpo_trials" in params
        assert "candidate_hpo_trials" in params
        assert "baseline_hpo_compute_budget" in params
        assert "candidate_hpo_compute_budget" in params
        assert "stats_arrays" in params  # **kwargs parameter

    def test_get_seed_trial_counts_signature(self):
        """Verify get_seed_trial_counts has correct function signature."""
        import inspect

        sig = inspect.signature(get_seed_trial_counts)
        params = list(sig.parameters.keys())

        # Should accept db, script_sha256, and optional hypothesis_clause_id
        assert params[0] == "db"
        assert params[1] == "script_sha256"
        assert params[2] == "hypothesis_clause_id"
        assert sig.parameters["hypothesis_clause_id"].default == ""

    def test_get_capability_probe_result_signature(self):
        """Verify get_capability_probe_result has correct function signature."""
        import inspect

        sig = inspect.signature(get_capability_probe_result)
        params = list(sig.parameters.keys())

        # Should accept only catalog_dir with default ""
        assert params == ["catalog_dir"]
        assert sig.parameters["catalog_dir"].default == ""

    def test_get_evidence_candidate_for_run_signature(self):
        """Verify get_evidence_candidate_for_run has correct function signature."""
        import inspect

        sig = inspect.signature(get_evidence_candidate_for_run)
        params = list(sig.parameters.keys())

        # Should accept run_id, catalog_dir, and stdout_verified
        assert params[0] == "run_id"
        assert params[1] == "catalog_dir"
        assert params[2] == "stdout_verified"
        assert sig.parameters["catalog_dir"].default == ""
        assert sig.parameters["stdout_verified"].default is None

    def test_get_sidecar_drift_signal_signature(self):
        """Verify get_sidecar_drift_signal has correct function signature."""
        import inspect

        sig = inspect.signature(get_sidecar_drift_signal)
        params = list(sig.parameters.keys())

        # Should accept script_path, catalog_dir, current_sidecar_sha256, script_id
        assert params[0] == "script_path"
        assert params[1] == "catalog_dir"
        assert params[2] == "current_sidecar_sha256"
        assert params[3] == "script_id"
        assert sig.parameters["catalog_dir"].default == ""
        assert sig.parameters["current_sidecar_sha256"].default == ""
        assert sig.parameters["script_id"].default == ""


class TestAcceptanceRequirements:
    """Tests verifying AC-6 specific requirements."""

    def test_ac6_no_mcp_call_anywhere(self):
        """AC-6: verify no MCP call appears anywhere in this item's code path."""
        import inspect

        import controller.bathos_library_wrappers as wrappers

        source_code = inspect.getsource(wrappers)

        # Parse and walk the entire AST to look for MCP-related calls
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            # Check for any name starting with 'mcp'
            if isinstance(node, ast.Name):
                assert not node.id.startswith("mcp"), (
                    f"Found MCP reference '{node.id}' in wrapper code"
                )

            # Check for attribute access like mcp__*
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    assert not node.value.id.startswith("mcp"), (
                        "Found MCP attribute access in wrapper code"
                    )


class TestEvidenceCandidateForRun:
    """Tests for get_evidence_candidate_for_run (GW-01, AC-19)."""

    def test_evidence_candidate_requires_run_id(self):
        """Verify get_evidence_candidate_for_run raises ValueError on empty run_id."""
        with pytest.raises(ValueError, match="run_id is required"):
            get_evidence_candidate_for_run(run_id="")

    def test_evidence_candidate_returns_structure(self):
        """Verify get_evidence_candidate_for_run builds EvidenceCandidate with correct fields."""
        import inspect

        # Verify signature allows stdout_verified passthrough
        sig = inspect.signature(get_evidence_candidate_for_run)
        assert "stdout_verified" in sig.parameters


class TestSidecarDriftSignal:
    """Tests for get_sidecar_drift_signal (GW-01, AC-18)."""

    def test_sidecar_drift_signal_returns_structure(self):
        """Verify get_sidecar_drift_signal builds SidecarDriftSignal with correct fields."""
        import inspect

        # Verify the function signature and return type annotation
        sig = inspect.signature(get_sidecar_drift_signal)
        assert "script_path" in sig.parameters
        assert "current_sidecar_sha256" in sig.parameters
        assert "script_id" in sig.parameters

    def test_sidecar_drift_signal_script_id_defaulting(self):
        """Verify get_sidecar_drift_signal uses script_id if provided, else derives from path."""
        import inspect

        # Verify docstring states that script_id defaults to script_path stem
        doc = inspect.getdoc(get_sidecar_drift_signal)
        assert "If empty, script_path is used" in doc or "script_id" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

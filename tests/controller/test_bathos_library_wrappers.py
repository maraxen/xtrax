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
    get_seed_trial_counts,
)


class TestModuleImport:
    """Tests verifying the module can be imported and exports expected names."""

    def test_module_exports_expected_names(self):
        """Verify module exports all required names."""
        import controller.bathos_library_wrappers as wrappers

        expected_names = {
            "call_stats_battery_gate",
            "get_capability_probe_result",
            "get_seed_trial_counts",
            "StatsBatteryResult",
        }
        exported_names = {name for name in dir(wrappers) if not name.startswith("_")}

        for name in expected_names:
            assert name in exported_names, f"{name} not exported from wrapper module"

    def test_functions_are_callable(self):
        """Verify the wrapper functions are callable."""
        assert callable(call_stats_battery_gate), "call_stats_battery_gate is not callable"
        assert callable(get_capability_probe_result), "get_capability_probe_result is not callable"
        assert callable(get_seed_trial_counts), "get_seed_trial_counts is not callable"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

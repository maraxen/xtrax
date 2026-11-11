"""Tests for scripts/assemble_changelog.py --check-new-modules gate (#3265)."""

from tests.ci.test_changelog_gate import (
    test_gate_fails_when_new_module_added_without_fragment,
    test_gate_passes_when_new_module_and_fragment_added,
    test_gate_passes_when_no_new_modules,
    test_gate_raises_on_unresolvable_base,
    test_gate_subpackage_init_requires_fragment,
)

__all__ = [
    "test_gate_fails_when_new_module_added_without_fragment",
    "test_gate_passes_when_new_module_and_fragment_added",
    "test_gate_passes_when_no_new_modules",
    "test_gate_raises_on_unresolvable_base",
    "test_gate_subpackage_init_requires_fragment",
]

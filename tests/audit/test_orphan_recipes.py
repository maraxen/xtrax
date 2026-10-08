"""Contract tests for the orphan-recipe enumerator (#5001).

The enumerator's job is to derive, not to remember. These tests pin the two edge
kinds it has to follow, the regex trap it was actually written wrong for the first
time, and the one condition it treats as a failure.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_orphan_recipes.py"

sys.path.insert(0, str(ROOT / "scripts"))

from audit_orphan_recipes import (  # noqa: E402
    JUST_CALL,
    discover_ci_entrypoints,
    load_recipes,
    orphan_audit_recipes,
    reachable_from,
    requires_arguments,
    unreferenced_workflow_recipes,
)


def _recipe(name, deps=(), body=(), parameters=()):
    return {
        "name": name,
        "dependencies": [{"recipe": dep, "arguments": []} for dep in deps],
        "body": [[line] for line in body],
        "parameters": list(parameters),
    }


class TestReachability:
    def test_follows_declared_prerequisites(self):
        recipes = {
            "entry": _recipe("entry", deps=["audit-a"]),
            "audit-a": _recipe("audit-a"),
        }
        assert reachable_from(recipes, {"entry"}) == {"entry", "audit-a"}

    def test_follows_just_calls_inside_a_body(self):
        """The edge kind that is easy to miss.

        `audit-deterministic` declares 19 prerequisites and calls 11 more recipes
        from its body. Counting only the dependency line understates CI coverage by
        more than a third -- an error made once already in this sprint.
        """
        recipes = {
            "entry": _recipe("entry", body=["just audit-b"]),
            "audit-b": _recipe("audit-b"),
        }
        assert reachable_from(recipes, {"entry"}) == {"entry", "audit-b"}

    def test_transitive_through_both_edge_kinds(self):
        recipes = {
            "entry": _recipe("entry", deps=["mid"]),
            "mid": _recipe("mid", body=["just audit-deep"]),
            "audit-deep": _recipe("audit-deep", deps=["audit-deeper"]),
            "audit-deeper": _recipe("audit-deeper"),
        }
        assert reachable_from(recipes, {"entry"}) == {
            "entry",
            "mid",
            "audit-deep",
            "audit-deeper",
        }

    def test_terminates_on_a_dependency_cycle(self):
        recipes = {
            "entry": _recipe("entry", body=["just audit-a"]),
            "audit-a": _recipe("audit-a", body=["just entry"]),
        }
        assert reachable_from(recipes, {"entry"}) == {"entry", "audit-a"}

    def test_ignores_entry_points_that_do_not_exist(self):
        assert reachable_from({"audit-a": _recipe("audit-a")}, {"ghost"}) == set()


class TestJustCallPattern:
    def test_does_not_match_across_a_newline(self):
        """Regression: this exact YAML produced a phantom `run` entry point.

        `\\s+` matched the newline between the prose word "just" and the next
        line's `run:` key, so the enumerator reported a workflow invoking a
        nonexistent recipe named `run` and exited 1.
        """
        yaml = "      - name: Install just\n        run: uv tool install rust-just\n"
        assert JUST_CALL.findall(yaml) == []

    def test_matches_a_real_invocation(self):
        assert JUST_CALL.findall("        run: just audit-deterministic\n") == [
            "audit-deterministic"
        ]

    def test_does_not_match_a_hyphenated_suffix(self):
        assert JUST_CALL.findall("uv tool install rust-just\n") == []

    def test_does_not_capture_a_flag(self):
        assert JUST_CALL.findall("run: just --list\n") == []

    def test_captures_the_full_hyphenated_recipe_name(self):
        """`just audit-port` must not be read as the prefix of `audit-port-static`."""
        assert JUST_CALL.findall("run: just audit-port-static\n") == ["audit-port-static"]


class TestArgumentDetection:
    def test_star_parameter_is_runnable_bare(self):
        recipe = _recipe("x", parameters=[{"name": "args", "kind": "star", "default": None}])
        assert requires_arguments(recipe) is False

    def test_defaulted_parameter_is_runnable_bare(self):
        recipe = _recipe("x", parameters=[{"name": "a", "kind": "singular", "default": "1"}])
        assert requires_arguments(recipe) is False

    def test_bare_singular_parameter_is_not(self):
        recipe = _recipe("x", parameters=[{"name": "a", "kind": "singular", "default": None}])
        assert requires_arguments(recipe) is True

    def test_plus_parameter_is_not(self):
        recipe = _recipe("x", parameters=[{"name": "a", "kind": "plus", "default": None}])
        assert requires_arguments(recipe) is True


class TestOrphanSelection:
    def test_only_audit_prefixed_recipes_are_reported(self):
        recipes = {
            "audit-loose": _recipe("audit-loose"),
            "validate-loose": _recipe("validate-loose"),
        }
        runnable, needs_args = orphan_audit_recipes(recipes, covered=set())
        assert runnable == ["audit-loose"]
        assert needs_args == []

    def test_covered_recipes_are_excluded(self):
        recipes = {"audit-a": _recipe("audit-a"), "audit-b": _recipe("audit-b")}
        runnable, _ = orphan_audit_recipes(recipes, covered={"audit-a"})
        assert runnable == ["audit-b"]

    def test_argument_requiring_orphans_are_split_out(self):
        recipes = {
            "audit-needs": _recipe(
                "audit-needs", parameters=[{"name": "a", "kind": "plus", "default": None}]
            ),
        }
        runnable, needs_args = orphan_audit_recipes(recipes, covered=set())
        assert runnable == []
        assert needs_args == ["audit-needs"]


class TestDanglingWorkflowRecipes:
    def test_flags_a_recipe_no_longer_in_the_justfile(self):
        dangling = unreferenced_workflow_recipes(
            {"audit-a": _recipe("audit-a")}, {"audit-a": {"ci.yml"}, "audit-gone": {"ci.yml"}}
        )
        assert dangling == [("audit-gone", {"ci.yml"})]

    def test_clean_when_every_entry_point_resolves(self):
        assert (
            unreferenced_workflow_recipes({"audit-a": _recipe("audit-a")}, {"audit-a": {"ci.yml"}})
            == []
        )


class TestAgainstTheRealRepository:
    def test_every_workflow_entry_point_resolves(self):
        """The script's only hard assertion, run against the live tree."""
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        assert entrypoints, "no `just` invocations found in .github/workflows"
        assert unreferenced_workflow_recipes(recipes, entrypoints) == []

    def test_deterministic_body_calls_are_counted_as_coverage(self):
        """`audit-deterministic` reaches more than its prerequisite line declares."""
        recipes = load_recipes(ROOT)
        declared = {dep["recipe"] for dep in recipes["audit-deterministic"]["dependencies"]}
        reached = reachable_from(recipes, {"audit-deterministic"})
        assert len(reached) > len(declared) + 1

    def test_orphans_and_covered_recipes_do_not_overlap(self):
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        covered = reachable_from(recipes, set(entrypoints))
        runnable, needs_args = orphan_audit_recipes(recipes, covered)
        assert not (set(runnable) | set(needs_args)) & covered

    def test_known_covered_gate_is_not_reported_as_an_orphan(self):
        """audit-docs-build-contract was wired into the chain in #130."""
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        covered = reachable_from(recipes, set(entrypoints))
        assert "audit-docs-build-contract" in covered

    def test_known_orphan_gate_is_reported(self):
        """audit-docs-build (the full, venv-mutating gate) is deliberately unchained."""
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        covered = reachable_from(recipes, set(entrypoints))
        runnable, _ = orphan_audit_recipes(recipes, covered)
        assert "audit-docs-build" in runnable


class TestSelfExclusion:
    def test_the_orphan_workflow_is_not_an_entry_point(self, tmp_path):
        """Coverage must not come from the workflow that runs the orphans.

        If audit-orphans.yml counted, every recipe it runs would be "reached", the
        derived list would empty itself, and the run would keep looking busy while
        testing nothing.
        """
        (tmp_path / "audit-orphans.yml").write_text("run: just audit-docs-build\n")
        (tmp_path / "ci.yml").write_text("run: just audit-deterministic\n")
        assert discover_ci_entrypoints(tmp_path) == {"audit-deterministic": {"ci.yml"}}

    def test_live_orphan_list_has_not_collapsed(self):
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        covered = reachable_from(recipes, set(entrypoints))
        runnable, _ = orphan_audit_recipes(recipes, covered)
        assert len(runnable) > 20, "orphan list collapsed -- check for self-coverage"


class TestCommandLine:
    @pytest.mark.parametrize("fmt", ["lines", "json", "report"])
    def test_exits_zero_on_the_current_tree(self, fmt):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--format", fmt],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_lines_format_emits_one_recipe_per_line(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--format", "lines"],
            capture_output=True,
            text=True,
            check=False,
        )
        names = result.stdout.split()
        assert names
        assert all(name.startswith("audit-") for name in names)
        assert len(names) == len(result.stdout.strip().splitlines())

    def test_json_format_is_parseable_and_complete(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
        payload = json.loads(result.stdout)
        assert payload["dangling_workflow_recipes"] == {}
        assert payload["audit_covered"] < payload["audit_total"]
        assert set(payload["orphans"]).isdisjoint(payload["entrypoints"])

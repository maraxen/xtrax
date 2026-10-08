"""Phase 2 of the orphan-recipe run (#5002): gating on the pinned passing set.

The orphan set stays derived; only which orphans GATE is listed, in
`.github/audit_orphans_pinned.toml`. These tests pin the gate's semantics (a pinned
recipe that did not pass -- including one that never ran -- is a regression; an
unpinned or unclassified failure is reported, not gated) and check the real pin file
against the real Justfile.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_orphan_recipes.py"

sys.path.insert(0, str(ROOT / "scripts"))

from audit_orphan_recipes import (  # noqa: E402
    PINNED_PATH,
    check_pinned,
    discover_ci_entrypoints,
    load_pinned,
    load_recipes,
    orphan_audit_recipes,
    reachable_from,
    read_results,
)


def _recipe(name):
    return {"name": name, "dependencies": [], "body": [], "parameters": []}


RECIPES = {n: _recipe(n) for n in ("audit-a", "audit-b", "audit-c", "audit-wired")}


def _check(results, pinned=("audit-a", "audit-b"), not_pinned=None, orphans=None):
    return check_pinned(
        pinned=list(pinned),
        not_pinned=dict(not_pinned or {"audit-c": "reason"}),
        orphans=list(orphans or ["audit-a", "audit-b", "audit-c"]),
        recipes=RECIPES,
        results=results,
    )


class TestPinnedCheck:
    def test_all_pinned_pass_is_clean(self):
        check = _check({"audit-a": "PASS", "audit-b": "PASS", "audit-c": "FAIL"})
        assert check.regressions == []
        assert check.unknown == []

    def test_pinned_failure_is_a_regression(self):
        assert _check({"audit-a": "PASS", "audit-b": "FAIL"}).regressions == ["audit-b"]

    def test_pinned_recipe_that_never_ran_is_a_regression(self):
        """Silence is not a pass: a pinned recipe absent from the results gates."""
        assert _check({"audit-a": "PASS"}).regressions == ["audit-b"]

    def test_unpinned_failure_does_not_gate(self):
        check = _check({"audit-a": "PASS", "audit-b": "PASS", "audit-c": "FAIL"})
        assert "audit-c" not in check.regressions

    def test_new_orphan_is_unclassified_not_gated(self):
        check = _check(
            {"audit-a": "PASS", "audit-b": "PASS", "audit-new": "FAIL"},
            orphans=["audit-a", "audit-b", "audit-c", "audit-new"],
        )
        assert check.unclassified == ["audit-new"]
        assert check.regressions == []

    def test_pinning_a_nonexistent_recipe_is_unknown(self):
        assert _check({"audit-a": "PASS"}, pinned=("audit-a", "audit-typo")).unknown == [
            "audit-typo"
        ]

    def test_pinned_recipe_now_reached_by_ci_is_noted_not_gated(self):
        check = _check({"audit-a": "PASS"}, pinned=("audit-a", "audit-wired"))
        assert check.not_orphans == ["audit-wired"]
        assert check.regressions == []

    def test_pinned_recipe_needing_arguments_is_an_error_not_a_ci_note(self):
        """Unreached but unrunnable bare: it must not read as 'reached by CI now'."""
        check = check_pinned(
            pinned=["audit-a", "audit-wired"],
            not_pinned={},
            orphans=["audit-a"],
            recipes=RECIPES,
            results={"audit-a": "PASS"},
            needs_arguments=["audit-wired"],
        )
        assert check.needs_arguments == ["audit-wired"]
        assert check.not_orphans == []

    def test_overlap_between_tables_is_rejected(self, tmp_path):
        pin = tmp_path / "pin.toml"
        pin.write_text('pinned = ["audit-a"]\n[not_pinned]\naudit-a = "x"\n')
        with pytest.raises(ValueError, match="both pinned and not_pinned"):
            load_pinned(pin)

    def test_read_results_parses_tab_lines(self, tmp_path):
        tsv = tmp_path / "r.tsv"
        tsv.write_text("audit-a\tPASS\naudit-b\tFAIL\n\n")
        assert read_results(tsv) == {"audit-a": "PASS", "audit-b": "FAIL"}


class TestRealPinFile:
    def test_every_named_recipe_exists_and_is_still_an_orphan(self):
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        runnable, _ = orphan_audit_recipes(recipes, reachable_from(recipes, set(entrypoints)))
        pinned, not_pinned = load_pinned(ROOT / PINNED_PATH)
        check = check_pinned(
            pinned=pinned, not_pinned=not_pinned, orphans=runnable, recipes=recipes, results={}
        )
        assert check.unknown == []
        assert check.not_orphans == []

    def test_every_not_pinned_entry_states_a_reason(self):
        _, not_pinned = load_pinned(ROOT / PINNED_PATH)
        assert not_pinned
        assert all(reason.strip() for reason in not_pinned.values())


def _run_check(tmp_path, lines):
    tsv = tmp_path / "results.tsv"
    tsv.write_text("".join(f"{name}\t{status}\n" for name, status in lines))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--check-results", str(tsv)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


class TestCheckResultsCli:
    def test_exits_zero_when_every_pinned_recipe_passes(self, tmp_path):
        pinned, _ = load_pinned(ROOT / PINNED_PATH)
        proc = _run_check(tmp_path, [(n, "PASS") for n in pinned])
        assert proc.returncode == 0, proc.stderr
        assert "PASS: all" in proc.stdout

    def test_exits_one_on_a_pinned_regression(self, tmp_path):
        pinned, _ = load_pinned(ROOT / PINNED_PATH)
        lines = [(n, "PASS") for n in pinned[1:]] + [(pinned[0], "FAIL")]
        proc = _run_check(tmp_path, lines)
        assert proc.returncode == 1
        assert f"pinned recipe {pinned[0]} did not pass" in proc.stderr

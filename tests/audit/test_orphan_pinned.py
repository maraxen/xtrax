"""Phase 2 of the orphan-recipe run (#5002): gating on the pinned passing set.

The orphan set stays derived; only which orphans GATE is listed, in
`.github/audit_orphans_pinned.toml`. These tests pin the gate's semantics (a pinned
recipe that did not pass -- including one that never ran -- is a regression; an
unpinned or unclassified failure is reported, not gated) and check the real pin file
against the real Justfile.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_orphan_recipes.py"

sys.path.insert(0, str(ROOT / "scripts"))

from audit_orphan_recipes import (  # noqa: E402
    JOB_TIMEOUT_S,
    PINNED_PATH,
    STEP_DEADLINE_S,
    check_pinned,
    discover_ci_entrypoints,
    load_pinned,
    load_recipes,
    load_skipped,
    load_timeouts,
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
        skipped = load_skipped(ROOT / PINNED_PATH)
        assert not_pinned or skipped
        assert all(reason.strip() for reason in {**not_pinned, **skipped}.values())

    def test_every_skipped_recipe_is_a_live_orphan(self):
        recipes = load_recipes(ROOT)
        entrypoints = discover_ci_entrypoints(ROOT / ".github" / "workflows")
        runnable, _ = orphan_audit_recipes(recipes, reachable_from(recipes, set(entrypoints)))
        assert set(load_skipped(ROOT / PINNED_PATH)) <= set(runnable)

    def test_timeout_for_release_readiness_is_above_the_default(self):
        timeouts = load_timeouts(ROOT / PINNED_PATH)
        assert timeouts["audit-release-readiness"] > 900
        assert timeouts["audit-release-readiness"] < JOB_TIMEOUT_S


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


class TestTimeoutAndSkipTables:
    def _pin(self, tmp_path, body):
        pin = tmp_path / "pin.toml"
        pin.write_text(body)
        return pin

    def test_timeouts_default_is_absent_and_explicit_value_is_read(self, tmp_path):
        assert load_timeouts(self._pin(tmp_path, "pinned = []\n")) == {}
        assert load_timeouts(self._pin(tmp_path, "[timeouts]\naudit-a = 1200\n")) == {
            "audit-a": 1200
        }

    @pytest.mark.parametrize("value", ['"3000"', "0", "-5", "5400", "true", "2.5"])
    def test_malformed_timeout_fails_loudly(self, tmp_path, value):
        with pytest.raises(ValueError, match="\\[timeouts\\]"):
            load_timeouts(self._pin(tmp_path, f"[timeouts]\naudit-a = {value}\n"))

    def test_skip_requires_a_reason(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty reason"):
            load_skipped(self._pin(tmp_path, '[skipped]\naudit-a = "  "\n'))

    def test_pinned_and_skipped_overlap_is_rejected(self, tmp_path):
        pin = self._pin(tmp_path, 'pinned = ["audit-a"]\n[skipped]\naudit-a = "x"\n')
        with pytest.raises(ValueError, match="both pinned and skipped"):
            load_skipped(pin)

    def test_skipped_is_not_gated_even_when_it_fails(self):
        check = check_pinned(
            pinned=["audit-a"],
            not_pinned={"audit-skip": "praxia missing"},
            orphans=["audit-a", "audit-skip"],
            recipes=RECIPES,
            results={"audit-a": "PASS", "audit-skip": "SKIP"},
        )
        assert check.regressions == []
        assert check.unclassified == []


WORKFLOW = ROOT / ".github" / "workflows" / "audit-orphans.yml"


def _run_script(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


class TestTypoedTimeoutKeyFailsLoudly:
    def test_check_pinned_flags_a_timeout_key_that_is_not_an_orphan(self):
        check = check_pinned(
            pinned=["audit-a"],
            not_pinned={},
            orphans=["audit-a"],
            recipes=RECIPES,
            results={"audit-a": "PASS"},
            timeouts={"audit-relase-readiness": 3000},
        )
        assert check.bad_timeouts == ["audit-relase-readiness"]

    def test_check_pinned_flags_a_timeout_key_that_is_skipped(self):
        check = check_pinned(
            pinned=["audit-a"],
            not_pinned={},
            orphans=["audit-a", "audit-skip"],
            recipes=RECIPES,
            results={"audit-a": "PASS"},
            timeouts={"audit-skip": 1200},
            skipped={"audit-skip": "praxia missing"},
        )
        assert check.bad_timeouts == ["audit-skip"]

    def test_valid_timeout_key_is_not_flagged(self):
        check = check_pinned(
            pinned=["audit-a"],
            not_pinned={},
            orphans=["audit-a"],
            recipes=RECIPES,
            results={"audit-a": "PASS"},
            timeouts={"audit-a": 1200},
        )
        assert check.bad_timeouts == []

    def test_cli_main_fails_on_a_typoed_key_in_every_mode(self, tmp_path):
        pin = tmp_path / "pin.toml"
        pin.write_text("pinned = []\n[timeouts]\naudit-relase-readiness = 3000\n")
        for extra in (["--format", "lines"], ["--timeout-for", "audit-bootstrap"]):
            proc = _run_script("--pinned", str(pin), *extra)
            assert proc.returncode == 1, extra
            assert "audit-relase-readiness" in proc.stderr

    def test_cli_main_fails_on_a_timeout_key_listed_in_skipped(self, tmp_path):
        pin = tmp_path / "pin.toml"
        pin.write_text(
            '[timeouts]\naudit-drift-check = 1200\n[skipped]\naudit-drift-check = "why"\n'
        )
        proc = _run_script("--pinned", str(pin), "--format", "lines")
        assert proc.returncode == 1
        assert "audit-drift-check" in proc.stderr


class TestTimeoutForCli:
    def test_prints_the_pinned_budget(self):
        proc = _run_script("--timeout-for", "audit-release-readiness")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "3000"

    def test_prints_the_default_for_an_unlisted_recipe(self):
        proc = _run_script("--timeout-for", "audit-bootstrap")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "900"

    @pytest.mark.parametrize(
        "body", ['[timeouts]\naudit-bootstrap = "3000"\n', "[timeouts]\naudit-bootstrap = 0\n"]
    )
    def test_malformed_file_exits_nonzero(self, tmp_path, body):
        pin = tmp_path / "pin.toml"
        pin.write_text(body)
        proc = _run_script("--pinned", str(pin), "--timeout-for", "audit-bootstrap")
        assert proc.returncode != 0
        assert "[timeouts]" in proc.stderr


class TestSkipFormats:
    def test_lines_format_excludes_skipped_recipes(self):
        proc = _run_script("--format", "lines")
        assert proc.returncode == 0, proc.stderr
        assert "audit-drift-check" not in proc.stdout.split()

    def test_skipped_format_emits_recipe_tab_reason(self):
        proc = _run_script("--format", "skipped")
        assert proc.returncode == 0, proc.stderr
        rows = [line.split("\t", 1) for line in proc.stdout.splitlines()]
        assert rows and rows[0][0] == "audit-drift-check"
        assert "praxia" in rows[0][1]

    def test_report_counts_only_runnable_recipes(self):
        proc = _run_script("--format", "report")
        assert proc.returncode == 0, proc.stderr
        runnable = _run_script("--format", "lines").stdout.split()
        assert f"Unreached and runnable ({len(runnable)}):" in proc.stdout

    def test_json_lists_skipped_separately_from_orphans(self):
        payload = json.loads(_run_script("--format", "json").stdout)
        assert "audit-drift-check" not in payload["orphans"]
        assert "audit-drift-check" in payload["skipped"]


class TestSkipReasonValidation:
    @pytest.mark.parametrize("value", ["5", "[]", "true"])
    def test_non_string_reason_is_rejected(self, tmp_path, value):
        pin = tmp_path / "pin.toml"
        pin.write_text(f"[skipped]\naudit-a = {value}\n")
        with pytest.raises(ValueError, match="non-empty reason"):
            load_skipped(pin)

    @pytest.mark.parametrize(
        "value", ['"line one\\nline two"', '"tab\\there"', '"carriage\\rreturn"']
    )
    def test_multiline_or_tabbed_reason_is_rejected(self, tmp_path, value):
        pin = tmp_path / "pin.toml"
        pin.write_text(f"[skipped]\naudit-a = {value}\n")
        with pytest.raises(ValueError, match="single line"):
            load_skipped(pin)


class TestWorkflowMatchesTheScript:
    def test_job_timeout_constant_matches_the_yaml(self):
        data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        minutes = data["jobs"]["audit-orphans"]["timeout-minutes"]
        assert minutes * 60 == JOB_TIMEOUT_S

    def test_step_deadline_matches_the_workflow_and_sits_inside_the_job(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        match = re.search(r"deadline=\$\(\(SECONDS \+ (\d+) \* 60\)\)", text)
        assert match, "recipe loop has no deadline"
        assert int(match.group(1)) * 60 == STEP_DEADLINE_S
        assert STEP_DEADLINE_S < JOB_TIMEOUT_S

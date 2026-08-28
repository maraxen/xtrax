"""Standing gate: telemetry enforcement cannot be removed or silently skipped.

Half these tests prove the gate PASSES on the repo as it stands. The other half
prove it FAILS on the specific regressions it exists to catch -- a gate nobody
has ever seen fail is indistinguishable from a gate that cannot fail, and this
one guards a property (every executed run leaves a reconstructable record) whose
loss is invisible until someone needs the record and it is not there.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_telemetry_coverage import (  # noqa: E402
    check_engine_enforces,
    check_verbs_declared,
    load_contract,
    main,
)


@pytest.fixture
def contract():
    return load_contract()


# --- the repo as it stands --------------------------------------------------


def test_repo_currently_passes(contract):
    assert check_engine_enforces(contract) == []
    assert check_verbs_declared(contract) == []


def test_script_subprocess_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/audit_telemetry_coverage.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS: telemetry coverage" in result.stdout


def test_every_registry_verb_is_declared(contract):
    """The contract must actually cover the live verb surface."""
    from audit_telemetry_coverage import _registry_keys

    keys = set(_registry_keys())
    assert keys, "REGISTRY keys could not be read"
    assert keys <= set(contract["verbs"]), f"undeclared verbs: {keys - set(contract['verbs'])}"


def test_the_executing_verbs_are_marked_recording(contract):
    """run/resume/sweep/export execute or lower user code; none may be exempt."""
    for verb in ("run", "resume", "sweep", "export"):
        assert contract["verbs"][verb] == "records"


# --- the gate detects its own regressions -----------------------------------


def test_detects_removed_engine_enforcement(tmp_path, contract):
    """The regression this gate exists for: someone deletes the ledger wiring."""
    stripped = tmp_path / "engine.py"
    stripped.write_text(
        "class Engine:\n"
        "    async def fit(self, state):\n"
        "        return state\n"
        "    async def eval(self, state):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    problems = check_engine_enforces(contract, engine_path=stripped)
    assert len(problems) == 2
    assert all("no longer calls _resolve_ledger" in p for p in problems)
    assert any("provenance cannot be captured retroactively" in p for p in problems)


def test_detects_a_renamed_enforcing_method(tmp_path, contract):
    renamed = tmp_path / "engine.py"
    renamed.write_text(
        "class Engine:\n"
        "    async def train(self, state):\n"
        "        ledger, owns = _resolve_ledger(None, None, 'train')\n"
        "        return state\n",
        encoding="utf-8",
    )
    problems = check_engine_enforces(contract, engine_path=renamed)
    assert any("was it renamed?" in p for p in problems)


def test_accepts_a_method_that_does_enforce(tmp_path):
    ok = tmp_path / "engine.py"
    ok.write_text(
        "class Engine:\n"
        "    async def fit(self, state):\n"
        "        ledger, owns = _resolve_ledger(None, None, 'train')\n"
        "        return state\n",
        encoding="utf-8",
    )
    assert check_engine_enforces({"engine": {"enforcing_methods": ["fit"]}}, engine_path=ok) == []


def test_a_docstring_mentioning_the_resolver_does_not_satisfy_the_gate(tmp_path):
    """AST, not grep: prose about the ledger is not the ledger."""
    prose = tmp_path / "engine.py"
    prose.write_text(
        "class Engine:\n"
        "    async def fit(self, state):\n"
        '        """This method used to call _resolve_ledger before it was removed."""\n'
        "        return state\n",
        encoding="utf-8",
    )
    problems = check_engine_enforces({"engine": {"enforcing_methods": ["fit"]}}, engine_path=prose)
    assert len(problems) == 1


def test_detects_an_undeclared_verb(tmp_path, contract):
    """A new verb must not be able to skip the telemetry decision."""
    registry = tmp_path / "registry.py"
    registry.write_text(
        'REGISTRY = {\n    "run": (1, 2),\n    "brand_new_verb": (3, 4),\n}\n',
        encoding="utf-8",
    )
    problems = check_verbs_declared(contract, registry_path=registry)
    assert len(problems) == 1
    assert "brand_new_verb" in problems[0]
    assert "audit/telemetry_coverage.toml" in problems[0]


def test_detects_an_invalid_disposition():
    problems = check_verbs_declared({"verbs": {"run": "sometimes_maybe"}})
    assert any("unknown disposition" in p for p in problems)


def test_detects_an_unreadable_registry(tmp_path, contract):
    """Failing closed: if the verb surface cannot be read, the gate does not pass."""
    unreadable = tmp_path / "registry.py"
    unreadable.write_text("SOMETHING_ELSE = {}\n", encoding="utf-8")
    problems = check_verbs_declared(contract, registry_path=unreadable)
    assert any("could not read REGISTRY keys" in p for p in problems)


def test_main_returns_nonzero_on_a_broken_contract(tmp_path):
    broken = tmp_path / "telemetry_coverage.toml"
    broken.write_text('[verbs]\nrun = "records"\n[engine]\nenforcing_methods = []\n')
    assert main(["--contract", str(broken)]) == 1

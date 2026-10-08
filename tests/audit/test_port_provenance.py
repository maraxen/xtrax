"""Port provenance gate and port_init scaffold (#2591, #2592).

Record, reader, and renderer tests live in tests/test_provenance.py, which the
tier-1 coverage run includes (tests/audit is excluded there).
"""

from __future__ import annotations

import ast
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.audit_port_provenance import audit_port_provenance, known_ported
from scripts.audit_port_provenance import main as gate_main
from scripts.port_init import main as init_main
from xtrax.provenance import (
    collect_provenance,
    render_declaration,
)

ROOT = Path(__file__).resolve().parents[2]

_KNOWN_PORTED = [
    "src/xtrax/profiling/__init__.py",
    "src/xtrax/profiling/claims.py",
    "src/xtrax/profiling/emitters.py",
    "src/xtrax/profiling/record.py",
    "src/xtrax/profiling/report.py",
    "src/xtrax/profiling/trace.py",
    "src/xtrax/transforms/map.py",
]

_NETWORK_MODULES = {"socket", "urllib", "http", "requests", "urllib3", "httpx"}


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "xtrax").mkdir(parents=True)
    (repo / "port" / "manifests").mkdir(parents=True)
    return repo


def test_known_ported_is_manifest_kernels_or_origin_docstrings() -> None:
    found = known_ported(ROOT)
    rels = [path.relative_to(ROOT).as_posix() for path, _why in found]
    assert rels == _KNOWN_PORTED
    reasons = {path.name: why for path, why in found}
    assert "module_path in port/manifests/wave_001_example.toml" in reasons["map.py"]
    assert "docstring says ported/vendored/adapted/upstreamed from" in reasons["emitters.py"]
    assert "docstring says ported/vendored/adapted/upstreamed from" in reasons["claims.py"]


def test_planted_docstring_without_declaration_fails_the_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "xtrax" / "ported_mod.py").write_text(
        '"""Ported from example upstream."""\n\nx = 1\n',
        encoding="utf-8",
    )
    failures = audit_port_provenance(repo)
    assert failures == [
        "src/xtrax/ported_mod.py: known ported module lacks __provenance__ "
        "(docstring says ported/vendored/adapted/upstreamed from)"
    ]
    assert gate_main(["--root", str(repo)]) == 1


def test_planted_manifest_module_without_declaration_fails_the_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "xtrax" / "mapped.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (repo / "port" / "manifests" / "wave.toml").write_text(
        textwrap.dedent(
            """\
            [manifest]
            wave_id = "w"

            [[kernels]]
            order = 1
            qualname = "xtrax.mapped.f"
            module_path = "src/xtrax/mapped.py"
            depends_on = []
            """
        ),
        encoding="utf-8",
    )
    failures = audit_port_provenance(repo)
    assert failures == [
        "src/xtrax/mapped.py: known ported module lacks __provenance__ "
        "(module_path in port/manifests/wave.toml)"
    ]


def test_missing_manifest_target_fails_the_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "port" / "manifests" / "wave.toml").write_text(
        '[[kernels]]\nmodule_path = "src/xtrax/missing.py"\n',
        encoding="utf-8",
    )
    assert audit_port_provenance(repo) == [
        "src/xtrax/missing.py: known ported module is missing "
        "(module_path in port/manifests/wave.toml)"
    ]


def test_imported_from_wording_is_not_a_port_claim(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "xtrax" / "plain.py").write_text(
        '"""Helpers imported from the stdlib."""\n',
        encoding="utf-8",
    )
    assert audit_port_provenance(repo) == []


def test_declared_module_passes_the_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "xtrax" / "ported_mod.py").write_text(
        textwrap.dedent(
            """\
            \"\"\"Adapted from example upstream.\"\"\"

            __provenance__ = {
                "upstream": "example",
                "relationship": "derived",
                "waiver_reason": "no revision and no SPDX id in the fixture",
            }
            """
        ),
        encoding="utf-8",
    )
    assert audit_port_provenance(repo) == []


def test_invalid_declaration_fails_the_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "xtrax" / "ported_mod.py").write_text(
        '"""Vendored from example."""\n'
        '__provenance__ = {"upstream": "example", "relationship": "nope"}\n',
        encoding="utf-8",
    )
    failures = audit_port_provenance(repo)
    assert len(failures) == 1
    assert "invalid __provenance__" in failures[0]
    assert "relationship" in failures[0]


def test_real_tree_gate_is_green_and_waivers_are_not_invented_shas() -> None:
    assert audit_port_provenance(ROOT) == []
    found = collect_provenance(ROOT / "src" / "xtrax")
    emitters = found["xtrax.profiling.emitters"]
    assert emitters.upstream == "prolix"
    assert emitters.relationship == "derived"
    assert emitters.revision is None
    assert emitters.licence is None
    assert emitters.waiver_reason is not None
    assert "wt-20260807-132628" in emitters.waiver_reason
    record = found["xtrax.profiling.record"]
    assert record.relationship == "ported"
    assert record.revision is None
    assert record.waiver_reason is not None
    assert "SPDX" in record.waiver_reason
    mapped = found["xtrax.transforms.map"]
    assert mapped.upstream == "port/reference/safe_map"
    assert mapped.relationship == "ported"
    assert mapped.revision is None
    assert mapped.licence is None
    assert "52fd5458018d46d3c333287f803152eb38f66b618f540363521d825b518aea34" in (
        mapped.waiver_reason or ""
    )


def test_port_init_cli_emits_and_refuses(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        init_main(
            [
                "--upstream",
                "https://example.com/up",
                "--relationship",
                "ported",
                "--revision",
                "v1.2.3",
                "--licence",
                "Apache-2.0",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Apache-2.0" in out
    assert "__provenance__" in out
    assert (
        init_main(
            [
                "--upstream",
                "prolix",
                "--relationship",
                "ported",
            ]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "waiver_reason" in err


def test_scaffold_performs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    text = render_declaration(
        upstream="https://example.com/up",
        relationship="ported",
        revision="v1",
        licence="MIT",
    )
    assert "MIT" in text
    for rel in (
        "src/xtrax/provenance.py",
        "scripts/port_init.py",
        "scripts/audit_port_provenance.py",
    ):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in _NETWORK_MODULES


def test_gate_is_wired_into_audit_port() -> None:
    justfile = (ROOT / "Justfile").read_text(encoding="utf-8")
    deps = re.search(r"^audit-port:(.*)$", justfile, re.MULTILINE)
    assert deps is not None
    assert "audit-port-provenance" in deps.group(1).split()
    body = re.search(r"^audit-port-provenance:\n((?:[ \t]+.*\n)+)", justfile, re.MULTILINE)
    assert body is not None
    assert "scripts/audit_port_provenance.py" in body.group(1)
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "just audit-port" in workflow
    assert "scripts/audit_port_provenance.py" in workflow


def test_gate_script_exits_zero_on_the_real_tree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_port_provenance.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "port provenance ok" in result.stdout

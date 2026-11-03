"""writer_liveness stays stdlib-only and importable under Python 3.9 (spec 261030 §3.1, A8).

Its annotations are evaluated at import (the audit forbids ``from __future__ import annotations``
in ``src/``), so PEP 604 ``X | None`` would break a 3.9 node. Two pins, as for the fileguard core:
an ``ast`` stdlib-import check, and an isolated ``-I`` run of the module file under a Python 3.9
interpreter found with ``uv python find 3.9`` (skipped only when none exists; CI provides one).
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax.run import writer_liveness

MODULE = Path(writer_liveness.__file__)


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("<relative>" if node.level else (node.module or "").split(".")[0])
    return roots


def test_imports_only_the_standard_library() -> None:
    roots = _imported_roots(MODULE.read_text(encoding="utf-8"))
    assert roots <= set(sys.stdlib_module_names), roots - set(sys.stdlib_module_names)


def _python39() -> str | None:
    uv = shutil.which("uv")
    if uv is None:
        return None
    proc = subprocess.run(
        [uv, "python", "find", "3.9"], capture_output=True, text=True, check=False
    )
    path = proc.stdout.strip()
    return path if proc.returncode == 0 and path else None


_SMOKE = r"""
import json, os, sys, tempfile
assert sys.version_info[:2] == (3, 9), sys.version
ns = {"__name__": "writer_liveness_node"}
exec(compile(open(sys.argv[1]).read(), "writer_liveness_node", "exec"), ns)
d = tempfile.mkdtemp()
p = os.path.join(d, "node-1-777-0-abcdef12.jsonl")
with open(p, "w") as f:
    f.write('{"a": 1}\n')
old = os.stat(p).st_mtime - 1000.0
os.utime(p, (old, old))
def absent(argv, timeout):
    if argv[0] == "squeue":
        return (1, "", "slurm_load_jobs error: Invalid job id specified")
    return (0, "COMPLETED\n", "")
def broken(argv, timeout):
    raise OSError("broken scheduler config")
v = ns["segment_writer_dead"](p, grace_s=300, runner=absent)
assert v.dead, v
assert not ns["segment_writer_dead"](p, grace_s=300, runner=broken).dead
print("ok")
"""


def _run39(py: str, source: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    smoke = tmp_path / "smoke.py"
    smoke.write_text(_SMOKE)
    return subprocess.run(
        [py, "-I", str(smoke), str(source)], capture_output=True, text=True, check=False
    )


def test_runs_standalone_under_python39(tmp_path: Path) -> None:
    py = _python39()
    if py is None:
        pytest.skip("no Python 3.9 interpreter (uv python find 3.9)")
    proc = _run39(py, MODULE, tmp_path)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_negative_control_pep604_fails_under_python39(tmp_path: Path) -> None:
    # The pin must be able to fail: the pre-fix annotation style breaks a 3.9 import.
    py = _python39()
    if py is None:
        pytest.skip("no Python 3.9 interpreter (uv python find 3.9)")
    bad = tmp_path / "bad.py"
    bad.write_text(MODULE.read_text(encoding="utf-8").replace("Optional[str]", "str | None"))
    proc = _run39(py, bad, tmp_path)
    assert proc.returncode != 0 and "unsupported operand" in proc.stderr, proc.stderr

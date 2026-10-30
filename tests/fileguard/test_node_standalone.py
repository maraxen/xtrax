"""The fileguard core stays stdlib-only, Python 3.9 compatible and embeddable (spec 261030 §3.4).

Two pins: an ``ast`` import check (runs everywhere) and an isolated ``-I`` run of the source text
from :func:`xtrax.fileguard.node_source` under a Python 3.9 interpreter, found with ``uv python
find 3.9``. The 3.9 run is skipped only when no 3.9 interpreter can be found; CI provides one.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax import fileguard

NODE = Path(fileguard.__file__).with_name("_node.py")


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add("<relative>")
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_node_core_imports_only_the_standard_library():
    roots = _imported_roots(NODE.read_text(encoding="utf-8"))
    assert roots <= set(sys.stdlib_module_names), roots - set(sys.stdlib_module_names)


def test_negative_control_import_check_catches_a_non_stdlib_import():
    roots = _imported_roots("import xtrax.run\nfrom . import sibling\n")
    assert not roots <= set(sys.stdlib_module_names)


def test_node_source_is_the_module_text():
    assert fileguard.node_source() == NODE.read_text(encoding="utf-8")


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
import sys
assert sys.version_info[:2] == (3, 9), sys.version
ns = {"__name__": "fileguard_node"}
exec(compile(open(sys.argv[1]).read(), "fileguard_node", "exec"), ns)
LFS = (
    "Disk quotas for usr u (uid 1):\n"
    "     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace\n"
    "/scratch  1  0  0  -  799990  1000000 1000000  -\n"
)
def run(argv):
    return (0, "lustre\n") if argv[0] == "stat" else (0, LFS)
seen = []
g = ns["FileCapGuard"](sys.argv[2], runner=run, user="u",
                        emit=lambda n, f: seen.append(n), check_setgid_group=False)
try:
    g.check(100)
except ns["QuotaRefused"] as e:
    assert "R-X1" in str(e)
else:
    raise SystemExit("guard did not refuse")
ns["check_concurrency_slack"](1000000, n_jobs=10, reread_every=1000, cap_fraction=0.8)
print("ok", seen)
"""


def test_node_core_runs_isolated_under_python_39(tmp_path):
    py39 = _python39()
    if py39 is None:
        pytest.skip("no Python 3.9 interpreter (uv python find 3.9)")
    src = tmp_path / "embedded_fileguard.py"
    src.write_text(fileguard.node_source(), encoding="utf-8")
    proc = subprocess.run(
        [py39, "-I", "-c", _SMOKE, str(src), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("ok")

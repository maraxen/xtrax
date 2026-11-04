"""The pack core stays stdlib-only, Python 3.9 compatible and embeddable (spec 261030 §3.3).

Two pins: an ``ast`` import check (runs everywhere) and an isolated ``-I`` run of the source text
from :func:`xtrax.pack.node_source` under a Python 3.9 interpreter, found with ``uv python find
3.9``. The 3.9 run is skipped only when no 3.9 interpreter can be found; CI provides one.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax import pack

NODE = Path(pack.__file__).with_name("_node.py")


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
    assert pack.node_source() == NODE.read_text(encoding="utf-8")


def _python39() -> str | None:
    uv = shutil.which("uv")
    if uv is None:
        return None
    proc = subprocess.run(
        [uv, "python", "find", "3.9"], capture_output=True, text=True, check=False
    )
    path = proc.stdout.strip()
    return path if proc.returncode == 0 and path else None


# Runs from the source text alone: a whole commit, pack, read and sweep cycle, as a node does it.
_SMOKE = r"""
import os, sys
assert sys.version_info[:2] == (3, 9), sys.version
ns = {"__name__": "pack_node"}
exec(compile(open(sys.argv[1]).read(), "pack_node", "exec"), ns)
unit = sys.argv[2]
os.makedirs(unit + "/fold")
with open(unit + "/fold/a.txt", "w") as fh:
    fh.write("alpha\n")
with open(unit + "/.hidden", "w") as fh:
    fh.write("hid\n")
record = ns["write_unit_record"](unit, input_digest="in", run_id="r")
loose = ns["unit_content_digest"](unit)
assert loose == record["content_digest"]
done = ns["pack_unit"](unit)
assert done["record_digest"] == record["record_digest"]
assert sorted(os.listdir(unit)) == ["members.idx.json", "members.tar.zst", "unit.json"]
assert ns["read_unit_member"](unit, "fold/a.txt") == b"alpha\n"
assert ns["unit_content_digest"](unit) == loose
print("ok")
"""


def test_node_core_runs_isolated_under_python_39(tmp_path):
    py39 = _python39()
    if py39 is None:
        pytest.skip("no Python 3.9 interpreter (uv python find 3.9)")
    src = tmp_path / "embedded_pack.py"
    src.write_text(pack.node_source(), encoding="utf-8")
    proc = subprocess.run(
        [py39, "-I", "-c", _SMOKE, str(src), str(tmp_path / "unit")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("ok")

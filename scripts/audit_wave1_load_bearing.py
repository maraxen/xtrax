#!/usr/bin/env python3
"""T2-02 wave-1 load-bearing gate (#3070): S2/S3/S4 landed as real code, not stubs.

F-C dependency binding (`.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md:
216-233`): before #2181's P1 walking skeleton can start, this gate verifies the S2 (executed
io_callback boundaries, T1-04+T1-05), S3 (graph serialize/CLI, T1-08+T1-11), and S4 (sealed
EvaluateFn seam, T1-07) deliverables are real, load-bearing `src/xtrax` code, not test doubles
(PM-3 anti-stub).

Three layered checks, since a pure importability check does not satisfy PM-3's anti-stub framing
or PM1's caution that T1-04's flat-scan AC4 alone is necessary-not-sufficient -- only T1-05
(nested-ordering stress harness) being green actually certifies the executed-boundaries
deliverable:

1. Importability (subprocess-isolated, mirrors tests/test_import_isolation.py's pattern).
2. Symbol-presence + stub-marker detection: each required symbol's own source is inspected for
   NotImplementedError/TODO/STUB/FIXME markers, or a body that's nothing but `pass`/`...`.
3. Live re-run of each deliverable's own certifying test file -- the check PM1's caution
   specifically calls for. Checks 1/2 alone would NOT catch a broken nested-ordering assertion
   (the module still imports fine, the symbols still exist) -- only actually re-running
   tests/stages/test_nested_ordering.py catches that.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMPORT_TARGETS: tuple[str, ...] = (
    "xtrax.stages.executor",  # T1-04
    "xtrax.stages.evaluate",  # T1-07
    "xtrax.composition.serialize",  # T1-08
    "xtrax.cli.graph_plan_verb",  # T1-11
)

REQUIRED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "xtrax.stages.executor": ("execute_map_axis", "execute_scan_axis"),
    "xtrax.stages.evaluate": ("SealedEvaluatorRegistry", "EvaluateFn"),
    "xtrax.composition.serialize": ("serialize_graph", "deserialize_graph"),
    "xtrax.cli.graph_plan_verb": ("run_graph_plan",),
}

CERTIFYING_TEST_FILES: tuple[str, ...] = (
    "tests/stages/test_nested_ordering.py",  # T1-05 certifies T1-04's executor
    "tests/stages/test_evaluate.py",  # T1-07
    "tests/composition/test_serialize.py",  # T1-08
    "tests/cli/test_graph_plan_verb.py",  # T1-11, AC1 graph->plan parity
)

_STUB_MARKERS = ("NotImplementedError", "TODO", "STUB", "FIXME")


def _is_stub_source(source: str) -> bool:
    """A body containing an explicit stub marker, or consisting of nothing but a docstring
    and/or a single `pass`/`...` statement, is considered a stub.
    """
    if any(marker in source for marker in _STUB_MARKERS):
        return True
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    if not tree.body:
        return False
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return False
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # skip a leading docstring
    if not body:
        return True
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            return True
    return False


def check_importable(module_names: Sequence[str]) -> list[str]:
    """Return 'module: ...' hits for any module that fails to import in a fresh subprocess."""
    hits: list[str] = []
    for module_name in module_names:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            hits.append(f"{module_name}: import failed (returncode={result.returncode})")
    return hits


def check_no_stub_markers(required_symbols: Mapping[str, Sequence[str]]) -> list[str]:
    """Return 'module.symbol: ...' hits for a missing symbol or a stub-shaped body."""
    hits: list[str] = []
    for module_name, symbols in required_symbols.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            hits.append(f"{module_name}: could not import to check symbols ({exc})")
            continue
        for symbol in symbols:
            if not hasattr(module, symbol):
                hits.append(f"{module_name}.{symbol}: symbol not found")
                continue
            obj = getattr(module, symbol)
            try:
                source = inspect.getsource(obj)
            except (OSError, TypeError):
                continue  # not introspectable (e.g. a C extension) -- not a stub-detection failure
            if _is_stub_source(source):
                hits.append(f"{module_name}.{symbol}: stub-shaped source (marker or empty body)")
    return hits


_UV_SYNC_EXTRAS: tuple[str, ...] = ("dev", "cli", "eda", "io")


def check_certifying_tests_green(test_files: Sequence[str], root: Path) -> list[str]:
    """Return 'path: ...' hits for any certifying test file that does not pass live, right now.

    Explicit `--extra` flags on every `uv run` call: a bare `uv run pytest` resyncs the shared
    venv to whatever extras THAT command requests, silently uninstalling packages (e.g.
    beartype, tyro) a prior `--extra dev`/`--extra cli` sync installed -- confirmed directly by
    mutation-testing this exact function, which surfaced a `ModuleNotFoundError: No module
    named 'beartype'` collection error on every certifying test file, not just the mutated one,
    once a bare `uv run pytest` call silently resynced the venv underneath it.
    """
    hits: list[str] = []
    extra_flags = [flag for extra in _UV_SYNC_EXTRAS for flag in ("--extra", extra)]
    for rel_path in test_files:
        full_path = root / rel_path
        result = subprocess.run(
            ["uv", "run", *extra_flags, "pytest", str(full_path), "-q", "--no-cov"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if result.returncode != 0:
            hits.append(
                f"{rel_path}: certifying test suite FAILED (returncode={result.returncode})"
            )
    return hits


def audit_wave1_load_bearing(
    root: Path = ROOT,
    *,
    import_targets: Sequence[str] = IMPORT_TARGETS,
    required_symbols: Mapping[str, Sequence[str]] = REQUIRED_SYMBOLS,
    certifying_test_files: Sequence[str] = CERTIFYING_TEST_FILES,
) -> tuple[bool, list[str]]:
    hits: list[str] = []
    hits += check_importable(import_targets)
    hits += check_no_stub_markers(required_symbols)
    hits += check_certifying_tests_green(certifying_test_files, root)
    return len(hits) == 0, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repo root to resolve certifying test file paths against (default: repo root)",
    )
    args = parser.parse_args(argv)

    passed, hits = audit_wave1_load_bearing(args.root)
    if passed:
        print(
            "PASS: wave1 load-bearing gate "
            "(T1-04/05/07/08/11 import clean, no stub markers, certifying tests green)"
        )
        return 0

    print("FAIL: wave1 load-bearing gate", file=sys.stderr)
    for hit in hits:
        print(f"  - {hit}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

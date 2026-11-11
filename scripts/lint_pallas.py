#!/usr/bin/env python3
"""JL-PALLAS static lint rules for Pallas kernels in xtrax (Debt #3197)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class PallasLinter(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.is_src = "src/xtrax" in filename.replace("\\", "/")
        self.findings: list[dict[str, Any]] = []
        # Map of function name -> FunctionDef AST node
        self.function_defs: dict[str, ast.FunctionDef] = {}
        # Track kernels checked for python control flow to avoid duplicate findings
        self.checked_kernel_bodies: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_defs[node.name] = node
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check BlockSpec calls (JL-PALLAS03)
        if func_name == "BlockSpec":
            self._check_block_spec(node)

        # Check pallas_call calls (JL-PALLAS01, JL-PALLAS02, JL-PALLAS04)
        if func_name == "pallas_call":
            self._check_pallas_call(node)

        self.generic_visit(node)

    def _check_block_spec(self, node: ast.Call) -> None:
        block_shape_node = None
        # Check positional arg 0 or keyword arg 'block_shape'
        for kw in node.keywords:
            if kw.arg == "block_shape":
                block_shape_node = kw.value
                break
        if block_shape_node is None and node.args:
            block_shape_node = node.args[0]

        if block_shape_node is None:
            return

        dims = []
        if isinstance(block_shape_node, (ast.Tuple, ast.List)):
            dims = block_shape_node.elts
        elif isinstance(block_shape_node, ast.Constant):
            dims = [block_shape_node]

        for dim in dims:
            if isinstance(dim, ast.Constant):
                val = dim.value
                if val is None:
                    continue  # None means full/unblocked dimension
                if isinstance(val, int) and not isinstance(val, bool):
                    if not _is_power_of_two(val):
                        self.findings.append(
                            {
                                "rule_id": "JL-PALLAS03",
                                "severity": "minor",
                                "line": dim.lineno,
                                "path": self.filename,
                                "message": (
                                    f"Block dimension {val} in BlockSpec is not a power of 2"
                                ),
                            }
                        )
            # Symbolic expressions (e.g. BLOCK_SIZE, TILE_K // 2) safely ignored (Finding M4)

    def _check_pallas_call(self, node: ast.Call) -> None:
        interpret_is_true = False
        has_compiler_params = False

        for kw in node.keywords:
            if kw.arg == "interpret":
                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    interpret_is_true = True
            elif kw.arg == "compiler_params":
                has_compiler_params = True

        # JL-PALLAS01: No interpret=True in src/xtrax
        if interpret_is_true and self.is_src:
            self.findings.append(
                {
                    "rule_id": "JL-PALLAS01",
                    "severity": "major",
                    "line": node.lineno,
                    "path": self.filename,
                    "message": ("pallas_call with interpret=True is prohibited in production code"),
                }
            )

        # JL-PALLAS02: Explicit compiler_params required when interpret is not True
        if not interpret_is_true and not has_compiler_params:
            self.findings.append(
                {
                    "rule_id": "JL-PALLAS02",
                    "severity": "minor",
                    "line": node.lineno,
                    "path": self.filename,
                    "message": (
                        "pallas_call must supply explicit compiler_params to guarantee lowering"
                    ),
                }
            )

        # JL-PALLAS04: Check kernel body function for Python control flow
        body_node = node.args[0] if node.args else None
        if body_node is None:
            for kw in node.keywords:
                if kw.arg in ("f", "kernel", "body"):
                    body_node = kw.value
                    break

        kernel_fn_name = None
        kernel_fn_def = None
        if isinstance(body_node, ast.Name):
            kernel_fn_name = body_node.id
            kernel_fn_def = self.function_defs.get(kernel_fn_name)
        elif isinstance(body_node, ast.FunctionDef):
            kernel_fn_def = body_node

        if kernel_fn_def is not None:
            key = f"{self.filename}:{kernel_fn_def.lineno}"
            if key not in self.checked_kernel_bodies:
                self.checked_kernel_bodies.add(key)
                self._check_control_flow(kernel_fn_def)

    def _check_control_flow(self, func_def: ast.FunctionDef) -> None:
        for inner in ast.walk(func_def):
            if isinstance(inner, ast.If):
                self.findings.append(
                    {
                        "rule_id": "JL-PALLAS04",
                        "severity": "major",
                        "line": inner.lineno,
                        "path": self.filename,
                        "message": (
                            "Python 'if' statement inside Pallas kernel body; use pl.when instead"
                        ),
                    }
                )
            elif isinstance(inner, (ast.For, ast.While)):
                loop_type = "for" if isinstance(inner, ast.For) else "while"
                self.findings.append(
                    {
                        "rule_id": "JL-PALLAS04",
                        "severity": "major",
                        "line": inner.lineno,
                        "path": self.filename,
                        "message": (
                            f"Python '{loop_type}' loop inside Pallas kernel body; use pl.loop"
                        ),
                    }
                )


def lint_pallas_source(source: str, path: Path | str) -> list[dict[str, Any]]:
    """Parse python source and return JL-PALLAS lint findings."""
    str_path = str(path)
    try:
        tree = ast.parse(source, filename=str_path)
    except SyntaxError:
        return []
    linter = PallasLinter(str_path)
    linter.visit(tree)
    return linter.findings


def lint_pallas_file(path: Path) -> list[dict[str, Any]]:
    """Lint a single python file for JL-PALLAS rules."""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return lint_pallas_source(source, path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Lint Pallas kernels for JL-PALLAS rules.")
    parser.add_argument(
        "targets", nargs="*", default=["src/xtrax"], help="Directories or files to lint"
    )
    args = parser.parse_args()

    findings: list[dict[str, Any]] = []
    for target_str in args.targets:
        target = Path(target_str)
        if target.is_file() and target.suffix == ".py":
            findings.extend(lint_pallas_file(target))
        elif target.is_dir():
            for py_file in target.rglob("*.py"):
                findings.extend(lint_pallas_file(py_file))

    for f in findings:
        sev = f["severity"].upper()
        print(f"{f['path']}:{f['line']}: [{sev}] {f['rule_id']}: {f['message']}")

    major_count = sum(1 for f in findings if f["severity"] == "major")
    if major_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

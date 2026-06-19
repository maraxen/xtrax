#!/usr/bin/env python3
"""Distribution N5 output-sink docs gate — chapter + re-export doctests (#1459)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "output_sink_docs.toml"


@dataclass(frozen=True)
class OutputSinkDocsConfig:
    version: str
    chapter_path: str
    index_path: str
    ci_workflow: str
    callbacks_module: str
    engine_io_module: str
    checkpoint_init: str
    min_chapter_bytes: int
    chapter_markers: tuple[str, ...]
    index_markers: tuple[str, ...]
    ci_markers: tuple[str, ...]
    doctest_modules: tuple[str, ...]
    doctest_markers: tuple[str, ...]
    reexport_source: str
    reexport_symbols: tuple[str, ...]


def load_output_sink_docs_config(config_path: Path) -> OutputSinkDocsConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("output_sink")
    if not isinstance(section, dict):
        raise ValueError(f"missing [output_sink] section in {config_path}")

    version = section.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("output_sink.version must be a non-empty string")

    def _req_str(key: str) -> str:
        value = section.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"output_sink.{key} must be a non-empty string")
        return value

    min_chapter_bytes = section.get("min_chapter_bytes")
    if not isinstance(min_chapter_bytes, int) or min_chapter_bytes <= 0:
        raise ValueError("output_sink.min_chapter_bytes must be a positive integer")

    def _nested_markers(parent: dict[str, object], table_name: str) -> tuple[str, ...]:
        table = parent.get(table_name, {})
        if not isinstance(table, dict):
            raise ValueError(f"output_sink.{table_name} must be a table")
        markers = table.get("markers")
        if not isinstance(markers, list) or not markers:
            raise ValueError(f"output_sink.{table_name}.markers must be a non-empty list")
        return tuple(str(item) for item in markers)

    doctest_section = section.get("doctest_modules", {})
    if not isinstance(doctest_section, dict):
        raise ValueError("output_sink.doctest_modules must be a table")
    doctest_paths = doctest_section.get("paths")
    if not isinstance(doctest_paths, list) or not doctest_paths:
        raise ValueError("output_sink.doctest_modules.paths must be a non-empty list")

    reexport_section = section.get("reexport", {})
    if not isinstance(reexport_section, dict):
        raise ValueError("output_sink.reexport must be a table")
    reexport_source = reexport_section.get("source")
    symbols = reexport_section.get("symbols")
    if not isinstance(reexport_source, str) or not reexport_source:
        raise ValueError("output_sink.reexport.source must be a non-empty string")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("output_sink.reexport.symbols must be a non-empty list")

    return OutputSinkDocsConfig(
        version=version,
        chapter_path=_req_str("chapter_path"),
        index_path=_req_str("index_path"),
        ci_workflow=_req_str("ci_workflow"),
        callbacks_module=_req_str("callbacks_module"),
        engine_io_module=_req_str("engine_io_module"),
        checkpoint_init=_req_str("checkpoint_init"),
        min_chapter_bytes=min_chapter_bytes,
        chapter_markers=_nested_markers(section, "chapter_markers"),
        index_markers=_nested_markers(section, "index_markers"),
        ci_markers=_nested_markers(section, "ci_markers"),
        doctest_modules=tuple(str(item) for item in doctest_paths),
        doctest_markers=_nested_markers(section, "doctest_markers"),
        reexport_source=reexport_source,
        reexport_symbols=tuple(str(item) for item in symbols),
    )


def _check_markers(
    failures: list[str],
    *,
    rel_path: str,
    text: str,
    markers: tuple[str, ...],
) -> None:
    for marker in markers:
        if marker not in text:
            failures.append(f"{rel_path} missing marker: {marker!r}")


def audit_output_sink_docs(
    root: Path,
    config_path: Path,
    *,
    skip_doctest: bool = False,
) -> tuple[bool, list[str]]:
    config = load_output_sink_docs_config(config_path)
    failures: list[str] = []

    chapter = root / config.chapter_path
    if not chapter.is_file():
        failures.append(f"missing output-sink chapter: {config.chapter_path}")
    else:
        chapter_text = chapter.read_text(encoding="utf-8")
        if chapter.stat().st_size < config.min_chapter_bytes:
            failures.append(
                f"{config.chapter_path} too small "
                f"({chapter.stat().st_size} bytes < {config.min_chapter_bytes})"
            )
        _check_markers(
            failures,
            rel_path=config.chapter_path,
            text=chapter_text,
            markers=config.chapter_markers,
        )

    index = root / config.index_path
    if not index.is_file():
        failures.append(f"missing docs index: {config.index_path}")
    else:
        _check_markers(
            failures,
            rel_path=config.index_path,
            text=index.read_text(encoding="utf-8"),
            markers=config.index_markers,
        )

    workflow = root / config.ci_workflow
    if not workflow.is_file():
        failures.append(f"missing CI workflow: {config.ci_workflow}")
    else:
        _check_markers(
            failures,
            rel_path=config.ci_workflow,
            text=workflow.read_text(encoding="utf-8"),
            markers=config.ci_markers,
        )

    callbacks = root / config.callbacks_module
    if not callbacks.is_file():
        failures.append(f"missing callbacks module: {config.callbacks_module}")
    else:
        callbacks_text = callbacks.read_text(encoding="utf-8")
        if config.reexport_source not in callbacks_text:
            failures.append(
                f"{config.callbacks_module} missing re-export: {config.reexport_source!r}"
            )
        for symbol in config.reexport_symbols:
            if symbol not in callbacks_text:
                failures.append(f"{config.callbacks_module} missing symbol reference: {symbol!r}")
        _check_markers(
            failures,
            rel_path=config.callbacks_module,
            text=callbacks_text,
            markers=config.doctest_markers,
        )
        if ">>>" not in callbacks_text:
            failures.append(f"{config.callbacks_module} missing doctest examples")

    engine_io = root / config.engine_io_module
    if not engine_io.is_file():
        failures.append(f"missing engine io module: {config.engine_io_module}")
    else:
        engine_text = engine_io.read_text(encoding="utf-8")
        _check_markers(
            failures,
            rel_path=config.engine_io_module,
            text=engine_text,
            markers=config.doctest_markers,
        )
        if ">>>" not in engine_text:
            failures.append(f"{config.engine_io_module} missing doctest examples")

    checkpoint_init = root / config.checkpoint_init
    if not checkpoint_init.is_file():
        failures.append(f"missing checkpoint init: {config.checkpoint_init}")
    else:
        checkpoint_text = checkpoint_init.read_text(encoding="utf-8")
        for symbol in (
            "get_checkpoint_manager",
            "save_checkpoint",
            "load_checkpoint",
        ):
            if symbol not in checkpoint_text:
                failures.append(f"{config.checkpoint_init} missing export: {symbol!r}")

    if not skip_doctest:
        doctest_paths = [str(root / rel) for rel in config.doctest_modules]
        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "--doctest-modules",
                *doctest_paths,
                "-q",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append("pytest --doctest-modules failed for output-sink modules")
            failures.append(result.stdout[-2000:] if result.stdout else "")
            failures.append(result.stderr[-2000:] if result.stderr else "")

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to output_sink_docs.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    parser.add_argument(
        "--skip-doctest",
        action="store_true",
        help="Skip running pytest --doctest-modules",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_output_sink_docs(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        skip_doctest=args.skip_doctest,
    )
    if passed:
        print("PASS: output-sink docs gate")
        return 0

    print("FAIL: output-sink docs gate", file=sys.stderr)
    for failure in failures:
        if failure:
            print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

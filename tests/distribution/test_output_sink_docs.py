"""Tests for distribution N5 output-sink docs gate (#1459)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from scripts.audit_output_sink_docs import (
    audit_output_sink_docs,
    load_output_sink_docs_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "output_sink_docs.toml"


def _write_config(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "output_sink_docs.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_output_sink_files(repo_root: Path) -> None:
    chapter = textwrap.dedent(
        """
        # Output Sinks

        Streaming Callbacks and Checkpoint Persistence.

        ```python
        from xtrax.io import BoundedCallbackHandler, async_indexed_stream
        from xtrax.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
            load_checkpoint,
        )
        ```

        xtrax.engine.io implementation.

        ```{autosummary}
        xtrax.io.BoundedCallbackHandler
        ```

        ```{automodule} xtrax.io
        ```

        ```{automodule} xtrax.checkpoint
        ```
        """
    ).strip() + "\n" * 80
    (repo_root / "docs" / "api").mkdir(parents=True)
    (repo_root / "docs" / "api" / "output-sinks.md").write_text(
        chapter, encoding="utf-8"
    )
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "index.md").write_text(
        "api/output-sinks\n", encoding="utf-8"
    )
    workflow_dir = repo_root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text(
        textwrap.dedent(
            """
            - name: Doctest canonical import paths
              run: uv run pytest --doctest-modules src/xtrax/io/ -q
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    callbacks_dir = repo_root / "src" / "xtrax" / "io"
    callbacks_dir.mkdir(parents=True)
    (callbacks_dir / "callbacks.py").write_text(
        textwrap.dedent(
            """
            \"\"\"Callbacks.

            >>> from xtrax.io import BoundedCallbackHandler
            >>> import asyncio
            >>> async def test():
            ...     handler = BoundedCallbackHandler(max_concurrent=1)
            ...     await handler.wait_all()
            ...     return True
            >>> asyncio.run(test())
            True
            \"\"\"
            from xtrax.engine.io import BoundedCallbackHandler, async_indexed_stream
            __all__ = ["BoundedCallbackHandler", "async_indexed_stream"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    engine_io_dir = repo_root / "src" / "xtrax" / "engine"
    engine_io_dir.mkdir(parents=True)
    (engine_io_dir / "io.py").write_text(
        textwrap.dedent(
            """
            import asyncio

            class BoundedCallbackHandler:
                def __init__(self, max_concurrent: int = 4):
                    self._semaphore = asyncio.Semaphore(max_concurrent)
                    self._pending_tasks = set()

                async def submit(self, coro):
                    async def bounded_coro():
                        async with self._semaphore:
                            await coro
                    task = asyncio.create_task(bounded_coro())
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)

                async def wait_all(self):
                    if self._pending_tasks:
                        await asyncio.gather(
                            *self._pending_tasks,
                            return_exceptions=True,
                        )

            async def async_indexed_stream(iterable, buffer_size: int = 2):
                for index, item in enumerate(iterable):
                    yield (index, item)

            class BoundedCallbackHandlerDoc:
                \"\"\"Example:

                >>> from xtrax.io import BoundedCallbackHandler
                >>> import asyncio
                >>> async def example():
                ...     handler = BoundedCallbackHandler(max_concurrent=1)
                ...     await handler.wait_all()
                ...     return True
                >>> asyncio.run(example())
                True
                \"\"\"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = repo_root / "src" / "xtrax" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "__init__.py").write_text(
        textwrap.dedent(
            """
            from xtrax.checkpoint.orbax import (
                get_checkpoint_manager,
                save_checkpoint,
                load_checkpoint,
            )
            __all__ = ["get_checkpoint_manager", "save_checkpoint", "load_checkpoint"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_output_sink_docs_config_reads_committed_toml() -> None:
    config = load_output_sink_docs_config(CONFIG_PATH)
    assert config.version == "0.1.0"
    assert config.chapter_path == "docs/api/output-sinks.md"
    assert "BoundedCallbackHandler" in config.reexport_symbols


def test_audit_output_sink_docs_passes_on_repo() -> None:
    passed, failures = audit_output_sink_docs(
        ROOT,
        CONFIG_PATH,
        skip_doctest=False,
    )
    assert passed is True, failures
    assert failures == []


def test_audit_output_sink_docs_fails_on_missing_chapter(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_output_sink_files(tmp_path)
    (tmp_path / "docs" / "api" / "output-sinks.md").unlink()
    passed, failures = audit_output_sink_docs(
        tmp_path,
        config_path,
        skip_doctest=True,
    )
    assert passed is False
    assert any("missing output-sink chapter" in item for item in failures)


def test_audit_output_sink_docs_fails_on_missing_ci_marker(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_output_sink_files(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.write_text("name: CI\n", encoding="utf-8")
    passed, failures = audit_output_sink_docs(
        tmp_path,
        config_path,
        skip_doctest=True,
    )
    assert passed is False
    assert any("ci.yml missing marker" in item for item in failures)


def test_main_cli_passes_on_repo() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_output_sink_docs.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout


def test_justfile_defines_audit_output_sink_docs() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-output-sink-docs:" in text

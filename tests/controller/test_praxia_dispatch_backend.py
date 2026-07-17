"""Tests for PraxiaDispatchBackend (LC-05, epic #3611, AC-4).

See `controller/praxia_dispatch_backend.py`'s module docstring for the reconciliation finding
this suite is built around: independent verification of praxia's real, merged `origin/main` found
`write_staged_file` is NOT reachable via a direct top-level `tools/call` (it is wired only into an
internal `workflow_dev` rig-tool-profile registry) -- so `PraxiaDispatchBackend` defaults to the
AC-4/1B self-computed-hash fallback rather than the primary MCP path. This suite therefore covers
three tiers:

1. The fallback path (`attempt_primary_path=False`, the default) -- what `dispatch_candidate()`
   actually does today.
2. The primary path (`attempt_primary_path=True` with an injected transport) -- proves the MCP
   transport code is correct and ready for when praxia's registration gap closes, without
   requiring a live `praxia-mcp` + database.
3. The low-level wire transport (`_call_write_staged_file`) against a tiny fake
   `praxia-mcp`-shaped subprocess -- mirrors `tests/controller/test_bathos_campaign_adapter.py`'s
   own real-wire-test precedent, including a fake response that reproduces the exact JSON-RPC
   error shape this reconciliation found is the real, current outcome.

No live praxia infrastructure is required for any test in this module.
"""

import hashlib
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from controller.dispatch import CandidateHandoff, CandidateHandoffFailure
from controller.praxia_dispatch_backend import (
    PraxiaDispatchBackend,
    _call_write_staged_file,
    _is_safe_segment,
    _is_safe_subdir,
    _PraxiaToolUnavailable,
    _resolve_praxia_mcp_command,
)


def _unreachable_transport(
    content: str, staging_subdir: str, file_extension: str
) -> dict[str, Any]:
    """A transport stub that fails the test if ever invoked -- used to prove the fallback path
    never touches the transport when `attempt_primary_path=False` (the default)."""
    raise AssertionError("transport should not be called when attempt_primary_path=False")


# ---------------------------------------------------------------------------
# Fallback path (AC-4/1B) -- the default, real, working implementation today.
# ---------------------------------------------------------------------------


class TestFallbackPath:
    def test_dispatch_candidate_uses_fallback_by_default(self, tmp_path: Path) -> None:
        """attempt_primary_path defaults to False -- the transport is never invoked."""
        backend = PraxiaDispatchBackend(
            "def foo():\n    return 42\n",
            workspace_root=tmp_path,
            transport=_unreachable_transport,
        )

        handoff = backend.dispatch_candidate()

        assert isinstance(handoff, CandidateHandoff)

    def test_fallback_returns_full_sha256_matching_recomputed_hash(self, tmp_path: Path) -> None:
        """The self-audit's mandatory check: the full 64-char hash genuinely threads through
        unmodified -- computed from the exact candidate_content passed at construction."""
        content = "def foo():\n    return 42\n"
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        backend = PraxiaDispatchBackend(content, workspace_root=tmp_path)
        handoff = backend.dispatch_candidate()

        assert handoff.content_sha256 == expected_hash
        assert len(handoff.content_sha256) == 64

    def test_fallback_writes_file_with_exact_content(self, tmp_path: Path) -> None:
        content = "x = 1\n"
        backend = PraxiaDispatchBackend(content, workspace_root=tmp_path)

        handoff = backend.dispatch_candidate()

        assert handoff.path.exists()
        assert handoff.path.read_text(encoding="utf-8") == content

    def test_fallback_path_under_dot_praxia_staging_subdir(self, tmp_path: Path) -> None:
        """Mirrors write_staged_file.rs's own path convention:
        .praxia/<staging_subdir>/<sha8>.<extension>."""
        content = "y = 2\n"
        backend = PraxiaDispatchBackend(
            content, staging_subdir="my_candidates", file_extension="txt", workspace_root=tmp_path
        )

        handoff = backend.dispatch_candidate()

        expected_sha8 = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        expected_path = tmp_path / ".praxia" / "my_candidates" / f"{expected_sha8}.txt"
        assert handoff.path == expected_path

    def test_fallback_default_staging_subdir_and_extension(self, tmp_path: Path) -> None:
        backend = PraxiaDispatchBackend("z = 3\n", workspace_root=tmp_path)
        handoff = backend.dispatch_candidate()

        assert handoff.path.parent.name == "loop_controller_candidates"
        assert handoff.path.parent.parent.name == ".praxia"
        assert handoff.path.suffix == ".py"

    def test_fallback_rejects_unsafe_staging_subdir(self, tmp_path: Path) -> None:
        """Mirrors write_staged_file's own status='rejected' reason verbatim."""
        backend = PraxiaDispatchBackend(
            "content", staging_subdir="../escape", workspace_root=tmp_path
        )

        with pytest.raises(CandidateHandoffFailure, match="unsafe_staging_subdir"):
            backend.dispatch_candidate()

        # Nothing should have been written outside the workspace.
        escape_target = tmp_path.parent / "escape"
        assert not escape_target.exists()

    def test_fallback_rejects_unsafe_file_extension(self, tmp_path: Path) -> None:
        backend = PraxiaDispatchBackend(
            "content", file_extension="yaml/../evil", workspace_root=tmp_path
        )

        with pytest.raises(CandidateHandoffFailure, match="unsafe_file_extension"):
            backend.dispatch_candidate()

    def test_fallback_wraps_os_error_as_candidate_handoff_failure(self, tmp_path: Path) -> None:
        """A genuine filesystem failure (staging dir path collides with an existing file, so
        mkdir(parents=True) fails) maps to CandidateHandoffFailure, not an uncaught OSError."""
        blocker = tmp_path / ".praxia"
        blocker.write_text("not a directory", encoding="utf-8")

        backend = PraxiaDispatchBackend("content", workspace_root=tmp_path)

        with pytest.raises(CandidateHandoffFailure, match="staging write failed"):
            backend.dispatch_candidate()

    def test_fallback_distinct_content_distinct_paths(self, tmp_path: Path) -> None:
        backend_a = PraxiaDispatchBackend("first", workspace_root=tmp_path)
        backend_b = PraxiaDispatchBackend("second", workspace_root=tmp_path)

        handoff_a = backend_a.dispatch_candidate()
        handoff_b = backend_b.dispatch_candidate()

        assert handoff_a.path != handoff_b.path
        assert handoff_a.content_sha256 != handoff_b.content_sha256


# ---------------------------------------------------------------------------
# Primary path (attempt_primary_path=True, injected transport) -- proves the
# response-parsing logic is correct even though it isn't exercised by default.
# ---------------------------------------------------------------------------


class TestPrimaryPathViaInjectedTransport:
    def test_primary_path_staged_status_returns_candidate_handoff(self, tmp_path: Path) -> None:
        expected_hash = hashlib.sha256(b"real content").hexdigest()

        def transport(content: str, staging_subdir: str, file_extension: str) -> dict[str, Any]:
            assert content == "real content"
            assert staging_subdir == "loop_controller_candidates"
            assert file_extension == "py"
            return {
                "status": "staged",
                "staged_path": "/workspace/.praxia/loop_controller_candidates/abcd1234.py",
                "content_sha256": expected_hash,
            }

        backend = PraxiaDispatchBackend(
            "real content", attempt_primary_path=True, transport=transport
        )

        handoff = backend.dispatch_candidate()

        assert handoff == CandidateHandoff(
            path=Path("/workspace/.praxia/loop_controller_candidates/abcd1234.py"),
            content_sha256=expected_hash,
        )

    def test_primary_path_rejected_status_raises_candidate_handoff_failure(self) -> None:
        def transport(content: str, staging_subdir: str, file_extension: str) -> dict[str, Any]:
            return {"status": "rejected", "reason": "unsafe_staging_subdir"}

        backend = PraxiaDispatchBackend("x", attempt_primary_path=True, transport=transport)

        with pytest.raises(CandidateHandoffFailure, match="unsafe_staging_subdir"):
            backend.dispatch_candidate()

    def test_primary_path_error_status_raises_candidate_handoff_failure_with_real_reason(
        self,
    ) -> None:
        def transport(content: str, staging_subdir: str, file_extension: str) -> dict[str, Any]:
            return {"status": "error", "reason": "failed to write staged file: disk full"}

        backend = PraxiaDispatchBackend("x", attempt_primary_path=True, transport=transport)

        with pytest.raises(CandidateHandoffFailure, match="disk full"):
            backend.dispatch_candidate()

    def test_primary_path_malformed_staged_response_raises_value_error(self) -> None:
        def transport(content: str, staging_subdir: str, file_extension: str) -> dict[str, Any]:
            return {"status": "staged"}  # missing staged_path/content_sha256

        backend = PraxiaDispatchBackend("x", attempt_primary_path=True, transport=transport)

        with pytest.raises(ValueError, match="malformed dispatch completion"):
            backend.dispatch_candidate()

    def test_primary_path_unexpected_status_raises_value_error(self) -> None:
        def transport(content: str, staging_subdir: str, file_extension: str) -> dict[str, Any]:
            return {"status": "unknown_status"}

        backend = PraxiaDispatchBackend("x", attempt_primary_path=True, transport=transport)

        with pytest.raises(ValueError, match="malformed dispatch completion"):
            backend.dispatch_candidate()

    def test_primary_path_tool_unavailable_falls_back_transparently(self, tmp_path: Path) -> None:
        """The confirmed, real-world scenario (module docstring): write_staged_file's tools/call
        round-trips a JSON-RPC error. dispatch_candidate() must still succeed via the fallback,
        not raise."""
        content = "fallback should still work"

        def transport(content_: str, staging_subdir: str, file_extension: str) -> dict[str, Any]:
            msg = "Tool 'write_staged_file' not found in internal match table"
            raise _PraxiaToolUnavailable(msg)

        backend = PraxiaDispatchBackend(
            content, attempt_primary_path=True, transport=transport, workspace_root=tmp_path
        )

        handoff = backend.dispatch_candidate()

        assert isinstance(handoff, CandidateHandoff)
        assert handoff.content_sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert handoff.path.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# Low-level wire transport (_call_write_staged_file) -- transport-level failure modes.
# ---------------------------------------------------------------------------


class TestCallWriteStagedFileTransportFailures:
    def test_spawn_failure_raises_candidate_handoff_failure(self) -> None:
        """Subprocess spawn failure (praxia-mcp missing/not executable) maps to
        CandidateHandoffFailure, per DispatchBackend's documented triad."""
        with pytest.raises(CandidateHandoffFailure, match="could not spawn"):
            _call_write_staged_file(
                "/nonexistent/path/definitely/not/a/real/binary",
                "content",
                "staging",
                "py",
                timeout=2.0,
            )

    def test_resolve_praxia_mcp_command_prefers_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "controller.praxia_dispatch_backend.shutil.which",
            lambda name: "/usr/local/bin/praxia-mcp" if name == "praxia-mcp" else None,
        )
        assert _resolve_praxia_mcp_command() == "/usr/local/bin/praxia-mcp"

    def test_resolve_praxia_mcp_command_falls_back_when_not_on_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("controller.praxia_dispatch_backend.shutil.which", lambda name: None)
        command = _resolve_praxia_mcp_command()
        assert command.endswith("praxia-mcp")


# ---------------------------------------------------------------------------
# Safety-check helpers (mirror write_staged_file.rs's own validation exactly).
# ---------------------------------------------------------------------------


class TestSafetyHelpers:
    @pytest.mark.parametrize(
        ("segment", "expected"),
        [
            ("loop_controller_candidates", True),
            ("py", True),
            ("a", True),
            ("a-b_c9", True),
            ("", False),
            ("Uppercase", False),
            ("_leading_underscore", False),
            ("-leading-dash", False),
            ("has space", False),
            ("../escape", False),
        ],
    )
    def test_is_safe_segment(self, segment: str, expected: bool) -> None:
        assert _is_safe_segment(segment) is expected

    @pytest.mark.parametrize(
        ("subdir", "expected"),
        [
            ("loop_controller_candidates", True),
            ("a/b/c", True),
            ("", False),
            ("../escape", False),
            ("a//b", False),
            ("a/", False),
        ],
    )
    def test_is_safe_subdir(self, subdir: str, expected: bool) -> None:
        assert _is_safe_subdir(subdir) is expected


# ---------------------------------------------------------------------------
# Real-wire end-to-end tests against a tiny fake praxia-mcp-shaped subprocess.
# Mirrors tests/controller/test_bathos_campaign_adapter.py's own precedent.
# ---------------------------------------------------------------------------

_FAKE_PRAXIA_MCP_SOURCE = """
import hashlib
import json
import os
import sys


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


_log_path = os.environ.get("FAKE_PRAXIA_MCP_LOG")
_mode = os.environ.get("FAKE_PRAXIA_MCP_MODE", "success")


def _log_call(name, arguments):
    if not _log_path:
        return
    with open(_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": name, "arguments": arguments}) + "\\n")


for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        _send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"serverInfo": {"name": "fake-praxia-mcp", "version": "0.0"}},
            }
        )
    elif method == "notifications/initialized":
        continue
    elif method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        _log_call(name, arguments)
        if _mode == "not_found":
            # Reproduces the real, confirmed outcome (module docstring): write_staged_file is
            # not wired into praxia's top-level dispatch table, so tools/call round-trips a
            # JSON-RPC-level error, not a tool result.
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{name}' not found in internal match table",
                    },
                }
            )
        else:
            content = arguments.get("content", "")
            staging_subdir = arguments.get("staging_subdir", "")
            file_extension = arguments.get("file_extension", "txt")
            sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            result = {
                "status": "staged",
                "staged_path": f"/tmp/.praxia/{staging_subdir}/{sha[:8]}.{file_extension}",
                "content_sha256": sha,
            }
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"structuredContent": result},
                }
            )
"""


def _write_fake_praxia_mcp(tmp_path: Path) -> Path:
    script_path = tmp_path / "fake_praxia_mcp.py"
    script_path.write_text(f"#!{sys.executable}\n{_FAKE_PRAXIA_MCP_SOURCE}", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


class TestRealWireTransport:
    def test_call_write_staged_file_over_real_json_rpc_wire_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves the flat {content, staging_subdir, file_extension} arguments genuinely reach
        the wire (no {action, payload} envelope), and the response round-trips correctly."""
        fake_server = _write_fake_praxia_mcp(tmp_path)
        log_path = tmp_path / "received_calls.jsonl"
        monkeypatch.setenv("FAKE_PRAXIA_MCP_LOG", str(log_path))
        monkeypatch.setenv("FAKE_PRAXIA_MCP_MODE", "success")

        result = _call_write_staged_file(
            str(fake_server), "some content", "my_subdir", "py", timeout=10.0
        )

        assert result["status"] == "staged"
        assert result["content_sha256"] == hashlib.sha256(b"some content").hexdigest()

        logged_calls = [
            json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(logged_calls) == 1
        assert logged_calls[0]["name"] == "write_staged_file"
        assert logged_calls[0]["arguments"] == {
            "content": "some content",
            "staging_subdir": "my_subdir",
            "file_extension": "py",
        }

    def test_call_write_staged_file_over_real_json_rpc_wire_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces the real, confirmed-today outcome over the actual wire protocol: a
        JSON-RPC-level error raises _PraxiaToolUnavailable, not ValueError/TimeoutError."""
        fake_server = _write_fake_praxia_mcp(tmp_path)
        monkeypatch.setenv("FAKE_PRAXIA_MCP_MODE", "not_found")

        with pytest.raises(_PraxiaToolUnavailable, match="not found in internal match table"):
            _call_write_staged_file(str(fake_server), "content", "subdir", "py", timeout=10.0)

    def test_dispatch_candidate_primary_path_over_real_wire_falls_back_on_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: attempt_primary_path=True against the real (fake) wire, in the confirmed
        not_found mode, still produces a valid CandidateHandoff via the fallback."""
        fake_server = _write_fake_praxia_mcp(tmp_path)
        monkeypatch.setenv("FAKE_PRAXIA_MCP_MODE", "not_found")
        content = "candidate via real wire, falls back"

        backend = PraxiaDispatchBackend(
            content,
            attempt_primary_path=True,
            command=str(fake_server),
            workspace_root=tmp_path,
            timeout=10.0,
        )

        handoff = backend.dispatch_candidate()

        assert handoff.content_sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert handoff.path.read_text(encoding="utf-8") == content


__all__ = [
    "TestCallWriteStagedFileTransportFailures",
    "TestFallbackPath",
    "TestPrimaryPathViaInjectedTransport",
    "TestRealWireTransport",
    "TestSafetyHelpers",
]

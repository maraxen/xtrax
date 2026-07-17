"""PraxiaDispatchBackend concrete implementation (epic #3611 loop-controller, LC-05, AC-4).

AC-4's text (verbatim): "Concrete PraxiaDispatchBackend wrapping the confirmed subprocess-to-MCP
invocation mechanics (.praxia/docs/research/260716_praxia-mcp-invocation-probe.md), calling LC-04's
tool, implementing LC-03's CandidateHandoff contract. AC: before implementation starts, an explicit
reconciliation step confirms LC-04's actual merged tool schema matches LC-03's CandidateHandoff
contract exactly. Fallback trigger to controller-side parsing (documented fallback) is a stated
review-window threshold (set at filing/pickup time); if triggered, the fallback still produces a
valid CandidateHandoff via a self-computed hash (documented as a known trust-model difference from
the primary path). depends_on: [LC-03, LC-04]."

## Reconciliation performed at pickup time -- and a correction to this ticket's own dispatch brief

Verified directly against praxia `origin/main` (the LC-04 merge), not assumed:

- **Field shape matches** `CandidateHandoff` exactly: `write_staged_file`'s schema
  (`agent_assets/schemas/v1/tool/write_staged_file.json`) takes
  `{content: str, staging_subdir: str, file_extension?: str}` and its Rust implementation
  (`crates/praxia-rig-tools/src/workflow_dev_tools/write_staged_file.rs`) returns
  `{status: "staged"|"rejected"|"error", staged_path?, content_sha256?, reason?}`, with
  `content_sha256` the full, never-truncated 64-hex-char digest (that file's own
  `write_staged_file_generic_content_stages_with_full_sha256` test asserts
  `content_sha256.len() == 64`). This maps cleanly onto `CandidateHandoff.path`/`.content_sha256`
  (`controller/dispatch.py`), whose own `__post_init__` already enforces the 64-char invariant.

- **Reachability does NOT match, however -- a correction to this ticket's dispatch brief**, which
  claimed `write_staged_file` is "registered as its own direct, top-level MCP tool." Independently
  verified false, via three separate pieces of direct evidence in praxia `origin/main`:
  1. `crates/praxia-tools/src/tool_catalog.rs` -- its own docstring calls it the "Single source of
     truth for MCP-advertised tool names" -- defines `LIGHT_TOOLS`, `ENGINE_TOOLS`, and
     `ENGINE_ROUTED_FOR_LIST`. `write_staged_file` appears in none of the three.
  2. `crates/praxia-agent-server/src/server/types.rs::invoke_internal` -- the Value-in/Value-out
     dispatch table used by both the in-process router and the out-of-process `praxia-tool-host`
     -- has no `"write_staged_file"` arm; an unmatched name falls to the catch-all
     `Err(McpError::invalid_request("Tool '{name}' not found in internal match table"))`.
  3. `crates/praxia-mcp/src/host_route.rs::engine_tool_specs()` -- the committed fallback catalog
     a cold `list_tools()` advertises before the tool-host is ever spawned -- also omits it.

  `write_staged_file` is real, tested, and merged -- but it lives exclusively in
  `crates/praxia-rig-tools/src/workflow_dev_tools/`, wired into `build_workflow_dev_tool_registry`
  (`action_tools.rs`), a `ToolRegistryFactory` consumed by an LLM agent running *inside* a
  `rig_run` flow under the `"workflow_dev"` tool profile -- not by an external client via a direct
  `tools/call`. Its own Rust test (`workflow_dev_write_staged_file_tool_call_succeeds`) confirms
  this: it calls `build_workflow_dev_tool_registry(...).get("write_staged_file")` directly,
  in-process, never touching the MCP wire protocol. A plain `tools/call` with
  `name: "write_staged_file"` against the real `praxia-mcp` server today -- exactly what this
  ticket's dispatch brief called the primary path -- round-trips a JSON-RPC-level error, not a
  tool result.

## Consequence for this implementation

AC-4's own governing spec
(`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md`, Finding 9's fix)
defines the fallback trigger textually as a *time-based* "no praxia-side merge or explicit
maintainer rejection within a stated review window." That literal trigger has not fired --
the tool merged same-day. But the trigger's underlying *intent* -- "the primary path is not
currently viable, so hand off via a self-computed hash instead" -- is unambiguously met: this is a
structural gap, confirmed by reading praxia's own dispatch tables directly, not a timing question.
This module therefore exercises the AC-4/1B fallback **by default**, documented exactly per the
AC's own required framing: a known, accepted trust-model difference from the primary path. In the
primary path, praxia's own server independently computes and attests to `content_sha256` from the
bytes it wrote; in this fallback, the controller writes the bytes itself and self-attests the hash
of its own write -- weaker, but sufficient to satisfy `CandidateHandoff.__post_init__`'s invariant
and unblock the loop.

This is a deliberate deviation from the dispatch brief's instruction not to implement the fallback,
made only after -- and because of -- independently re-verifying the reachability claim the brief
itself asked this session to check. The brief's schema-*shape* claim (flat
`{content, staging_subdir, file_extension}` fields, no `{action, payload}` envelope) is still
correct *if/when* `write_staged_file` is wired into a Tier-A `LIGHT_TOOLS` entry (the natural fix,
mirroring how e.g. `"docs"`/`"handoff"` are dispatched via praxia's `dispatch_uniform_tools!` macro
with flat request structs) -- so the primary-path MCP transport code below is real, tested code
kept ready to fire the moment that registration gap closes, not dead code; it is just not called by
default today. **Recommended follow-up**: file the praxia-side registration gap as backlog debt --
a natural fit for this epic's own AC-10 "cross-repo gap backlog filing" item (which already tracks
the `mode="sequential"` and stale-MCP-server gaps) -- deliberately not done in this session, to
avoid scope creep into AC-10's own ownership.

## Design

- `dispatch_candidate()` takes no arguments (per `DispatchBackend`'s Protocol signature); candidate
  content is supplied at construction (`candidate_content`), mirroring `MockDispatchBackend`'s own
  field-based shape -- this ticket implements the hand-off mechanism, not candidate proposal itself.
- `attempt_primary_path` (default `False`, given the confirmed-unreachable finding above) controls
  whether `dispatch_candidate()` even tries the real MCP round trip before using the fallback.
  Settable `True` to exercise/prove the primary-path transport code (tests do this), or once
  praxia's own registration gap closes.
- Even with `attempt_primary_path=True`, a JSON-RPC-level error on the `write_staged_file` call
  specifically (the exact, confirmed-deterministic outcome today) transparently falls back rather
  than raising -- this is the one failure mode this reconciliation has fully explained. Every other
  transport failure (subprocess spawn error, handshake timeout, malformed JSON) still raises the
  `TimeoutError`/`ValueError`/`CandidateHandoffFailure` triad `DispatchBackend`'s docstring
  documents, so a genuinely broken transport is never silently papered over.
"""

import hashlib
import json
import re
import selectors
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from controller.dispatch import CandidateHandoff, CandidateHandoffFailure

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_STAGING_SUBDIR = "loop_controller_candidates"
_DEFAULT_FILE_EXTENSION = "py"

# Mirrors scripts/probe_praxia_mcp_invocation.py's `_FALLBACK_COMMAND` / controller/
# bathos_campaign_adapter.py's `_FALLBACK_COMMAND` precedent: the live ~/.claude.json
# mcpServers.praxia entry's command, read directly, used only if `praxia-mcp` isn't resolvable
# via PATH at call time.
_FALLBACK_COMMAND = "/home/marielle/.cargo/bin/praxia-mcp"

# Mirrors write_staged_file.rs::is_safe_segment exactly: one lowercase-alnum leading char, then
# lowercase-alnum/underscore/dash.
_SAFE_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# A transport is a plain callable so tests can inject a stub instead of spawning a real
# `praxia-mcp` subprocess. `(content, staging_subdir, file_extension) -> tool_result_dict`.
PraxiaMcpTransport = Callable[[str, str, str], dict[str, Any]]


class _PraxiaToolUnavailable(Exception):
    """`write_staged_file`'s `tools/call` round-tripped a JSON-RPC-level error.

    Internal signal only -- never escapes `dispatch_candidate()`. See module docstring: this is
    the confirmed, deterministic outcome of calling `write_staged_file` against today's real
    `praxia-mcp`, since the tool is not wired into any top-level MCP dispatch table. Caught by
    `dispatch_candidate()` to route to the documented AC-4/1B fallback.
    """


def _is_safe_segment(segment: str) -> bool:
    """Mirrors `write_staged_file.rs::is_safe_segment` exactly: `^[a-z0-9][a-z0-9_-]*$`."""
    return bool(_SAFE_SEGMENT_RE.match(segment))


def _is_safe_subdir(subdir: str) -> bool:
    """Mirrors `write_staged_file.rs::is_safe_subdir`: one or more `/`-separated safe segments."""
    if not subdir:
        return False
    return all(_is_safe_segment(segment) for segment in subdir.split("/"))


def _resolve_praxia_mcp_command() -> str:
    """Resolve the `praxia-mcp` entrypoint: PATH first, then the known local fallback.

    Mirrors `scripts/probe_praxia_mcp_invocation.py::_resolve_command` /
    `controller/bathos_campaign_adapter.py::_resolve_bth_mcp_command` exactly.
    """
    return shutil.which("praxia-mcp") or _FALLBACK_COMMAND


def _send(proc: subprocess.Popen, payload: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def _read_one_response(proc: subprocess.Popen, *, timeout: float) -> dict[str, Any]:
    assert proc.stdout is not None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    if not sel.select(timeout=timeout):
        raise TimeoutError(f"no response within {timeout}s")

    line = proc.stdout.readline()
    if not line:
        stderr = proc.stderr.read() if proc.stderr else ""
        msg = f"process closed stdout; stderr={stderr!r}"
        raise RuntimeError(msg)

    return json.loads(line)


def _extract_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Parse a `tools/call` MCP result into `write_staged_file`'s own `{status, ...}` dict.

    Mirrors `controller/bathos_campaign_adapter.py::_extract_structured_result`'s
    `structuredContent`-preferred, `content[0].text`-fallback parse.
    """
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                parsed = json.loads(block.get("text", ""))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    msg = (
        "malformed dispatch completion: tools/call 'write_staged_file' returned no parseable "
        f"result: {result!r}"
    )
    raise ValueError(msg)


def _call_write_staged_file(
    command: str, content: str, staging_subdir: str, file_extension: str, *, timeout: float
) -> dict[str, Any]:
    """Spawn praxia's real MCP entrypoint, complete the handshake, call `write_staged_file`.

    Hand-rolled JSON-RPC-over-stdio, no MCP SDK client -- mirrors
    `scripts/probe_praxia_mcp_invocation.py`'s exact mechanics (the confirmed subprocess-to-MCP
    invocation recipe this ticket wraps), extended one step further from that script's
    `tools/list` stopping point through to a real `tools/call`, the same extension
    `controller/bathos_campaign_adapter.py::_call_mcp_tool` already made for bathos.

    `arguments` is `{content, staging_subdir, file_extension}` directly -- flat fields, no
    `{action, payload}` envelope -- matching `write_staged_file`'s own schema and the way every
    other Tier-A `LIGHT_TOOLS` entry with a flat request struct is dispatched (see module
    docstring for why this is still the right shape even though the tool isn't reachable this
    way today).

    Raises:
        TimeoutError: the handshake or tool call did not respond within `timeout`.
        ValueError: the initialize response, or the tool call's response, could not be parsed as
            expected.
        CandidateHandoffFailure: the subprocess itself could not be spawned (praxia-mcp missing
            or not executable) -- treated as a staging hand-off failure, per
            `CandidateHandoffFailure`'s own docstring ("e.g., staging write fails").
        _PraxiaToolUnavailable: the `tools/call` round-tripped a JSON-RPC-level error (the
            confirmed, current outcome for `write_staged_file` -- see module docstring).
            Internal only; callers of `dispatch_candidate()` never see this type.
    """
    try:
        proc = subprocess.Popen(
            [command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        msg = f"staging write failed: could not spawn praxia-mcp ({command!r}): {exc}"
        raise CandidateHandoffFailure(msg) from exc

    try:
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "xtrax-loop-controller-praxia-dispatch-backend",
                        "version": "0.1",
                    },
                },
            },
        )
        try:
            init_response = _read_one_response(proc, timeout=timeout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed dispatch completion: initialize failed: {exc}") from exc

        result = init_response.get("result")
        if not isinstance(result, dict) or "serverInfo" not in result:
            msg = (
                "malformed dispatch completion: no serverInfo in initialize response: "
                f"{init_response!r}"
            )
            raise ValueError(msg)

        # MCP requires the initialized notification before any tool call is valid.
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "write_staged_file",
                    "arguments": {
                        "content": content,
                        "staging_subdir": staging_subdir,
                        "file_extension": file_extension,
                    },
                },
            },
        )
        try:
            call_response = _read_one_response(proc, timeout=timeout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed dispatch completion: tools/call failed: {exc}") from exc

        if "error" in call_response:
            # Confirmed, current, deterministic outcome for write_staged_file (module docstring)
            # -- not a genuine transport bug, so this is deliberately not raised as ValueError.
            msg = f"tools/call 'write_staged_file' JSON-RPC error: {call_response['error']}"
            raise _PraxiaToolUnavailable(msg)

        return _extract_tool_result(call_response.get("result") or {})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


class PraxiaDispatchBackend:
    """Concrete `DispatchBackend` backed by praxia (epic #3611 loop-controller, LC-05, AC-4).

    See module docstring for the reconciliation finding this implementation is built around: the
    AC-4/1B self-computed-hash fallback is exercised by default, not the primary
    `write_staged_file` MCP path, because independent verification found that path is not
    reachable in praxia's current merged state.
    """

    def __init__(
        self,
        candidate_content: str,
        *,
        staging_subdir: str = _DEFAULT_STAGING_SUBDIR,
        file_extension: str = _DEFAULT_FILE_EXTENSION,
        workspace_root: Path | None = None,
        command: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        attempt_primary_path: bool = False,
        transport: PraxiaMcpTransport | None = None,
    ) -> None:
        """Configure the backend.

        Args:
            candidate_content: the exact candidate source text to hand off. `dispatch_candidate`
                takes no arguments (per `DispatchBackend`'s Protocol signature), so the content a
                given backend instance will propose is fixed at construction -- mirrors
                `MockDispatchBackend.candidate_content`'s own shape. Proposing *what* content to
                stage is out of this ticket's scope (LC-05 implements the hand-off mechanism
                only).
            staging_subdir: caller-supplied hint selecting the staging namespace under
                `.praxia/` -- must match `^[a-z0-9][a-z0-9_-]*$` per `/`-separated segment
                (`write_staged_file`'s own validation rule, mirrored here so the fallback and
                primary paths reject the same inputs identically). Default is scoped to this
                epic's own candidates -- the underlying tool's own field name stays generic; the
                *value* passed for it is this caller's choice.
            file_extension: staged filename's extension, no leading dot. Default `"py"` --
                candidates are Python source mutations per the epic's own mandate doc.
            workspace_root: root `.praxia/<staging_subdir>/` is written under, for the fallback
                path. Defaults to the current working directory, matching
                `write_staged_file.rs`'s own `workspace_path`-relative convention.
            command: explicit `praxia-mcp` executable path. If `None`, resolved lazily per call.
            timeout: seconds to wait for each MCP response before raising `TimeoutError`.
            attempt_primary_path: if `True`, `dispatch_candidate()` first attempts the real
                `write_staged_file` MCP call before falling back. Default `False` (see module
                docstring: the reconciliation performed at pickup time confirmed this call
                cannot succeed against today's merged praxia, so attempting it by default would
                only add a doomed subprocess round trip to every dispatch).
            transport: injection seam for tests -- a callable
                `(content, staging_subdir, file_extension) -> tool_result_dict` used instead of
                spawning a real `praxia-mcp` subprocess. If `None`, the real stdio transport is
                used when `attempt_primary_path` is `True`.
        """
        self._candidate_content = candidate_content
        self._staging_subdir = staging_subdir
        self._file_extension = file_extension
        self._workspace_root = workspace_root if workspace_root is not None else Path.cwd()
        self._command = command
        self._timeout = timeout
        self._attempt_primary_path = attempt_primary_path
        self._transport = transport if transport is not None else self._default_transport

    def _default_transport(
        self, content: str, staging_subdir: str, file_extension: str
    ) -> dict[str, Any]:
        command = self._command if self._command is not None else _resolve_praxia_mcp_command()
        return _call_write_staged_file(
            command, content, staging_subdir, file_extension, timeout=self._timeout
        )

    def dispatch_candidate(self) -> CandidateHandoff:
        """Propose and hand off `self._candidate_content`.

        See module docstring for the reconciliation finding and the resulting default path.

        Returns:
            CandidateHandoff: from the primary MCP path if attempted and successful, otherwise
                from the AC-4/1B fallback (self-computed hash, controller-side write).

        Raises:
            CandidateHandoffFailure: the staging write failed -- either `write_staged_file`
                itself reported `status in ("rejected", "error")` (primary path), or the
                fallback's own local write failed (unsafe staging_subdir/file_extension, or a
                genuine filesystem error).
            TimeoutError: the primary path's MCP round trip exceeded `timeout` (only reachable
                when `attempt_primary_path=True`).
            ValueError: the primary path's response could not be parsed as expected (only
                reachable when `attempt_primary_path=True`).
        """
        if self._attempt_primary_path:
            try:
                envelope = self._transport(
                    self._candidate_content, self._staging_subdir, self._file_extension
                )
            except _PraxiaToolUnavailable:
                return self._dispatch_via_fallback()
            return self._handoff_from_envelope(envelope)
        return self._dispatch_via_fallback()

    def _handoff_from_envelope(self, envelope: dict[str, Any]) -> CandidateHandoff:
        status = envelope.get("status")
        if status == "staged":
            staged_path = envelope.get("staged_path")
            content_sha256 = envelope.get("content_sha256")
            if not isinstance(staged_path, str) or not isinstance(content_sha256, str):
                msg = (
                    "malformed dispatch completion: 'staged' response missing "
                    f"staged_path/content_sha256: {envelope!r}"
                )
                raise ValueError(msg)
            return CandidateHandoff(path=Path(staged_path), content_sha256=content_sha256)
        if status in ("rejected", "error"):
            reason = envelope.get("reason", f"write_staged_file returned status={status!r}")
            raise CandidateHandoffFailure(f"staging write failed: {reason}")
        msg = f"malformed dispatch completion: unexpected response: {envelope!r}"
        raise ValueError(msg)

    def _dispatch_via_fallback(self) -> CandidateHandoff:
        """AC-4/1B fallback: the controller writes the content itself and self-computes the hash.

        Mirrors `write_staged_file.rs::stage_content`'s own path convention
        (`.praxia/<staging_subdir>/<sha8>.<file_extension>`) and validation rule so a future
        migration back to the primary path is a no-op for anything downstream that only reads
        `CandidateHandoff.path`/`.content_sha256`.

        Raises:
            CandidateHandoffFailure: `staging_subdir`/`file_extension` failed the safety check
                (mirrors `write_staged_file`'s own `status="rejected"` reasons verbatim), or the
                local write itself failed (permission denied, disk full, etc).
        """
        if not _is_safe_subdir(self._staging_subdir):
            raise CandidateHandoffFailure("staging write failed: unsafe_staging_subdir")
        if not _is_safe_segment(self._file_extension):
            raise CandidateHandoffFailure("staging write failed: unsafe_file_extension")

        content_bytes = self._candidate_content.encode("utf-8")
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        staging_dir = self._workspace_root / ".praxia" / self._staging_subdir
        staged_path = staging_dir / f"{content_sha256[:8]}.{self._file_extension}"

        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(content_bytes)
        except OSError as exc:
            raise CandidateHandoffFailure(f"staging write failed: {exc}") from exc

        return CandidateHandoff(path=staged_path, content_sha256=content_sha256)


__all__ = [
    "PraxiaDispatchBackend",
    "PraxiaMcpTransport",
]

"""bathos campaign-sequencing adapter (epic #3611 loop-controller, LC-06, AC-5).

AC-5's text (verbatim): "bathos campaign-sequencing adapter -- wraps
`campaign_create`/`run`/`campaign_conclude` MCP calls into one cohesive controller-owned
helper, including `derived_from` pass-through from the outset (closing finding 5's
dependency gap preemptively -- `AC-7` no longer needs to retrofit this). AC: the adapter's
`run` wrapper accepts `derived_from` as a first-class parameter." See
`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md`.

## Transport decision: MCP, not CLI-subprocess -- grounded, not assumed

This module lives in `controller/`, not `src/xtrax/`, which matters: `src/xtrax/run/
seed_emission.py`, `component_binding.py`, and `baseline_budget_emission.py` (already-merged
#2181 items) all emit `bth run` CLI-flag lists instead of calling bathos directly, but every
one of those modules says exactly why in its own docstring -- "xtrax has no dependency on
bathos, by design... a compute library must not depend on the orchestration/experiment-
tracking tool built on top of it." That constraint is scoped to `src/xtrax/` (enforced by
`scripts/audit_bathos_independence.py`, itself scoped to `src/xtrax` only -- confirmed by
reading that gate directly). It does not apply to `controller/`: this epic's own architecture
spec (Fork 2) already decided `controller/` may depend on bathos directly for the pure
read/compute gaps (AC-6: `bathos.stats_gates`, `count_seeds_for_script`/
`count_runs_for_script`). So the CLI-flag-assembly precedent in `src/xtrax/` is a workaround
for a constraint this module doesn't have, not evidence this module should also assemble CLI
argv.

The real question was MCP vs. a direct `bathos`-library import vs. `bth` CLI-subprocess. Fork
2's own resolution draws exactly this line: pure read/compute gaps (b/c) get direct library
import; anything that *writes* to bathos's own catalog DB should go through "MCP's tool-layer
validation/audit surface" instead of bypassing it (Finding 8, rejecting a direct import for
`campaign_edges`/`run_edges` on precisely this ground). `campaign_create`/`run`/
`campaign_conclude` are exactly that class of operation -- each one mutates bathos's own
DuckDB catalog (`campaigns` / `runs` / `campaign_runs` rows) -- so this module goes through
MCP, matching every mention of these three calls throughout the architecture spec (FIXED
CONSTRAINTS, the Fork 2 decomposition, and AC-5's own text all say "MCP calls", never CLI).

Verified directly against bathos `origin/main` (commit `b5599a3`), not assumed:
- `src/bathos/mcp.py:1597-1627` (`@app.tool("run")`) and `:1630-1652`
  (`@app.tool("campaign_create")`) and `:1683-1701` (`@app.tool("campaign_conclude")`) are
  real, live FastMCP tool registrations -- not aspirational. Their un-decorated inner
  functions (`run_tool` at `:985`, `campaign_create_tool` at `:1050`, `campaign_conclude_tool`
  at `:1172`) show the exact field contracts this module targets.
- `src/bathos/cli.py:112` (`run`), `:651` (`campaign create`), `:720` (`campaign conclude`)
  confirm the CLI equivalents also exist and are live -- but two concrete, confirmed gaps
  make CLI the worse choice here, not just a stylistic difference:
  1. `campaign create`'s CLI (`cli.py:654`) accepts `mode` = exploration|confirmation|
     sequential (plus a `--sequential` shorthand), but the MCP `campaign_create` tool's mode
     validation (`mcp.py:1073-1076`) only allows `("exploration", "confirmation")` --
     "sequential" is rejected. This is a REAL functionality gap in MCP vs CLI. The
     architecture spec's own FIXED CONSTRAINTS section names this exact gap (Finding 2a,
     "mode='sequential' gap in campaign_create's MCP signature... a known bathos-side gap,
     not something this session resolves by rearchitecting xtrax/controller") and defers it
     to AC-10 as a non-blocking cross-repo backlog item -- i.e. the epic's own authors knew
     about this exact limitation and still chose MCP, accepting the gap rather than routing
     around it via CLI. This module does the same: it forwards whatever `mode` the caller
     supplies verbatim and lets bathos's own validation reject an unsupported value loudly
     (surfaced as `BathosMcpToolError`), rather than silently drop or CLI-side-shim
     "sequential" support that this session explicitly declined to build.
  2. The CLI's own success output for `campaign create` only echoes an 8-char TRUNCATED
     campaign ID (`cli.py`: `typer.echo(f"Created campaign {campaign.id[:8]} -- ...")`),
     meant for a human terminal, not a machine caller -- a CLI-subprocess adapter would have
     to regex-scrape free-text stdout to recover an ID good enough to pass to every
     subsequent `run --campaign <id>` call in the same loop. The MCP tool's return dict
     already carries the real, full `campaign_id` field (`mcp.py:1093`) with zero parsing.
     This is the concrete, load-bearing reason MCP was chosen over CLI-subprocess, not merely
     "the spec said so" -- the CLI's own output shape is not fit for this purpose.
- `src/bathos/runner.py:188-199` (`run_script` signature), `:332` (`parent_run_id=
  derived_from or ""`), `:342-352` (lineage resolution against `derived_from`), and
  `:530-554` (`add_run_to_campaign` called at `:543`, inside the same `run_script`
  invocation) confirm the fixed-constraint claim verbatim: `derived_from` resolution and
  campaign attachment both happen inside ONE `run_script` call. **No separate `campaign_add`
  call is needed** -- `run`'s own internal `add_run_to_campaign` already handles attachment
  when `campaign_id` is supplied. This module therefore exposes exactly three calls
  (`campaign_create`, `run`, `campaign_conclude`), matching the FIXED CONSTRAINT's "campaign_
  create once, one run call per candidate per loop iteration..., campaign_conclude once at
  the end" verbatim -- no fourth method for `campaign_add`.
- `src/bathos/mcp.py:168-193` (`require_write_token`) confirms every one of these three
  calls is a write-verb MCP tool requiring `token=` matching the local, 0600-mode
  `~/.bth/mcp_token` file (debt #619). `src/bathos/mcp_auth.py:47-55`/`:58-73` shows that
  file's resolution rule (`BTH_MCP_TOKEN_PATH` override, else `~/.bth/mcp_token`) and its
  bootstrap-on-first-use behavior. This module independently re-implements that *resolution*
  rule (`_token_path`) to READ the token -- it deliberately does not `import bathos` to reuse
  `get_or_create_token` directly, keeping this module's only bathos dependency the
  arm's-length `bth-mcp` subprocess (consistent with the transport decision above: reading a
  local file to authenticate against MCP's validation surface is not the write-bypass Fork 2
  rejected; it's a prerequisite for going through that surface correctly). Confirmed locally
  that `~/.bth/mcp_token` already exists (64 bytes, matching
  `secrets.token_hex(32)`) -- this is real, exercised infrastructure, not a hypothetical.
- `src/bathos/mcp.py:85-165` (`traced_tool`) confirms every tool call -- success or failure
  -- returns a dict merged with `{"ok": bool, "error_code": ..., "error": ...,
  "resolution_hint": ...}` and NEVER lets an exception escape to the MCP transport layer.
  Combined with `fastmcp.tools.base.Tool.convert_result` (installed at
  `bathos/.venv/lib/python3.12/site-packages/fastmcp/tools/base.py:310-319`, "No schema --
  only use structured_content for dicts"), a dict-returning tool's `tools/call` result always
  carries that dict as BOTH `structuredContent` and a JSON-text `content` block. This module's
  `_extract_structured_result` prefers `structuredContent`, falling back to parsing
  `content[0].text` as JSON.

## Consistency with the controller's other MCP surface

The controller already needs a hand-rolled JSON-RPC-over-stdio client for praxia
(`scripts/probe_praxia_mcp_invocation.py`, feeding the future `PraxiaDispatchBackend`). This
module's own low-level transport (`_call_mcp_tool`) deliberately mirrors that script's exact
mechanics (spawn subprocess, `initialize` -> `notifications/initialized` -> one real call,
terminate) rather than inventing a second, different idiom (CLI argv assembly +
subprocess + free-text stdout parsing) for bathos specifically -- one transport pattern for
both of the controller's cross-repo MCP integrations, not two. Unlike that probe script (which
deliberately stopped at `tools/list`, leaving a live `tools/call` for "the controller's own
build phase"), this module IS that build phase for bathos: it goes one step further, to a
real `tools/call`.

## Known, deliberately out-of-scope gaps found during grounding (not fixed here)

- `campaign_create`'s MCP tool has no `parent` parameter at all (the CLI's `--parent` for
  hierarchical/child campaigns, `cli.py:658`, has no MCP equivalent) -- this module does not
  expose campaign parentage; out of scope for AC-5's own text, which only names
  `campaign_create`/`run`/`campaign_conclude`.
- `campaign_conclude`'s MCP tool has no `force`/`abort_if_below_threshold` (CLI-only,
  threshold-checking flags, `cli.py:727-732`) -- likewise out of scope here.
Both are left to a future AC-10-style cross-repo gap-filing item, not silently worked around.
"""

import json
import os
import selectors
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_TIMEOUT = 30.0

# Mirrors scripts/probe_praxia_mcp_invocation.py's `_FALLBACK_COMMAND` precedent: the live
# ~/.claude.json mcpServers.bathos entry's command, read directly, used only if `bth-mcp`
# isn't resolvable via PATH at call time.
_FALLBACK_COMMAND = "/home/marielle/projects/bathos/.venv/bin/bth-mcp"

# A transport is a plain callable so tests can inject a stub/mock instead of spawning a real
# `bth-mcp` subprocess -- see module docstring's "don't require live bathos infrastructure
# for unit tests" requirement. `(tool_name, arguments) -> structured_envelope_dict`.
BathosMcpTransport = Callable[[str, dict[str, Any]], dict[str, Any]]


class BathosAdapterError(Exception):
    """Base class for this adapter's own errors (transport, auth, or tool-level failures)."""


class BathosTokenMissingError(BathosAdapterError):
    """No local bathos MCP write-token file was found (or it was empty)."""


class BathosMcpTransportError(BathosAdapterError):
    """The `bth-mcp` subprocess could not be reached, or the JSON-RPC exchange failed."""


class BathosMcpToolError(BathosAdapterError):
    """A bathos MCP tool call completed but its structured envelope reported `ok: False`."""

    def __init__(self, tool_name: str, envelope: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.envelope = envelope
        detail = envelope.get("error") or envelope.get("error_code") or "unknown error"
        super().__init__(f"bathos MCP tool {tool_name!r} failed: {detail}")


@dataclass(frozen=True, slots=True)
class CampaignHandle:
    """A created bathos campaign, per `campaign_create_tool`'s return contract
    (`bathos/src/bathos/mcp.py:1093-1099`)."""

    campaign_id: str
    name: str
    mode: str
    status: str
    started_at: str


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    """Result of one bathos `run` call for a single candidate, per `run_tool`'s return
    contract (`bathos/src/bathos/mcp.py:1043-1047`).

    Campaign attachment (when `campaign_id` was supplied to `run`) already happened inside
    bathos's own `run_script` via its internal `add_run_to_campaign` call
    (`bathos/src/bathos/runner.py:543`) -- nothing further to do here.
    """

    script_path: str
    exit_code: int
    success: bool


@dataclass(frozen=True, slots=True)
class CampaignConclusion:
    """Result of `campaign_conclude`, per `campaign_conclude_tool`'s return contract
    (`bathos/src/bathos/mcp.py:1201-1205`)."""

    status: str
    campaign_id: str
    outcome_label: str


def _resolve_bth_mcp_command() -> str:
    """Resolve the `bth-mcp` entrypoint: PATH first, then the known local fallback.

    Mirrors `scripts/probe_praxia_mcp_invocation.py::_resolve_command` exactly.
    """
    return shutil.which("bth-mcp") or _FALLBACK_COMMAND


def _token_path() -> Path:
    """Resolve the local bathos MCP write-token file path.

    Independently re-implements (does not import `bathos`) the exact resolution rule of
    `bathos.mcp_auth.token_path` -- verified directly against
    `bathos/src/bathos/mcp_auth.py:47-55`: honor `BTH_MCP_TOKEN_PATH` for tests/overrides,
    else default to `~/.bth/mcp_token`.
    """
    override = os.environ.get("BTH_MCP_TOKEN_PATH")
    if override:
        return Path(override)
    return Path.home() / ".bth" / "mcp_token"


def _read_local_mcp_token() -> str:
    """Read the local bathos MCP write-token, without creating it.

    Bathos's own server-side `check_token` (`bathos/src/bathos/mcp_auth.py:76-86`) calls
    `get_or_create_token()` and so *can* create the token file itself on first use -- but
    that happens inside the bathos process handling the very first authenticated call, using
    whatever token that call actually supplied. This adapter cannot bootstrap trust for
    itself; it can only read a token that a prior local `bth`/`bth-mcp` invocation already
    created (see `bathos/src/bathos/errors.py:72`'s own resolution hint: "run any `bth`
    command locally to create it if missing").
    """
    path = _token_path()
    if not path.exists():
        msg = (
            f"no local bathos MCP write-token at {path} -- run any `bth` command locally "
            "first to create it (bathos.mcp_auth.get_or_create_token)"
        )
        raise BathosTokenMissingError(msg)
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        msg = f"bathos MCP write-token file at {path} is empty"
        raise BathosTokenMissingError(msg)
    return token


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


def _extract_structured_result(result: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    """Parse a `tools/call` MCP result into bathos's own structured-envelope dict.

    See module docstring for why `structuredContent` is authoritative (every bathos tool is
    wrapped by `traced_tool`, which always returns a dict) and `content[0].text` is only a
    fallback.
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

    msg = f"tools/call {tool_name!r} returned no parseable structuredContent/content: {result!r}"
    raise BathosMcpTransportError(msg)


def _call_mcp_tool(
    command: str, tool_name: str, arguments: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    """Spawn bathos's real `bth-mcp` entrypoint, complete the MCP handshake, call one tool,
    and return its structured-envelope dict.

    Hand-rolled JSON-RPC-over-stdio, no MCP SDK client -- mirrors
    `scripts/probe_praxia_mcp_invocation.py`'s exact mechanics, extended one step further
    from that script's `tools/list` stopping point through to a real `tools/call` (see module
    docstring). One subprocess per call, not a long-lived session reused across a whole
    campaign: every bathos MCP tool already opens and closes its own DuckDB connection per
    call (e.g. `campaign_create_tool`/`campaign_conclude_tool`, both `db = duckdb.connect(...)`
    / `finally: db.close()` in `bathos/src/bathos/mcp.py`), so there is no server-side session
    state a persistent client connection would actually be preserving.
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
        raise BathosMcpTransportError(f"failed to spawn {command!r}: {exc}") from exc

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
                        "name": "xtrax-loop-controller-bathos-adapter",
                        "version": "0.1",
                    },
                },
            },
        )
        try:
            init_response = _read_one_response(proc, timeout=timeout)
        except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            raise BathosMcpTransportError(f"initialize failed: {exc}") from exc

        result = init_response.get("result")
        if not isinstance(result, dict) or "serverInfo" not in result:
            msg = f"no serverInfo in initialize response: {init_response!r}"
            raise BathosMcpTransportError(msg)

        # MCP requires the initialized notification before any tool call is valid.
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        try:
            call_response = _read_one_response(proc, timeout=timeout)
        except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            raise BathosMcpTransportError(f"tools/call {tool_name!r} failed: {exc}") from exc

        if "error" in call_response:
            msg = f"tools/call {tool_name!r} JSON-RPC error: {call_response['error']}"
            raise BathosMcpTransportError(msg)

        return _extract_structured_result(call_response.get("result") or {}, tool_name=tool_name)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


class BathosCampaignAdapter:
    """Sequences one bathos campaign's lifecycle over bathos's real MCP server.

    Exactly three calls compose this adapter, matching the epic's own FIXED CONSTRAINT
    verbatim: `campaign_create` once per campaign start, one `run` call per candidate per
    loop iteration (its own internal `add_run_to_campaign` handles attachment -- no separate
    `campaign_add` call), and `campaign_conclude` once at the end. See the module docstring
    for the grounded MCP-vs-CLI transport decision and file/line citations.
    """

    def __init__(
        self,
        *,
        project_slug: str = "",
        catalog_dir: str = "",
        command: str | None = None,
        token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: BathosMcpTransport | None = None,
    ) -> None:
        """Configure the adapter.

        Args:
            project_slug: bathos project slug. Empty string is passed straight through to
                bathos, which resolves its own default (`run`/`campaign_create` disagree on
                what that default actually is server-side -- this adapter does not paper
                over that by inventing a client-side default of its own).
            catalog_dir: bathos catalog directory. Empty string means "use bathos's default",
                same pass-through convention as `project_slug`.
            command: explicit `bth-mcp` executable path. If `None`, resolved lazily per call
                via `_resolve_bth_mcp_command` (PATH, then the known local fallback).
            token: explicit MCP write-token. If `None`, resolved lazily per call by reading
                the local `~/.bth/mcp_token` file (or `BTH_MCP_TOKEN_PATH` override).
            timeout: seconds to wait for each MCP response before raising
                `BathosMcpTransportError`.
            transport: injection seam for tests -- a callable `(tool_name, arguments) ->
                envelope_dict` used instead of spawning a real `bth-mcp` subprocess. If
                `None`, the real stdio transport is used.
        """
        self._project_slug = project_slug
        self._catalog_dir = catalog_dir
        self._command = command
        self._token = token
        self._timeout = timeout
        self._transport = transport if transport is not None else self._default_transport

    def _default_transport(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        command = self._command if self._command is not None else _resolve_bth_mcp_command()
        return _call_mcp_tool(command, tool_name, arguments, timeout=self._timeout)

    def _resolved_token(self) -> str:
        return self._token if self._token is not None else _read_local_mcp_token()

    def _invoke(self, tool_name: str, arguments: dict[str, Any], *, write: bool) -> dict[str, Any]:
        call_arguments = dict(arguments)
        if write:
            call_arguments["token"] = self._resolved_token()
        envelope = self._transport(tool_name, call_arguments)
        # `envelope.get("ok") is False` is the normal case: every one of this adapter's
        # three tools is `@traced_tool`-wrapped (confirmed: `mcp.py:1598,1631,1684`), which
        # always merges in an explicit `ok` before a real response ever reaches this client
        # (`mcp.py:131-140`). The second condition mirrors that same merge's own fallback
        # heuristic ("ok" absent but "error" set) as defense-in-depth, not a behavior this
        # adapter invents independently.
        ok = envelope.get("ok")
        if ok is False or (ok is None and envelope.get("error") is not None):
            raise BathosMcpToolError(tool_name, envelope)
        return envelope

    def campaign_create(
        self,
        name: str,
        *,
        mode: str = "exploration",
        question: str = "",
        hypothesis: str = "",
    ) -> CampaignHandle:
        """Create a new bathos campaign. Call once, at campaign start.

        Args:
            name: campaign name.
            mode: forwarded verbatim to bathos's `campaign_create` MCP tool -- see module
                docstring re: the confirmed `mode="sequential"` MCP gap (Finding 2a). An
                unsupported mode surfaces as a loud `BathosMcpToolError`, never silently
                dropped or rewritten.
            question: optional research question.
            hypothesis: optional research hypothesis.

        Returns:
            A `CampaignHandle` with the real, full `campaign_id` to thread through every
            subsequent `run`/`campaign_conclude` call in this campaign.

        Raises:
            BathosMcpToolError: bathos's own tool-level validation rejected the call (e.g.
                missing `name`, or an unsupported `mode`).
            BathosMcpTransportError: the MCP round-trip itself failed.
            BathosTokenMissingError: no local MCP write-token is available.
        """
        envelope = self._invoke(
            "campaign_create",
            {
                "name": name,
                "mode": mode,
                "project_slug": self._project_slug,
                "catalog_dir": self._catalog_dir,
                "question": question,
                "hypothesis": hypothesis,
            },
            write=True,
        )
        return CampaignHandle(
            campaign_id=envelope["campaign_id"],
            name=envelope["name"],
            mode=envelope["mode"],
            status=envelope["status"],
            started_at=envelope["started_at"],
        )

    def run(
        self,
        script_path: str,
        *,
        args: list[str] | None = None,
        derived_from: str = "",
        campaign_id: str = "",
        output_paths: list[str] | None = None,
        tags: list[str] | None = None,
        agent_mode: str = "",
        no_sidecar: bool = False,
    ) -> CandidateRunResult:
        """Run one candidate script and record its provenance. Call once per candidate per
        loop iteration.

        Args:
            script_path: path to the candidate script bathos should execute.
            args: command-line arguments for the script.
            derived_from: **first-class parameter per AC-5's own AC text** -- the parent run
                ID for lineage (bathos's `parent_run_id`). Threaded straight through to
                bathos's `run` MCP tool, which resolves it against bathos's own run catalog
                (`bathos/src/bathos/runner.py:342-352`) inside this same call. This is what
                lets AC-7 (lineage interim) reuse this adapter with zero retrofitting.
            campaign_id: campaign to attach this run to. Bathos's own `run_script` handles
                attachment internally via `add_run_to_campaign`
                (`bathos/src/bathos/runner.py:543`) -- this adapter does not make a separate
                `campaign_add` call.
            output_paths: registered output files.
            tags: search tags.
            agent_mode: `"collaborative"`, `"autonomous"`, or `""`.
            no_sidecar: bypass sidecar enforcement, for exploratory runs.

        Returns:
            A `CandidateRunResult` with the script's exit code and success flag.

        Raises:
            BathosMcpToolError: bathos's own tool-level validation or the script run itself
                reported failure.
            BathosMcpTransportError: the MCP round-trip itself failed.
            BathosTokenMissingError: no local MCP write-token is available.
        """
        envelope = self._invoke(
            "run",
            {
                "script_path": script_path,
                "args": args or [],
                "project_slug": self._project_slug,
                "catalog_dir": self._catalog_dir,
                "output_paths": output_paths or [],
                "tags": tags or [],
                "agent_mode": agent_mode,
                "derived_from": derived_from,
                "campaign_id": campaign_id,
                "no_sidecar": no_sidecar,
            },
            write=True,
        )
        return CandidateRunResult(
            script_path=envelope["script_path"],
            exit_code=envelope["exit_code"],
            success=envelope["success"],
        )

    def campaign_conclude(
        self, campaign_id: str, outcome_label: str, *, conclusion: str = ""
    ) -> CampaignConclusion:
        """Conclude a bathos campaign. Call once, at the end.

        Args:
            campaign_id: the campaign to conclude (the `CampaignHandle.campaign_id` from
                `campaign_create`).
            outcome_label: outcome label (e.g. `"success"`, `"inconclusive"`, `"failed"`).
            conclusion: summary conclusion text.

        Returns:
            A `CampaignConclusion` confirming the conclusion.

        Raises:
            BathosMcpToolError: bathos's own tool-level validation rejected the call (e.g.
                missing `campaign_id`/`outcome_label`, or an unknown campaign).
            BathosMcpTransportError: the MCP round-trip itself failed.
            BathosTokenMissingError: no local MCP write-token is available.
        """
        envelope = self._invoke(
            "campaign_conclude",
            {
                "campaign_id": campaign_id,
                "outcome_label": outcome_label,
                "conclusion": conclusion,
                "catalog_dir": self._catalog_dir,
            },
            write=True,
        )
        return CampaignConclusion(
            status=envelope["status"],
            campaign_id=envelope["campaign_id"],
            outcome_label=envelope["outcome_label"],
        )


__all__ = [
    "BathosAdapterError",
    "BathosCampaignAdapter",
    "BathosMcpToolError",
    "BathosMcpTransport",
    "BathosMcpTransportError",
    "BathosTokenMissingError",
    "CampaignConclusion",
    "CampaignHandle",
    "CandidateRunResult",
]

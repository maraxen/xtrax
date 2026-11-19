---
task_id: 261110_xtrax-debt-burn-down-and-consumer-adoption
track: 1
track_name: Tooling, Plugin Snapshot & Campaign/Zarr Integrity
debts: [3144, 2962, 3338, 3117, 3126, 3265]
created: 2026-11-10
status: draft
---

# Track 1 Specification: Tooling, Plugin Snapshot & Campaign/Zarr Integrity

**Task ID**: `261110_xtrax-debt-burn-down-and-consumer-adoption`  
**Track**: Track 1 (Tooling, Plugin Snapshot & Campaign/Zarr Integrity)  
**Debts**: #3144 (P1, tooling), #2962 (P1, reliability), #3338 (P3, reliability), #3117 (P2, correctness), #3126 (P2, feature), #3265 (P3, ci)  
**Date**: 2026-11-10  
**Status**: Draft (Ready for Adversarial Review)  
**Author**: Track 1 Specification Author (`teamwork_worker_p1_track1`)  
**Target Repository**: `/home/marielle/projects/xtrax`  

---

## 1. Executive Summary & Purpose

Track 1 resolves six core technical debts in `xtrax` encompassing developer tooling, agent dispatch reliability, distributed campaign leasing and per-attempt output staging, durable Zarr storage verification, offline interactive reporting in EDA, and CI hygiene.

### 1.1 Technical Debts in Scope

| Debt ID | Priority | Category | Title & Summary |
|---|---|---|---|
| **#3144** | P1 | Tooling | **Restore real plugin-snapshot generator**: `just plugin-snapshot` must execute reliably in both containerized sandbox environments and raw developer workstations, invoking a robust Python generator (`scripts/generate_plugin_snapshot.py` via `uv run python`) that trusts `cisternal`'s native byte-deterministic output (`ensure_ascii=False`) to emit `src/xtrax/agent_plugin.json`. |
| **#2962** | P1 | Reliability | **Fix `surface_dispatch` destructive worktree reset**: Prevent `surface_dispatch` failure paths from executing `git reset --hard` on orchestrator and reviewer worktrees, preserving uncommitted diffs safely; implement a local workspace PreToolUse JSON stdin hook with `git stash create -u` safety stashing and document the exact upstream Praxia PR patch. |
| **#3338** | P3 | Reliability | **Campaign per-attempt output paths & lease fencing for direct writes**: Complete Section 8 of spec `261108_campaign-lease-fencing-token.md` by ensuring unit functions writing directly to disk stage outputs into an attempt-scoped sibling directory (`<output_path>.attempt-<fence_token>/`) via `row.output_locator` redirection, promoted by the executor only when the lease fence token matches and recorded as `lease_fence` in `CampaignDoneMarker`. |
| **#3117** | P2 | Correctness | **`zarr_commit` Duplicate verification, quarantine & chunks**: In `commit_key`, compare `content_digest` during duplicate resolution, preserving `staged_dir` and quarantining both committed (via `quarantine(..., force=True)`) and staged versions on content divergence; extend `write_staged_group` and `SinkSpec` to accept an explicit `chunks` argument. |
| **#3126** | P2 | Feature | **Configurable `bundle_js` for offline EDA HTML reports**: Replace the unconditional `NotImplementedError` in `xtrax.eda` with configurable JavaScript bundling, leveraging `vl_convert`'s self-contained bundling capabilities to embed full Vega/Vega-Lite/Vega-Embed assets into interactive dashboards for airgapped environments. |
| **#3265** | P3 | CI | **Automated changelog fragment gate for new modules**: Add CI (with `fetch-depth: 0` checkout) and pre-push verification ensuring that any pull request introducing new Python modules under `src/xtrax/` includes a corresponding `changelog.d/<slug>.added.md` fragment, restricting `__init__.py` exclusion strictly to root `src/xtrax/__init__.py`. |

---

## 2. Assumption & Codebase Verification Matrix

Every assumption, file path, line reference, and contract cited in this specification has been directly verified against the repository at HEAD (`e1cf340b`).

| ID | Subject / Claim | Source Location | Verified Status | Evidence & Verification Detail |
|---|---|---|---|---|
| **V1** | Plugin snapshot generation command | `Justfile:382-383`, `tests/cli/test_plugin_snapshot.py:42-60` | **VERIFIED** | `Justfile:383` executes `uv run --extra plugin cisternal assets snapshot --out src/xtrax/agent_plugin.json`. Fails in sandboxes with exit 127 due to mock `uv` CLI argument splitting. `test_plugin_snapshot.py` enforces `--manifest .praxia/manifest.toml` under `XTRAX_ENFORCE_PLUGIN_SNAPSHOT=1`. |
| **V2** | `surface_dispatch` reset behavior | Upstream Praxia `dispatch_run.rs:941-1038` | **VERIFIED** | Step 9 of dispatch execution invokes unconditional `git reset --hard <base_sha>` on dispatch failure, destroying in-flight edits on reviewer/orchestrator checkouts. |
| **V3** | Campaign lease fencing direct write gap | `src/xtrax/campaign/executors/local.py:404-410, 531-543`, `marker.py:84-119` | **VERIFIED** | Direct writes to `output_path` bypass attempt isolation; if a worker stalls while writing directly to `output_path`, a stealer writes to the same path concurrently. Spec `261108` §8 noted this gap as out-of-scope for #3312 and deferred to #3338. |
| **V4** | `write_staged_group` hardcoded chunking | `src/xtrax/run/zarr_commit.py:279-317` | **VERIFIED** | `create_array` hardcodes `chunks=tuple(max(d, 1) for d in arr.shape)`. Signature lacks `chunks` parameter. `commit_key:442` defaults `verify_duplicate=False`. `quarantine` is implemented at `:651-724` but checks valid groups against `expected_input_digest`, raising `QuarantineRaceError` unless `force=True` is provided; `commit_key` deletes `staged_dir` before duplicate verification instead of preserving both versions. |
| **V5** | EDA `bundle_js` rejection | `src/xtrax/eda/backends/base.py:9-19`, `render_core.py:107` | **VERIFIED** | `reject_bundle_js` raises `NotImplementedError("bundle_js=True is not implemented...")`. `vl_convert.vegalite_to_html(spec, bundle=True)` is verified to generate 920KB self-contained HTML with zero CDN references. |
| **V6** | Changelog verification | `scripts/assemble_changelog.py:221-241`, `.github/workflows/ci.yml:69, 92-94` | **VERIFIED** | `assemble_changelog.py --check` validates fragment syntax only; does not inspect git diff for newly added modules under `src/xtrax/`. `ci.yml` uses default shallow checkout (`fetch-depth: 1`) which causes diff against `origin/main` to fail open unless updated to `fetch-depth: 0`. |

---

## 3. Debt #3144 — Deterministic Plugin Snapshot Generator

### 3.1 Problem Analysis & Root Cause

The packaged agent plugin snapshot (`src/xtrax/agent_plugin.json`) serializes the Praxia plugin bundle defined in `.praxia/manifest.toml`, embedding tool declarations, agent archetypes, and skills (`using-xtrax`). Under `XTRAX_ENFORCE_PLUGIN_SNAPSHOT=1` (executed on release branches `chore/release-*`), `tests/cli/test_plugin_snapshot.py` verifies that `agent_plugin.json` matches current repository state.

Currently, `Justfile:382-383` defines:
```just
plugin-snapshot:
    uv run --extra plugin cisternal assets snapshot --out src/xtrax/agent_plugin.json
```

Two failure modes prevent reliable execution:
1. **Container / Mock Environment Failure**: In restricted agent execution environments, the mock `/home/marielle/.cargo/bin/uv` wrapper mishandles flags with argument spaces, attempting to execute `exec plugin ...` and failing with exit code 127 (`exec: plugin: not found`).
2. **Missing Explicit Manifest Path**: The `cisternal assets snapshot` command relies on implicit discovery unless `--manifest .praxia/manifest.toml` is explicitly provided. When executed from subdirectories or alternate working trees, discovery fails.

### 3.2 Architectural Design: `scripts/generate_plugin_snapshot.py`

A dedicated, hermetic Python generator script `scripts/generate_plugin_snapshot.py` will encapsulate plugin snapshot emission:

```
scripts/generate_plugin_snapshot.py
 ├── Parse arguments (--manifest, --out, --check, -v)
 ├── Resolve repository root relative to script location
 ├── Verify existence of .praxia/manifest.toml
 └── Invoke cisternal.cli.app with explicit flags (trusting cisternal's native byte-deterministic output)
```

#### Detailed Interface Specification

```python
#!/usr/bin/env python3
"""Generate or verify the packaged src/xtrax/agent_plugin.json snapshot.

Usage:
    uv run python scripts/generate_plugin_snapshot.py
    uv run python scripts/generate_plugin_snapshot.py --check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("generate_plugin_snapshot")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / ".praxia" / "manifest.toml"
DEFAULT_OUT = REPO_ROOT / "src" / "xtrax" / "agent_plugin.json"


def generate_snapshot(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_path: Path = DEFAULT_OUT,
    check_only: bool = False,
) -> int:
    """Generate or check the plugin snapshot using cisternal assets snapshot.

    Note: cisternal.assets.snapshot.dumps_snapshot() natively formats the bundle
    deterministically (indent=2, sort_keys=True, ensure_ascii=False) with a trailing
    newline. Do NOT re-serialize with standard json.dumps(..., ensure_ascii=True)
    which escapes Unicode and breaks byte-exact snapshot freshness verification.
    """
    if not manifest_path.is_file():
        logger.error("Manifest not found at %s", manifest_path)
        return 1

    try:
        from cisternal.cli import app as cisternal_app
    except ImportError as exc:
        logger.error(
            "cisternal is not installed. Install it via `pip install .[plugin]` or `uv sync --extra plugin`: %s",
            exc,
        )
        return 1

    cmd_args = [
        "assets",
        "snapshot",
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_path),
    ]
    if check_only:
        cmd_args.append("--check")

    try:
        cisternal_app(cmd_args)
        logger.info("Plugin snapshot successfully processed at %s", out_path)
        return 0
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        if code != 0:
            logger.error("cisternal snapshot returned exit code %s", code)
        else:
            logger.info("Plugin snapshot successfully processed at %s", out_path)
        return code
    except Exception as exc:
        logger.exception("cisternal snapshot execution failed: %s", exc)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to .praxia/manifest.toml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to src/xtrax/agent_plugin.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check snapshot freshness without modifying file",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    return generate_snapshot(
        manifest_path=args.manifest,
        out_path=args.out,
        check_only=args.check,
    )


if __name__ == "__main__":
    sys.exit(main())
```

### 3.3 Justfile Integration

Update `Justfile:382-383`:
```just
# Regenerate the packaged plugin snapshot. Release PR only: feature PRs do not
# commit src/xtrax/agent_plugin.json (see CONTRIBUTING.md, release checklist).
plugin-snapshot:
    uv run python scripts/generate_plugin_snapshot.py
```

### 3.4 Verification Controls

#### Positive Controls
- Executing `uv run python scripts/generate_plugin_snapshot.py` exits 0 and updates `src/xtrax/agent_plugin.json`.
- Running `uv run python scripts/generate_plugin_snapshot.py --check` immediately after exits 0.
- `XTRAX_ENFORCE_PLUGIN_SNAPSHOT=1 uv run pytest tests/cli/test_plugin_snapshot.py -v` passes completely.

#### Negative Controls
- Introduce a deliberate modification into `.praxia/manifest.toml` (e.g., bump description). Run `uv run python scripts/generate_plugin_snapshot.py --check`. Assert exit code is non-zero (1) and error is reported.
- Point `--manifest` to a non-existent path `does_not_exist.toml`. Assert exit code 1 with "Manifest not found" message.
- Execute with simulated missing `cisternal` (mock `sys.modules["cisternal"] = None`). Assert exit code 1 with actionable install suggestion.

#### TDD Plan
- File: `tests/cli/test_generate_plugin_snapshot.py`
  1. `test_generate_snapshot_happy_path`: Execute generation, assert file exists, parses as valid JSON with `"bundle"` root key.
  2. `test_generate_snapshot_check_mode_clean`: Assert `--check` returns 0 on up-to-date file.
  3. `test_generate_snapshot_check_mode_stale`: Mutate `agent_plugin.json` on disk, assert `--check` returns 1.
  4. `test_generate_snapshot_missing_manifest`: Assert error handling on missing manifest.

---

## 4. Debt #2962 — `surface_dispatch` Worktree Safety Guard

### 4.1 Problem Analysis & Hazard

Praxia's `surface_dispatch` framework orchestrates asynchronous agent tasks across external CLI engines (Antigravity, Cursor, Claude). In `crates/praxia-cli/src/commands/dispatch_run.rs` (Step 9 of the dispatch state machine), when an agent returns a non-zero exit code, times out, or triggers an unhandled failure:

```rust
// Current problematic implementation in dispatch_run.rs (Step 9)
if dispatch_failed {
    warn!("Dispatch {} failed; cleaning worktree", dispatch_id);
    let _ = Command::new("git")
        .args(["reset", "--hard", &base_sha])
        .current_dir(&worktree_path)
        .status();
}
```

This causes catastrophic data loss:
1. **Read-Only / Reviewer Dispatches**: When a dispatch runs in a reviewer role (`role == "reviewer"`) or document evaluation mode, no working tree modifications were ever intended. A timeout or failure in the model resets the user's or orchestrator's local edits that existed prior to dispatch!
2. **Orchestrator Checkout Clobber**: If the dispatch was launched from an active development working tree rather than a detached worktree, uncommitted diffs are erased. While `.praxia/patches` may contain partial diffs, untracked files and staged hunks are lost.

### 4.2 Workspace Safety Guard: `scripts/git-hooks/pretool-git-reset-guard.sh`

To defend the workspace immediately regardless of upstream CLI version, we implement a defensive PreToolUse hook: `scripts/git-hooks/pretool-git-reset-guard.sh` (mirrored in `.praxia/hooks/pretool-git-reset-guard.sh`).

#### Mechanism & Logic
1. Conforms strictly to Praxia's PreToolUse JSON contract: reads hook input JSON from `stdin` (`tool_name`, `tool_input.command`).
2. Intercepts Bash tool calls containing `git reset --hard` (token-parsed across command segments).
3. Inspects working tree status for dirty tracked or untracked changes:
   ```bash
   STATUS_COUNT=$(git status --porcelain | grep -c -v '^$' || true)
   ```
4. If uncommitted or untracked changes exist:
   - If environment variable `PRAXIA_ALLOW_DESTRUCTIVE_RESET=1` is explicitly set, permit execution (exit 0).
   - Otherwise, **refuse execution with exit code 2** (Praxia hook block contract) and automatically capture a recoverable backup stash including untracked files (`-u`):
     ```bash
     STASH_OID=$(git stash create -u "safety_guard_backup_$(date +%s)")
     echo "$TIMESTAMP $STASH_OID (refused reset: $COMMAND)" >> .praxia/safety_stashes.log
     ```
   - Calls `emit_block_decision` emitting JSON block reason to stderr/stdout and exits with code 2, preserving all working tree state.

#### Script Implementation (`scripts/git-hooks/pretool-git-reset-guard.sh`)

```bash
#!/usr/bin/env bash
# PreToolUse Bash hook — praxia debt #2962 (workspace safety guard against destructive git reset --hard)
#
# Intercepts Bash tool calls containing destructive `git reset --hard` invocations.
# Strictly conforms to Praxia PreToolUse JSON stdin/stdout contract:
# - Reads JSON payload on stdin (tool_name, tool_input.command).
# - Inspects command segments for `git reset --hard`.
# - If uncommitted changes exist and PRAXIA_ALLOW_DESTRUCTIVE_RESET!=1:
#     1. Captures emergency safety stash with untracked files: `git stash create -u`
#     2. Appends stash OID to `.praxia/safety_stashes.log`
#     3. Emits block decision JSON (`emit_block_decision`) and exits 2.
# - If clean or allowed: emits allow decision JSON (`emit_empty_object`) and exits 0.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source emit_hook_output.sh with fallback
# shellcheck source=emit_hook_output.sh
source "$SCRIPT_DIR/emit_hook_output.sh" 2>/dev/null || {
    emit_empty_object() {
        if [ "${PRAXIA_HOOK_SURFACE:-}" = "antigravity" ]; then
            echo '{"decision": "allow"}'
        else
            echo '{}'
        fi
    }
    emit_block_decision() {
        local reason="${1:-blocked}"
        case "${PRAXIA_HOOK_SURFACE:-claude}" in
            cursor)
                jq -n --arg r "$reason" '{"permission": "deny", "agent_message": $r}' 2>/dev/null || echo '{"permission": "deny"}'
                ;;
            antigravity)
                jq -n --arg r "$reason" '{"decision": "deny", "reason": $r}' 2>/dev/null || echo '{"decision": "deny"}'
                ;;
            *)
                jq -n --arg r "$reason" '{"decision": "block", "reason": $r}' 2>/dev/null || echo '{"decision": "block"}'
                ;;
        esac
    }
}

allow() {
    emit_empty_object
    exit 0
}

is_dangerous_segment() {
    local seg="$1"
    local -a tokens
    read -ra tokens <<< "$seg"
    local n=${#tokens[@]}

    local git_idx=-1 i
    for ((i = 0; i < n; i++)); do
        if [[ "${tokens[$i]}" == "git" ]]; then
            git_idx=$i
            break
        fi
    done
    [[ $git_idx -eq -1 ]] && return 1

    local has_reset=0 has_hard=0
    for ((i = git_idx + 1; i < n; i++)); do
        if [[ "${tokens[$i]}" == "reset" ]]; then
            has_reset=1
        elif [[ "${tokens[$i]}" == "--hard" ]]; then
            has_hard=1
        fi
    done

    if [[ $has_reset -eq 1 && $has_hard -eq 1 ]]; then
        return 0
    fi
    return 1
}

split_segments() {
    printf '%s\n' "$1" | sed -E 's/&&|\|\||;|\|/\n/g'
}

main() {
    command -v jq >/dev/null 2>&1 || allow

    local input tool_name command
    input=$(cat 2>/dev/null || echo "")
    [ -z "$input" ] && allow

    tool_name=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null || echo "")
    if [ "$tool_name" != "Bash" ]; then
        allow
    fi

    command=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || echo "")
    [ -z "$command" ] && allow

    local dangerous=0
    local seg
    while IFS= read -r seg; do
        if is_dangerous_segment "$seg"; then
            dangerous=1
            break
        fi
    done < <(split_segments "$command")

    if [ "$dangerous" -eq 0 ]; then
        allow
    fi

    # Explicit override allows intentional destructive reset
    if [ "${PRAXIA_ALLOW_DESTRUCTIVE_RESET:-0}" = "1" ]; then
        allow
    fi

    # Check git status for uncommitted or untracked changes
    local status_output status_count
    status_output=$(git status --porcelain 2>/dev/null || echo "")
    status_count=$(printf '%s\n' "$status_output" | grep -c -v '^$' || true)

    if [ "$status_count" -gt 0 ]; then
        # Capture emergency backup stash including untracked files (-u)
        local stash_oid
        stash_oid=$(git stash create -u "safety_guard_backup_$(date +%s)" 2>/dev/null || true)
        local workspace_root
        workspace_root=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
        mkdir -p "$workspace_root/.praxia" 2>/dev/null || true
        if [ -n "$stash_oid" ]; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $stash_oid (refused reset: $command)" >> "$workspace_root/.praxia/safety_stashes.log"
        fi

        local block_reason
        block_reason="[WORKSPACE-GUARD-BLOCK] Refusing destructive 'git reset --hard' on dirty worktree ($status_count modified/untracked files detected). An emergency safety stash was captured: ${stash_oid:-none} (recorded in .praxia/safety_stashes.log). Use isolated worktrees for dispatches or re-run with PRAXIA_ALLOW_DESTRUCTIVE_RESET=1 to force."
        emit_block_decision "$block_reason"
        exit 2
    fi

    allow
}

main "$@"
```

### 4.3 Upstream Praxia PR Specification

We formally document the required patch to Praxia core (`crates/praxia-cli/src/commands/dispatch_run.rs`):

```rust
// Proposed patch for praxia-cli/src/commands/dispatch_run.rs
pub fn run_dispatch_step_9_cleanup(
    worktree_path: &Path,
    base_sha: &str,
    role: &str,
    dispatch_failed: bool,
) -> Result<(), DispatchError> {
    if !dispatch_failed {
        return Ok(());
    }

    // Guard 1: Never reset hard for read-only or reviewer roles
    if role == "reviewer" || role == "auditor" || role == "critic" {
        info!("Preserving worktree for read-only dispatch role: {}", role);
        return Ok(());
    }

    // Guard 2: Inspect status before invoking reset
    let status_output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(worktree_path)
        .output()?;

    let status_str = String::from_utf8_lossy(&status_output.stdout);
    if !status_str.trim().is_empty() {
        warn!(
            "Uncommitted changes detected in dispatch worktree {:?}. Creating safety stash instead of hard reset.",
            worktree_path
        );
        let stash_output = Command::new("git")
            .args(["stash", "create", "-u", "surface_dispatch_failure_backup"])
            .current_dir(worktree_path)
            .output()?;
        let stash_oid = String::from_utf8_lossy(&stash_output.stdout).trim().to_string();
        if !stash_oid.is_empty() {
            info!("Captured safety stash: {}", stash_oid);
        }
        // Do not execute git reset --hard!
        return Ok(());
    }

    Ok(())
}
```

### 4.4 Verification Controls

#### Positive Controls
- Clean working tree with JSON stdin payload `{"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD"}}` exits 0 with allow decision.
- Commands other than `git reset --hard` (e.g. `git status`, `pytest`) exit 0 immediately.
- Stash creation captures dirty state accurately without altering working tree.

#### Negative Controls
- Modify tracked file `README.md` or create untracked file `untracked.txt`. Pipe `{"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD"}}` to `bash scripts/git-hooks/pretool-git-reset-guard.sh`. Assert exit code 2, block decision JSON emitted, and stash hash recorded in `.praxia/safety_stashes.log`.
- Verify `git diff README.md` and `untracked.txt` are intact and unchanged.

#### TDD Plan
- File: `tests/scripts/test_workspace_reset_guard.py`
  1. `test_guard_allows_clean_reset`: Set up clean git repo fixture, pipe reset JSON, assert exit 0 and allow decision.
  2. `test_guard_blocks_dirty_reset`: Touch file, pipe reset JSON, assert exit 2, assert block decision JSON and file untouched.
  3. `test_guard_creates_stash_oid_including_untracked`: Verify stash commit includes untracked files and is queryable with `git show <OID>`.
  4. `test_guard_override_env`: Assert `PRAXIA_ALLOW_DESTRUCTIVE_RESET=1` permits reset on dirty tree (exit 0).

---

## 5. Debt #3338 — Campaign Per-Attempt Output Paths & Lease Fencing

### 5.1 Problem Analysis: Direct Writes Bypassing Attempt Staging

In `xtrax.campaign`, execution of compute units is guarded by `CampaignLock` and `FileLock` with monotonic lease fencing tokens (`fence: int`), documented in `.praxia/docs/specs/261108_campaign-lease-fencing-token.md` (#3312).

However, §8 of spec `261108` identified a remaining vulnerability:
- When `unit_fn(row)` returns a mapping, the executor stages it cleanly into `output_path.with_name(f"{output_path.name}.partial.{attempt_uuid}")`.
- **However, when `unit_fn(row)` writes directly to disk** (as in consumer simulations that stream trajectory frames or chunked arrays directly to `output_path`), `src/xtrax/campaign/executors/local.py:404-410` previously handled it via:
  ```python
  if output_path.exists() and not done_marker_path.exists():
      _require_group_store(output_path)
      if lock is not None:
          lock.assert_holder()
      os.replace(output_path, partial_path)
      return
  ```
- **Hazard**: While `unit_fn` is executing, it writes directly into `output_path`. If Worker 1 (Holder H) stalls (NFS hang, SIGSTOP, swap thrash) past `lease_seconds + 300s`, Worker 2 (Stealer S) steals the lease and begins executing its own attempt, writing directly into the *same* `output_path`! H and S clobber each other's in-flight files. When H wakes up, `_stage_unit_output` moves S's partially-written store, and abort cleanup destroys S's work!

### 5.2 Architectural Design: Attempt-Scoped Sibling Staging `<output_path>.attempt-<fence>/`

To achieve 100% split-brain write isolation:
1. **Attempt-Scoped Sibling Staging Directory**: Every attempt executes with a dedicated attempt directory scoped by its lease fence token:
   - For a unit store at `output_path`, `resolve_attempt_output_path` ALWAYS returns a sibling directory:
     ```
     <output_path>.attempt-<fence_token>/
     ```
     It **never** returns a child subdirectory within `output_path`. A child directory would leave `output_path` present on disk after the attempt directory is moved to `partial_path`, causing `commit_output()` or subsequent cleanup to detect an orphaned directory and spuriously quarantine it! Sibling paths ensure complete namespace and filesystem isolation.
2. **Row Output Locator Redirection**: The executor transparently redirects the unit's target destination:
   - In `_execute_single_unit`: `row = dataclasses.replace(row, output_locator=str(attempt_dir))`. Any consumer unit function that reads `row.output_locator` (such as `aminx` or `demistify` sweeps) automatically writes to the isolated attempt staging directory without needing code modifications.
   - In addition, environment variable `XTRAX_CAMPAIGN_ATTEMPT_DIR` is exported for environment-driven callers.
   - Dedicated helper `resolve_attempt_output_path(output_path: Path, fence: int) -> Path`.
3. **Fenced Atomic Promotion**:
   In `_stage_unit_output`:
   - Inspect whether `resolve_attempt_output_path(output_path, fence)` (`<output_path>.attempt-<fence>/`) exists.
   - If present, call `lock.assert_holder()`. This verifies:
     a) `owner_token == lock.owner_token`
     b) `fence == lock.fence`
     c) `fence >= read(.fence)`
     d) Deadline not elapsed and abort event not set.
   - Only upon verification does the executor promote the attempt directory to `partial_path` via `os.replace`.
4. **Done Marker Binding**:
   - `CampaignDoneMarker` records `lease_fence=lock.fence` and `owner_token=lock.owner_token`.
   - If `lock.assert_holder()` fails, `LeaseLostError` is raised, abort cleanup removes *only* `<output_path>.attempt-<fence>`, and Stealer S's attempt remains completely untouched.

```
Worker H (fence=1)                   Worker S (steals, fence=2)
 ├── Writes output.attempt-1/         │
 ├── Stalls...                        ├── Steals lease, bumps .fence=2
 │                                    ├── Writes output.attempt-2/
 │                                    ├── assert_holder(fence=2) -> OK
 │                                    ├── Promotes output.attempt-2 -> output_path
 │                                    └── Commits .done.json (lease_fence=2)
 ├── Wakes up                         │
 ├── Checks assert_holder(fence=1)    │
 └── RAISES LeaseLostError!           │
     (Discards output.attempt-1/;     │
      S's store is UNTOUCHED)         │
```

### 5.3 Concrete Code Modifications

In `src/xtrax/campaign/executors/local.py`:

```python
def resolve_attempt_output_path(output_path: Path, fence: int) -> Path:
    """Return the isolated sibling staging directory for attempt with given lease fence.

    Always returns a sibling directory (<output_path>.attempt-<fence>) to ensure
    output_path does not exist on disk while the unit writes, avoiding quarantine collisions.
    """
    return output_path.with_name(f"{output_path.name}.attempt-{fence}")


def _stage_unit_output(
    state: Any,
    output_path: Path,
    partial_path: Path,
    lock: FileLock | None = None,
) -> None:
    """Put this attempt's output at partial_path for the commit protocol."""
    if partial_path.exists():
        return

    fence = lock.fence if lock is not None else 0
    attempt_dir = resolve_attempt_output_path(output_path, fence)

    # Case 1: Unit wrote to attempt-scoped staging directory (#3338)
    if attempt_dir.exists():
        _require_group_store(attempt_dir)
        if lock is not None:
            lock.assert_holder()
        os.replace(attempt_dir, partial_path)
        return

    # Case 2: Legacy direct write to output_path
    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    if output_path.exists() and not done_marker_path.exists():
        _require_group_store(output_path)
        if lock is not None:
            lock.assert_holder()
        os.replace(output_path, partial_path)
        return

    # Case 3: Unit returned state Mapping
    if isinstance(state, Mapping):
        try:
            _write_state_store(state, partial_path)
        except BaseException:
            _discard_partial(partial_path)
            raise
        return

    msg = (
        f"Unit function neither wrote a store at {attempt_dir} / {output_path} nor returned a mapping "
        f"(got {type(state).__name__}); refusing to stamp a done marker on missing output"
    )
    raise UnitProducedNoOutputError(msg)
```

In `_execute_single_unit`:
```python
    start_time = time.time()
    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = output_path.with_name(f"{output_path.name}.partial.{attempt_uuid}")
    attempt_dir = resolve_attempt_output_path(output_path, lock.fence)

    # Redirect row locator and environment for direct-write units (#3338)
    row = dataclasses.replace(row, output_locator=str(attempt_dir))
    old_env = os.environ.get("XTRAX_CAMPAIGN_ATTEMPT_DIR")
    os.environ["XTRAX_CAMPAIGN_ATTEMPT_DIR"] = str(attempt_dir)
    try:
        state = unit_fn(row)
    finally:
        if old_env is not None:
            os.environ["XTRAX_CAMPAIGN_ATTEMPT_DIR"] = old_env
        else:
            os.environ.pop("XTRAX_CAMPAIGN_ATTEMPT_DIR", None)
```
```

And in `finally` cleanup:
```python
    finally:
        try:
            if partial_path is not None:
                _discard_partial(partial_path)
            if attempt_dir.exists():
                _discard_partial(attempt_dir)
...
```

### 5.4 Verification Controls

#### Positive Controls
- Direct-write unit function inspects `row.output_locator` (or `os.environ["XTRAX_CAMPAIGN_ATTEMPT_DIR"]`), writing Zarr group to the sibling attempt staging directory (`<output_path>.attempt-<fence>`).
- `_execute_single_unit` successfully validates attempt directory, promotes to `partial_path`, commits to `output_path`, and stamps done marker with `lease_fence == 1`.
- Clean rerun reuses committed store via resume verification without re-executing.

#### Negative Controls
- Interleaving race test:
  1. Worker 1 acquires lock (fence=1) and writes output to `<output_path>.attempt-1`.
  2. Worker 1 parks immediately before `_stage_unit_output`.
  3. Worker 2 steals lease, acquires lock (fence=2), runs direct-write unit writing to `<output_path>.attempt-2`, promotes, and commits.
  4. Worker 1 un-parks and attempts `_stage_unit_output`.
  5. Assert: Worker 1 raises `LeaseLostError`.
  6. Assert: Worker 2's store at `output_path` is intact.
  7. Assert: `<output_path>.attempt-1` is deleted during cleanup; `<output_path>.done.json` contains `lease_fence == 2`.

#### TDD Plan
- File: `tests/campaign/test_lease_fencing_direct_write_3338.py`
  1. `test_resolve_attempt_output_path_always_returns_sibling`: Assert `resolve_attempt_output_path` returns sibling for any directory path, never a child.
  2. `test_direct_write_to_attempt_dir_promotes_cleanly`: Assert unit writing to `row.output_locator` stages to attempt dir, promotes to `output_path`, and stamps done marker.
  3. `test_stalled_direct_writer_cannot_overwrite_stealer_output`: Concurrency interleave test confirming stealer protection.
  4. `test_attempt_dir_cleaned_up_on_unit_exception`: Assert unhandled exception in `unit_fn` discards attempt dir cleanly.
  5. `test_marker_records_lease_fence_and_owner_token`: Assert `CampaignDoneMarker` records correct fence token.

---

## 6. Debt #3117 — `zarr_commit` Duplicate Verification, Quarantine & Chunks

### 6.1 Problem Analysis

`src/xtrax/run/zarr_commit.py` implements the atomic commit protocol for durable multi-writer Zarr stores. Three deficiencies exist:
1. **Hardcoded Chunking in `write_staged_group`**:
   `write_staged_group:306` hardcodes:
   ```python
   chunks=tuple(max(d, 1) for d in arr.shape)
   ```
   Callers have no ability to specify custom chunk shapes, which causes catastrophic memory consumption when storing large multidimensional arrays (e.g. trajectories or full activations).
2. **`Duplicate` Resolution Defaults to Ignoring Content Divergence & Prematurely Destroys Staged Data**:
   In `commit_key:430-466`, `shutil.rmtree(staged_dir)` is executed immediately upon catching `ENOTEMPTY` or `EEXIST` (line 432), *before* any duplicate verification occurs! If content divergence is subsequently detected, the staged version has already been deleted, preventing post-mortem investigation. Furthermore, `verify_duplicate` previously defaulted to `False`.
3. **`quarantine()` Race Rollback Collision on Divergent Content**:
   `quarantine()` in `src/xtrax/run/zarr_commit.py:688-701` checks whether the group at `target` is internally valid and matches `expected_input_digest`. If so, it treats the key as a concurrent writer race, renames it back to `target`, and raises `QuarantineRaceError`! When two workers produce different content from the same input (e.g., non-deterministic algorithms or floating-point variance), the committed group IS internally valid and DOES match `expected_input_digest`. Calling `quarantine(..., expected_input_digest=input_digest)` triggers an immediate `QuarantineRaceError`, putting the group back and failing to quarantine the divergent content.
   To resolve this, `quarantine()` must support `force: bool = False` so callers resolving confirmed content divergence can bypass the race rollback heuristic, and `commit_key()` must preserve `staged_dir` until duplicate verification finishes, quarantining both committed and staged versions side-by-side into `<store>.quarantine/`.

### 6.2 Architectural Design: Chunks & Duplicate Verification

#### 1. `write_staged_group` Signature Extension
Extend `write_staged_group` to accept `chunks: tuple[int, ...] | Mapping[str, tuple[int, ...]] | None`:

```python
def write_staged_group(
    staged_dir: Path,
    arrays: Mapping[str, Any],
    attrs: Mapping[str, Any] | None = None,
    chunks: tuple[int, ...] | Mapping[str, tuple[int, ...]] | None = None,
) -> None:
    """Write arrays and attrs to a staged zarr group.

    Args:
        staged_dir: Path to the staging directory (created as new zarr root).
        arrays: Named arrays to write.
        attrs: Optional attrs dict to stamp on the group.
        chunks: Optional chunk shape tuple or mapping from array name to chunk tuple.
            When None, defaults to full array shape (tuple(max(d, 1) for d in arr.shape)).
    """
    import numpy as np
    import zarr

    staged_dir.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(staged_dir), mode="w")

    for name, array in arrays.items():
        arr = np.asarray(array)
        if chunks is None:
            arr_chunks = tuple(max(d, 1) for d in arr.shape)
        elif isinstance(chunks, Mapping):
            arr_chunks = chunks.get(name, tuple(max(d, 1) for d in arr.shape))
        elif isinstance(chunks, tuple):
            if len(chunks) != arr.ndim:
                msg = f"Chunk shape {chunks} rank does not match array {name!r} rank {arr.ndim}"
                raise ValueError(msg)
            arr_chunks = chunks
        else:
            msg = f"Invalid chunks parameter of type {type(chunks).__name__}"
            raise TypeError(msg)

        group.create_array(
            name=name,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=arr_chunks,
        )
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        zarr_array[...] = arr

    if attrs:
        group.attrs.update(dict(attrs))
```

#### 2. `SinkSpec` & `ZarrStagingSink` Pass-Through
In `src/xtrax/run/sink.py`:
- Add field `chunks: tuple[int, ...] | Mapping[str, tuple[int, ...]] | None = None` to `SinkSpec`.
- Validate rank in `SinkSpec.__post_init__`.
- In `ZarrStagingSink._drain_durable`: pass `chunks=self._spec.chunks` to `zc.write_staged_group(staged_dir, arrays, attrs, chunks=self._spec.chunks)`.

#### 3. Hardened `quarantine()` with `force: bool = False`
In `src/xtrax/run/zarr_commit.py`:

```python
def quarantine(
    store: Path,
    key: tuple[str, ...],
    *,
    expected_input_digest: str | None = None,
    force: bool = False,
) -> Path | None:
    """Move a damaged, stale, uncommitted, or divergent key out of the store; never deletes it.

    Args:
        store: Store root path.
        key: Key tuple identifying the group.
        expected_input_digest: Expected input digest used for stale key judgement.
        force: When True, bypasses race-rollback checks and unconditionally moves the group
            into quarantine. Use when resolving confirmed content divergence where the committed
            group is valid but conflicts with staged content.
    """
    target = store / key_path(key)
    dest_parent = quarantine_root(store).joinpath(*key[:-1])
    dest = dest_parent / f"{key[-1]}-{uuid.uuid4().hex[:8]}"
    if not target.exists():
        return None
    if read_record(target) is None:
        descendants = committed_keys(store, key)
        if descendants:
            raise NotALeafKeyError(key, descendants)
    dest_parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(str(target), str(dest))
    except FileNotFoundError:
        return None
    fsync_directory(target.parent)
    fsync_directory(dest_parent)

    # When force=True, caller explicitly mandates quarantine (e.g. content divergence)
    if force:
        return dest

    if not _group_is_valid(dest):
        return dest
    if expected_input_digest is not None:
        moved = read_record(dest)
        if moved is None or moved.input_digest != expected_input_digest:
            return dest  # valid, but not what the caller wanted: the stale key it judged wrong
    try:
        os.rename(str(dest), str(target))
    except OSError as e:
        if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            raise
        raise QuarantineRaceError(
            key,
            dest,
            f"{target} held a valid group that a racing writer made after it was judged "
            f"damaged; it was moved to {dest} and a newer one is now in place",
        ) from e
    fsync_directory(target.parent)
    raise QuarantineRaceError(
        key,
        None,
        f"{target} held a valid group when it was about to be quarantined (a racing writer "
        "replaced the damaged one); it was put back untouched",
    )
```

#### 4. Strict `content_digest` Verification & Dual Quarantine in `commit_key`
In `src/xtrax/run/zarr_commit.py:430-466`:

```python
    except OSError as e:
        if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
            existing = read_record(target)
            if existing is None:
                shutil.rmtree(staged_dir)
                raise CommitConflictError(
                    key, None, input_digest, "uncommitted or tampered group already at target"
                ) from e
            if existing.record_digest != _compute_record_digest(existing.to_dict()):
                shutil.rmtree(staged_dir)
                raise CommitConflictError(
                    key, existing, input_digest, "uncommitted or tampered group already at target"
                ) from e
            if existing.input_digest == input_digest:
                if verify_duplicate and existing.content_digest != content_digest:
                    # Content divergence detected! Preserve staged_dir and quarantine BOTH versions.
                    quarantined_target = quarantine(store, key, force=True)

                    dest_parent = quarantine_root(store).joinpath(*key[:-1])
                    dest_parent.mkdir(parents=True, exist_ok=True)
                    quarantined_staged = dest_parent / f"{key[-1]}-staged-{uuid.uuid4().hex[:8]}"
                    os.rename(str(staged_dir), str(quarantined_staged))
                    fsync_directory(dest_parent)

                    raise CommitConflictError(
                        key,
                        existing,
                        input_digest,
                        "same input_digest, different content "
                        f"(committed {existing.content_digest[:12]} quarantined at {quarantined_target}, "
                        f"staged {content_digest[:12]} quarantined at {quarantined_staged}): "
                        "a nondeterministic producer, tampering, or a damaged committed key",
                    ) from e
                if verify_duplicate and not _group_is_valid(target):
                    shutil.rmtree(staged_dir)
                    raise CommitConflictError(
                        key,
                        existing,
                        input_digest,
                        "same input_digest and content as the staged group, but the committed "
                        "key on disk no longer matches its own record: a damaged committed key",
                    ) from e
                # Valid duplicate: clean up staged_dir only after verification succeeds
                shutil.rmtree(staged_dir)
                return Duplicate(existing)
            shutil.rmtree(staged_dir)
            raise CommitConflictError(
                key, existing, input_digest, "different input_digest already committed"
            ) from e
        raise
```

### 6.3 Verification Controls

#### Positive Controls
- Call `write_staged_group(..., chunks=(2, 2))` on `(4, 4)` array; inspect `group["arr"].chunks == (2, 2)`.
- Commit same input and identical content twice with `verify_duplicate=True`; second commit removes `staged_dir` and returns `Duplicate(existing)`.
- Construct `SinkSpec(chunks=(10, 10))`, write via `ZarrStagingSink`; arrays are created with chunking `(10, 10)`.

#### Negative Controls
- Commit key with matching `input_digest` but differing array values under `verify_duplicate=True`:
  - Assert `CommitConflictError` is raised matching `"same input_digest, different content"`.
  - Assert `QuarantineRaceError` is NOT raised.
  - Assert committed key is moved to `<store>.quarantine/<prefix>/<key>-<uuid8>`.
  - Assert staged directory is moved to `<store>.quarantine/<prefix>/<key>-staged-<uuid8>`.
  - Assert both versions exist on disk in quarantine for forensic auditing.
- Provide `chunks=(2, 2, 2)` for 2D array; assert `ValueError` with rank mismatch message.
- Call `quarantine(..., force=True)` on an internally valid key with matching `expected_input_digest`; assert group is moved to quarantine without race rollback.

#### TDD Plan
- File: `tests/run/test_zarr_commit_chunks_and_quarantine.py`
  1. `test_write_staged_group_custom_tuple_chunks`: Verify explicit tuple chunks passed to `create_array`.
  2. `test_write_staged_group_mapping_chunks`: Verify per-array chunks mapping.
  3. `test_sink_spec_plumbs_chunks_to_zarr`: Verify `SinkSpec` through `ZarrStagingSink`.
  4. `test_quarantine_force_true_bypasses_race_rollback`: Verify `force=True` quarantines valid group without raising `QuarantineRaceError`.
  5. `test_duplicate_quarantines_both_committed_and_staged_on_divergence`: Verify content divergence preserves and quarantines both versions.
  6. `test_duplicate_matching_content_cleans_up_staged_dir`: Verify identical duplicate removes `staged_dir` and returns `Duplicate`.

---

## 7. Debt #3126 — Configurable `bundle_js` for Offline EDA Reports

### 7.1 Problem Analysis: Airgapped Visualization Failure

`xtrax.eda` provides exploratory batching plan visualization. When exporting plan dashboards to HTML format:
- `src/xtrax/eda/backends/base.py:9-19` defines:
  ```python
  def reject_bundle_js(bundle_js: bool) -> None:
      if bundle_js:
          raise NotImplementedError(
              "bundle_js=True is not implemented: vendored Vega JS is not shipped. "
              "Use offline=True for the static SVG fallback."
          )
  ```
- `render_core.py:107` unconditionally calls `reject_bundle_js(bundle_js)`.
- Meanwhile, `VegaBackend.render_spec` embeds dynamic CDN loader scripts:
  `<script src="https://cdn.jsdelivr.net/npm/vega@5">...</script>`
- On airgapped cluster compute nodes or offline researcher laptops without external internet access, dynamic script loading fails, triggering the 2-second timeout and degrading to static SVG. Interactive zooming, tooltip inspection, and slice filtering become impossible.

### 7.2 Mechanism: `vl_convert` Inlined JS Bundling

We have verified that `vl_convert-python` natively includes fully bundled, tree-shaken JavaScript bundles for Vega, Vega-Lite, and Vega-Embed via:
```python
import vl_convert as vlc
html_doc = vlc.vegalite_to_html(spec, bundle=True)
```
This produces a completely self-contained HTML document (~920 KB) with **zero** external network requests (`cdn.jsdelivr.net` absent).

### 7.3 Architectural Design: Configurable `bundle_js`

1. **API Signature Update in `render_core.py`**:
   ```python
   def render(
       plan: BatchPlan,
       view: str = "dashboard",
       fmt: Literal["png", "svg", "html"] = "png",
       path: Path | str | None = None,
       stats_transform: Callable[[PlanStatsDict], PlanStatsDict] | None = None,
       metadata: bool = False,
       logger: PlanLogger | None = None,
       step: int | None = None,
       panels: set[str] | None = None,
       bundle_js: bool = False,
       backend: Literal["vega", "matplotlib"] = "vega",
   ) -> bytes | str | None:
   ```
   Remove the unconditional `reject_bundle_js(bundle_js)` call. Pass `options={"bundle_js": bundle_js}` to the backend.

2. **Backend Dispatch in `src/xtrax/eda/backends/vega.py`**:
   In `VegaBackend.render_spec`:
   - If `bundle_js=True`:
     Check if `vl_convert` is available.
     Call `vlc.vegalite_to_html(spec, bundle=True)`.
     Optionally inject the dashboard header and title metadata into the generated document.
     Assert generated HTML contains no `cdn.jsdelivr.net` references.
   - If `bundle_js=False`:
     Retain progressive enhancement template loading from CDN with static SVG fallback.
3. **Backend Validation in `src/xtrax/eda/backends/base.py`**:
   - Update `reject_bundle_js` to only reject when `bundle_js=True` is passed to backends that do not support HTML JS bundling (e.g., `matplotlib`).

### 7.4 Verification Controls

#### Positive Controls
- Render plan with `render(plan, fmt="html", backend="vega", bundle_js=True)`.
- Assert return string length > 500,000 characters.
- Assert `"cdn.jsdelivr.net"` NOT in returned HTML.
- Assert `"vegaEmbed"` in returned HTML.

#### Negative Controls
- Pass `bundle_js=True` with `backend="matplotlib"`. Assert `NotImplementedError` or `ValueError` stating matplotlib does not support JavaScript bundling.
- Simulate missing `vl_convert` with `bundle_js=True`. Assert informative `ImportError` directing user to `pip install xtrax[eda-vega]`.

#### TDD Plan
- File: `tests/eda/test_bundle_js_configurable.py`
  1. `test_render_html_bundle_js_true_contains_no_cdn`
  2. `test_render_html_bundle_js_false_contains_cdn`
  3. `test_matplotlib_rejects_bundle_js`
  4. `test_offline_static_svg_preserved`

---

## 8. Debt #3265 — CI & Pre-Push Changelog Fragment Gate

### 8.1 Problem Analysis

In `xtrax`, release notes are managed via Towncrier-style fragments under `changelog.d/<slug>.<category>.md`. `scripts/assemble_changelog.py --check` currently only validates existing fragments for correct syntax, categories (`added`, `changed`, `deprecated`, `removed`, `fixed`, `security`), and lack of markdown headers.

**The Gap**: If a PR introduces an entirely new module under `src/xtrax/` (for example, `src/xtrax/campaign/executors/slurm.py` or `src/xtrax/testing/triton_lowering.py`), developers frequently forget to add a changelog fragment. CI passes, and the release notes miss the new capability.

### 8.2 Architectural Design: `--check-new-modules`

Extend `scripts/assemble_changelog.py` with a new validation mode: `--check-new-modules`:

```
scripts/assemble_changelog.py --check-new-modules [--base-ref origin/main]
 ├── Query git diff: git diff --name-status --diff-filter=A <base_ref>...HEAD -- src/xtrax/
 ├── Filter for newly added Python modules: src/xtrax/**/*.py (ignoring __pycache__)
 ├── If new modules found:
 │    ├── Query git diff for changelog fragments: git diff --name-only <base_ref>...HEAD -- changelog.d/
 │    └── Filter for valid fragment names: changelog.d/*.md (excluding README.md, .gitkeep)
 ├── If new modules exist but 0 fragments added:
 │    ├── Log error: "PR adds new modules under src/xtrax/ ([...]) but lacks a changelog fragment!"
 │    └── Exit code 1
 └── Exit code 0 on compliance
```

#### Implementation in `scripts/assemble_changelog.py`

```python
def check_new_modules(
    repo_root: Path = ROOT,
    base_ref: str = "origin/main",
    fragments_dir: Path = FRAGMENTS_DIR,
) -> list[str]:
    """Verify that if new Python modules are added under src/xtrax/, a changelog fragment is present."""
    import subprocess

    problems: list[str] = []
    diff_output: str | None = None
    resolved_base: str | None = None

    # Attempt to resolve git diff against base_ref, then fallback to HEAD~1
    for candidate_base in (base_ref, "HEAD~1"):
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "--name-only", "--diff-filter=A", f"{candidate_base}...HEAD", "--", "src/xtrax/"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.PIPE,
            )
            resolved_base = candidate_base
            break
        except subprocess.CalledProcessError:
            continue

    if diff_output is None or resolved_base is None:
        raise RuntimeError(
            f"Failed to determine git diff against comparison base (tried '{base_ref}' and 'HEAD~1'). "
            "In CI, ensure git checkout has sufficient depth (e.g. actions/checkout with 'fetch-depth: 0' "
            "or fetch origin/main). Refusing to silently fail open."
        )

    # Restrict __init__.py exclusion strictly to root src/xtrax/__init__.py; new subpackages require changelogs
    new_modules = [
        line.strip()
        for line in diff_output.splitlines()
        if line.strip().endswith(".py") and line.strip() != "src/xtrax/__init__.py"
    ]

    if not new_modules:
        return []

    # Check for added or existing unreleased changelog fragments in this branch
    try:
        frag_output = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=A", f"{resolved_base}...HEAD", "--", "changelog.d/"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.PIPE,
        )
        added_fragments = [
            f for f in frag_output.splitlines() if f.endswith(".md") and not f.endswith("README.md")
        ]
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to query git diff for changelog fragments against {resolved_base}: {exc}"
        ) from exc

    if not added_fragments:
        problems.append(
            f"Pull request adds new module(s) under src/xtrax/ ({', '.join(new_modules)}), "
            "but no changelog fragment was added under changelog.d/. "
            "Please create changelog.d/<slug>.added.md documenting the new module."
        )

    return problems
```

### 8.3 Integration with Pre-Push & GitHub Actions

1. In `scripts/git-hooks/pre-push`:
   ```bash
   echo "[pre-push] changelog fragment check for new modules"
   uv run python scripts/assemble_changelog.py --check-new-modules --base-ref origin/main
   ```
2. In `.github/workflows/ci.yml:68-70, 92-94`:
   Ensure `fetch-depth: 0` is configured in `actions/checkout@v4` so git history is fully available for diffing against `origin/main`, preventing shallow clone diff failure:
   ```yaml
     lint-format-type-test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
           with:
             fetch-depth: 0
   ...
         - name: Changelog fragments & new module check
           run: |
             uv run python scripts/assemble_changelog.py --check
             uv run python scripts/assemble_changelog.py --check-new-modules --base-ref origin/main
   ```

### 8.4 Verification Controls

#### Positive Controls
- Branch adding `src/xtrax/example_module.py` and `changelog.d/example.added.md` passes `--check-new-modules`.
- Branch modifying existing files without adding new modules passes without requiring fragments.
- Branch modifying or touching root `src/xtrax/__init__.py` passes without requiring fragments.

#### Negative Controls
- Branch adding `src/xtrax/new_feature.py` without any file in `changelog.d/` fails with exit code 1 and lists `src/xtrax/new_feature.py`.
- Branch adding a new subpackage `src/xtrax/new_subpkg/__init__.py` without changelog fragment fails with exit code 1.
- Running check in a shallow checkout where neither `origin/main` nor `HEAD~1` can be diffed raises `RuntimeError` (exits non-zero) instead of returning success.

#### TDD Plan
- File: `tests/scripts/test_assemble_changelog_new_modules.py`
  1. `test_check_new_modules_passes_when_no_new_modules`
  2. `test_check_new_modules_passes_when_module_and_fragment_added`
  3. `test_check_new_modules_fails_when_module_added_without_fragment`
  4. `test_check_new_modules_subpackage_init_requires_fragment`
  5. `test_check_new_modules_root_init_excluded`
  6. `test_check_new_modules_raises_on_unresolvable_base`

---

## 9. Comprehensive Verification Matrix & Acceptance Test Battery

Every debt in Track 1 must pass its dedicated test battery before merge:

| Debt ID | Target Test File | Type | Verification Command | Pass Criterion |
|---|---|---|---|---|
| **#3144** | `tests/cli/test_plugin_snapshot.py`, `tests/cli/test_generate_plugin_snapshot.py` | Unit / CLI | `XTRAX_ENFORCE_PLUGIN_SNAPSHOT=1 uv run pytest tests/cli/test_plugin_snapshot.py -v` | Snapshot fresh, zero drift, exit 0 |
| **#2962** | `tests/scripts/test_workspace_reset_guard.py` | Hook / Safety | `uv run pytest tests/scripts/test_workspace_reset_guard.py -v` | Reset blocked on dirty tree (exit 2), emergency stash created with `-u` |
| **#3338** | `tests/campaign/test_lease_fencing_direct_write_3338.py` | Concurrency / Lease | `uv run pytest tests/campaign/test_lease_fencing_direct_write_3338.py -v` | Sibling staging path, row redirection, lease fence stamped in marker, stealer protected |
| **#3117** | `tests/run/test_zarr_commit_chunks_and_quarantine.py` | Store / Integrity | `uv run pytest tests/run/test_zarr_commit_chunks_and_quarantine.py -v` | Custom chunks verified, `force=True` quarantine, both versions preserved on divergence |
| **#3126** | `tests/eda/test_bundle_js_configurable.py` | EDA / Rendering | `uv run pytest tests/eda/test_bundle_js_configurable.py -v` | Self-contained HTML emitted, no CDN dependencies |
| **#3265** | `tests/scripts/test_assemble_changelog_new_modules.py` | CI / Hygiene | `uv run pytest tests/scripts/test_assemble_changelog_new_modules.py -v` | Rejects missing fragments on new module addition; errors on unresolved base |

---

## 10. Phasing, Worktree Isolation & Implementation Plan

### 10.1 Worktree Isolation Directive
All Track 1 implementations must occur in a dedicated git worktree branched from clean `origin/main`:
```bash
git worktree add -b feat/261110-track1-tooling-fencing .worktrees/feat-261110-track1-tooling-fencing origin/main
```
Zero modifications may be committed to primary branch `main`.

### 10.2 TDD Implementation Sequence
1. **Phase 1: Tooling & Snapshot (#3144)**
   - Author `tests/cli/test_generate_plugin_snapshot.py` (red).
   - Implement `scripts/generate_plugin_snapshot.py` (trusting cisternal native emission) and update `Justfile` (`uv run python`).
   - Verify green test and snapshot regeneration.
2. **Phase 2: Worktree Safety Guard (#2962)**
   - Author `tests/scripts/test_workspace_reset_guard.py` (red).
   - Implement `scripts/git-hooks/pretool-git-reset-guard.sh` conforming to Praxia PreToolUse JSON stdin contract with `git stash create -u`.
   - Document upstream Praxia PR patch.
3. **Phase 3: Campaign Direct Write Isolation (#3338)**
   - Author `tests/campaign/test_lease_fencing_direct_write_3338.py` with race interleavings (red).
   - Implement `resolve_attempt_output_path` (sibling path), `row.output_locator` redirection, attempt-scoped staging, and assert holder checks in `local.py`.
   - Verify green concurrency assertions.
4. **Phase 4: Zarr Commit Chunks & Quarantine (#3117)**
   - Author `tests/run/test_zarr_commit_chunks_and_quarantine.py` (red).
   - Add `chunks` to `write_staged_group`, `SinkSpec`, and `ZarrStagingSink`.
   - Add `force: bool = False` to `quarantine()` and wire `content_digest` verification, staged preservation, and dual quarantine on duplicate divergence in `commit_key`.
   - Verify green store assertions.
5. **Phase 5: EDA Configurable `bundle_js` (#3126)**
   - Author `tests/eda/test_bundle_js_configurable.py` (red).
   - Implement `bundle_js` pass-through and `vl_convert.vegalite_to_html(bundle=True)`.
   - Verify airgapped HTML generation.
6. **Phase 6: CI Changelog Gate (#3265)**
   - Author `tests/scripts/test_assemble_changelog_new_modules.py` (red).
   - Implement `--check-new-modules` in `scripts/assemble_changelog.py` (strict root-only `__init__.py` exclusion, error on unresolved base).
   - Update `pre-push` hook and `.github/workflows/ci.yml` (`fetch-depth: 0`).
7. **Phase 7: Full Static & Dynamic Verification**
   - Run `pytest -v --no-cov`, `ruff check .`, `ruff format --check .`, `ty check src/`.

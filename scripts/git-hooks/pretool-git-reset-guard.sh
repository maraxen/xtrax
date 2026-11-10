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

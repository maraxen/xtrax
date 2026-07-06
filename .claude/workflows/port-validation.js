// port_validation PCW stub v0.1 — base from `praxia dw emit port_validation`
// Source of truth: agent_assets/workflows/port_validation.yaml
// Regenerate base: PRAXIA_WORKFLOWS_DIR=agent_assets/workflows praxia dw emit port_validation
// Hook schema: port/docs/hook_schema_port_validation.md (flat Appendix A payload)
// Routing matrix: audit/routing.toml (domain=port rows, P5-ROUTE)
// Spec: .praxia/docs/specs/260602_pcw-dw-js-emission.md

// Dispatch metadata (informational — matched at the dispatch layer, not here):
//   keywords: port, parity, oracle, jax-port, validation, implementation
//   required_capabilities: implementation, auditing
export const meta = {
  name: "port-validation",
  description: "Implementation validation pipeline: oracle seal → spec → topo manifest → static gates → graded parity → emit → route (design §1.3)",
  whenToUse: "Use for: port, parity, oracle, jax-port, validation, implementation",
  phases: [
    { title: "P0-ORACLE", detail: "reference-vendor — seal port/reference/" },
    { title: "P1-SPEC", detail: "specification-specialist — jaxtyping contracts" },
    { title: "P1.5-TOPO", detail: "Topo Manifest artifact (not a gate)" },
    { title: "P2-STATIC", detail: "jax-purity-reviewer — jaxlint + trace count" },
    { title: "P3-PARITY", detail: "test-designer — graded T1–T5 blocking pytest" },
    { title: "P4-EMIT", detail: "graph-auditor — port_emit.py → audits.jsonl" },
    { title: "P5-ROUTE", detail: "composer-orchestrator — audit/routing.toml domain=port" },
  ],
};

const verdictLine =
  "End your reply with a line 'verdict: <label>' on its own line so the workflow can route.";
const MAX_FIX_RETRIES = 2;

// §6.3 — gated agent prompts must end with `verdict: <label>`; otherwise routing
// falls back to "advance" (silently advancing past a gate). The emitted prompts
// below append that instruction.
function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

// --- P3-PARITY hook validation (flat Appendix A — port/docs/hook_schema_port_validation.md) ---
const HOOK_SCHEMA_DOC = "port/docs/hook_schema_port_validation.md";
const ROUTING_TOML = "audit/routing.toml";

/** Unwrap optional transport wrapper; walker validates flat payload only. */
function unwrapSubagentStopHook(raw) {
  if (!raw || typeof raw !== "object") return null;
  if (raw.payload && typeof raw.payload === "object") return raw.payload;
  return raw;
}

/** SHA-256 hex of UTF-8 summary line (pytest short-summary format). */
async function sha256Utf8(line) {
  const data = new TextEncoder().encode(line);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Walker FAIL conditions (design §1.5 / hook schema doc):
 * 1) PASS claimed but pytest_exit_code != 0
 * 2) stdout_sha256 mismatch vs captured summary line
 * 3) oracle_id != port_target.toml lock
 */
async function validatePortHookPayload(flat, { oracleIdLock, stdoutSummaryLine }) {
  const required = [
    "tier_verdict",
    "port_parity_tier",
    "oracle_id",
    "pytest_nodeid",
    "pytest_exit_code",
    "stdout_sha256",
  ];
  for (const key of required) {
    if (flat[key] === undefined || flat[key] === null || flat[key] === "") {
      return { ok: false, reason: `missing field: ${key}` };
    }
  }
  if (flat.tier_verdict === "PASS" && flat.pytest_exit_code !== 0) {
    return { ok: false, reason: "PASS claimed but pytest_exit_code != 0" };
  }
  if (oracleIdLock && flat.oracle_id !== oracleIdLock) {
    return { ok: false, reason: "oracle_id mismatch vs port/port_target.toml lock" };
  }
  if (stdoutSummaryLine) {
    const expected = await sha256Utf8(stdoutSummaryLine);
    if (flat.stdout_sha256 !== expected) {
      return { ok: false, reason: "stdout_sha256 mismatch" };
    }
  }
  return { ok: true };
}

let _p3RepairCycles = 0;

const BUDGET = { maxRewinds: 6, maxCostUsd: 12.0, maxWalltimeS: 9000 };
const _budget = { rewinds: 0 };

function _checkBudget() {
  if (_budget.rewinds > BUDGET.maxRewinds) {
    throw new Error(`FAIL: max_rewinds exceeded (${_budget.rewinds}/${BUDGET.maxRewinds})`);
  }
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < 50000) {
    throw new Error("FAIL: token budget exhausted");
  }
  // Walltime guard removed — harness forbids Date.now() (breaks resume). Use budget.remaining() for cost-based limits.
  // TODO(#854-deferred, see spec §6.4): maxCostUsd is logged, not enforced —
  // no .js session cost API exists in USD denomination yet.
}

function _rewind(label) {
  _budget.rewinds++;
  log(`[budget] rewind ${_budget.rewinds}/${BUDGET.maxRewinds} \u2014 returning to ${label}`);
  _checkBudget();
}

log(`[budget] cost ceiling ${BUDGET.maxCostUsd} USD \u2014 informational only (spec §6.4 deferred)`);

// §6.2 — single labeled loop; `_jump` re-enters a prior phase on a back-edge.
let _jump = null;
outer: while (true) {
  if (_jump === null || _jump === "p0_oracle") {
    _jump = null;
    phase("P0 Oracle");
    const P0_ORACLE_PROMPT = `[role: reference-vendor] [phase: p0_oracle]\nP0-ORACLE v0.1 stub: seal port/reference/; exclusive write; baseline I/O from oracle execution.\nVerdicts: advance|needs_work.` + "\n" + verdictLine;
    const PRE_FLIGHT = "Pre-flight: run git status --short on your target files first. If the target file is already dirty, emit verdict: needs_work — do not blend edits.\n";
    let prevFix = null;
    let fixPassed = false;
    for (let attempt = 0; attempt <= MAX_FIX_RETRIES && !fixPassed; attempt++) {
      const p0OracleResult = await agent(
        PRE_FLIGHT + P0_ORACLE_PROMPT + (attempt > 0 ? `\nPrior attempt result:\n${prevFix}\n` : ""),
        {
        agentType: "reference-vendor",
        label: `fix:p0_oracle#${attempt}`,
        phase: "P0-ORACLE"
      }
      );
      const fv = extractVerdict(p0OracleResult);
      if (fv !== "needs_work") { fixPassed = true; prevFix = p0OracleResult; break; }
      prevFix = p0OracleResult;
    }
    const p0OracleVerdict = extractVerdict(prevFix ?? "");
    if (p0OracleVerdict === "needs_work") { _rewind("P0 Oracle"); _jump = "p0_oracle"; continue outer; }

  }
  if (_jump === null || _jump === "p1_spec") {
    _jump = null;
    if (p0OracleVerdict !== "advance") {
      log("skipping p1_spec — prerequisite did not advance");
      return { verdict: "skipped" };
    }
    phase("P1 Spec");
    const P1_SPEC_PROMPT = `[role: specification-specialist] [phase: p1_spec]\nTODO(#854): author the real dispatch prompt for this PCW role_step.\nVerdicts for this phase: advance|needs_revision.` + "\n" + verdictLine;
    const p1SpecResult = await agent(P1_SPEC_PROMPT, {
      agentType: "specification-specialist",
      label: "P1 Spec",
      phase: "P1 Spec",
    });
    const p1SpecVerdict = extractVerdict(p1SpecResult);
    if (p1SpecVerdict === "needs_revision") { _rewind("P1 Spec"); _jump = "p1_spec"; continue outer; }

  }
  if (_jump === null || _jump === "p1_5_topo") {
    _jump = null;
    phase("P1 5 Topo");
    const P1_5_TOPO_PROMPT = `[role: recon] [phase: p1_5_topo]\nTODO(#854): author the real dispatch prompt for this PCW role_step.\nVerdicts for this phase: advance.` + "\n" + verdictLine;
    const p15TopoResult = await agent(P1_5_TOPO_PROMPT, {
      agentType: "recon",
      label: "P1 5 Topo",
      phase: "P1 5 Topo",
    });
    const p15TopoVerdict = extractVerdict(p15TopoResult);

  }
  if (_jump === null || _jump === "p2_static") {
    _jump = null;
    phase("P2 Static");
    const P2_STATIC_PROMPT = `[role: jax-purity-reviewer] [phase: p2_static]\nTODO(#854): author the real dispatch prompt for this PCW role_step.\nVerdicts for this phase: advance|needs_work.` + "\n" + verdictLine;
    const p2StaticResult = await agent(P2_STATIC_PROMPT, {
      agentType: "jax-purity-reviewer",
      label: "P2 Static",
      phase: "P2 Static",
    });
    const p2StaticVerdict = extractVerdict(p2StaticResult);
    if (p2StaticVerdict === "needs_work") { _rewind("P2 Static"); _jump = "p2_static"; continue outer; }

  }
  if (_jump === null || _jump === "p3_parity") {
    _jump = null;
    phase("P3-PARITY");
    const P3_PARITY_PROMPT = `[role: test-designer] [phase: p3_parity]\nP3-PARITY v0.1 stub: run blocking T1→T5 pytest in port/tests/.\nEmit subagent-stop flat payload per ${HOOK_SCHEMA_DOC}.\nVerdicts: advance|port_repair|human_escalation.` + "\n" + verdictLine;
    let p3ParityResult = null;
    let p3ParityVerdict = "advance";
    for (let attempt = 0; attempt <= MAX_FIX_RETRIES; attempt++) {
      p3ParityResult = await agent(
        P3_PARITY_PROMPT +
          (attempt > 0 ? `\nPrior parity attempt:\n${p3ParityResult}\n` : "") +
          `\nOn tier completion, include hook JSON (flat Appendix A). Walker validates per ${HOOK_SCHEMA_DOC}.`,
        {
          agentType: "test-designer",
          label: `P3-PARITY#${attempt}`,
          phase: "P3-PARITY",
        }
      );
      p3ParityVerdict = extractVerdict(p3ParityResult);
      // v0.1: if agent embeds hook JSON, validate when parseable
      const hookMatch = String(p3ParityResult ?? "").match(/```json\s*([\s\S]*?)\s*```/);
      if (hookMatch) {
        try {
          const raw = JSON.parse(hookMatch[1]);
          const flat = unwrapSubagentStopHook(raw);
          const check = await validatePortHookPayload(flat, {
            oracleIdLock: args?.oracle_id ?? null,
            stdoutSummaryLine: args?.stdout_summary_line ?? null,
          });
          if (!check.ok) {
            log(`[P3-PARITY] hook validation FAIL: ${check.reason} (see ${HOOK_SCHEMA_DOC})`);
            p3ParityVerdict = "port_repair";
          }
        } catch (e) {
          log(`[P3-PARITY] hook JSON parse skipped: ${e}`);
        }
      }
      if (p3ParityVerdict !== "port_repair") break;
      if (attempt >= MAX_FIX_RETRIES) {
        log(`[P3-PARITY] repair cycles exhausted (${MAX_FIX_RETRIES}); escalate`);
        p3ParityVerdict = "human_escalation";
        break;
      }
      _p3RepairCycles++;
      log(`[P3-PARITY] port-repair cycle ${_p3RepairCycles}/${MAX_FIX_RETRIES}`);
    }
    if (p3ParityVerdict === "port_repair") { _rewind("P3-PARITY"); _jump = "p3_parity"; continue outer; }
    if (p3ParityVerdict === "human_escalation") {
      log(`[route] human_escalation_reason — jumping to P5-ROUTE (${ROUTING_TOML})`);
      _jump = "p5_route";
      continue outer;
    }

  }
  if (_jump === null || _jump === "p4_emit") {
    _jump = null;
    phase("P4 Emit");
    const P4_EMIT_PROMPT = `[role: graph-auditor] [phase: p4_emit]\nTODO(#854): author the real dispatch prompt for this PCW role_step.\nVerdicts for this phase: advance|needs_work.` + "\n" + verdictLine;
    const p4EmitResult = await agent(P4_EMIT_PROMPT, {
      agentType: "graph-auditor",
      label: "P4 Emit",
      phase: "P4 Emit",
    });
    const p4EmitVerdict = extractVerdict(p4EmitResult);
    if (p4EmitVerdict === "needs_work") { _rewind("P3 Parity"); _jump = "p3_parity"; continue outer; }

  }
  if (_jump === null || _jump === "p5_route") {
    _jump = null;
    phase("P5 Route");
    const P5_ROUTE_PROMPT = `[role: composer-orchestrator] [phase: p5_route]\nP5-ROUTE v0.1 stub: apply ${ROUTING_TOML} domain=port CC5 rows.\nWrite .praxia/port/routing/<wave_id>_<finding_id>.toml.\nVerdicts: advance.` + "\n" + verdictLine;
    const p5RouteResult = await agent(P5_ROUTE_PROMPT, {
      agentType: "composer-orchestrator",
      label: "P5 Route",
      phase: "P5 Route",
    });
    const p5RouteVerdict = extractVerdict(p5RouteResult);
    if (p5RouteVerdict === "advance") { break outer; }

  }
  const _finalVerdict = p5RouteVerdict;
  return { workflow: meta.name, verdict: _finalVerdict, rewinds: _budget.rewinds };
  break outer;
}

"""D8 Structure-complexity gate: complexipy max + ruff C901/PLR091x (N2.8 / #1588)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.devtools.baseline import (
    DEFAULT_BASELINE_PATH,
    evaluate_metric,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.devtools.emit import append_finding, emit_metric_finding

COGNITIVE_METRIC = "structure.cognitive_complexity_max"
RUFF_METRIC = "structure.ruff_complexity_violation_count"
DIMENSION = "structure_complexity"
COGNITIVE_RULE_ID = "structure_complexity.cognitive"
RUFF_RULE_ID = "structure_complexity.ruff"


@dataclass(frozen=True, slots=True)
class ComplexityHit:
    qualname: str
    file_line: str
    complexity: int


@dataclass(frozen=True, slots=True)
class RuffHit:
    rule_id: str
    file_line: str
    symbol: str
    message: str


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    cognitive_complexity_max: float
    ruff_violation_count: int
    cognitive_hits: tuple[ComplexityHit, ...]
    ruff_hits: tuple[RuffHit, ...]
    findings_emitted: int
    baseline_updated: bool
    cognitive_metric_key: str = COGNITIVE_METRIC
    ruff_metric_key: str = RUFF_METRIC


def _file_line(path: Path, lineno: int) -> str:
    return f"{path}:{lineno}"


def parse_complexipy_json(
    data: list[dict[str, Any]],
    *,
    root: Path,
    threshold: int,
) -> tuple[float, list[ComplexityHit]]:
    """Parse complexipy JSON report -> (max score, hits above ``threshold``)."""
    if not data:
        return 0.0, []

    max_score = 0.0
    hits: list[ComplexityHit] = []
    for entry in data:
        complexity = int(entry["complexity"])
        max_score = max(max_score, float(complexity))
        if complexity <= threshold:
            continue
        rel_path = str(entry["path"])
        file_path = (root / rel_path).resolve()
        qualname = str(entry["function_name"])
        line_start = int(entry.get("line_start", 1))
        hits.append(
            ComplexityHit(
                qualname=qualname,
                file_line=_file_line(file_path, line_start),
                complexity=complexity,
            )
        )
    return max_score, hits


def parse_ruff_complexity_json(
    data: list[dict[str, Any]],
) -> tuple[int, list[RuffHit]]:
    """Parse ruff JSON diagnostics for C901/PLR0912/PLR0915."""
    hits: list[RuffHit] = []
    for entry in data:
        code = str(entry.get("code", ""))
        if code not in {"C901", "PLR0912", "PLR0915"}:
            continue
        location = entry["location"]
        filename = str(entry["filename"])
        lineno = int(location["row"])
        message = str(entry.get("message", ""))
        symbol = ""
        if "`" in message:
            symbol = message.split("`", 2)[1]
        hits.append(
            RuffHit(
                rule_id=code,
                file_line=f"{filename}:{lineno}",
                symbol=symbol,
                message=message,
            )
        )
    return len(hits), hits


def _workspace_root(start: Path) -> Path:
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return current


def _complexipy_scan_root(target: Path) -> Path:
    resolved = target.resolve()
    if resolved.is_file():
        return resolved.parent
    return resolved.parent


def run_complexipy_scan(
    root: Path,
    *,
    threshold: int = 15,
) -> tuple[float, list[ComplexityHit]]:
    """Run complexipy; return max cognitive score and hits above threshold."""
    resolved_root = root.resolve()
    with tempfile.NamedTemporaryFile(
        suffix=".json",
        prefix="complexipy-",
        delete=False,
    ) as handle:
        report_path = Path(handle.name)

    cmd = [
        "uv",
        "run",
        "complexipy",
        str(resolved_root),
        "--output-format",
        "json",
        "--output",
        str(report_path),
        "--ignore-complexity",
        f"--max-complexity-allowed={threshold}",
    ]
    workspace = _workspace_root(resolved_root)
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if not report_path.is_file():
            msg = (
                "complexipy JSON report missing; "
                f"exit={result.returncode}: {(result.stdout + result.stderr).strip()}"
            )
            raise RuntimeError(msg)
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        scan_root = _complexipy_scan_root(resolved_root)
        if isinstance(raw, dict) and "files" in raw:
            flattened: list[dict[str, Any]] = []
            for file_entry in raw["files"]:
                file_path = str(file_entry["path"])
                for fn in file_entry.get("functions", []):
                    flattened.append(
                        {
                            "path": file_path,
                            "function_name": fn["name"],
                            "complexity": fn["complexity"],
                            "line_start": fn.get("line_start", 1),
                        }
                    )
            raw = flattened
        if not isinstance(raw, list):
            msg = f"unexpected complexipy JSON shape: {type(raw).__name__}"
            raise RuntimeError(msg)
        return parse_complexipy_json(raw, root=scan_root, threshold=threshold)
    finally:
        report_path.unlink(missing_ok=True)


def run_ruff_complexity_scan(root: Path) -> tuple[int, list[RuffHit]]:
    """Run ruff C901/PLR0912/PLR0915 and return violation count + hits."""
    resolved_root = root.resolve()
    cmd = [
        "uv",
        "run",
        "ruff",
        "check",
        str(resolved_root),
        "--select",
        "C901,PLR0912,PLR0915",
        "--output-format",
        "json",
    ]
    result = subprocess.run(
        cmd,
        cwd=_workspace_root(resolved_root),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    if not stdout:
        return 0, []
    data = json.loads(stdout)
    if not isinstance(data, list):
        msg = f"unexpected ruff JSON shape: {type(data).__name__}"
        raise RuntimeError(msg)
    return parse_ruff_complexity_json(data)


def run_structure_complexity_gate(
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    root: Path | None = None,
    *,
    cognitive_threshold: int = 15,
    cognitive_ceiling: int = 25,
    run_id: str | None = None,
    write_baseline: bool = True,
) -> GateResult:
    """Run complexipy + ruff scans, emit findings, evaluate baseline ratchet."""
    resolved_root = root or Path.cwd()
    cognitive_max, cognitive_hits = run_complexipy_scan(
        resolved_root,
        threshold=cognitive_threshold,
    )
    ruff_count, ruff_hits = run_ruff_complexity_scan(resolved_root)

    emitted = 0
    record = emit_metric_finding(
        dim=DIMENSION,
        severity="info",
        file_line=str(resolved_root / "src" / "xtrax"),
        evidence=f"cognitive complexity max {cognitive_max:.0f}",
        rule_id=COGNITIVE_RULE_ID,
        symbol_qualname="",
        payload={
            "violation_kind": "cognitive_complexity_max",
            "cognitive_complexity_max": cognitive_max,
            "cognitive_threshold": cognitive_threshold,
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)
    emitted += 1

    record = emit_metric_finding(
        dim=DIMENSION,
        severity="info",
        file_line=str(resolved_root / "src" / "xtrax"),
        evidence=f"ruff complexity violations {ruff_count}",
        rule_id=RUFF_RULE_ID,
        symbol_qualname="",
        payload={
            "violation_kind": "ruff_complexity_violation_count",
            "ruff_complexity_violation_count": ruff_count,
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)
    emitted += 1

    for hit in cognitive_hits:
        if hit.complexity <= cognitive_ceiling:
            continue
        record = emit_metric_finding(
            dim=DIMENSION,
            severity="minor",
            file_line=hit.file_line,
            evidence=(
                f"{hit.qualname} cognitive complexity {hit.complexity} "
                f"(ceiling {cognitive_ceiling})"
            ),
            rule_id=COGNITIVE_RULE_ID,
            symbol_qualname=hit.qualname,
            payload={
                "violation_kind": "cognitive_complexity",
                "complexity": hit.complexity,
                "cognitive_ceiling": cognitive_ceiling,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    for hit in ruff_hits:
        record = emit_metric_finding(
            dim=DIMENSION,
            severity="minor",
            file_line=hit.file_line,
            evidence=hit.message,
            rule_id=hit.rule_id,
            symbol_qualname=hit.symbol,
            payload={
                "violation_kind": "ruff_complexity",
                "rule_id": hit.rule_id,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    baseline = load_baseline(path=baseline_path)
    passes_cognitive, update_cognitive = evaluate_metric(
        baseline,
        COGNITIVE_METRIC,
        cognitive_max,
    )
    passes_ruff, update_ruff = evaluate_metric(
        baseline,
        RUFF_METRIC,
        float(ruff_count),
    )
    passed = passes_cognitive and passes_ruff

    baseline_updated = False
    if passed and write_baseline and (update_cognitive or update_ruff):
        tightened = baseline
        if update_cognitive:
            tightened = update_metric(
                tightened,
                COGNITIVE_METRIC,
                cognitive_max,
                "minimize",
            )
        if update_ruff:
            tightened = update_metric(
                tightened,
                RUFF_METRIC,
                float(ruff_count),
                "minimize",
            )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passed,
        cognitive_complexity_max=cognitive_max,
        ruff_violation_count=ruff_count,
        cognitive_hits=tuple(cognitive_hits),
        ruff_hits=tuple(ruff_hits),
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
    )

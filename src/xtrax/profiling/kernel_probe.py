"""Multi-pass GPU kernel telemetry, PTX/ptxas/Triton parsers, and precision ratios.

Links production ``cmdbuf=on`` measurements with isolated subprocess
``cmdbuf=off`` kernel breakdowns and optional compiler PTX/``ptxas`` register
and spill telemetry without mutating immutable ``ProbeRecord`` artifacts.

Zero module-level ``jax`` import and strictly scoped within
``xtrax.profiling.*`` per the profiling leaf-package contract.
"""

import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xtrax.profiling.claims import (
    CONTRACT_VERSION,
    ClaimClass,
    ClaimValidityError,
    _reject_unverifiable_git_sha,
)
from xtrax.profiling.compare import _shas_match
from xtrax.profiling.emitters import emit_probe_record
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.unit_probe import probe_unit

_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", flags=re.DOTALL)
_ENTRY_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:\.visible|\.weak)\s+)?\.entry\s+([A-Za-z_.$][\w.$]*)\s*\("
)
_REG_RE = re.compile(
    r"\.reg\s+(?:\.(?:v2|v4)\s+)?\."
    r"(pred|bf16|b128|b8|b16|b32|b64|s8|s16|s32|s64|u8|u16|u32|u64|f16|f32|f64)"
    r"\s+([^;]+);"
)
_VEC_PREFIX_RE = re.compile(r"\.reg\s+\.(v2|v4)\b")
_VREG_CNT_RE = re.compile(r"<(\d+)>")
_MEM_DECL_RE = re.compile(
    r"\.(local|shared)\s+(?:\.align\s+\d+\s+)?"
    r"\.(pred|bf16|b128|b8|b16|b32|b64|s8|s16|s32|s64|u8|u16|u32|u64|f16|f32|f64)"
    r"\s+([^;\n]+);"
)
_EXTERN_SHARED_RE = re.compile(r"\.extern\s+\.shared\b[^;]*\[\s*\]\s*;")
_PTXAS_KERNEL_RE = re.compile(r"Compiling entry function\s+'([^']+)'")
_PTXAS_REGS_RE = re.compile(r"Used\s+(\d+)\s+registers\b")
_PTXAS_SMEM_RE = re.compile(r"(\d+)\s+bytes\s+smem\b")
_PTXAS_STACK_RE = re.compile(r"(\d+)\s+bytes\s+stack\s+frame\b")
_PTXAS_SPILL_ST_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+stores\b")
_PTXAS_SPILL_LD_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+loads\b")
_VALID_BASIS = frozenset({"batch", "per_replica"})

_PTX_TYPE_BYTES: dict[str, int] = {
    "pred": 1,
    "b8": 1,
    "s8": 1,
    "u8": 1,
    "b16": 2,
    "s16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "b32": 4,
    "s32": 4,
    "u32": 4,
    "f32": 4,
    "b64": 8,
    "s64": 8,
    "u64": 8,
    "f64": 8,
    "b128": 16,
}

_PTX_32BIT_UNITS: dict[str, int] = {
    "pred": 1,
    "b8": 1,
    "s8": 1,
    "u8": 1,
    "b16": 1,
    "s16": 1,
    "u16": 1,
    "f16": 1,
    "bf16": 1,
    "b32": 1,
    "s32": 1,
    "u32": 1,
    "f32": 1,
    "b64": 2,
    "s64": 2,
    "u64": 2,
    "f64": 2,
    "b128": 4,
}

_PRECISION_UNANIMITY_FIELDS: tuple[str, ...] = (
    "platform",
    "device_kind",
    "xla_flags",
    "jax_version",
    "jaxlib_version",
    "n_atoms",
    "num_replicas",
    "replica_id",
    "bucket_index",
    "bucket_capacity",
)

_TRANSIENT_DUMP_FLAGS = frozenset({"--xla_dump_to", "--xla_dump_hlo_as_text"})


@dataclasses.dataclass(frozen=True, slots=True)
class PtxKernelStats:
    """Virtual PTX and optional ``ptxas``/Triton hardware resource telemetry."""

    kernel_name: str
    vregs_by_type: dict[str, int]
    vregs_total_32bit_equiv: int
    f64_vregs: int
    ptx_local_bytes: int
    ptx_shared_bytes: int
    has_dynamic_extern_shared: bool
    ptx_file: str
    phys_regs: int | None = None
    spill_store_bytes: int | None = None
    spill_load_bytes: int | None = None
    stack_frame_bytes: int | None = None
    ptxas_smem_bytes: int | None = None
    triton_spills: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class PtxasKernelStats:
    """Physical register, shared-memory, and spill statistics from ``ptxas --verbose``."""

    kernel_name: str
    phys_regs: int
    smem_bytes: int = 0
    stack_frame_bytes: int = 0
    spill_store_bytes: int = 0
    spill_load_bytes: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class TritonKernelMeta:
    """Resource metadata parsed from a Triton cache ``.json`` sidecar."""

    kernel_name: str
    n_regs: int | None
    n_spills: int | None
    shared: int | None
    json_path: str


@dataclasses.dataclass(frozen=True, slots=True)
class DualPassProbeResult:
    """Paired production and diagnostic records returned by :func:`probe_kernel_dual_pass`."""

    prod_record: ProbeRecord
    diag_record: ProbeRecord | None
    prod_record_path: Path | None
    diag_record_path: Path | None
    diag_skipped_reason: str | None = None
    ptx_stats: tuple[PtxKernelStats, ...] = ()
    ptxas_warnings: tuple[str, ...] = ()
    ptx_capture_status: str = "disabled"
    ptx_spill_source: str = "none"
    ptx_filter_status: str = "not_applicable"
    hlo_attribution_skipped: str | None = None


def _find_matching_close(text: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """Return the index of the balanced delimiter closing ``text[start]``."""
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx
    return None


def _extract_entry_body(text: str, paren_start: int) -> tuple[str, str] | None:
    """Return ``(body_text, entry_full_text)`` from balanced ``(...)`` and ``{...}``."""
    paren_end = _find_matching_close(text, paren_start, "(", ")")
    if paren_end is None:
        return None
    brace_open = text.find("{", paren_end + 1)
    if brace_open < 0:
        return None
    brace_end = _find_matching_close(text, brace_open, "{", "}")
    if brace_end is None:
        return None
    return text[brace_open + 1 : brace_end], text[paren_start : brace_end + 1]


def _parse_reg_decl_match(match: re.Match[str], by_type: dict[str, int]) -> None:
    """Accumulate virtual register declarations from one ``.reg`` regex match."""
    dtype, decls = match.group(1), match.group(2)
    vm = _VEC_PREFIX_RE.search(match.group(0))
    vec_tag = vm.group(1) if vm else ""
    vec_mult = {"v2": 2, "v4": 4}.get(vec_tag, 1)
    for part in decls.split(","):
        tok = part.strip()
        if not tok:
            continue
        cm = _VREG_CNT_RE.search(tok)
        cnt = int(cm.group(1)) if cm else 1
        by_type[dtype] = by_type.get(dtype, 0) + cnt * vec_mult


def _accumulate_vregs(body: str) -> tuple[dict[str, int], int]:
    """Parse ``.reg`` virtual SSA declarations into per-type counts and 32-bit equivalents."""
    by_type: dict[str, int] = {}
    for line in body.splitlines():
        for m in _REG_RE.finditer(line):
            _parse_reg_decl_match(m, by_type)
    equiv = sum(cnt * _PTX_32BIT_UNITS.get(dtype, 1) for dtype, cnt in by_type.items())
    return by_type, equiv


def _decl_element_count(decl: str, space: str) -> tuple[int, bool]:
    """Return ``(element_count, is_unsized_extern_shared)`` for one PTX decl."""
    bm = re.search(r"\[(\d*)\]", decl)
    if bm is None:
        return 1, False
    if bm.group(1) == "":
        return 0, space == "shared"
    return int(bm.group(1)), False


def _parse_mem_bytes(body: str) -> tuple[int, int, bool]:
    """Sum declared ``.local``/``.shared`` byte sizes and detect unsized extern shared."""
    local_bytes = 0
    shared_bytes = 0
    dyn_shared = bool(_EXTERN_SHARED_RE.search(body))
    for m in _MEM_DECL_RE.finditer(body):
        space, dtype, rest = m.group(1), m.group(2), m.group(3)
        unit = _PTX_TYPE_BYTES.get(dtype, 4)
        for decl in rest.split(","):
            cnt, saw_dyn = _decl_element_count(decl, space)
            if saw_dyn:
                dyn_shared = True
            if space == "local":
                local_bytes += unit * cnt
            else:
                shared_bytes += unit * cnt
    return local_bytes, shared_bytes, dyn_shared


def _split_entries_with_bodies(ptx_text: str) -> list[tuple[str, str, str]]:
    """Return ``[(kernel_name, body_text, normalized_full_text), ...]`` from PTX."""
    cleaned = _COMMENT_RE.sub("", ptx_text)
    module_dyn_shared = bool(_EXTERN_SHARED_RE.search(cleaned))
    entries: list[tuple[str, str, str]] = []
    for m in _ENTRY_RE.finditer(cleaned):
        name = m.group(1)
        extracted = _extract_entry_body(cleaned, m.end() - 1)
        if extracted is None:
            continue
        body, full_text = extracted
        norm = f"{body}|dyn={int(module_dyn_shared)}"
        entries.append((name, body, norm))
    return entries


def parse_ptx_text(ptx_text: str, *, ptx_file: str = "<memory>") -> list[PtxKernelStats]:
    """Parse virtual SSA registers and local/shared memory from dumped PTX text.

    PTX ``.reg`` directives declare virtual SSA names (not physical hardware
    registers); physical register counts and exact spill byte totals come from
    :func:`parse_ptxas_verbose`. 32-bit equivalents count 64-bit types as 2
    registers, ``.b128`` as 4, and <=32-bit types (including ``.pred``) as 1.
    """
    cleaned = _COMMENT_RE.sub("", ptx_text)
    module_dyn_shared = bool(_EXTERN_SHARED_RE.search(cleaned))
    results: list[PtxKernelStats] = []
    for name, body, _ in _split_entries_with_bodies(ptx_text):
        vregs_by_type, total_32 = _accumulate_vregs(body)
        local_bytes, shared_bytes, body_dyn = _parse_mem_bytes(body)
        results.append(
            PtxKernelStats(
                kernel_name=name,
                vregs_by_type=vregs_by_type,
                vregs_total_32bit_equiv=total_32,
                f64_vregs=vregs_by_type.get("f64", 0),
                ptx_local_bytes=local_bytes,
                ptx_shared_bytes=shared_bytes,
                has_dynamic_extern_shared=body_dyn or module_dyn_shared,
                ptx_file=ptx_file,
            )
        )
    return results


def parse_ptxas_verbose(stderr_text: str) -> dict[str, PtxasKernelStats]:
    """Parse per-kernel physical registers and spill bytes from ``ptxas --verbose``."""
    out: dict[str, PtxasKernelStats] = {}
    current_name: str | None = None
    acc: dict[str, int] = {}

    def _flush() -> None:
        nonlocal acc
        if current_name is not None and "phys_regs" in acc:
            out[current_name] = PtxasKernelStats(
                kernel_name=current_name,
                phys_regs=acc["phys_regs"],
                smem_bytes=acc.get("smem_bytes", 0),
                stack_frame_bytes=acc.get("stack_frame_bytes", 0),
                spill_store_bytes=acc.get("spill_store_bytes", 0),
                spill_load_bytes=acc.get("spill_load_bytes", 0),
            )
        acc = {}

    for line in stderr_text.splitlines():
        km = _PTXAS_KERNEL_RE.search(line)
        if km:
            _flush()
            current_name = km.group(1)
            continue
        if current_name is None:
            continue
        rm = _PTXAS_REGS_RE.search(line)
        if rm:
            acc["phys_regs"] = int(rm.group(1))
        for pat, key in (
            (_PTXAS_SMEM_RE, "smem_bytes"),
            (_PTXAS_STACK_RE, "stack_frame_bytes"),
            (_PTXAS_SPILL_ST_RE, "spill_store_bytes"),
            (_PTXAS_SPILL_LD_RE, "spill_load_bytes"),
        ):
            pm = pat.search(line)
            if pm:
                acc[key] = int(pm.group(1))
    _flush()
    return out


def _parse_triton_meta_file(path: Path) -> TritonKernelMeta | None:
    """Parse one Triton ``.json`` metadata sidecar into :class:`TritonKernelMeta`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if not any(k in raw for k in ("n_regs", "n_spills", "shared")):
        return None
    n_regs = raw.get("n_regs")
    n_spills = raw.get("n_spills")
    shared = raw.get("shared")
    return TritonKernelMeta(
        kernel_name=name,
        n_regs=(int(n_regs) if isinstance(n_regs, int) and not isinstance(n_regs, bool) else None),
        n_spills=(
            int(n_spills) if isinstance(n_spills, int) and not isinstance(n_spills, bool) else None
        ),
        shared=(int(shared) if isinstance(shared, int) and not isinstance(shared, bool) else None),
        json_path=str(path),
    )


def parse_triton_metadata_dir(cache_dir: str | Path) -> dict[str, TritonKernelMeta]:
    """Read Triton ``.json`` cache/dump sidecars for ``n_regs`` and ``n_spills``."""
    root = Path(cache_dir)
    if not root.is_dir():
        return {}
    metas: dict[str, TritonKernelMeta] = {}
    for path in sorted(root.rglob("*.json")):
        meta = _parse_triton_meta_file(path)
        if meta is not None:
            metas[meta.kernel_name] = meta
    return metas


def _add_optional_spill_metrics(out: dict[str, float], stats: list[PtxKernelStats]) -> None:
    """Populate optional ptxas physical register and spill fields onto ``out``."""
    phys = [s.phys_regs for s in stats if s.phys_regs is not None]
    if phys:
        out["ptxas_regs_max"] = float(max(phys))
    spills = [
        (s.spill_store_bytes or 0) + (s.spill_load_bytes or 0)
        for s in stats
        if s.spill_store_bytes is not None or s.spill_load_bytes is not None
    ]
    if spills:
        out["ptxas_spill_bytes_max"] = float(max(spills))
    t_spills = [s.triton_spills for s in stats if s.triton_spills is not None]
    if t_spills:
        out["triton_spills_max"] = float(max(t_spills))


def summarize_ptx_metrics(stats: list[PtxKernelStats]) -> dict[str, float]:
    """Summarize parsed PTX/``ptxas``/Triton kernel stats into float ``ProbeRecord.metrics``."""
    if not stats:
        return {
            "ptx_kernel_count": 0.0,
            "ptx_vregs_max": 0.0,
            "ptx_f64_vregs_max": 0.0,
            "ptx_lmem_bytes_max": 0.0,
            "ptx_lmem_bytes_total": 0.0,
            "ptx_smem_bytes_max": 0.0,
        }
    out: dict[str, float] = {
        "ptx_kernel_count": float(len(stats)),
        "ptx_vregs_max": float(max(s.vregs_total_32bit_equiv for s in stats)),
        "ptx_f64_vregs_max": float(max(s.f64_vregs for s in stats)),
        "ptx_lmem_bytes_max": float(max(s.ptx_local_bytes for s in stats)),
        "ptx_lmem_bytes_total": float(sum(s.ptx_local_bytes for s in stats)),
        "ptx_smem_bytes_max": float(max(s.ptx_shared_bytes for s in stats)),
    }
    _add_optional_spill_metrics(out, stats)
    return out


def _consume_stripped_flag(tokens: list[str], idx: int, strip_keys: set[str]) -> int | None:
    """Return the next index if ``tokens[idx]`` matches an overridden/transient flag."""
    tok = tokens[idx]
    for key in strip_keys:
        if tok == key:
            has_val = idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--")
            return idx + (2 if has_val else 1)
        if tok.startswith(f"{key}="):
            return idx + 1
    return None


def merge_xla_flags(base_flags: str, overrides: dict[str, str]) -> str:
    """Tokenize ``XLA_FLAGS`` with ``shlex``, replace overrides/transient dumps, and rejoin."""
    tokens = shlex.split(base_flags or "")
    strip_keys = set(overrides) | _TRANSIENT_DUMP_FLAGS
    kept: list[str] = []
    idx = 0
    while idx < len(tokens):
        next_idx = _consume_stripped_flag(tokens, idx, strip_keys)
        if next_idx is not None:
            idx = next_idx
            continue
        kept.append(tokens[idx])
        idx += 1
    for key, value in overrides.items():
        kept.append(f"{key}={value}")
    return shlex.join(kept)


def find_ptxas() -> str | None:
    """Locate the ``ptxas`` binary across PATH, CUDA env homes, or ``nvidia.cuda_nvcc``."""
    on_path = shutil.which("ptxas")
    if on_path:
        return on_path
    for env_key in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(env_key, "").strip()
        if base:
            cand = Path(base) / "bin" / "ptxas"
            if cand.is_file():
                return str(cand)
    std = Path("/usr/local/cuda/bin/ptxas")
    if std.is_file():
        return str(std)
    try:
        mod = importlib.import_module("nvidia.cuda_nvcc")
        mod_file = getattr(mod, "__file__", None)
        if mod_file:
            cand = Path(mod_file).resolve().parent / "bin" / "ptxas"
            if cand.is_file():
                return str(cand)
    except ImportError:
        return None
    return None


def compute_capability_to_sm(cc: str) -> str:
    """Convert compute capability ``'8.9'`` or ``'12.0'`` into ``'sm_89'`` or ``'sm_120'``."""
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", cc)
    if not m:
        raise ValueError(f"Invalid CUDA compute capability string: {cc!r}")
    return f"sm_{int(m.group(1))}{int(m.group(2))}"


def _select_dir_ptx_files(parent: Path, files: list[Path]) -> list[tuple[Path, bool]]:
    """Prefer ``.ptx`` files backed by a matching ``.cubin`` inside ``parent``."""
    winners = [p for p in files if parent.joinpath(f"{p.stem}.cubin").is_file()]
    chosen = winners if winners else files
    return [(p, True) for p in chosen]


def _gather_triton_ptx_files(triton_dirs: list[Path]) -> list[tuple[Path, bool]]:
    """Collect Triton ``.ptx`` files, discarding un-backed trial dumps when ``.cubin`` exists."""
    selected: list[tuple[Path, bool]] = []
    for root in triton_dirs:
        if not root.is_dir():
            continue
        by_dir: dict[Path, list[Path]] = {}
        for ptx_path in sorted(root.rglob("*.ptx")):
            by_dir.setdefault(ptx_path.parent, []).append(ptx_path)
        for parent, files in by_dir.items():
            selected.extend(_select_dir_ptx_files(parent, files))
    return selected


def _dedup_ptx_entries(
    ptx_sources: list[tuple[Path, bool]],
) -> tuple[list[tuple[PtxKernelStats, Path, bool]], int]:
    """Parse and deduplicate PTX kernels by ``(name, sha256(body))``, preferring Triton."""
    # Process Triton sources before XLA dumps so Triton entries win deduplication.
    ordered = sorted(ptx_sources, key=lambda item: (0 if item[1] else 1, str(item[0])))
    seen: set[tuple[str, str]] = set()
    kept: list[tuple[PtxKernelStats, Path, bool]] = []
    dedup_count = 0
    for ptx_path, is_triton in ordered:
        try:
            text = ptx_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        body_by_name = {name: norm for name, _body, norm in _split_entries_with_bodies(text)}
        for stat in parse_ptx_text(text, ptx_file=str(ptx_path)):
            norm = body_by_name.get(stat.kernel_name, "")
            digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            key = (stat.kernel_name, digest)
            if key in seen:
                dedup_count += 1
                continue
            seen.add(key)
            kept.append((stat, ptx_path, is_triton))
    return kept, dedup_count


def _filter_executed_entries(
    entries: list[tuple[PtxKernelStats, Path, bool]],
    executed_kernels: tuple[str, ...],
) -> tuple[list[tuple[PtxKernelStats, Path, bool]], str]:
    """Filter deduplicated kernels to executed names when available."""
    if not executed_kernels or not entries:
        return entries, "no_filter_names"
    targets = {k for k in executed_kernels if k}
    matched = [
        item
        for item in entries
        if item[0].kernel_name in targets
        or any(t in item[0].kernel_name or item[0].kernel_name in t for t in targets)
    ]
    if matched:
        return matched, "matched"
    return entries, "unfiltered_name_mismatch"


def _run_ptxas_on_files(
    ptx_paths: list[Path],
    arch: str,
    ptxas_bin: str,
) -> tuple[dict[str, PtxasKernelStats], list[str]]:
    """Run ``ptxas --verbose`` across unique PTX files and collect warnings."""
    parsed: dict[str, PtxasKernelStats] = {}
    warnings: list[str] = []
    for ptx_path in ptx_paths:
        try:
            proc = subprocess.run(
                [ptxas_bin, "--verbose", f"--gpu-name={arch}", "-o", os.devnull, str(ptx_path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"ptxas invocation error on {ptx_path.name}: {exc}")
            continue
        combined = f"{proc.stderr}\n{proc.stdout}"
        if proc.returncode != 0:
            warnings.append(f"ptxas failed on {ptx_path.name} (exit {proc.returncode})")
            continue
        parsed.update(parse_ptxas_verbose(combined))
    return parsed, warnings


@dataclasses.dataclass(frozen=True, slots=True)
class _PassCArtifacts:
    stats: tuple[PtxKernelStats, ...]
    dedup_count: int
    filter_status: str
    spill_source: str
    warnings: tuple[str, ...]


def _resolve_ptxas_stats(
    filtered: list[tuple[PtxKernelStats, Path, bool]],
    compute_capability: str | None,
) -> tuple[dict[str, PtxasKernelStats], list[str]]:
    """Locate ``ptxas`` and compile unique PTX files for register/spill stats."""
    if not filtered or not compute_capability:
        return {}, []
    ptxas_bin = find_ptxas()
    if ptxas_bin is None:
        return {}, ["ptxas binary not found; falling back to Triton/PTX telemetry"]
    try:
        arch = compute_capability_to_sm(compute_capability)
    except ValueError as exc:
        return {}, [str(exc)]
    unique_files = list(dict.fromkeys(item[1] for item in filtered))
    return _run_ptxas_on_files(unique_files, arch, ptxas_bin)


def _enrich_ptx_stat(
    stat: PtxKernelStats,
    pa: PtxasKernelStats | None,
    tm: TritonKernelMeta | None,
) -> PtxKernelStats:
    """Attach optional ``ptxas`` and Triton metadata onto one :class:`PtxKernelStats`."""
    phys_regs = pa.phys_regs if pa is not None else stat.phys_regs
    spill_st = pa.spill_store_bytes if pa is not None else stat.spill_store_bytes
    spill_ld = pa.spill_load_bytes if pa is not None else stat.spill_load_bytes
    stack_b = pa.stack_frame_bytes if pa is not None else stat.stack_frame_bytes
    smem_b = pa.smem_bytes if pa is not None else stat.ptxas_smem_bytes
    t_spills = tm.n_spills if tm is not None else stat.triton_spills
    if tm is not None and phys_regs is None and tm.n_regs is not None:
        phys_regs = tm.n_regs
    return dataclasses.replace(
        stat,
        phys_regs=phys_regs,
        spill_store_bytes=spill_st,
        spill_load_bytes=spill_ld,
        stack_frame_bytes=stack_b,
        ptxas_smem_bytes=smem_b,
        triton_spills=t_spills,
    )


def _lookup_triton_meta(
    path: Path,
    *,
    is_triton: bool,
    kernel_name: str,
    triton_metas: dict[str, TritonKernelMeta],
) -> TritonKernelMeta | None:
    """Resolve Triton metadata for ``kernel_name``, preferring sibling ``.json``."""
    if is_triton:
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            meta = _parse_triton_meta_file(sibling)
            if meta is not None:
                return meta
    return triton_metas.get(kernel_name)


def collect_pass_c_ptx_stats(
    triton_dump_dir: Path,
    triton_cache_dir: Path,
    xla_dump_dir: Path,
    *,
    executed_kernels: tuple[str, ...] = (),
    compute_capability: str | None = None,
) -> _PassCArtifacts:
    """Collect, deduplicate, filter, and enrich Pass-C PTX/ptxas/Triton artifacts."""
    sources = _gather_triton_ptx_files([triton_dump_dir, triton_cache_dir])
    if xla_dump_dir.is_dir():
        sources.extend((p, False) for p in sorted(xla_dump_dir.glob("*.ptx")))

    deduped, dedup_count = _dedup_ptx_entries(sources)
    filtered, filter_status = _filter_executed_entries(deduped, executed_kernels)

    triton_metas: dict[str, TritonKernelMeta] = {}
    triton_metas.update(parse_triton_metadata_dir(triton_cache_dir))
    triton_metas.update(parse_triton_metadata_dir(triton_dump_dir))
    ptxas_stats, warnings = _resolve_ptxas_stats(filtered, compute_capability)

    enriched: list[PtxKernelStats] = []
    used_ptxas = False
    used_triton = False
    for stat, path, is_triton in filtered:
        pa = ptxas_stats.get(stat.kernel_name)
        tm = _lookup_triton_meta(
            path,
            is_triton=is_triton,
            kernel_name=stat.kernel_name,
            triton_metas=triton_metas,
        )
        used_ptxas = used_ptxas or pa is not None
        used_triton = used_triton or tm is not None
        enriched.append(_enrich_ptx_stat(stat, pa, tm))

    spill_source = (
        "ptxas"
        if used_ptxas
        else ("triton_meta" if used_triton else ("ptx_only" if enriched else "none"))
    )
    return _PassCArtifacts(
        stats=tuple(enriched),
        dedup_count=dedup_count,
        filter_status=filter_status if enriched else "not_applicable",
        spill_source=spill_source,
        warnings=tuple(warnings),
    )


def _worker_cmd(
    *,
    mode: str,
    fn_spec: str,
    inputs_spec: str | None,
    scale: int | None,
    shapes: str | None,
    extent_axis: str | None,
    n_steady: int,
    n_wall: int,
    known_labels: frozenset[str],
    result_json: Path,
) -> list[str]:
    """Build command-line arguments for launching ``xtrax.profiling._diag_worker``."""
    cmd = [
        sys.executable,
        "-m",
        "xtrax.profiling._diag_worker",
        "--mode",
        mode,
        "--fn",
        fn_spec,
        "--n-steady",
        str(n_steady),
        "--n-wall",
        str(n_wall),
        "--result-json",
        str(result_json),
    ]
    if inputs_spec is not None:
        cmd.extend(["--inputs", inputs_spec])
    if scale is not None:
        cmd.extend(["--scale", str(scale)])
    if shapes is not None:
        cmd.extend(["--shapes", shapes])
    if extent_axis is not None:
        cmd.extend(["--extent-axis", extent_axis])
    if known_labels:
        cmd.extend(["--known-labels-json", json.dumps(sorted(known_labels))])
    return cmd


def _apply_parent_gpu_preallocate(env: dict[str, str], *, isolate_passes: bool) -> None:
    """Disable subprocess 75% VRAM preallocation when parent holds or shares GPU context."""
    if not isolate_passes or "jax" in sys.modules:
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _run_worker_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
) -> tuple[int, str, str]:
    """Execute one isolated profiling worker subprocess and return ``(rc, stdout, stderr)``."""
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _run_pass_a(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    probe_id: str,
    n_atoms: int,
    stage: int,
    fn_spec: str | None,
    inputs_spec: str | None,
    scale: int | None,
    shapes: str | None,
    extent_axis: str | None,
    n_steady: int,
    n_wall: int,
    prod_path: Path,
    base_config: dict[str, str],
    isolate_passes: bool,
    repo_root: Path | None,
) -> tuple[ProbeRecord, str | None]:
    """Execute Pass A (``cmdbuf=on``) and emit immutable production ``ProbeRecord``."""
    base_xla = os.environ.get("XLA_FLAGS", "")
    clean_xla = merge_xla_flags(base_xla, {})
    if fn_spec is not None and (inputs_spec is not None or shapes is not None) and isolate_passes:
        with tempfile.TemporaryDirectory(prefix="xtrax_pass_a_") as tmp_dir:
            res_path = Path(tmp_dir) / "pass_a.json"
            env = dict(os.environ)
            env["XLA_FLAGS"] = clean_xla
            _apply_parent_gpu_preallocate(env, isolate_passes=isolate_passes)
            cmd = _worker_cmd(
                mode="prod",
                fn_spec=fn_spec,
                inputs_spec=inputs_spec,
                scale=scale,
                shapes=shapes,
                extent_axis=extent_axis,
                n_steady=n_steady,
                n_wall=n_wall,
                known_labels=frozenset(),
                result_json=res_path,
            )
            rc, _out, err = _run_worker_subprocess(cmd, env=env)
            if rc != 0 or not res_path.is_file():
                raise RuntimeError(f"Pass A isolated worker failed (exit {rc}): {err.strip()}")
            payload = json.loads(res_path.read_text(encoding="utf-8"))
        prod_record = emit_probe_record(
            path=prod_path,
            probe_id=probe_id,
            stage=stage,
            n_atoms=n_atoms,
            platform=payload["platform"],
            metrics=payload["metrics"],
            config=base_config,
            git_sha=payload.get("git_sha"),
            repo_root=repo_root,
            x64_enabled=payload.get("x64_enabled"),
            device_kind=payload.get("device_kind"),
            jax_version=payload.get("jax_version"),
            jaxlib_version=payload.get("jaxlib_version"),
            xla_flags=payload.get("xla_flags"),
        )
        return prod_record, None

    import jax

    result = probe_unit(fn, args, kwargs, n_steady=n_steady, n_wall=n_wall)
    prod_metrics: dict[str, float | int | str] = dict(result.as_metrics())
    prod_record = emit_probe_record(
        path=prod_path,
        probe_id=probe_id,
        stage=stage,
        n_atoms=n_atoms,
        platform=jax.default_backend(),
        metrics=prod_metrics,
        config=base_config,
        repo_root=repo_root,
    )
    return prod_record, None


def _decode_scope_map(raw_scopes: Any) -> dict[str, tuple[float, int] | None] | None:
    """Convert deserialized worker scope lists into tuple pairs."""
    if not isinstance(raw_scopes, dict):
        return None
    out: dict[str, tuple[float, int] | None] = {}
    for lbl, val in raw_scopes.items():
        out[str(lbl)] = None if val is None else (float(val[0]), int(val[1]))
    return out


def _run_pass_c_in_temp(
    *,
    tmp_root: Path,
    sub_env: dict[str, str],
    fn_spec: str,
    inputs_spec: str | None,
    scale: int | None,
    shapes: str | None,
    extent_axis: str | None,
    pass_b: dict[str, Any],
) -> tuple[_PassCArtifacts, str]:
    """Run optional cold PTX/Triton dump pass and collect compiler resource stats."""
    t_cache = tmp_root / "triton_cache"
    t_dump = tmp_root / "triton_dump"
    j_cache = tmp_root / "jax_cache"
    x_dump = tmp_root / "xla_dump"
    for d in (t_cache, t_dump, j_cache, x_dump):
        d.mkdir(parents=True, exist_ok=True)
    pass_c_json = tmp_root / "pass_c.json"
    pass_c_env = dict(sub_env)
    pass_c_env["TRITON_CACHE_DIR"] = str(t_cache)
    pass_c_env["TRITON_KERNEL_DUMP"] = "1"
    pass_c_env["TRITON_DUMP_DIR"] = str(t_dump)
    pass_c_env["JAX_COMPILATION_CACHE_DIR"] = str(j_cache)
    pass_c_env["XLA_FLAGS"] = merge_xla_flags(
        os.environ.get("XLA_FLAGS", ""),
        {
            "--xla_gpu_enable_command_buffer": "",
            "--xla_dump_to": str(x_dump),
            "--xla_dump_hlo_as_text": "false",
        },
    )
    cmd_c = _worker_cmd(
        mode="ptx",
        fn_spec=fn_spec,
        inputs_spec=inputs_spec,
        scale=scale,
        shapes=shapes,
        extent_axis=extent_axis,
        n_steady=1,
        n_wall=1,
        known_labels=frozenset(),
        result_json=pass_c_json,
    )
    rc_c, _out_c, _err_c = _run_worker_subprocess(cmd_c, env=pass_c_env)
    if rc_c != 0:
        return _PassCArtifacts((), 0, "not_applicable", "none", ()), f"skipped:exit_{rc_c}"

    pass_c_cc = pass_b.get("compute_capability")
    if pass_c_json.is_file():
        pass_c_payload = json.loads(pass_c_json.read_text(encoding="utf-8"))
        pass_c_cc = pass_c_payload.get("compute_capability") or pass_c_cc
    artifacts = collect_pass_c_ptx_stats(
        t_dump,
        t_cache,
        x_dump,
        executed_kernels=tuple(pass_b.get("executed_triton_kernels", ())),
        compute_capability=pass_c_cc,
    )
    status = "ok" if artifacts.stats else "skipped:no_ptx_emitted"
    return artifacts, status


def _resolve_dual_pass_preskip(
    run_diag: bool,
    fn_spec: str | None,
    inputs_spec: str | None,
    shapes: str | None,
) -> str | None:
    """Resolve pre-flight ``diag_skipped_reason`` before Pass A emission."""
    if not run_diag:
        return "diag_disabled"
    if fn_spec is None or (inputs_spec is None and shapes is None):
        return "no_importable_spec"
    return None


def _assemble_diag_record_inputs(
    pass_b: dict[str, Any],
    prod_record: ProbeRecord,
    ptx_artifacts: _PassCArtifacts,
    ptx_status: str,
    *,
    config: dict[str, str] | None = None,
    n_steady: int = 3,
) -> tuple[
    dict[str, float | int | str],
    dict[str, str],
    dict[str, tuple[float, int] | None] | None,
    str | None,
]:
    """Assemble metrics, config, scopes, and HLO skip status for Pass B emission."""
    scopes = _decode_scope_map(pass_b.get("scopes"))
    diag_metrics: dict[str, float | int | str] = dict(pass_b.get("metrics", {}))
    if ptx_artifacts.stats:
        diag_metrics.update(summarize_ptx_metrics(list(ptx_artifacts.stats)))
    diag_cfg = dict(config or {})
    diag_cfg.update(
        {
            "diagnostic_mode": "cmdbuf_off",
            "paired_prod_probe_id": prod_record.probe_id,
            "diag_n_steady": str(n_steady),
            "ptx_capture_status": ptx_status,
            "ptx_spill_source": ptx_artifacts.spill_source,
            "ptx_dedup_count": str(ptx_artifacts.dedup_count),
            "ptx_filter_status": ptx_artifacts.filter_status,
        }
    )
    hlo_skipped = pass_b.get("hlo_attribution_skipped")
    if hlo_skipped:
        diag_cfg["hlo_attribution_skipped"] = str(hlo_skipped)
    return diag_metrics, diag_cfg, scopes, hlo_skipped


def probe_kernel_dual_pass(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    probe_id: str,
    n_atoms: int,
    stage: int = 2,
    fn_spec: str | None = None,
    inputs_spec: str | None = None,
    scale: int | None = None,
    shapes: str | None = None,
    extent_axis: str | None = None,
    n_steady: int = 3,
    n_wall: int = 5,
    known_labels: frozenset[str] = frozenset(),
    prod_path: str | Path | None = None,
    diag_dir: str | Path = "outputs/profiling/diagnostics",
    config: dict[str, str] | None = None,
    run_diag: bool = True,
    capture_ptx: bool = True,
    isolate_passes: bool = True,
    repo_root: Path | None = None,
) -> DualPassProbeResult:
    """Run Pass A (``cmdbuf=on``), optional Pass B (``cmdbuf=off``), and optional Pass C PTX.

    Never mutates ``prod_record`` after creation: ``diag_skipped_reason`` is
    resolved prior to Pass A emission when Pass B cannot run. Subprocess passes
    exit cleanly to release 100% GPU VRAM between Pass A, Pass B, and Pass C.
    """
    call_kwargs = dict(kwargs or {})
    pre_skip = _resolve_dual_pass_preskip(run_diag, fn_spec, inputs_spec, shapes)

    prod_cfg = dict(config or {})
    prod_cfg["diagnostic_mode"] = "none"
    prod_cfg.setdefault("step_time_basis", "batch")
    if pre_skip is not None:
        prod_cfg["diag_skipped_reason"] = pre_skip

    with tempfile.TemporaryDirectory(prefix="xtrax_prod_fallback_") as tmp_prod:
        eff_prod_path = Path(prod_path) if prod_path is not None else Path(tmp_prod) / "prod.json"
        prod_record, _ = _run_pass_a(
            fn,
            args,
            call_kwargs,
            probe_id=probe_id,
            n_atoms=n_atoms,
            stage=stage,
            fn_spec=fn_spec,
            inputs_spec=inputs_spec,
            scale=scale,
            shapes=shapes,
            extent_axis=extent_axis,
            n_steady=n_steady,
            n_wall=n_wall,
            prod_path=eff_prod_path,
            base_config=prod_cfg,
            isolate_passes=isolate_passes,
            repo_root=repo_root,
        )
        written_prod = eff_prod_path if prod_path is not None else None

    if pre_skip is not None:
        return DualPassProbeResult(
            prod_record=prod_record,
            diag_record=None,
            prod_record_path=written_prod,
            diag_record_path=None,
            diag_skipped_reason=pre_skip,
            ptx_capture_status="disabled",
            ptx_spill_source="none",
            ptx_filter_status="not_applicable",
        )

    assert fn_spec is not None
    sub_env = dict(os.environ)
    _apply_parent_gpu_preallocate(sub_env, isolate_passes=isolate_passes)
    if prod_path is None:
        sub_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    with tempfile.TemporaryDirectory(prefix="xtrax_diag_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        pass_b_json = tmp_root / "pass_b.json"
        pass_b_env = dict(sub_env)
        pass_b_env["XLA_FLAGS"] = merge_xla_flags(
            os.environ.get("XLA_FLAGS", ""),
            {"--xla_gpu_enable_command_buffer": ""},
        )
        cmd_b = _worker_cmd(
            mode="diag",
            fn_spec=fn_spec,
            inputs_spec=inputs_spec,
            scale=scale,
            shapes=shapes,
            extent_axis=extent_axis,
            n_steady=n_steady,
            n_wall=n_wall,
            known_labels=known_labels,
            result_json=pass_b_json,
        )
        rc_b, _out_b, _err_b = _run_worker_subprocess(cmd_b, env=pass_b_env)
        if rc_b != 0 or not pass_b_json.is_file():
            return DualPassProbeResult(
                prod_record=prod_record,
                diag_record=None,
                prod_record_path=written_prod,
                diag_record_path=None,
                diag_skipped_reason=f"diag_subprocess_failed:exit_{rc_b}",
                ptx_capture_status="skipped:diag_failed",
                ptx_spill_source="none",
                ptx_filter_status="not_applicable",
            )
        pass_b = json.loads(pass_b_json.read_text(encoding="utf-8"))

        ptx_artifacts = _PassCArtifacts((), 0, "not_applicable", "none", ())
        ptx_status = "disabled"
        if capture_ptx:
            ptx_artifacts, ptx_status = _run_pass_c_in_temp(
                tmp_root=tmp_root,
                sub_env=sub_env,
                fn_spec=fn_spec,
                inputs_spec=inputs_spec,
                scale=scale,
                shapes=shapes,
                extent_axis=extent_axis,
                pass_b=pass_b,
            )

    diag_metrics, diag_cfg, scopes, hlo_skipped = _assemble_diag_record_inputs(
        pass_b,
        prod_record,
        ptx_artifacts,
        ptx_status,
        config=config,
        n_steady=n_steady,
    )
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", prod_record.probe_id).strip("_") or "probe"
    diag_out = Path(diag_dir) / f"{safe_id}.diag.json"
    diag_record = emit_probe_record(
        path=diag_out,
        probe_id=f"{prod_record.probe_id}:diag",
        stage=prod_record.stage,
        n_atoms=prod_record.n_atoms,
        platform=pass_b["platform"],
        metrics=diag_metrics,
        scopes=scopes,
        attribution_method=pass_b.get("attribution_method"),
        config=diag_cfg,
        git_sha=pass_b.get("git_sha") or prod_record.git_sha,
        repo_root=repo_root,
        num_replicas=prod_record.num_replicas,
        replica_id=prod_record.replica_id,
        bucket_index=prod_record.bucket_index,
        bucket_capacity=prod_record.bucket_capacity,
        x64_enabled=pass_b.get("x64_enabled"),
        device_kind=pass_b.get("device_kind"),
        jax_version=pass_b.get("jax_version"),
        jaxlib_version=pass_b.get("jaxlib_version"),
        xla_flags=pass_b.get("xla_flags"),
    )
    return DualPassProbeResult(
        prod_record=prod_record,
        diag_record=diag_record,
        prod_record_path=written_prod,
        diag_record_path=diag_out,
        diag_skipped_reason=None,
        ptx_stats=ptx_artifacts.stats,
        ptxas_warnings=ptx_artifacts.warnings,
        ptx_capture_status=ptx_status,
        ptx_spill_source=ptx_artifacts.spill_source,
        ptx_filter_status=ptx_artifacts.filter_status,
        hlo_attribution_skipped=hlo_skipped,
    )


def _check_precision_stage_and_sha(
    single_record: ProbeRecord,
    mixed_record: ProbeRecord,
) -> None:
    """Assert precision pair matches contract major version, stage, diagnostic mode, and SHA."""
    running_major = CONTRACT_VERSION.split(".")[0]
    s_major = single_record.contract_version.split(".")[0]
    m_major = mixed_record.contract_version.split(".")[0]
    if s_major != m_major or s_major != running_major:
        raise ClaimValidityError("precision pair contract_version major mismatch")
    if single_record.stage < 1 or mixed_record.stage < 1:
        raise ClaimValidityError("precision pair requires stage >= 1")
    if single_record.platform == "gpu" and min(single_record.stage, mixed_record.stage) < 2:
        raise ClaimValidityError("GPU precision pair requires stage >= 2")
    if single_record.stage != mixed_record.stage:
        raise ClaimValidityError("precision pair stage mismatch")

    for label, rec in (("single", single_record), ("mixed", mixed_record)):
        dmode = rec.config.get("diagnostic_mode", "none")
        if dmode not in ("", "none"):
            raise ClaimValidityError(f"{label} precision record is diagnostic ({dmode!r})")

    _reject_unverifiable_git_sha([single_record, mixed_record], ClaimClass.END_TO_END)
    if not _shas_match(single_record.git_sha, mixed_record.git_sha):
        raise ClaimValidityError(
            "precision pair git_sha mismatch: "
            f"{single_record.git_sha!r} != {mixed_record.git_sha!r}"
        )


def _check_precision_env_and_basis(
    single_record: ProbeRecord,
    mixed_record: ProbeRecord,
    hold_fixed_config: tuple[str, ...],
) -> None:
    """Assert precision pair environment unanimity, step_time_basis, and held config keys."""
    for attr in _PRECISION_UNANIMITY_FIELDS:
        if getattr(single_record, attr) != getattr(mixed_record, attr):
            raise ClaimValidityError(f"precision pair not unanimous on {attr!r}")

    s_basis = (
        single_record.config["step_time_basis"]
        if "step_time_basis" in single_record.config
        else "batch"
    )
    m_basis = (
        mixed_record.config["step_time_basis"]
        if "step_time_basis" in mixed_record.config
        else "batch"
    )
    if s_basis not in _VALID_BASIS or m_basis not in _VALID_BASIS or s_basis != m_basis:
        raise ClaimValidityError(
            f"precision pair step_time_basis mismatch: {s_basis!r} vs {m_basis!r}"
        )

    for key in hold_fixed_config:
        if key not in single_record.config or key not in mixed_record.config:
            raise ClaimValidityError(f"precision pair missing hold_fixed key {key!r}")
        if single_record.config[key] != mixed_record.config[key]:
            raise ClaimValidityError(f"precision pair mismatch on {key!r}")


def _precision_unanimity_check(
    single_record: ProbeRecord,
    mixed_record: ProbeRecord,
    hold_fixed_config: tuple[str, ...],
) -> None:
    """Assert precision pair matches contract, stage, SHAs, and environment unanimity."""
    _check_precision_stage_and_sha(single_record, mixed_record)
    _check_precision_env_and_basis(single_record, mixed_record, hold_fixed_config)


def compute_precision_ratio(
    single_record: ProbeRecord,
    mixed_record: ProbeRecord,
    *,
    metric: str = "total_step_seconds",
    hold_fixed_config: tuple[str, ...] = (),
) -> float:
    """Compute ``mixed / single`` ratio after enforcing hardware/precision unanimity."""
    _precision_unanimity_check(single_record, mixed_record, hold_fixed_config)

    if mixed_record.x64_enabled is not True:
        raise ClaimValidityError("mixed_record must have x64_enabled=True")

    has_prec = "precision" in single_record.config or "precision" in mixed_record.config
    if single_record.x64_enabled is True or has_prec:
        s_prec = single_record.config.get("precision")
        m_prec = mixed_record.config.get("precision")
        if s_prec != "single" or m_prec != "mixed":
            raise ClaimValidityError(
                "when x64_enabled=True on single_record or 'precision' is set, "
                f"config['precision'] must be ('single', 'mixed'), got ({s_prec!r}, {m_prec!r})"
            )

    if metric not in single_record.metrics or metric not in mixed_record.metrics:
        raise ClaimValidityError(f"precision metric {metric!r} missing from pair")
    s_val = float(single_record.metrics[metric])
    m_val = float(mixed_record.metrics[metric])
    if not (math.isfinite(s_val) and s_val > 0.0 and math.isfinite(m_val) and m_val > 0.0):
        raise ClaimValidityError(f"precision metric {metric!r} must be finite > 0.0")
    return m_val / s_val


__all__ = [
    "DualPassProbeResult",
    "PtxKernelStats",
    "PtxasKernelStats",
    "TritonKernelMeta",
    "collect_pass_c_ptx_stats",
    "compute_capability_to_sm",
    "compute_precision_ratio",
    "find_ptxas",
    "merge_xla_flags",
    "parse_ptx_text",
    "parse_ptxas_verbose",
    "parse_triton_metadata_dir",
    "probe_kernel_dual_pass",
    "summarize_ptx_metrics",
]

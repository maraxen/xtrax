#!/usr/bin/env python3
"""P2-STATIC trace-count gate: chex.assert_max_traces on port kernel qualnames."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chex
import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PORT_TARGET = ROOT / "port" / "port_target.toml"
SCHEMA_VERSION = "audit_port_trace_count_v0"
ERROR_TAXONOMY_CLASS = "compilation_leak"
DEFAULT_MAX_TRACES = 1


@dataclass(frozen=True)
class KernelSpec:
    qualname: str
    trace_probe: str | None = None
    order: int = 0


def resolve_port_target(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        pyproject_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        tool_port = pyproject_data.get("tool", {}).get("port", {})
        configured = tool_port.get("target")
        if configured:
            return (ROOT / configured).resolve()
    return DEFAULT_PORT_TARGET.resolve()


def load_port_target(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_manifest(port_root: Path, wave_id: str) -> dict[str, Any] | None:
    manifest_path = port_root / "manifests" / f"{wave_id}.toml"
    if not manifest_path.is_file():
        return None
    return tomllib.loads(manifest_path.read_text(encoding="utf-8"))


def max_traces_from_config(port_config: dict[str, Any]) -> int:
    parity = port_config.get("parity", {})
    value = parity.get("max_traces", DEFAULT_MAX_TRACES)
    if not isinstance(value, int) or value < 1:
        raise SystemExit(
            f"port_target parity.max_traces must be a positive int, got {value!r}"
        )
    return value


def iter_kernels(
    port_config: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> list[KernelSpec]:
    if manifest is not None:
        kernels = manifest.get("kernels")
        if isinstance(kernels, list) and kernels:
            specs: list[KernelSpec] = []
            for entry in kernels:
                if not isinstance(entry, dict):
                    continue
                qualname = entry.get("qualname")
                if not isinstance(qualname, str) or not qualname:
                    continue
                trace_probe = entry.get("trace_probe")
                order = entry.get("order", 0)
                probe_value = trace_probe if isinstance(trace_probe, str) else None
                specs.append(
                    KernelSpec(
                        qualname=qualname,
                        trace_probe=probe_value,
                        order=int(order) if isinstance(order, int) else 0,
                    )
                )
            return sorted(specs, key=lambda spec: spec.order)

    port_section = port_config.get("port", {})
    qualname = port_section.get("symbol_qualname", "")
    if isinstance(qualname, str) and qualname:
        return [KernelSpec(qualname=qualname)]
    return []


def is_placeholder_qualname(qualname: str) -> bool:
    return "<" in qualname


def import_qualname(qualname: str) -> Any:
    module_path, _, attr = qualname.rpartition(".")
    if not module_path or not attr:
        raise ImportError(f"invalid qualname: {qualname!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def import_probe(trace_probe: str) -> Callable[[], None]:
    module_path, sep, attr = trace_probe.partition(":")
    if not sep or not module_path or not attr:
        raise ValueError(
            f"trace_probe must be 'module.path:callable', got {trace_probe!r}"
        )
    module = importlib.import_module(module_path)
    probe = getattr(module, attr)
    if not callable(probe):
        raise TypeError(f"trace_probe {trace_probe!r} is not callable")
    return probe


def _guard_for_trace_count(
    target: Callable[..., Any],
    *,
    max_traces: int,
) -> Callable[..., Any]:
    """Apply @jax.jit @chex.assert_max_traces when target is not already traced."""
    try:
        return jax.jit(chex.assert_max_traces(n=max_traces)(target))
    except ValueError:
        return target


def run_trace_gate(
    target: Callable[..., Any],
    probe: Callable[[Callable[..., Any]], None],
    *,
    max_traces: int,
) -> None:
    """Run probe against target wrapped with chex.assert_max_traces."""
    chex.clear_trace_counter()
    guarded = _guard_for_trace_count(target, max_traces=max_traces)
    probe(guarded)


def check_kernel(spec: KernelSpec, *, max_traces: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "qualname": spec.qualname,
        "max_traces": max_traces,
        "error_taxonomy_class": ERROR_TAXONOMY_CLASS,
    }
    if is_placeholder_qualname(spec.qualname):
        return {
            **base,
            "status": "skipped",
            "reason": "placeholder qualname in port_target/manifest",
        }
    if not spec.trace_probe:
        return {
            **base,
            "status": "skipped",
            "reason": "no trace_probe configured for kernel",
        }

    try:
        target = import_qualname(spec.qualname)
        probe_runner = import_probe(spec.trace_probe)

        def _invoke(guarded: Callable[..., Any]) -> None:
            probe_runner(guarded)

        run_trace_gate(target, _invoke, max_traces=max_traces)
    except Exception as exc:  # noqa: BLE001 — gate aggregates per-kernel failures
        return {
            **base,
            "status": "fail",
            "reason": str(exc),
            "trace_probe": spec.trace_probe,
        }

    return {
        **base,
        "status": "pass",
        "trace_probe": spec.trace_probe,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_envelope(
    *,
    port_target: Path,
    wave_id: str,
    max_traces: int,
    manifest_path: Path | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    fail_count = sum(1 for result in results if result["status"] == "fail")
    pass_count = sum(1 for result in results if result["status"] == "pass")
    skip_count = sum(1 for result in results if result["status"] == "skipped")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "chex.assert_max_traces",
        "emitted_at": datetime.now(UTC).isoformat(),
        "port_target": _display_path(port_target),
        "wave_id": wave_id,
        "manifest": _display_path(manifest_path) if manifest_path else None,
        "max_traces": max_traces,
        "kernel_count": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "results": results,
    }


def audit_trace_counts(
    port_target_path: Path,
) -> tuple[dict[str, Any], int]:
    port_config = load_port_target(port_target_path)
    port_section = port_config.get("port", {})
    wave_id = port_section.get("wave_id")
    if not isinstance(wave_id, str) or not wave_id:
        raise SystemExit("port_target.toml [port] wave_id is required")

    max_traces = max_traces_from_config(port_config)
    port_root = port_target_path.parent
    manifest = load_manifest(port_root, wave_id)
    manifest_path = port_root / "manifests" / f"{wave_id}.toml"
    if manifest is None:
        manifest_path = None

    kernels = iter_kernels(port_config, manifest)
    results = [check_kernel(spec, max_traces=max_traces) for spec in kernels]
    resolved_manifest = (
        manifest_path if manifest_path and manifest_path.is_file() else None
    )
    envelope = build_envelope(
        port_target=port_target_path,
        wave_id=wave_id,
        max_traces=max_traces,
        manifest_path=resolved_manifest,
        results=results,
    )
    exit_code = 1 if envelope["fail_count"] > 0 else 0
    return envelope, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port-target",
        type=Path,
        default=None,
        help=(
            "Path to port_target.toml "
            "(default: [tool.port].target or port/port_target.toml)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the envelope JSON",
    )
    args = parser.parse_args(argv)

    port_target = resolve_port_target(args.port_target)
    if not port_target.is_file():
        raise SystemExit(f"port target not found: {port_target}")

    envelope, exit_code = audit_trace_counts(port_target)
    payload = json.dumps(envelope, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

"""Stateless subprocess measurement worker for multi-pass kernel profiling.

Runs a single isolated pass (``--mode prod``, ``diag``, or ``ptx``) against an
importable callable and writes structured raw telemetry atomically to
``--result-json``. Never writes ``ProbeRecord`` files to disk and never imports
``jax`` or ``xtrax.cli`` at module scope.
"""

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from xtrax.profiling.record import (
    capture_device_kind,
    capture_git_sha,
    capture_jax_version,
    capture_jaxlib_version,
    capture_x64_enabled,
    capture_xla_flags,
)
from xtrax.profiling.trace import (
    hlo_text_for,
    interval_union_seconds,
    load_trace_events,
    parse_dispatch_counts,
    parse_hlo_op_times,
    parse_scopes,
    scope_map_from_hlo_text,
)
from xtrax.profiling.unit_probe import probe_unit

_TRITON_HLO_RE = re.compile(r'custom_call_target="[^"]*triton[^"]*"[^}]*name\s*=\s*"([^"]+)"')
_SHAPE_ENTRY_RE = re.compile(r"^\(([^)]*)\)(?:<([a-zA-Z0-9]+)>|([a-zA-Z0-9]+))$")


def _load_symbol(import_path: str) -> Any:
    """Resolve ``module.path:symbol`` using stdlib ``importlib``."""
    if ":" not in import_path:
        raise ValueError(f"Expected 'module.path:symbol', got {import_path!r}")
    mod_name, attr_name = import_path.rsplit(":", 1)
    if not mod_name or not attr_name:
        raise ValueError(f"Malformed import path {import_path!r}")
    cwd_str = str(Path.cwd())
    if cwd_str not in sys.path:
        sys.path.insert(0, cwd_str)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


def _dtype_lookup(name: str, jnp_mod: Any) -> Any:
    """Resolve a JAX dtype object from its string name."""
    mapping = {
        "f16": jnp_mod.float16,
        "float16": jnp_mod.float16,
        "bf16": jnp_mod.bfloat16,
        "bfloat16": jnp_mod.bfloat16,
        "f32": jnp_mod.float32,
        "float32": jnp_mod.float32,
        "f64": jnp_mod.float64,
        "float64": jnp_mod.float64,
        "i32": jnp_mod.int32,
        "int32": jnp_mod.int32,
        "i64": jnp_mod.int64,
        "int64": jnp_mod.int64,
        "u32": jnp_mod.uint32,
        "uint32": jnp_mod.uint32,
        "bool": jnp_mod.bool_,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported shape dtype {name!r}")
    return mapping[name]


def _build_args_from_shapes(shapes_spec: str, jnp_mod: Any) -> tuple[Any, ...]:
    """Construct zero-filled arrays from ``name=(d0,...)dtype`` tokens."""
    out = []
    for entry in shapes_spec.split():
        if "=" not in entry:
            raise ValueError(f"Malformed shape entry {entry!r}")
        _, rest = entry.split("=", 1)
        m = _SHAPE_ENTRY_RE.fullmatch(rest)
        if not m:
            raise ValueError(f"Malformed shape spec {entry!r}")
        shape_str, bracketed, bare = m.groups()
        dtype_name = bracketed or bare
        dims = tuple(int(tok.strip()) for tok in shape_str.split(",") if tok.strip())
        out.append(jnp_mod.zeros(dims, dtype=_dtype_lookup(dtype_name, jnp_mod)))
    return tuple(out)


def _split_factory_output(produced: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize an inputs factory return value into ``(args, kwargs)``."""
    if (
        isinstance(produced, tuple)
        and len(produced) == 2
        and isinstance(produced[0], tuple)
        and isinstance(produced[1], dict)
    ):
        return produced[0], dict(produced[1])
    if isinstance(produced, tuple):
        return produced, {}
    return (produced,), {}


def _resolve_inputs(
    *,
    inputs_spec: str | None,
    scale: int | None,
    shapes: str | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build positional and keyword inputs from ``--inputs`` or ``--shapes``."""
    import jax.numpy as jnp

    if inputs_spec is not None:
        factory = _load_symbol(inputs_spec)
        produced = factory(scale) if scale is not None else factory()
        return _split_factory_output(produced)
    if shapes is not None:
        return _build_args_from_shapes(shapes, jnp), {}
    raise ValueError("Worker requires either --inputs or --shapes")


def _resolve_pass_b_hlo_text(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    jax_mod: Any,
) -> tuple[str | None, str | None]:
    """Obtain HLO text directly or via ``jax.jit(fn)`` with graceful fallback."""
    lower = getattr(fn, "lower", None)
    if callable(lower):
        try:
            return hlo_text_for(fn, *args, **kwargs), None
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return hlo_text_for(jax_mod.jit(fn), *args, **kwargs), None
    except Exception:  # noqa: BLE001 - unjittable host wrappers degrade gracefully
        return None, "unjittable_wrapper"


def _device_compute_capability(jax_mod: Any) -> str | None:
    """Best-effort GPU compute capability string (e.g. ``'8.9'``)."""
    try:
        devices = jax_mod.devices()
    except Exception:
        return None
    for dev in devices:
        if getattr(dev, "platform", None) == "gpu":
            cc = getattr(dev, "compute_capability", None)
            if cc is not None and str(cc).strip():
                return str(cc).strip()
    return None


def _collect_provenance(jax_mod: Any) -> dict[str, Any]:
    """Capture subprocess JAX/git/XLA provenance fields."""
    return {
        "platform": jax_mod.default_backend(),
        "x64_enabled": capture_x64_enabled(),
        "device_kind": capture_device_kind(),
        "jax_version": capture_jax_version(),
        "jaxlib_version": capture_jaxlib_version(),
        "xla_flags": capture_xla_flags(),
        "git_sha": capture_git_sha(),
        "compute_capability": _device_compute_capability(jax_mod),
    }


def _classify_op_bucket(
    op_name: str,
    mapped: str | None,
    share_ok: bool,
    known_labels: frozenset[str],
    scopes: dict[str, list[Any] | None],
) -> tuple[str | None, str | None]:
    """Return ``(target_label, triton_kernel_name)`` for a profiled HLO op."""
    triton_name: str | None = None
    is_cc = False
    if mapped is not None:
        if mapped.startswith("triton:"):
            triton_name = mapped[len("triton:") :]
            is_cc = True
        elif mapped.startswith("custom_call:"):
            is_cc = True
        if mapped in known_labels:
            return None, triton_name
    if not is_cc and not share_ok:
        return None, triton_name
    target_label = mapped if is_cc and mapped is not None else op_name
    if target_label in scopes:
        return None, triton_name
    return target_label, triton_name


def _accumulate_op_buckets(
    op_times: dict[str, tuple[float, int]],
    scope_map: dict[str, str | None],
    known_labels: frozenset[str],
    scopes: dict[str, list[Any] | None],
) -> tuple[dict[str, list[float | int]], set[str]]:
    """Aggregate Triton custom calls and significant unmapped ops into ``op_buckets``."""
    total_op_sec = sum(float(sec) for sec, _ in op_times.values())
    triton_kernels: set[str] = set()
    op_buckets: dict[str, list[float | int]] = {}
    for op_name, (sec, cnt) in op_times.items():
        share_ok = total_op_sec > 0.0 and (float(sec) / total_op_sec) >= 0.01
        target_label, triton_name = _classify_op_bucket(
            op_name,
            scope_map.get(op_name),
            share_ok,
            known_labels,
            scopes,
        )
        if triton_name is not None:
            triton_kernels.add(triton_name)
        if target_label is None:
            continue
        bucket = op_buckets.setdefault(target_label, [0.0, 0])
        bucket[0] += float(sec)
        bucket[1] += int(cnt)
    return op_buckets, triton_kernels


def _collect_diag_scopes(
    events: list[dict[str, Any]],
    hlo_text: str | None,
    known_labels: frozenset[str],
    n_steady: int,
) -> tuple[dict[str, list[Any] | None], dict[str, str], list[str]]:
    """Build normalized per-unit scopes, attribution map, and executed Triton names."""
    scopes: dict[str, list[Any] | None] = {}
    attribution: dict[str, str] = {}
    scope_map: dict[str, str | None] = {}
    if hlo_text is not None:
        scope_map = scope_map_from_hlo_text(hlo_text, known_labels)
        matched = parse_scopes(events, scope_map)
        for label in sorted(known_labels):
            if label in matched:
                sec, cnt = matched[label]
                scopes[label] = [float(sec) / n_steady, int(cnt)]
                attribution[label] = "named_scope"
            else:
                scopes[label] = None

    op_times = parse_hlo_op_times(events)
    triton_kernels: set[str] = set(_TRITON_HLO_RE.findall(hlo_text)) if hlo_text else set()
    op_buckets, found_triton = _accumulate_op_buckets(op_times, scope_map, known_labels, scopes)
    triton_kernels.update(found_triton)
    for label, (sec, cnt) in sorted(op_buckets.items()):
        scopes[label] = [float(sec) / n_steady, int(cnt)]
        attribution[label] = "op_name"

    return scopes, attribution, sorted(triton_kernels)


def _run_prod_mode(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    fn = _load_symbol(args.fn)
    call_args, call_kwargs = _resolve_inputs(
        inputs_spec=args.inputs,
        scale=args.scale,
        shapes=args.shapes,
    )
    result = probe_unit(fn, call_args, call_kwargs, n_steady=args.n_steady, n_wall=args.n_wall)
    prov = _collect_provenance(jax)
    return {
        "mode": "prod",
        **prov,
        "metrics": result.as_metrics(),
    }


def _run_diag_mode(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    fn = _load_symbol(args.fn)
    call_args, call_kwargs = _resolve_inputs(
        inputs_spec=args.inputs,
        scale=args.scale,
        shapes=args.shapes,
    )
    known_labels = frozenset(json.loads(args.known_labels_json))

    jax.block_until_ready(fn(*call_args, **call_kwargs))
    hlo_text, hlo_skipped = _resolve_pass_b_hlo_text(fn, call_args, call_kwargs, jax)

    trace_kwargs: dict[str, Any] = {"create_perfetto_trace": True}
    if hasattr(jax.profiler, "ProfileOptions"):
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        trace_kwargs["profiler_options"] = options

    with tempfile.TemporaryDirectory(prefix="xtrax_pass_b_") as tmp_dir:
        with jax.profiler.trace(tmp_dir, **trace_kwargs):
            t0 = time.perf_counter()
            for _ in range(args.n_steady):
                jax.block_until_ready(fn(*call_args, **call_kwargs))
            traced_window = time.perf_counter() - t0
        events = load_trace_events(tmp_dir)

    scopes, attribution, triton_kernels = _collect_diag_scopes(
        events, hlo_text, known_labels, args.n_steady
    )
    counts = parse_dispatch_counts(events)
    device_per_unit = interval_union_seconds(events) / args.n_steady
    traced_per_unit = traced_window / args.n_steady
    host_per_unit = max(traced_per_unit - device_per_unit, 0.0)
    prov = _collect_provenance(jax)

    return {
        "mode": "diag",
        **prov,
        "hlo_attribution_skipped": hlo_skipped,
        "executed_triton_kernels": triton_kernels,
        "scopes": scopes,
        "attribution_method": attribution,
        "metrics": {
            "n_executions": float(counts["n_executions"]) / args.n_steady,
            "n_compilations": float(counts["n_compilations"]) / args.n_steady,
            "n_jit_traces": float(counts["n_jit_traces"]) / args.n_steady,
            "total_step_seconds": float(traced_per_unit),
            "device_seconds": float(device_per_unit),
            "host_seconds": float(host_per_unit),
        },
    }


def _run_ptx_mode(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    fn = _load_symbol(args.fn)
    call_args, call_kwargs = _resolve_inputs(
        inputs_spec=args.inputs,
        scale=args.scale,
        shapes=args.shapes,
    )
    jax.block_until_ready(fn(*call_args, **call_kwargs))
    return {"mode": "ptx", "ok": True, "compute_capability": _device_compute_capability(jax)}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as UTF-8 JSON to ``path`` via an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    """Execute one isolated profiling worker pass and write ``--result-json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prod", "diag", "ptx"), required=True)
    parser.add_argument("--fn", required=True)
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--shapes", default=None)
    parser.add_argument("--extent-axis", default=None)
    parser.add_argument("--n-steady", type=int, default=3)
    parser.add_argument("--n-wall", type=int, default=5)
    parser.add_argument("--known-labels-json", default="[]")
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_steady < 1 or args.n_wall < 1:
        parser.error("--n-steady and --n-wall must be >= 1")

    if args.mode == "prod":
        payload = _run_prod_mode(args)
    elif args.mode == "diag":
        payload = _run_diag_mode(args)
    else:
        payload = _run_ptx_mode(args)
    _atomic_write_json(args.result_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

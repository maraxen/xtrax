"""Tests for port trace-count gate (AC-5 / #2268)."""

import importlib.metadata
import json
import subprocess
import textwrap
from pathlib import Path

import jax.numpy as jnp
import pytest

from scripts.audit_port_trace_count import (
    DEFAULT_MAX_TRACES,
    KernelSpec,
    audit_trace_counts,
    check_kernel,
    is_placeholder_qualname,
    iter_kernels,
    max_traces_from_config,
    run_trace_gate,
)

ROOT = Path(__file__).resolve().parents[2]


def test_chex_is_dev_dependency() -> None:
    version = importlib.metadata.version("chex")
    assert version


def test_max_traces_defaults_to_one() -> None:
    assert max_traces_from_config({}) == DEFAULT_MAX_TRACES
    assert max_traces_from_config({"parity": {}}) == 1
    assert max_traces_from_config({"parity": {"max_traces": 2}}) == 2


def test_placeholder_qualname_detection() -> None:
    assert is_placeholder_qualname("xtrax.<module>.<fn>")
    assert not is_placeholder_qualname("xtrax.sparse.inference.sparse_filter_jit")


def test_iter_kernels_prefers_manifest_order() -> None:
    port_config = {"port": {"symbol_qualname": "xtrax.fallback.fn"}}
    manifest = {
        "kernels": [
            {"order": 2, "qualname": "xtrax.second.fn"},
            {"order": 1, "qualname": "xtrax.first.fn", "trace_probe": "pkg:probe"},
        ]
    }
    specs = iter_kernels(port_config, manifest)
    assert [spec.qualname for spec in specs] == [
        "xtrax.first.fn",
        "xtrax.second.fn",
    ]
    assert specs[0].trace_probe == "pkg:probe"


def test_run_trace_gate_passes_for_stable_input_structure() -> None:
    def target(x: jnp.ndarray) -> jnp.ndarray:
        return x + 1

    def probe(guarded) -> None:
        x = jnp.zeros(3)
        guarded(x)
        guarded(x)

    run_trace_gate(target, probe, max_traces=1)


def test_run_trace_gate_fails_when_shape_structure_changes() -> None:
    def target(x: jnp.ndarray) -> jnp.ndarray:
        return x + 1

    def probe(guarded) -> None:
        guarded(jnp.zeros(3))
        guarded(jnp.zeros((4, 5)))

    with pytest.raises(AssertionError):
        run_trace_gate(target, probe, max_traces=1)


def test_check_kernel_skips_placeholder_without_import() -> None:
    result = check_kernel(
        KernelSpec(qualname="xtrax.<module>.<fn>"),
        max_traces=1,
    )
    assert result["status"] == "skipped"
    assert "placeholder" in result["reason"]


def test_audit_trace_counts_template_port_target(tmp_path: Path) -> None:
    port_root = tmp_path / "port"
    port_root.mkdir()
    (port_root / "manifests").mkdir()
    port_target = port_root / "port_target.toml"
    port_target.write_text(
        textwrap.dedent(
            """
            [port]
            wave_id = "wave_test"
            symbol_qualname = "xtrax.<module>.<fn>"

            [parity]
            max_traces = 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    envelope, exit_code = audit_trace_counts(port_target)
    assert exit_code == 0
    assert envelope["wave_id"] == "wave_test"
    assert envelope["max_traces"] == 1
    assert envelope["skip_count"] == 1
    assert envelope["fail_count"] == 0


def test_audit_trace_counts_runs_configured_probe(tmp_path: Path, monkeypatch) -> None:
    probe_module = tmp_path / "trace_probe_fixture.py"
    probe_module.write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def kernel(x):
                return x + 1

            def probe_kernel(guarded):
                x = jnp.zeros(3)
                guarded(x)
                guarded(x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    port_root = tmp_path / "port"
    port_root.mkdir()
    manifests = port_root / "manifests"
    manifests.mkdir()
    wave_id = "wave_probe"
    (manifests / f"{wave_id}.toml").write_text(
        textwrap.dedent(
            f"""
            [manifest]
            wave_id = "{wave_id}"

            [[kernels]]
            order = 1
            qualname = "trace_probe_fixture.kernel"
            trace_probe = "trace_probe_fixture:probe_kernel"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    port_target = port_root / "port_target.toml"
    port_target.write_text(
        textwrap.dedent(
            f"""
            [port]
            wave_id = "{wave_id}"
            symbol_qualname = "trace_probe_fixture.kernel"

            [parity]
            max_traces = 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    envelope, exit_code = audit_trace_counts(port_target)
    assert exit_code == 0
    assert envelope["pass_count"] == 1
    assert envelope["results"][0]["status"] == "pass"


def test_script_subprocess_on_repo_template_exits_zero() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_port_trace_count.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    envelope = json.loads(result.stdout)
    assert envelope["schema_version"] == "audit_port_trace_count_v0"
    assert envelope["max_traces"] == 1
    assert envelope["wave_id"] == "wave_001_example"

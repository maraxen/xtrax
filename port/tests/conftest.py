"""Graded parity harness fixtures and tier gates (P3-PARITY, AC-3/AC-4/AC-11)."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parent
PORT_TARGET_PATH = PORT_ROOT / "port_target.toml"
REFERENCE_ROOT = PORT_ROOT / "reference"
EMIT_PATH = PORT_ROOT / "emit" / "port_emit.py"

TIER_MARKERS = ("tier_1", "tier_2", "tier_3", "tier_4", "tier_5")
TIER_TO_EMIT = {
    "tier_1": "parity_tier_1",
    "tier_2": "parity_tier_2",
    "tier_3": "parity_tier_3",
    "tier_4": "parity_tier_4",
    "tier_5": "parity_tier_5",
}
DEFAULT_TASK_ID = "260617_xtrax-composition-mission"
TIER_TIMEOUT_SECONDS = 120

# backlog #3493 sub-task: numpy's assert_allclose/assert_array_almost_equal
# failure message reports the real discrepancy as free text; this regex covers
# both the legacy phrasing ("Max absolute difference: X") and the numpy>=2.x
# phrasing ("Max absolute difference among violations: X").
_MAX_ABS_DIFF_RE = re.compile(r"Max absolute difference(?: among violations)?:\s*([0-9.eE+\-]+)")


@dataclass(frozen=True)
class PortWaveConfig:
    port: dict[str, Any]
    capabilities: dict[str, Any]
    parity: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    blocking_tiers: tuple[str, ...]


def _load_port_emit_module():
    spec = importlib.util.spec_from_file_location("port_emit", EMIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load port emit module from {EMIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("port_emit", module)
    spec.loader.exec_module(module)
    return module


port_emit = _load_port_emit_module()


def _resolve_port_target_path() -> Path:
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        configured = data.get("tool", {}).get("port", {}).get("target")
        if configured:
            return (REPO_ROOT / configured).resolve()
    return PORT_TARGET_PATH.resolve()


def load_port_target(path: Path | None = None) -> dict[str, Any]:
    target_path = path or _resolve_port_target_path()
    return tomllib.loads(target_path.read_text(encoding="utf-8"))


def load_manifest(port_root: Path, wave_id: str) -> tuple[dict[str, Any], Path]:
    manifest_path = port_root / "manifests" / f"{wave_id}.toml"
    if not manifest_path.is_file():
        msg = f"manifest not found for wave_id={wave_id!r}: {manifest_path}"
        raise FileNotFoundError(msg)
    return tomllib.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def canonical_manifest_bytes(manifest_text: str) -> bytes:
    """Canonical TOML bytes for manifest_hash (excludes manifest_hash field)."""
    lines: list[str] = []
    for line in manifest_text.splitlines():
        if re.match(r"^\s*manifest_hash\s*=", line):
            continue
        lines.append(line)
    canonical = "\n".join(lines).rstrip() + "\n"
    return canonical.encode("utf-8")


def compute_manifest_hash(manifest_text: str) -> str:
    digest = hashlib.sha256(canonical_manifest_bytes(manifest_text)).hexdigest()
    return f"sha256:{digest}"


def verify_manifest_hash(manifest_path: Path) -> None:
    text = manifest_path.read_text(encoding="utf-8")
    manifest = tomllib.loads(text)
    recorded = manifest.get("manifest", {}).get("manifest_hash")
    if not isinstance(recorded, str) or not recorded:
        pytest.fail(f"{manifest_path}: missing [manifest].manifest_hash")
    expected = compute_manifest_hash(text)
    if recorded != expected:
        pytest.fail(
            f"{manifest_path}: manifest_hash mismatch "
            f"(recorded {recorded!r}, expected {expected!r})"
        )


def blocking_tiers_from_config(port_config: dict[str, Any]) -> tuple[str, ...]:
    parity = port_config.get("parity", {})
    ad_critical = bool(parity.get("ad_critical", False))
    justification = parity.get("ad_critical_justification", "")
    if ad_critical:
        if not isinstance(justification, str) or not justification.strip():
            msg = "port_target parity.ad_critical=true requires non-empty ad_critical_justification"
            raise ValueError(msg)
        return TIER_MARKERS
    return tuple(tier for tier in TIER_MARKERS if tier != "tier_4")


def build_wave_config(port_config: dict[str, Any], manifest_path: Path) -> PortWaveConfig:
    port_section = port_config.get("port", {})
    wave_id = port_section.get("wave_id")
    if not isinstance(wave_id, str) or not wave_id:
        raise ValueError("port_target.toml [port] wave_id is required")
    manifest, _ = load_manifest(PORT_ROOT, wave_id)
    return PortWaveConfig(
        port=port_section,
        capabilities=port_config.get("capabilities", {}),
        parity=port_config.get("parity", {}),
        manifest=manifest,
        manifest_path=manifest_path,
        blocking_tiers=blocking_tiers_from_config(port_config),
    )


def _tier_marker_for_item(item: pytest.Item) -> str | None:
    for tier in TIER_MARKERS:
        if item.get_closest_marker(tier) is not None:
            return tier
    return None


def _tier_sort_key(item: pytest.Item) -> tuple[int, str]:
    tier = _tier_marker_for_item(item)
    if tier is None:
        return (len(TIER_MARKERS), item.nodeid)
    return (TIER_MARKERS.index(tier), item.nodeid)


def _import_reference_algo(reference_subtree: str):
    ref_path = PORT_ROOT / reference_subtree / "algo.py"
    if not ref_path.is_file():
        raise FileNotFoundError(f"reference oracle missing: {ref_path}")
    spec = importlib.util.spec_from_file_location(
        f"port_reference_{reference_subtree.replace('/', '_')}",
        ref_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import reference oracle: {ref_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def port_wave() -> PortWaveConfig:
    port_config = load_port_target()
    port_section = port_config.get("port", {})
    wave_id = port_section["wave_id"]
    _, manifest_path = load_manifest(PORT_ROOT, wave_id)
    verify_manifest_hash(manifest_path)
    return build_wave_config(port_config, manifest_path)


@pytest.fixture(scope="session")
def port_target(port_wave: PortWaveConfig) -> dict[str, Any]:
    return {
        "port": port_wave.port,
        "capabilities": port_wave.capabilities,
        "parity": port_wave.parity,
    }


@pytest.fixture(scope="session")
def oracle(port_wave: PortWaveConfig):
    subtree = port_wave.port.get("reference_subtree", "")
    if not isinstance(subtree, str) or not subtree:
        raise ValueError("port_target [port] reference_subtree is required")
    return _import_reference_algo(subtree)


@pytest.fixture(scope="session")
def tier_gate() -> dict[str, Any]:
    return {"failed_at": None, "completed": set()}


def pytest_configure(config: pytest.Config) -> None:
    for tier in TIER_MARKERS:
        config.addinivalue_line(
            "markers",
            f"{tier}: graded parity tier ({TIER_TO_EMIT[tier]})",
        )
    config.addinivalue_line(
        "markers",
        f"timeout({TIER_TIMEOUT_SECONDS}): per-tier CPU budget (AC-4)",
    )


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del session, config
    port_config = load_port_target()
    blocking = blocking_tiers_from_config(port_config)
    for item in items:
        tier = _tier_marker_for_item(item)
        if tier is None:
            continue
        if tier not in blocking:
            item.add_marker(pytest.mark.skip(reason=f"{tier} skipped (ad_critical=false)"))
        item.add_marker(pytest.mark.timeout(TIER_TIMEOUT_SECONDS))
    items.sort(key=_tier_sort_key)


def pytest_runtest_setup(item: pytest.Item) -> None:
    tier = _tier_marker_for_item(item)
    if tier is None:
        return
    gate = item.session.stash.get(_TIER_GATE_STASH_KEY, None)
    if gate is None:
        return
    failed_at = gate.get("failed_at")
    if failed_at is None:
        return
    failed_index = TIER_MARKERS.index(failed_at)
    current_index = TIER_MARKERS.index(tier)
    if current_index > failed_index:
        pytest.skip(f"blocked after {failed_at} failure")


def _emit_tier_result(
    *,
    item: pytest.Item,
    port_wave: PortWaveConfig,
    tier: str,
    status: port_emit.TierStatus,
    error_taxonomy_class: str,
    max_discrepancy: float | None = None,
    traceback_excerpt: str = "",
) -> None:
    tolerance_policy = str(port_wave.parity.get("tolerance_policy", ""))
    task_id = str(port_wave.manifest.get("manifest", {}).get("task_id", DEFAULT_TASK_ID))
    symbol_qualname = str(port_wave.port.get("symbol_qualname", ""))
    oracle_id = str(port_wave.port.get("oracle_id", ""))
    port_parity_tier = TIER_TO_EMIT[tier]
    verdict = port_emit.TierVerdict(
        status=status,
        tolerance_policy=tolerance_policy,
        error_taxonomy_class=error_taxonomy_class,
        max_discrepancy=max_discrepancy,
    )
    evidence = port_emit.Evidence(
        pytest_nodeid=item.nodeid,
        traceback_excerpt=traceback_excerpt,
    )
    audits_path = REPO_ROOT / ".praxia" / "audits.jsonl"
    port_emit.emit_tier_verdict(
        task_id=task_id,
        symbol_qualname=symbol_qualname,
        port_parity_tier=port_parity_tier,
        oracle_id=oracle_id,
        tier_verdict=verdict,
        evidence=evidence,
        audits_path=audits_path,
    )


def _extract_max_discrepancy(exc: BaseException | None) -> float | None:
    """Best-effort extraction of the real numeric discrepancy from a failed
    parity assertion's exception message (backlog #3493 sub-task).

    The graded parity tiers fail exclusively via
    ``numpy.testing.assert_allclose``/``assert_array_almost_equal``, whose
    message embeds the actual max absolute difference as free text (there is
    no structured field numpy exposes for this). Returns ``None`` when there
    is no exception, or its message doesn't match the expected phrasing —
    callers must treat ``None`` as "unknown", not "zero".
    """
    if exc is None:
        return None
    match = _MAX_ABS_DIFF_RE.search(str(exc))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _max_discrepancy_for_call(call: pytest.CallInfo[None]) -> float | None:
    """Adapter over pytest.CallInfo for the FAIL branch of
    pytest_runtest_makereport — factored out so it (and, transitively,
    _extract_max_discrepancy) is unit-testable without hand-driving the
    hookwrapper generator."""
    if call.excinfo is None:
        return None
    return _extract_max_discrepancy(call.excinfo.value)


_TIER_GATE_STASH_KEY = pytest.StashKey[dict[str, Any]]()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    tier = _tier_marker_for_item(item)
    if tier is None or call.when != "call":
        return

    if _TIER_GATE_STASH_KEY not in item.session.stash:
        item.session.stash[_TIER_GATE_STASH_KEY] = {"failed_at": None, "completed": set()}

    gate = item.session.stash[_TIER_GATE_STASH_KEY]
    try:
        port_wave = item.session._port_wave_config  # type: ignore[attr-defined]
    except AttributeError:
        port_config = load_port_target()
        wave_id = port_config["port"]["wave_id"]
        _, manifest_path = load_manifest(PORT_ROOT, wave_id)
        port_wave = build_wave_config(port_config, manifest_path)
        item.session._port_wave_config = port_wave  # type: ignore[attr-defined]

    if report.passed:
        gate["completed"].add(tier)
        _emit_tier_result(
            item=item,
            port_wave=port_wave,
            tier=tier,
            status="PASS",
            error_taxonomy_class="none",
        )
    elif report.failed:
        if gate["failed_at"] is None:
            gate["failed_at"] = tier
        tb_excerpt = ""
        if call.excinfo is not None:
            tb_excerpt = str(call.excinfo.value)[:500]
        _emit_tier_result(
            item=item,
            port_wave=port_wave,
            tier=tier,
            status="FAIL",
            error_taxonomy_class="numeric_drift",
            max_discrepancy=_max_discrepancy_for_call(call),
            traceback_excerpt=tb_excerpt,
        )


@pytest.fixture(autouse=True)
def _ensure_repo_on_path() -> Iterator[None]:
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    port_str = str(PORT_ROOT.parent)
    if port_str not in sys.path:
        sys.path.insert(0, port_str)
    yield

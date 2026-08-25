"""D3 type-hardening gate: annotation coverage + shape specificity (N2.3 / #1583)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.baseline import (
    DEFAULT_BASELINE_PATH,
    evaluate_metric,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.findings import append_finding, emit_metric_finding

COVERAGE_METRIC_KEY = "type_hardening.annotation_coverage_pct"
SHAPE_METRIC_KEY = "type_hardening.shape_specificity_pct"
DIMENSION = "type_hardening"

JAXTYPING_ARRAY_NAMES = frozenset(
    {
        "Array",
        "ArrayLike",
        "Bool",
        "Complex",
        "Float",
        "Float32",
        "Float64",
        "Int",
        "Int32",
        "Int64",
        "Key",
        "Num",
        "PRNGKeyArray",
        "Scalar",
        "Shaped",
        "UInt",
    }
)


@dataclass(frozen=True, slots=True)
class AnnotationStats:
    public_params: int
    annotated_params: int
    array_annotations: int
    shape_typed_arrays: int

    @property
    def annotation_coverage_pct(self) -> float:
        if self.public_params == 0:
            return 0.0
        return 100.0 * self.annotated_params / self.public_params

    @property
    def shape_specificity_pct(self) -> float:
        if self.array_annotations == 0:
            return 100.0
        return 100.0 * self.shape_typed_arrays / self.array_annotations


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    stats: AnnotationStats
    annotation_coverage_pct: float
    shape_specificity_pct: float
    findings_emitted: int
    baseline_updated: bool
    coverage_metric_key: str = COVERAGE_METRIC_KEY
    shape_metric_key: str = SHAPE_METRIC_KEY


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _annotation_root_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_root_name(node.value)
    return None


def _is_array_annotation(node: ast.expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_array_annotation(node.left) or _is_array_annotation(node.right)
    root = _annotation_root_name(node)
    if root in JAXTYPING_ARRAY_NAMES:
        return True
    if isinstance(node, ast.Subscript):
        return _is_array_annotation(node.value) or _subscript_has_shape_string(node)
    return False


def _subscript_has_shape_string(node: ast.Subscript) -> bool:
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple):
        return any(_slice_elt_has_shape(elt) for elt in slice_node.elts)
    return _slice_elt_has_shape(slice_node)


def _slice_elt_has_shape(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) or node.value is Ellipsis
    return False


def _annotation_has_shape(node: ast.expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_has_shape(node.left) or _annotation_has_shape(node.right)
    if isinstance(node, ast.Subscript):
        return _subscript_has_shape_string(node)
    return False


def _iter_signature_params(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.arg]:
    for arg in (
        *fn.args.posonlyargs,
        *fn.args.args,
        *fn.args.kwonlyargs,
    ):
        if arg.arg in {"self", "cls"}:
            continue
        yield arg
    if fn.args.vararg is not None and fn.args.vararg.arg not in {"self", "cls"}:
        yield fn.args.vararg
    if fn.args.kwarg is not None and fn.args.kwarg.arg not in {"self", "cls"}:
        yield fn.args.kwarg


def _record_annotation(
    stats: AnnotationStats,
    node: ast.expr | None,
    *,
    annotated: bool,
) -> AnnotationStats:
    public_params = stats.public_params + 1
    annotated_params = stats.annotated_params + (1 if annotated else 0)
    array_annotations = stats.array_annotations
    shape_typed_arrays = stats.shape_typed_arrays
    if annotated and node is not None and _is_array_annotation(node):
        array_annotations += 1
        if _annotation_has_shape(node):
            shape_typed_arrays += 1
    return AnnotationStats(
        public_params=public_params,
        annotated_params=annotated_params,
        array_annotations=array_annotations,
        shape_typed_arrays=shape_typed_arrays,
    )


def inspect_public_callable(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    qualname: str,
) -> list[str]:
    """Return human-readable violations for a single public callable."""
    violations: list[str] = []
    for arg in _iter_signature_params(fn):
        if arg.annotation is None:
            violations.append(f"{qualname}: parameter `{arg.arg}` missing annotation")
        elif _is_array_annotation(arg.annotation) and not _annotation_has_shape(arg.annotation):
            violations.append(
                f"{qualname}: parameter `{arg.arg}` array annotation lacks shape axes"
            )
    if fn.returns is None:
        violations.append(f"{qualname}: missing return annotation")
    elif _is_array_annotation(fn.returns) and not _annotation_has_shape(fn.returns):
        violations.append(f"{qualname}: return array annotation lacks shape axes")
    return violations


def _scan_function(
    stats: AnnotationStats,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    public: bool,
) -> AnnotationStats:
    if not public:
        return stats
    for arg in _iter_signature_params(fn):
        stats = _record_annotation(
            stats,
            arg.annotation,
            annotated=arg.annotation is not None,
        )
    stats = _record_annotation(
        stats,
        fn.returns,
        annotated=fn.returns is not None,
    )
    return stats


def _scan_class(stats: AnnotationStats, node: ast.ClassDef) -> AnnotationStats:
    if not _is_public(node.name):
        return stats
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stats = _scan_function(stats, child, public=_is_public(child.name))
    return stats


def _scan_module(tree: ast.Module) -> AnnotationStats:
    stats = AnnotationStats(0, 0, 0, 0)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stats = _scan_function(stats, node, public=_is_public(node.name))
        elif isinstance(node, ast.ClassDef):
            stats = _scan_class(stats, node)
    return stats


def scan_package_annotations(root: Path) -> AnnotationStats:
    """AST-scan public parameters/returns under ``root`` for jaxtyping coverage."""
    total = AnnotationStats(0, 0, 0, 0)
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        module_stats = _scan_module(tree)
        total = AnnotationStats(
            public_params=total.public_params + module_stats.public_params,
            annotated_params=(total.annotated_params + module_stats.annotated_params),
            array_annotations=(total.array_annotations + module_stats.array_annotations),
            shape_typed_arrays=(total.shape_typed_arrays + module_stats.shape_typed_arrays),
        )
    return total


def run_type_hardening_gate(
    target: Path,
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    *,
    run_id: str | None = None,
    write_baseline: bool = True,
) -> GateResult:
    """Scan annotation metrics, emit findings, evaluate baseline ratchet."""
    stats = scan_package_annotations(target.resolve())
    coverage_pct = stats.annotation_coverage_pct
    shape_pct = stats.shape_specificity_pct

    emitted = 0
    if stats.public_params > stats.annotated_params:
        record = emit_metric_finding(
            dim=DIMENSION,
            severity="info",
            file_line=str(target),
            evidence=(
                f"annotation coverage {coverage_pct:.1f}% "
                f"({stats.annotated_params}/{stats.public_params} public slots)"
            ),
            rule_id="type_hardening.missing_annotations",
            symbol_qualname="",
            payload={
                "violation_kind": "annotation_coverage",
                "annotation_coverage_pct": coverage_pct,
                "public_params": stats.public_params,
                "annotated_params": stats.annotated_params,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    if stats.array_annotations > stats.shape_typed_arrays:
        record = emit_metric_finding(
            dim=DIMENSION,
            severity="info",
            file_line=str(target),
            evidence=(
                f"shape specificity {shape_pct:.1f}% "
                f"({stats.shape_typed_arrays}/{stats.array_annotations} array slots)"
            ),
            rule_id="type_hardening.missing_shape_axes",
            symbol_qualname="",
            payload={
                "violation_kind": "shape_specificity",
                "shape_specificity_pct": shape_pct,
                "array_annotations": stats.array_annotations,
                "shape_typed_arrays": stats.shape_typed_arrays,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    baseline = load_baseline(path=baseline_path)
    passes_coverage, update_coverage = evaluate_metric(
        baseline,
        COVERAGE_METRIC_KEY,
        coverage_pct,
    )
    passes_shape, update_shape = evaluate_metric(
        baseline,
        SHAPE_METRIC_KEY,
        shape_pct,
    )
    passed = passes_coverage and passes_shape

    baseline_updated = False
    if passed and write_baseline and (update_coverage or update_shape):
        tightened = baseline
        if update_coverage:
            tightened = update_metric(
                tightened,
                COVERAGE_METRIC_KEY,
                coverage_pct,
                "maximize",
            )
        if update_shape:
            tightened = update_metric(
                tightened,
                SHAPE_METRIC_KEY,
                shape_pct,
                "maximize",
            )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passed,
        stats=stats,
        annotation_coverage_pct=coverage_pct,
        shape_specificity_pct=shape_pct,
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
    )

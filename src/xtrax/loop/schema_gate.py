"""Schema-gate (T2-12, #2181, AC-2).

AC-2's claim: a candidate slotted into a StageBundle must have its `extract_schema`-derived
schema (`jax.eval_shape`, zero FLOPs) equal the slot's declared `BundleSchema` before any real
execution -- reject pre-compute on any mismatch. This module is the glue, not new machinery: the
actual trace-and-schema-extraction primitive (`xtrax.inference.extract_schema`/`BundleSchema`) and
its sibling `verify_structure`/`StructureMismatchError` (consumed by T2-13) already exist, fully
built, and are already composed together by `xtrax.inference.api.infer_bundle`. What did not exist
is (a) a way to resolve a `candidate_path` (T2-11's own scoped concept -- a source-file `Path`)
into a live callable, and (b) a loud, diagnostic schema-equality check with this epic's fast/loud
exception convention.

`resolve_candidate_callable` deliberately does its own fresh import rather than reusing
`xtrax.loop.candidate_static`'s private `_check_clean_import` -- that helper is unexported, and
discards the imported module (it only needs pass/fail for T2-11's own narrower check). Keeping
this module's import logic self-contained matches every other T2-0X module's stance of not
reaching into a merged sibling's private internals.

Extension seam: this module does not decide how a caller obtains `declared_schema` for a given
`StageBundle` slot -- there is currently no "declared schema per slot" concept on `StageBundle`
itself (`xtrax.stages.bundle`), and inventing one here would be speculative scope creep onto a
type this module doesn't own. `assert_schema_gate` takes `declared_schema` as an explicit
caller-supplied parameter; associating a StageBundle slot with its declared schema remains
real, separate, not-yet-scoped work for whatever eventually becomes the #2181 loop controller.
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jax import ShapeDtypeStruct

from xtrax.inference.schema import BundleSchema, extract_schema


class SchemaGateError(Exception):
    """Base for AC-2's schema-gate rejection errors. Reject pre-compute; do not execute."""


class CandidateResolutionError(SchemaGateError):
    """`candidate_path` could not be imported, or `callable_name` isn't a callable attribute."""


class SchemaMismatchError(SchemaGateError):
    """The candidate's `extract_schema`-derived schema does not equal `declared_schema`."""


def resolve_candidate_callable(candidate_path: Path, callable_name: str) -> Callable[..., Any]:
    """Import `candidate_path` fresh and return its `callable_name` attribute.

    Raises:
        CandidateResolutionError: the file can't be imported (syntax error, missing dependency,
            etc.), or `callable_name` is missing or not callable on the imported module.
    """
    try:
        spec = importlib.util.spec_from_file_location("__schema_gate_candidate__", candidate_path)
        if spec is None or spec.loader is None:
            msg = f"failed to create module spec for {candidate_path}"
            raise CandidateResolutionError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except CandidateResolutionError:
        raise
    except Exception as exc:
        msg = f"failed to import candidate {candidate_path}: {type(exc).__name__}: {exc}"
        raise CandidateResolutionError(msg) from exc

    fn = getattr(module, callable_name, None)
    if fn is None:
        msg = f"candidate {candidate_path} has no attribute {callable_name!r}"
        raise CandidateResolutionError(msg)
    if not callable(fn):
        msg = (
            f"candidate {candidate_path}.{callable_name} is not callable (got {type(fn).__name__})"
        )
        raise CandidateResolutionError(msg)
    return fn


def _diff_schemas(derived: BundleSchema, declared: BundleSchema) -> list[str]:
    """Return a list of human-readable field-level diffs, empty iff the schemas match."""
    diffs: list[str] = []
    derived_keys = set(derived.fields)
    declared_keys = set(declared.fields)

    for name in sorted(declared_keys - derived_keys):
        diffs.append(f"field {name!r} declared but not produced by the candidate")
    for name in sorted(derived_keys - declared_keys):
        diffs.append(f"field {name!r} produced by the candidate but not declared")

    for name in sorted(derived_keys & declared_keys):
        derived_leaf = derived.fields[name]
        declared_leaf = declared.fields[name]
        if derived_leaf.shape != declared_leaf.shape or derived_leaf.dtype != declared_leaf.dtype:
            diffs.append(
                f"field {name!r}: declared shape={declared_leaf.shape} "
                f"dtype={declared_leaf.dtype}, derived shape={derived_leaf.shape} "
                f"dtype={derived_leaf.dtype}"
            )

    if derived.carry_specs != declared.carry_specs:
        diffs.append(
            f"carry_specs mismatch: declared={declared.carry_specs!r} "
            f"derived={derived.carry_specs!r}"
        )

    return diffs


def assert_schema_gate(
    candidate_path: Path,
    callable_name: str,
    *,
    abstract_inputs: list[ShapeDtypeStruct] | tuple[ShapeDtypeStruct, ...],
    declared_schema: BundleSchema,
) -> BundleSchema:
    """The composed AC-2 gate: resolve, trace (zero FLOPs), and compare against `declared_schema`.

    `jax.eval_shape` (inside `extract_schema`) never executes real ops, so a rejection here costs
    no compute -- exactly AC-2's "reject before any execution" claim.

    Returns:
        The derived `BundleSchema`, once confirmed to match `declared_schema` -- useful for a
        caller to pass forward (e.g. to T2-13's structure-tripwire) without re-tracing.

    Raises:
        CandidateResolutionError: see `resolve_candidate_callable`.
        SchemaMismatchError: the derived schema does not equal `declared_schema`, naming every
            mismatched/missing/extra field.
    """
    fn = resolve_candidate_callable(candidate_path, callable_name)
    derived_schema = extract_schema(fn, abstract_inputs)

    diffs = _diff_schemas(derived_schema, declared_schema)
    if diffs:
        msg = "candidate schema does not match declared BundleSchema:\n  - " + "\n  - ".join(diffs)
        raise SchemaMismatchError(msg)

    return derived_schema


__all__ = [
    "CandidateResolutionError",
    "SchemaGateError",
    "SchemaMismatchError",
    "assert_schema_gate",
    "resolve_candidate_callable",
]

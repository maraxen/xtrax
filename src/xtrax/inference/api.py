"""Public entrypoint for signature inference: infer_bundle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xtrax.inference.axes import synthesize_axes
from xtrax.inference.config import get_axis_config
from xtrax.inference.schema import BundleSchema, extract_schema
from xtrax.inference.verify import verify_structure
from xtrax.tiling.plan import AxisSpec


def infer_bundle(
    fn: Any,
    abstract_inputs: Sequence[Any],
    *,
    verify_against: Sequence[Any] | None = None,
) -> tuple[BundleSchema, list[AxisSpec]]:
    """Infer a BundleSchema and list of AxisSpec objects for *fn*.

    Given a traceable JAX function and a sequence of abstract inputs
    (ShapeDtypeStruct or (shape, dtype) specs), this function:

    1. Extracts the output schema via :func:`~xtrax.inference.schema.extract_schema`.
    2. Retrieves any axis overrides from the ``@axis_config`` sidecar attached
       to *fn* (or ``None`` if the function is not decorated).
    3. Synthesizes :class:`~xtrax.tiling.plan.AxisSpec` objects from the
       abstract inputs and overrides.
    4. Optionally verifies structural consistency against concrete inputs.

    Args:
        fn: A pure, traceable JAX function.  Its positional arguments must
            match the elements of *abstract_inputs*.
        abstract_inputs: Sequence of abstract input specs.  Each element is
            either a :class:`jax.ShapeDtypeStruct` (passed through as-is) or a
            ``(shape, dtype)`` tuple (normalised via
            :func:`~xtrax.inference.abstract.build_abstract_inputs`).
        verify_against: Optional sequence of concrete inputs.  When provided,
            :func:`~xtrax.inference.verify.verify_structure` is called before
            returning; :class:`~xtrax.inference.errors.StructureMismatchError`
            is raised if the abstract and concrete output structures diverge.

    Returns:
        A two-tuple ``(schema, axes)`` where:
        - *schema* is a :class:`~xtrax.inference.schema.BundleSchema` whose
          ``fields`` map leaf names to :class:`jax.ShapeDtypeStruct`.
        - *axes* is a list of :class:`~xtrax.tiling.plan.AxisSpec` objects,
          one per qualifying (ndim >= 1) input leaf.  When no ``@axis_config``
          decorator is present every axis has ``role == AxisRole.UNKNOWN``.

    Raises:
        StructureMismatchError: If *verify_against* is provided and the
            abstract output structure differs from the concrete output.
        Any error raised by :func:`jax.eval_shape` during schema extraction.
    """
    # Normalise abstract_inputs: elements may already be ShapeDtypeStruct
    # or (shape, dtype) tuples.  build_abstract_inputs handles each leaf;
    # we apply it element-wise so the list structure is preserved.
    from xtrax.inference.abstract import build_abstract_inputs

    normalised: list[Any] = [build_abstract_inputs(inp) for inp in abstract_inputs]

    # 1. Extract output schema.
    schema: BundleSchema = extract_schema(fn, normalised)

    # 2. Retrieve @axis_config overrides (None if bare fn).
    overrides = get_axis_config(fn)

    # 3. Synthesize axes from abstract inputs and optional overrides.
    axes: list[AxisSpec] = synthesize_axes(normalised, overrides)

    # 4. Optional structure verification against concrete inputs.
    if verify_against is not None:
        verify_structure(fn, normalised, list(verify_against))

    return schema, axes

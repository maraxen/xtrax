"""Load safetensors weights and bring their dtypes inside a target's envelope.

HuggingFace checkpoints ship bf16 and f16 routinely, and ``native`` -- the only
target that executes, and therefore the only one that can verify anything --
cannot run bf16 at all: IREE's runtime has no bf16 buffer-to-numpy mapping. So a
bf16 checkpoint has to be cast before it can be exported and checked. Doing that
explicitly, and reporting every leaf it happened to, is the point of this
module; an implicit cast would change the exported artifact's dtypes silently.

Scope note: the spike this is promoted from also carried a ``TinyMLP`` and a
``_fit`` helper that sliced or zero-padded real checkpoint tensors to whatever
shape the caller asked for. Neither is promoted. They were scaffolding for a
driver script, and silently reshaping someone's weights to fit is a footgun
rather than library surface. What is promoted is the part with a reason to
exist: load, cast within a target's envelope, and report.

``huggingface_hub`` and ``safetensors`` are imported lazily, so importing this
module needs only the base install.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.export.safety import DtypeNotSupportedError, dtype_name
from xtrax.export.targets import Target

__all__ = [
    "HFWeightsError",
    "LoadedWeights",
    "WeightReport",
    "load_hf_weights",
]

_MISSING_EXTRA = "install the export toolchain with: pip install xtrax[export]"


class HFWeightsError(Exception):
    """Raised when a checkpoint cannot be fetched or read."""


@dataclass(frozen=True)
class WeightReport:
    """What the load actually did, in enough detail to audit it.

    Attributes:
        source: Where the tensors came from, e.g. ``"org/model/model.safetensors"``.
        tensors_seen: How many tensors the file held.
        dtypes_cast: One ``"name: from -> to"`` entry per cast tensor. Never
            truncated -- the spike kept only the first two, which turns a report
            meant for auditing into a sample.
    """

    source: str
    tensors_seen: int
    dtypes_cast: tuple[str, ...]


@dataclass(frozen=True)
class LoadedWeights:
    """Tensors ready for export, plus the record of what was done to them.

    Attributes:
        tensors: Tensor name -> JAX array, dtypes already inside the target's
            envelope.
        report: The accompanying WeightReport.
    """

    tensors: dict[str, Any]
    report: WeightReport


def _accepts(target: Target, name: str, request_features: frozenset[str]) -> bool:
    """Whether ``target`` takes this dtype outright or via a requested feature."""
    if name in target.supported_dtypes:
        return True
    if name in target.optional_dtypes:
        feature = target.optional_dtype_features.get(name)
        return feature is not None and feature in request_features
    return False


def _read_safetensors(repo_id: str | None, filename: str, local_path: Path | None) -> Any:
    """Return the tensors in a safetensors file, downloading it only if needed."""
    try:
        from safetensors.numpy import load_file  # ty: ignore[unresolved-import]
    except ImportError as exc:
        msg = f"safetensors is required to read checkpoints: {_MISSING_EXTRA}"
        raise HFWeightsError(msg) from exc

    if local_path is not None:
        if not local_path.is_file():
            msg = f"no such checkpoint: {local_path}"
            raise HFWeightsError(msg)
        return load_file(str(local_path))

    try:
        from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]
    except ImportError as exc:
        msg = f"huggingface_hub is required to fetch {repo_id!r}: {_MISSING_EXTRA}"
        raise HFWeightsError(msg) from exc

    if repo_id is None:  # pragma: no cover - guarded by load_hf_weights
        msg = "no repo_id to download from, and no local_path to read instead."
        raise HFWeightsError(msg)

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:  # noqa: BLE001 - hub raises a wide family
        msg = f"could not fetch {repo_id}/{filename}: {exc}"
        raise HFWeightsError(msg) from exc
    return load_file(path)


def load_hf_weights(
    repo_id: str | None = None,
    target: Target | None = None,
    *,
    filename: str = "model.safetensors",
    local_path: Path | None = None,
    request_features: frozenset[str] = frozenset(),
    cast_to: str = "f32",
) -> LoadedWeights:
    """Load safetensors weights, casting any dtype ``target`` will not carry.

    Args:
        repo_id: HuggingFace repo, e.g. ``"org/model"``. Ignored when
            ``local_path`` is given.
        target: The target these weights will be exported for; supplies the
            dtype envelope. Required.
        filename: File within the repo.
        local_path: Read this file instead of downloading anything.
        request_features: Device features unlocking the target's optional dtypes.
        cast_to: Dtype to cast rejected tensors to. Must itself be one the
            target accepts.

    Returns:
        The tensors with acceptable dtypes, and a report naming every cast.

    Raises:
        HFWeightsError: The checkpoint could not be read, or was empty.
        DtypeNotSupportedError: A tensor is f64, or ``cast_to`` is itself a dtype
            the target rejects. f64 is never silently downcast -- losing half
            your precision should be a decision you made, not one made for you.
        ValueError: Neither ``repo_id`` nor ``local_path`` was given, or no
            target was.
    """
    if target is None:
        msg = "target is required: the dtype envelope to cast into comes from it."
        raise ValueError(msg)
    if repo_id is None and local_path is None:
        msg = "pass repo_id= to download a checkpoint, or local_path= to read one."
        raise ValueError(msg)

    if not _accepts(target, cast_to, request_features):
        supported = ", ".join(sorted(target.supported_dtypes))
        msg = (
            f"cast_to={cast_to!r} is not accepted by target {target.name!r}, so "
            f"casting into it would not help. Supported: {supported}."
        )
        raise DtypeNotSupportedError(msg)

    import jax.numpy as jnp

    tensors = _read_safetensors(repo_id, filename, local_path)
    source = str(local_path) if local_path is not None else f"{repo_id}/{filename}"
    if not tensors:
        msg = f"{source} contained no tensors"
        raise HFWeightsError(msg)

    cast: list[str] = []
    out: dict[str, Any] = {}
    for name in sorted(tensors):
        arr = tensors[name]
        current = dtype_name(arr.dtype)
        if current == "f64":
            msg = (
                f"tensor {name!r} is f64, which no target carries: IREE demotes it "
                f"to f32 and rewrites the artifact's signature. Cast it yourself, "
                f"so the precision loss is a decision rather than a surprise."
            )
            raise DtypeNotSupportedError(msg)
        if _accepts(target, current, request_features):
            out[name] = jnp.asarray(arr)
            continue
        cast.append(f"{name}: {current} -> {cast_to}")
        out[name] = jnp.asarray(arr).astype(jnp.dtype(_numpy_name(cast_to)))

    return LoadedWeights(
        tensors=out,
        report=WeightReport(source=source, tensors_seen=len(tensors), dtypes_cast=tuple(cast)),
    )


def _numpy_name(short: str) -> str:
    """Expand a short dtype name back to the one numpy/JAX construct from."""
    if short == "bool":
        return "bool"
    for long, prefix in (("bfloat", "bf"), ("float", "f"), ("int", "i"), ("uint", "u")):
        if short.startswith(prefix) and short[len(prefix) :].isdigit():
            return long + short[len(prefix) :]
    return short

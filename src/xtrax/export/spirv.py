"""SPIR-V extraction and shader validation.

Extraction is implemented; validation is not, and deliberately so.

Measured caveat, recorded here because it governs what this module can ever
claim (see ``.praxia/docs/research/260901_webgpu-export-measurement-pass.md``):
IREE's Vulkan HAL passes dispatch parameters through push constants, and push
constants are not part of the W3C WebGPU feature set. A validator configured
the way a browser is configured therefore rejects IREE's SPIR-V. A validation
device must never be constructed with extra features enabled to get around
that -- doing so produces a passing check that establishes nothing. That is why
no ``validate_webgpu`` exists here and why no target is registered at
``VerificationLevel.VALIDATED``.

This module holds no IREE import. It reads bytes IREE already wrote.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SPIRV_MAGIC_BE",
    "SPIRV_MAGIC_LE",
    "SpirvValidationResult",
    "is_spirv",
    "spirv_binaries_in",
]

# SPIR-V's 0x07230203 magic, as the four bytes actually found at the head of a
# module. Both orderings are recognised: the word is stored in the producer's
# endianness, and the reader is supposed to detect which from the magic itself.
SPIRV_MAGIC_LE = b"\x03\x02\x23\x07"
SPIRV_MAGIC_BE = b"\x07\x23\x02\x03"


def is_spirv(data: bytes) -> bool:
    """True if ``data`` begins with the SPIR-V magic number.

    The filter exists because an IREE executable dump directory is not
    homogeneous: ``metal-spirv`` writes Metal Shading Language source there,
    whose head is the ASCII ``#inc`` of an ``#include`` line, and treating that
    as SPIR-V would hand a caller bytes no shader tool can read.

    Args:
        data: The head of a candidate file; only the first four bytes are read.

    Returns:
        Whether the bytes carry the SPIR-V magic in either endianness.
    """
    return data[:4] in (SPIRV_MAGIC_LE, SPIRV_MAGIC_BE)


def spirv_binaries_in(dump_dir: Path) -> dict[str, bytes]:
    """Collect the SPIR-V modules IREE dumped into ``dump_dir``.

    Args:
        dump_dir: Directory given to ``--iree-hal-dump-executable-binaries-to``.

    Returns:
        Filename -> module bytes, for every file carrying the SPIR-V magic,
        sorted by name so the mapping is stable across runs. Empty when the
        backend emitted no SPIR-V, which is the normal result for a backend that
        emits something else.
    """
    found: dict[str, bytes] = {}
    if not dump_dir.is_dir():
        return found
    for path in sorted(dump_dir.iterdir()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if is_spirv(data):
            found[path.name] = data
    return found


@dataclass(frozen=True)
class SpirvValidationResult:
    """Outcome of validating one SPIR-V module against a shader validator.

    Attributes:
        valid: Whether the validator accepted the module.
        adapter_type: Adapter class reported by the validator, e.g. ``"CPU"``.
        backend: Validator backend, e.g. ``"Vulkan"``.
        device_name: Human-readable device name, e.g. ``"llvmpipe"``.
        error: The validator's message when ``valid`` is False, else None.
    """

    valid: bool
    adapter_type: str
    backend: str
    device_name: str
    error: str | None

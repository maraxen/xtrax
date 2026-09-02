"""SPIR-V extraction and shader validation.

PR1 defines only the result record, so ``ExportResult`` can reference it in a
frozen-dataclass field. Extraction and validation land with the SPIR-V targets.

Measured caveat, recorded here because it governs what this module can ever
claim (see ``.praxia/docs/research/260901_webgpu-export-measurement-pass.md``):
IREE's Vulkan HAL passes dispatch parameters through push constants, and push
constants are not part of the W3C WebGPU feature set. A validator configured
the way a browser is configured therefore rejects IREE's SPIR-V. A validation
device must never be constructed with extra features enabled to get around
that -- doing so produces a passing check that establishes nothing.
"""

from dataclasses import dataclass

__all__ = ["SpirvValidationResult"]


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

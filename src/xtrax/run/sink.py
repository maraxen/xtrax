"""Output sink routing configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SinkSpec:
    """Routing config for output sinks."""

    output_dir: Path | None = None
    format: Literal["jsonl", "h5", "none"] = "jsonl"
    flush_every: int = 1

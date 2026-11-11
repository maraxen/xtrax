"""Base protocol and abstract interface for xtrax EDA visualization backends."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from xtrax.eda.types import PlanStatsDict


def reject_bundle_js(bundle_js: bool, backend: str = "matplotlib") -> None:
    """Fail loudly on bundle_js=True for backends that do not support JS bundling."""
    if bundle_js:
        raise NotImplementedError(
            f"bundle_js=True is not implemented for backend {backend!r}. "
            "Use backend='vega' with fmt='html' for self-contained JavaScript bundling."
        )


class BaseBackend(ABC):
    """Abstract base class for EDA rendering backends."""

    name: str

    @abstractmethod
    def render_spec(
        self,
        spec: dict[str, Any],
        fmt: Literal["png", "svg", "html"],
        stats: PlanStatsDict,
        options: dict[str, Any] | None = None,
    ) -> bytes | str:
        """Render a compiled Vega-Lite specification or plan stats dictionary.

        Args:
            spec: Compiled Vega-Lite specification dictionary.
            fmt: Output format ("png", "svg", or "html").
            stats: Raw intermediate plan statistics.
            options: Backend-specific rendering options.

        Returns:
            Rendered bytes (PNG/SVG) or string (HTML).
        """
        ...

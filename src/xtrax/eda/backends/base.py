"""Base protocol and abstract interface for xtrax EDA visualization backends."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from xtrax.eda.types import PlanStatsDict


def reject_bundle_js(bundle_js: bool) -> None:
    """Fail loudly on bundle_js=True: vendored Vega JS is not shipped.

    A silent no-op here would return HTML that still loads its scripts from a CDN,
    which an airgapped node cannot reach.
    """
    if bundle_js:
        raise NotImplementedError(
            "bundle_js=True is not implemented: vendored Vega JS is not shipped. "
            "Use offline=True for the static SVG fallback."
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

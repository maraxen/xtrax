"""Domain analytical plugins for xtrax Campaign EDA.

Provides core domain plugin interfaces, registry management, schema validation
result structures, and builtin reference plugins.
"""

from dataclasses import dataclass
from typing import Any

from xtrax.eda.campaign.domains.aminx import AminxDomainPlugin
from xtrax.eda.campaign.plugins import (
    DomainPlugin,
    discover_domain_plugins,
    register_domain_plugin,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Diagnostic issue produced during domain plugin validation."""

    message: str
    severity: str = "error"
    field: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating a domain plugin or domain metric result against schema."""

    is_valid: bool
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def has_errors(self) -> bool:
        """Return True if any issues are of error severity."""
        return any(issue.severity == "error" for issue in self.issues)


class PluginRegistry:
    """Registry interface for discovering and managing domain plugins."""

    @classmethod
    def register(cls, plugin: DomainPlugin) -> None:
        """Register a domain plugin instance."""
        register_domain_plugin(plugin)

    @classmethod
    def discover(cls) -> dict[str, DomainPlugin]:
        """Discover registered domain plugins and entry points."""
        return discover_domain_plugins()

    @classmethod
    def get(cls, name: str) -> DomainPlugin | None:
        """Retrieve a registered domain plugin by name."""
        return discover_domain_plugins().get(name)


def validate_domain_plugin(plugin: Any) -> ValidationResult:
    """Validate that a plugin instance satisfies the DomainPlugin protocol."""
    issues: list[ValidationIssue] = []

    if not isinstance(plugin, DomainPlugin):
        issues.append(
            ValidationIssue(
                message=f"Plugin {type(plugin).__name__} does not implement DomainPlugin protocol",
                severity="error",
            )
        )
    if not hasattr(plugin, "domain_name") or not isinstance(plugin.domain_name, str):
        issues.append(
            ValidationIssue(
                message="Plugin must have a string 'domain_name' property",
                severity="error",
                field="domain_name",
            )
        )
    if not hasattr(plugin, "version") or not isinstance(plugin.version, str):
        issues.append(
            ValidationIssue(
                message="Plugin must have a string 'version' property",
                severity="error",
                field="version",
            )
        )
    if not hasattr(plugin, "extract_cell") or not callable(plugin.extract_cell):
        issues.append(
            ValidationIssue(
                message="Plugin must implement callable 'extract_cell'",
                severity="error",
                field="extract_cell",
            )
        )
    if not hasattr(plugin, "compute_metrics") or not callable(plugin.compute_metrics):
        issues.append(
            ValidationIssue(
                message="Plugin must implement callable 'compute_metrics'",
                severity="error",
                field="compute_metrics",
            )
        )

    return ValidationResult(is_valid=len(issues) == 0, issues=tuple(issues))


# Register builtin reference plugins
register_domain_plugin(AminxDomainPlugin())

__all__ = [
    "AminxDomainPlugin",
    "DomainPlugin",
    "PluginRegistry",
    "ValidationIssue",
    "ValidationResult",
    "validate_domain_plugin",
]

"""Domain analytical plugins for xtrax Campaign EDA."""

from xtrax.eda.campaign.domains.aminx import AminxDomainPlugin
from xtrax.eda.campaign.domains.demistify import DemistifyDomainPlugin
from xtrax.eda.campaign.plugins import register_domain_plugin

# Register builtin reference plugins
register_domain_plugin(AminxDomainPlugin())
register_domain_plugin(DemistifyDomainPlugin())

__all__ = ["AminxDomainPlugin", "DemistifyDomainPlugin"]

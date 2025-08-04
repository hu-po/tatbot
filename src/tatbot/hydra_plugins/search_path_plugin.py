"""Hydra search path plugin to allow external packages to contribute configs."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class TatbotSearchPathPlugin(SearchPathPlugin):
    """Plugin to add external package configs to Hydra search path."""
    
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Add search paths for external packages
        # This allows external packages to contribute their own configs
        # without editing the main tatbot repository
        search_path.append("provider", "pkg://tatbot_external.conf")
        search_path.append("provider", "pkg://tatbot_custom.conf")
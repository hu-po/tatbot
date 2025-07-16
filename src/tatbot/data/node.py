from dataclasses import dataclass

from tatbot.data import Yaml

@dataclass
class Node(Yaml):
    """Node in the tatbot network."""
    name: str
    """Name of the node."""
    ip: str
    """IP address of the node, used for SSH connection."""
    user: str
    """Username for SSH connection."""
    emoji: str = "🌐"
    """Emoji to use for logging."""
    deps: str = "."
    """Dependencies to install on the node, see pyproject.toml."""

    yaml_dir: str = "~/tatbot/config"
    """Directory containing the config yaml files."""
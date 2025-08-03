import ipaddress

from pydantic import field_validator

from tatbot.data.base import BaseCfg


class Node(BaseCfg):
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

    @field_validator('ip')
    def validate_ip(cls, v):
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")
        return v

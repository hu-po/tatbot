"""Pydantic models for MCP server settings."""

from typing import List, Optional

from pydantic_settings import BaseSettings

from tatbot.utils.constants import CONF_SCENES_DIR


class MCPConstants:
    """Configuration constants for MCP models."""
    DEFAULT_HOST: str = "0.0.0.0"
    DEFAULT_PORT: int = 8000
    DEFAULT_TRANSPORT: str = "streamable-http"
    DEFAULT_ALLOWED_IPS: List[str] = ["127.0.0.1", "::1"]
    DEFAULT_VERSION: str = "1.0"
    SCENES_CONFIG_PATH: str = str(CONF_SCENES_DIR)


class MCPSettings(BaseSettings):
    """MCP server settings with environment variable overrides."""
    host: str = MCPConstants.DEFAULT_HOST
    port: int = MCPConstants.DEFAULT_PORT
    transport: str = MCPConstants.DEFAULT_TRANSPORT
    debug: bool = False
    extras: List[str] = []
    tools: List[str] = []

    # Security settings
    auth_token: Optional[str] = None
    ip_allowlist: List[str] = MCPConstants.DEFAULT_ALLOWED_IPS
    require_auth: bool = True

    class Config:
        env_prefix = "MCP_"

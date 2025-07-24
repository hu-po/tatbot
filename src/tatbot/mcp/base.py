"""Base MCP server abstractions."""

from dataclasses import dataclass

from tatbot.utils.log import get_logger

log = get_logger("mcp.base", "🔌")

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""
    transport: str = "streamable-http"
    """Transport type for MCP server."""
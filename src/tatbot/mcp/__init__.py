"""MCP (Model Control Protocol) module for tatbot."""

from .models import MCPSettings
from tatbot.tools.system.models import (
    ListNodesInput,
    ListNodesOutput as ListNodesResponse,
    ListScenesInput,
    ListScenesOutput as ListScenesResponse,
    PingNodesInput,
    PingNodesOutput as PingNodesResponse,
)

__all__ = [
    "PingNodesInput", "PingNodesResponse", 
    "ListScenesInput", "ListScenesResponse",
    "ListNodesInput", "ListNodesResponse",
    "MCPSettings"
]

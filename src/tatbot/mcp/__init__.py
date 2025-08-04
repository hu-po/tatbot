"""MCP (Model Control Protocol) module for tatbot."""

from .models import (
    GetLatestRecordingInput,
    GetLatestRecordingResponse,
    ListNodesInput,
    ListNodesResponse,
    ListScenesInput,
    ListScenesResponse,
    MCPSettings,
    PingNodesInput,
    PingNodesResponse,
    RunOpInput,
    RunOpResult,
)

__all__ = [
    "RunOpInput", "RunOpResult",
    "PingNodesInput", "PingNodesResponse", 
    "ListScenesInput", "ListScenesResponse",
    "ListNodesInput", "ListNodesResponse",
    "GetLatestRecordingInput", "GetLatestRecordingResponse",
    "MCPSettings"
]

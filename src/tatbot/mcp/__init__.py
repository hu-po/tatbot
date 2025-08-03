"""MCP (Model Control Protocol) module for tatbot."""

from .models import (
    RunOpInput, RunOpResult,
    PingNodesInput, PingNodesResponse,
    ListScenesInput, ListScenesResponse,
    ListNodesInput, ListNodesResponse,
    GetNfsInfoInput, GetNfsInfoResponse,
    GetLatestRecordingInput, GetLatestRecordingResponse,
    MCPSettings
)

__all__ = [
    "RunOpInput", "RunOpResult",
    "PingNodesInput", "PingNodesResponse", 
    "ListScenesInput", "ListScenesResponse",
    "ListNodesInput", "ListNodesResponse",
    "GetNfsInfoInput", "GetNfsInfoResponse",
    "GetLatestRecordingInput", "GetLatestRecordingResponse",
    "MCPSettings"
]

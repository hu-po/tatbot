"""Pydantic models for MCP requests and responses."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class MCPSettings(BaseSettings):
    """MCP server settings with environment variable overrides."""
    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "streamable-http"
    debug: bool = False
    extras: List[str] = []
    tools: List[str] = []

    class Config:
        env_prefix = "MCP_"


# Request Models
class RunOpInput(BaseModel):
    """Input model for running robot operations."""
    op_name: str
    scene_name: str = "default"
    debug: bool = False

    @field_validator('op_name')
    @classmethod
    def validate_op_name(cls, v: str) -> str:
        """Validate that the operation name exists in available operations."""
        # Import here to avoid circular imports
        from tatbot.ops import NODE_AVAILABLE_OPS
        
        # Get all available operations across all nodes
        all_ops = set()
        for ops in NODE_AVAILABLE_OPS.values():
            all_ops.update(ops)
        
        if v not in all_ops:
            available_ops = sorted(list(all_ops))
            raise ValueError(f"Invalid op_name: {v}. Available: {available_ops}")
        return v

    @field_validator('scene_name')
    @classmethod
    def validate_scene_name(cls, v: str) -> str:
        """Validate that the scene name exists in the scenes directory."""
        # Use the same logic as the list_scenes function
        scenes_dir = Path("~/tatbot/config/scenes").expanduser().resolve()
        try:
            available_scenes = [f.replace(".yaml", "") for f in os.listdir(str(scenes_dir)) if f.endswith(".yaml")]
            if v not in available_scenes:
                raise ValueError(f"Invalid scene_name: {v}. Available scenes: {available_scenes}")
        except FileNotFoundError:
            # If scenes directory doesn't exist, allow any scene name
            pass
        return v


class PingNodesInput(BaseModel):
    """Input model for pinging network nodes."""
    nodes: Optional[List[str]] = None

    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate that the specified nodes exist in the network."""
        if v:
            # Import here to avoid circular imports
            from tatbot.utils.net import NetworkManager
            net = NetworkManager()
            
            # Get available nodes from NetworkManager
            available_nodes = [node.name for node in net.nodes]
            invalid = [n for n in v if n not in available_nodes]
            if invalid:
                raise ValueError(f"Invalid nodes: {invalid}. Available nodes: {available_nodes}")
        return v


class ListScenesInput(BaseModel):
    """Input model for listing available scenes."""
    pass


class ListNodesInput(BaseModel):
    """Input model for listing available nodes."""
    pass


# Response Models
class RunOpResult(BaseModel):
    """Response model for robot operation execution."""
    message: str
    success: bool
    op_name: str
    scene_name: str


class PingNodesResponse(BaseModel):
    """Response model for network node ping results."""
    status: str
    details: List[str]
    all_success: bool


class ListScenesResponse(BaseModel):
    """Response model for available scenes listing."""
    scenes: List[str]
    count: int


class ListNodesResponse(BaseModel):
    """Response model for available nodes listing."""
    nodes: List[str]
    count: int


class NodeInfo(BaseModel):
    """Information about a network node."""
    name: str
    emoji: str
    status: str


class NodesStatusResponse(BaseModel):
    """Response model for nodes status."""
    nodes: List[NodeInfo]
    total_count: int
    online_count: int


class GetNfsInfoInput(BaseModel):
    """Input model for getting NFS information."""
    pass


class GetNfsInfoResponse(BaseModel):
    """Response model for NFS information."""
    info: str


class GetLatestRecordingInput(BaseModel):
    """Input model for getting latest recording."""
    pass


class GetLatestRecordingResponse(BaseModel):
    """Response model for latest recording."""
    filename: str
    found: bool
"""Pydantic models for MCP requests and responses."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class MCPConstants:
    """Configuration constants for MCP models."""
    DEFAULT_HOST: str = "0.0.0.0"
    DEFAULT_PORT: int = 8000
    DEFAULT_TRANSPORT: str = "streamable-http"
    DEFAULT_ALLOWED_IPS: List[str] = ["127.0.0.1", "::1"]
    DEFAULT_VERSION: str = "1.0"
    SCENES_CONFIG_PATH: str = "~/tatbot/src/conf/scenes"


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and other non-serializable types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # For other non-serializable types, convert to string
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


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


# Request Models
class PingNodesInput(BaseModel):
    """Input model for pinging network nodes."""
    version: str = "1.0"  # Tool versioning
    nodes: Optional[List[str]] = None

    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate that the specified nodes exist in the network."""
        if v:
            try:
                # Import here to avoid circular imports
                from tatbot.utils.net import NetworkManager
                net = NetworkManager()
                
                # Get available nodes from NetworkManager
                available_nodes = [node.name for node in net.nodes]
                invalid = [n for n in v if n not in available_nodes]
                if invalid:
                    raise ValueError(f"Invalid nodes: {invalid}. Available nodes: {available_nodes}")
            except Exception as e:
                # If NetworkManager fails, log the error but don't fail validation
                # This allows the tool to work even if the network config is incomplete
                import logging
                logging.getLogger("tatbot.mcp.models").warning(f"NetworkManager validation failed: {e}")
        return v


class ListScenesInput(BaseModel):
    """Input model for listing available scenes."""
    version: str = "1.0"  # Tool versioning


class ListNodesInput(BaseModel):
    """Input model for listing available nodes."""
    version: str = "1.0"  # Tool versioning


# ListOpsInput removed - no longer needed with unified tools system


# Response Models
class BaseResponse(BaseModel):
    """Base response model with custom JSON serialization."""
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to use custom JSON encoder for numpy arrays."""
        return json.dumps(self.model_dump(**kwargs), cls=NumpyEncoder, ensure_ascii=False)


class PingNodesResponse(BaseResponse):
    """Response model for network node ping results."""
    version: str = "1.0"  # Tool versioning
    status: str
    details: List[str]
    all_success: bool


class ListScenesResponse(BaseResponse):
    """Response model for available scenes listing."""
    version: str = "1.0"  # Tool versioning
    scenes: List[str]
    count: int


class ListNodesResponse(BaseResponse):
    """Response model for available nodes listing."""
    version: str = "1.0"  # Tool versioning
    nodes: List[str]
    count: int


# ListOpsResponse removed - no longer needed with unified tools system


class NodeInfo(BaseModel):
    """Information about a network node."""
    name: str
    emoji: str
    status: str


class NodesStatusResponse(BaseResponse):
    """Response model for nodes status."""
    version: str = "1.0"  # Tool versioning
    nodes: List[NodeInfo]
    total_count: int
    online_count: int


class ConvertStrokeListInput(BaseModel):
    """Input model for converting StrokeList to StrokeBatch."""
    version: str = "1.0"  # Tool versioning
    strokes_file_path: str  # Path to strokes YAML file on shared NFS
    strokebatch_file_path: str  # Path where strokebatch should be saved on shared NFS
    scene_name: str  # Scene name for conversion parameters
    first_last_rest: bool = True  # Apply first/last rest positions
    use_ee_offsets: bool = True  # Apply end-effector offsets


class ConvertStrokeListResponse(BaseResponse):
    """Response model for StrokeList to StrokeBatch conversion."""
    version: str = "1.0"  # Tool versioning
    strokebatch_base64: str  # Base64-encoded safetensors data
    success: bool
    message: str
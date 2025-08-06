"""Pydantic models for MCP requests and responses."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


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
    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "streamable-http"
    debug: bool = False
    extras: List[str] = []
    tools: List[str] = []
    
    # Tool naming settings
    namespace_tools: bool = True  # Prefix tools with node name to avoid conflicts
    
    # Security settings
    auth_token: Optional[str] = None
    ip_allowlist: List[str] = ["127.0.0.1", "::1"]  # localhost by default
    require_auth: bool = True

    class Config:
        env_prefix = "MCP_"


# Request Models
class RunOpInput(BaseModel):
    """Input model for running robot operations."""
    version: str = "1.0"  # Tool versioning
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
        scenes_dir = Path("~/tatbot/src/conf/scenes").expanduser().resolve()
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


class ListOpsInput(BaseModel):
    """Input model for listing available operations."""
    version: str = "1.0"  # Tool versioning
    node_name: Optional[str] = None  # If provided, list ops for specific node only


# Response Models
class BaseResponse(BaseModel):
    """Base response model with custom JSON serialization."""
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to use custom JSON encoder for numpy arrays."""
        return json.dumps(self.model_dump(**kwargs), cls=NumpyEncoder, ensure_ascii=False)


class RunOpResult(BaseResponse):
    """Response model for robot operation execution."""
    version: str = "1.0"  # Tool versioning
    message: str
    success: bool
    op_name: str
    scene_name: str


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


class ListOpsResponse(BaseResponse):
    """Response model for available operations listing."""
    version: str = "1.0"  # Tool versioning
    ops: List[str]
    count: int
    node_name: Optional[str] = None  # If filtered by node


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
"""Pydantic models for MCP requests and responses."""

import json
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


# Response Models
class BaseResponse(BaseModel):
    """Base response model with custom JSON serialization."""
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to use custom JSON encoder for numpy arrays."""
        return json.dumps(self.model_dump(**kwargs), cls=NumpyEncoder, ensure_ascii=False)


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



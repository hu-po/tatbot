"""Base types and utilities for the unified tools system."""

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List

from mcp.server.fastmcp import Context
from pydantic import BaseModel, ConfigDict

from tatbot.utils.log import get_logger

log = get_logger("tools.base", "🔧")


class ToolInput(BaseModel):
    """Base class for all tool input models."""
    model_config = ConfigDict(extra="ignore")
    debug: bool = False


class ToolOutput(BaseModel):
    """Base class for all tool output models."""
    success: bool
    message: str


@dataclass
class ToolContext:
    """Unified context for tool execution, wrapping MCP context with convenience methods."""
    
    node_name: str
    mcp_context: Context
    
    async def report_progress(self, progress: float, message: str) -> None:
        """Report progress to the client."""
        await self.mcp_context.report_progress(progress, 1.0, message)
    
    async def info(self, message: str) -> None:
        """Send info message to the client."""
        await self.mcp_context.info(message)
    
    async def error(self, message: str) -> None:
        """Send error message to the client."""
        log.error(message)


@dataclass
class ToolDefinition:
    """Definition of a tool including its metadata and function."""
    
    name: str
    func: Callable
    nodes: List[str]
    description: str
    input_model: type[ToolInput]
    output_model: type[ToolOutput]
    requires: List[str]
    
    def is_available_on_node(self, node_name: str) -> bool:
        """Check if this tool is available on the given node."""
        return "*" in self.nodes or node_name in self.nodes
    
    def check_requirements(self, node_config: dict) -> bool:
        """Check if node meets tool requirements."""
        if not self.requires:
            return True
        
        node_extras = set(node_config.get("extras", []))
        required_extras = set(self.requires)
        
        return required_extras.issubset(node_extras)


ToolFunction = Callable[[ToolInput, ToolContext], AsyncGenerator[Dict[str, Any], ToolOutput]]
"""Generic MCP server with Hydra configuration support."""

import socket
import types
from typing import List, Optional

import hydra
from mcp.server.fastmcp import FastMCP
from omegaconf import DictConfig, OmegaConf

from tatbot.mcp import handlers
from tatbot.mcp.models import MCPSettings
from tatbot.utils.exceptions import NetworkConnectionError
from tatbot.utils.log import get_logger


class ServerConstants:
    """Configuration constants for MCP server."""
    CONFIG_PATH: str = "../../conf"
    DEFAULT_CONFIG_NAME: str = "config"

log = get_logger("mcp.server", "🔌")


def _register_tools(mcp: FastMCP, tool_names: Optional[List[str]], node_name: str, namespace_tools: bool = True) -> None:
    """Register tools dynamically based on configuration."""
    available_tools = handlers.get_available_tools()
    tools_to_register = tool_names or list(available_tools.keys())
    
    log.info(f"Registering tools for {node_name}: {tools_to_register} (namespace_tools={namespace_tools})")
    
    for tool_name in tools_to_register:
        if tool_name in available_tools:
            tool_fn = available_tools[tool_name]
            
            if namespace_tools:
                # Create namespaced tool name: node_name_tool_name
                registered_name = f"{node_name}_{tool_name}"
                
                # Create a new function with the namespaced name but same signature
                wrapper_fn = types.FunctionType(
                    tool_fn.__code__,
                    tool_fn.__globals__,
                    registered_name,
                    tool_fn.__defaults__,
                    tool_fn.__closure__
                )
                wrapper_fn.__annotations__ = tool_fn.__annotations__
                wrapper_fn.__doc__ = tool_fn.__doc__
                
                # Register tool with namespaced name
                mcp.tool()(wrapper_fn)
                log.info(f"✅ Registered tool: {registered_name} (was {tool_name})")
            else:
                # Register tool with original name
                mcp.tool()(tool_fn)
                log.info(f"✅ Registered tool: {tool_name}")
        else:
            log.warning(f"⚠️ Tool {tool_name} not found in handlers")


@hydra.main(
    version_base=None, 
    config_path=ServerConstants.CONFIG_PATH, 
    config_name=ServerConstants.DEFAULT_CONFIG_NAME
)
def main(cfg: DictConfig) -> None:
    """Main server entry point with Hydra configuration."""
    # Extract MCP settings from Hydra config
    mcp_config = OmegaConf.to_object(cfg.mcp)
    settings = MCPSettings(**mcp_config)
    
    # Get node name from Hydra config, fallback to hostname
    node_name = cfg.get("node", socket.gethostname())
    
    log.info(f"Starting MCP server for node: {node_name}")
    log.info(f"Settings: host={settings.host}, port={settings.port}, transport={settings.transport}")
    log.info(f"Extras: {settings.extras}")
    log.info(f"Tools: {settings.tools}")
    
    # Create FastMCP server
    mcp = FastMCP(f"tatbot.{node_name}", host=str(settings.host), port=settings.port)
    
    # Note: MCP protocol handles authentication through its own mechanisms
    # FastMCP doesn't use HTTP middleware like FastAPI
    
    # Register tools
    _register_tools(mcp, settings.tools, node_name, settings.namespace_tools)
    
    # Add resource for listing nodes
    @mcp.resource("nodes://all")
    def get_nodes() -> str:
        """Get all available nodes."""
        try:
            from tatbot.utils.net import NetworkManager
            net = NetworkManager()
            return "\n".join(f"{node.emoji} {node.name}" for node in net.nodes)
        except ImportError as import_error:
            log.error(f"Failed to import NetworkManager: {import_error}")
            return f"NetworkManager not available: {import_error}"
        except NetworkConnectionError as network_error:
            log.error(f"Network connection error: {network_error}")
            return f"Network error: {network_error}"
        except Exception as unexpected_error:
            log.error(f"Unexpected error getting nodes: {unexpected_error}")
            return f"Unexpected error: {unexpected_error}"
    

    
    # Start server
    log.info(f"🚀 Starting MCP server on {settings.host}:{settings.port}")
    mcp.run(transport=settings.transport)


if __name__ == "__main__":
    main()
"""Generic MCP server with Hydra configuration support."""

import logging
import os
import socket
from typing import List, Optional

import hydra
from mcp.server.fastmcp import FastMCP
from omegaconf import DictConfig, OmegaConf

from tatbot.mcp.models import MCPSettings
from tatbot.state.manager import StateManager
from tatbot.tools import get_tools_for_node
from tatbot.utils.exceptions import NetworkConnectionError
from tatbot.utils.log import SUBMODULES, get_logger


class ServerConstants:
    """Configuration constants for MCP server."""
    CONFIG_PATH: str = "../../conf"
    DEFAULT_CONFIG_NAME: str = "config"

log = get_logger("mcp.server", "🔌")


def _register_tools(mcp: FastMCP, tool_names: Optional[List[str]], node_name: str) -> None:
    """Register tools dynamically based on configuration."""
    available_tools = get_tools_for_node(node_name)
    tools_to_register = tool_names or list(available_tools.keys())
    
    log.info(f"Registering tools for {node_name}: {tools_to_register}")
    
    for tool_name in tools_to_register:
        if tool_name in available_tools:
            tool_fn = available_tools[tool_name]
            mcp.tool()(tool_fn)
            log.info(f"✅ Registered tool: {tool_name}")
        else:
            log.warning(f"⚠️ Tool {tool_name} not found for node {node_name}")


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
    
    if settings.debug:    
        logging.getLogger().setLevel(logging.DEBUG)
        for submodule in SUBMODULES:
            logging.getLogger(f"tatbot.{submodule}").setLevel(logging.DEBUG)
        log.debug("🐛 MCP debug mode enabled (mcp.debug=true)")
    
    # Get node name from Hydra config, fallback to hostname
    node_name = cfg.get("node", socket.gethostname())
    
    log.info(f"Starting MCP server for node: {node_name}")
    log.info(f"Settings: host={settings.host}, port={settings.port}, transport={settings.transport}")
    log.info(f"Extras: {settings.extras}")
    log.info(f"Tools: {settings.tools}")
    
    # Configure Redis from Hydra (single source of truth)
    redis_host = str(cfg.redis.get("host", "eek"))
    redis_port = int(cfg.redis.get("port", 6379))
    os.environ.setdefault("REDIS_HOST", redis_host)
    os.environ.setdefault("REDIS_PORT", str(redis_port))
    log.info(f"Redis target: {redis_host}:{redis_port}")
    
    # Create FastMCP server (open mode - no authentication)
    mcp = FastMCP(
        f"tatbot.{node_name}", 
        host=str(settings.host), 
        port=settings.port
    )
    
    # Register tools
    _register_tools(mcp, settings.tools, node_name)
    
    # Initialize StateManager with configured Redis host/port
    state_manager = StateManager(node_id=node_name, redis_host=redis_host, redis_port=redis_port)
    
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
    
    # Add resource for global state status
    @mcp.resource("state://status")
    async def get_state_status() -> str:
        """Get global state status."""
        try:
            async with state_manager:
                status = await state_manager.get_system_status()
                return f"Redis Connected: {status['redis_connected']}\n" \
                       f"Active Sessions: {status['active_stroke_sessions']}\n" \
                       f"Nodes Online: {status['nodes_online']}/{status['total_nodes']}"
        except Exception as e:
            log.error(f"Failed to get state status: {e}")
            return f"State status unavailable: {e}"
    
    # Add resource for stroke progress
    @mcp.resource("state://stroke/progress")
    async def get_stroke_progress() -> str:
        """Get current stroke progress."""
        try:
            async with state_manager:
                progress = await state_manager.get_stroke_progress()
                if progress:
                    return f"Stroke: {progress.stroke_idx}/{progress.total_strokes}\n" \
                           f"Pose: {progress.pose_idx}/{progress.stroke_length}\n" \
                           f"Scene: {progress.scene_name}\n" \
                           f"Executing: {progress.is_executing}"
                return "No active stroke session"
        except Exception as e:
            log.error(f"Failed to get stroke progress: {e}")
            return f"Stroke progress unavailable: {e}"
    
    # Add resource for node health
    @mcp.resource("state://health/{node_id}")
    async def get_node_health(node_id: str) -> str:
        """Get health status for specific node."""
        try:
            async with state_manager:
                health = await state_manager.get_node_health(node_id)
                if health:
                    return f"Node: {health.node_id}\n" \
                           f"Reachable: {health.is_reachable}\n" \
                           f"MCP Server: {health.mcp_server_running}\n" \
                           f"Last Heartbeat: {health.last_heartbeat}"
                return f"No health data for node: {node_id}"
        except Exception as e:
            log.error(f"Failed to get node health: {e}")
            return f"Node health unavailable: {e}"
    

    
    # Start server
    log.info(f"🚀 Starting MCP server on {settings.host}:{settings.port}")
    mcp.run(transport=settings.transport)


if __name__ == "__main__":
    main()

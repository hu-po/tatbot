"""Generic MCP server with Hydra configuration support."""

from pathlib import Path

import hydra
from mcp.server.fastmcp import FastMCP
from omegaconf import DictConfig, OmegaConf

from tatbot.mcp import handlers
from tatbot.mcp.models import MCPSettings
from tatbot.utils.log import get_logger

log = get_logger("mcp.server", "🔌")


def _register_tools(mcp: FastMCP, tool_names: list[str] | None, node_name: str):
    """Register tools dynamically based on configuration."""
    available_tools = handlers.get_available_tools()
    tools_to_register = tool_names or list(available_tools.keys())
    
    log.info(f"Registering tools for {node_name}: {tools_to_register}")
    
    for tool_name in tools_to_register:
        if tool_name in available_tools:
            tool_fn = available_tools[tool_name]
            # Create a closure to capture the node_name for tools that need it
            if tool_name == "run_op":
                # Use a closure factory to properly capture node_name
                def make_run_op_wrapper(node_name_captured):
                    async def run_op_wrapper(input_data, ctx):
                        return await tool_fn(input_data, ctx, node_name_captured)
                    return run_op_wrapper
                
                run_op_wrapper = make_run_op_wrapper(node_name)
                run_op_wrapper.__name__ = tool_name
                mcp.tool()(run_op_wrapper)
            else:
                mcp.tool()(tool_fn)
            log.info(f"✅ Registered tool: {tool_name}")
        else:
            log.warning(f"⚠️ Tool {tool_name} not found in handlers")


@hydra.main(
    version_base=None, 
    config_path=str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "conf"), 
    config_name="config"
)
def main(cfg: DictConfig):
    """Main server entry point with Hydra configuration."""
    # Extract MCP settings from Hydra config
    mcp_config = OmegaConf.to_object(cfg.mcp)
    settings = MCPSettings(**mcp_config)
    
    # Get node name from Hydra config, fallback to hostname
    import socket
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
    _register_tools(mcp, settings.tools, node_name)
    
    # Add resource for listing nodes
    @mcp.resource("nodes://all")
    def get_nodes() -> str:
        """Get all available nodes."""
        try:
            from tatbot.utils.net import NetworkManager
            net = NetworkManager()
            return "\n".join(f"{node.emoji} {node.name}" for node in net.nodes)
        except Exception as e:
            log.error(f"Failed to get nodes: {e}")
            return f"Error getting nodes: {e}"
    

    
    # Start server
    log.info(f"🚀 Starting MCP server on {settings.host}:{settings.port}")
    mcp.run(transport=settings.transport)


if __name__ == "__main__":
    main()
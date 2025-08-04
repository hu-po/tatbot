"""Generic MCP server with Hydra configuration support."""


import hydra
from mcp.server.fastmcp import FastMCP
from omegaconf import DictConfig, OmegaConf

from tatbot.mcp import handlers
from tatbot.mcp.models import MCPSettings
from tatbot.utils.log import get_logger

log = get_logger("mcp.server", "🔌")


def _register_tools(mcp: FastMCP, tool_names: list[str] | None, node_name: str, namespace_tools: bool = True):
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
                
                # Create wrapper function with proper closure
                def make_wrapper(fn):
                    async def namespaced_tool_fn(input_data, ctx):
                        return await fn(input_data, ctx)
                    return namespaced_tool_fn
                
                wrapper_fn = make_wrapper(tool_fn)
                wrapper_fn.__name__ = registered_name
                
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
    config_path="../../conf", 
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
    _register_tools(mcp, settings.tools, node_name, settings.namespace_tools)
    
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
"""MCP server running on rpi2 node."""
from mcp.server.fastmcp import FastMCP

from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.mcp.base import MCPConfig

log = get_logger('mcp.rpi2', '🔌🍇')

mcp = FastMCP("tatbot.rpi2")

@mcp.tool(description="Get nfs information")
def get_nfs_info() -> str:
    return "nfs info"

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)
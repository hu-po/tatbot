"""MCP server running on ojo node."""
from mcp.server.fastmcp import FastMCP

from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.mcp.base import MCPConfig

log = get_logger('mcp.ojo', '🔌🦎')

mcp = FastMCP("tatbot.ojo", host="0.0.0.0", port=8000)

# TODO: add tools, resources, etc.

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)
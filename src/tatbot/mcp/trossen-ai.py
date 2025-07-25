"""MCP server running on trossen-ai node."""

import logging

from mcp.server.fastmcp import Context, FastMCP

from tatbot.mcp.base import MCPConfig, _run_op
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.trossen-ai", "🔌🦾")

mcp = FastMCP("tatbot.trossen-ai", host="localhost", port=8000)

@mcp.tool()
async def run_op(op_name: str, ctx: Context) -> str:
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    return await _run_op(op_name, ctx, "trossen-ai")

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

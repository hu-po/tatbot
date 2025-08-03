"""MCP server running on oop node."""

import logging

from mcp.server.fastmcp import Context, FastMCP

from tatbot.mcp.base import MCPConfig, RunOpInput, _run_op
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.oop", "🔌🦊")

mcp = FastMCP("tatbot.oop", host="192.168.1.51", port=8000)

@mcp.tool()
async def run_op(input: RunOpInput, ctx: Context) -> str:
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    return await _run_op(input, ctx, "oop")


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

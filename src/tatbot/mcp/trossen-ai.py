"""MCP server running on trossen-ai node."""

import logging
import traceback

from mcp.server.fastmcp import Context, FastMCP

from tatbot.bot.ops import get_op
from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.trossen-ai", "🔌🦾")

mcp = FastMCP("tatbot.trossen-ai", host="0.0.0.0", port=8000)


@mcp.tool(description="Run a robot operation.")
async def run_robot_op(op_name: str, ctx: Context) -> str:
    """Performs a robot operation with progress updates."""
    await ctx.info(f"Running robot op: {op_name}")
    try:
        op_class, op_config = get_op(op_name)
        config = op_config()
        op = op_class(config)
        await ctx.report_progress(
            progress=0.01, total=1.0, message=f"Created op class and config: {config}"
        )
    except Exception:
        _msg = f"❌ Exception when creating op: {traceback.format_exc()}"
        log.error(_msg)
        return _msg    
    
    try:
        async for result in op.run():
            log.info(f"Intermediate result: {result}")
            await ctx.report_progress(progress=result['progress'], total=1.0, message=result['message'])
        _msg = f"✅ Completed robot operation {op_name}"
        log.info(_msg)
        return _msg
        
    except Exception:
        _msg = f"❌ Exception when running op: {traceback.format_exc()}"
        log.error(_msg)
        return _msg
    

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)

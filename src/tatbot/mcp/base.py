"""Base MCP server abstractions (legacy - will be deprecated)."""

import asyncio
import traceback

from mcp.server.fastmcp import Context

from tatbot.data.base import BaseCfg
from tatbot.utils.log import get_logger

log = get_logger("mcp.base", "🔌")


class MCPConfig(BaseCfg):
    """MCP server configuration with debug and transport settings."""
    debug: bool = False
    transport: str = "streamable-http"

async def _run_op(input_data, ctx: Context, node_name: str) -> str:
    """
    Legacy _run_op function - DEPRECATED.
    Use tatbot.mcp.handlers.run_op instead.
    """
    # Import locally to avoid circular imports
    from tatbot.ops import get_op
    
    await ctx.info(f"Running robot op: {input_data.op_name}")
    try:
        op_class, op_config = get_op(input_data.op_name, node_name)
        config = op_config(scene=input_data.scene_name, debug=input_data.debug)
        op = op_class(config)
        await ctx.report_progress(
            progress=0.01, total=1.0, message=f"Created op: {config}"
        )
    except Exception:
        _msg = f"❌ Exception when creating op: {traceback.format_exc()}"
        log.error(_msg)
        return _msg    
    
    try:
        async for result in op.run():
            log.info(f"Intermediate result: {result}")
            await ctx.report_progress(progress=result['progress'], total=1.0, message=result['message'])
        _msg = f"✅ Completed {input_data.op_name}"
        log.info(_msg)
        return _msg
    except (KeyboardInterrupt, asyncio.CancelledError):
        _msg = "🛑⌨️ Keyboard/E-stop interrupt detected"
        log.error(_msg)
        return _msg
    except Exception:
        _msg = f"❌ Exception when running op: {traceback.format_exc()}"
        log.error(_msg)
        return _msg
    finally:
        if 'op' in locals():
            op.cleanup()
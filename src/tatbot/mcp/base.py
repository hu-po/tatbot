"""Base MCP server abstractions."""

import traceback
from dataclasses import dataclass

from mcp.server.fastmcp import Context

from tatbot.ops import get_op
from tatbot.utils.log import get_logger

log = get_logger("mcp.base", "🔌")

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""
    transport: str = "streamable-http"
    """Transport type for MCP server."""

async def _run_op(op_name: str, ctx: Context, node_name: str, debug: bool = False) -> str:
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    await ctx.info(f"Running robot op: {op_name}")
    try:
        op_class, op_config = get_op(op_name, node_name)
        config = op_config(debug=debug)
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
    except KeyboardInterrupt:
        _msg = "🛑⌨️ Keyboard/E-stop interrupt detected"
        log.error(_msg)
    except Exception:
        _msg = f"❌ Exception when running op: {traceback.format_exc()}"
        log.error(_msg)
    finally:
        op.cleanup()
        return _msg
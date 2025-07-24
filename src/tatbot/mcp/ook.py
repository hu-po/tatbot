"""MCP server running on ook node."""

import concurrent.futures
import logging
import traceback
from typing import List, Optional

from mcp.server.fastmcp import Context, FastMCP

from tatbot.bot.ops import get_op
from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.utils.net import NetworkManager

log = get_logger("mcp.ook", "🔌🦧")

mcp = FastMCP("tatbot.ook", host="127.0.0.1", port=8000)
net = NetworkManager()


@mcp.resource("nodes://all")
def get_nodes() -> str:
    return "\n".join(f"{node.emoji} {node.name}" for node in net.nodes)


@mcp.tool(description="Ping nodes and report connectivity status.")
def ping_nodes(nodes: Optional[List[str]] = None) -> str:
    log.info(f"🔌 Pinging nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to ping."

    messages = []
    all_success = True

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_node = {executor.submit(net._test_node_connection, node): node for node in target_nodes}
        for future in concurrent.futures.as_completed(future_to_node):
            _, success, message = future.result()
            messages.append(message)
            if not success:
                all_success = False

    header = (
        "✅ All specified nodes are responding"
        if all_success
        else "❌ Some specified nodes are not responding"
    )
    if not nodes:
        header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"

    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in sorted(messages))

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
    except Exception:
        _msg = f"❌ Exception when running op: {traceback.format_exc()}"
        log.error(_msg)
    finally:
        op.cleanup()
        return _msg

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger("server").setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

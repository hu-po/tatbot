"""MCP server running on ook node."""

import concurrent.futures
import logging
from typing import List, Optional

from pydantic import BaseModel, field_validator
from mcp.server.fastmcp import Context, FastMCP

from tatbot.mcp.base import MCPConfig, RunOpInput, _run_op
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.utils.net import NetworkManager

log = get_logger("mcp.ook", "🔌🦧")

mcp = FastMCP("tatbot.ook", host="192.168.1.90", port=8000)
net = NetworkManager()


class PingNodesInput(BaseModel):
    """Input model for pinging network nodes."""
    nodes: Optional[List[str]] = None

    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate that the specified nodes exist in the network."""
        if v:
            # Get available nodes from NetworkManager
            available_nodes = [node.name for node in net.nodes]
            invalid = [n for n in v if n not in available_nodes]
            if invalid:
                raise ValueError(f"Invalid nodes: {invalid}. Available nodes: {available_nodes}")
        return v


@mcp.resource("nodes://all")
def get_nodes() -> str:
    return "\n".join(f"{node.emoji} {node.name}" for node in net.nodes)


@mcp.tool()
def ping_nodes(input: PingNodesInput) -> str:
    """Ping nodes and report connectivity status."""
    log.info(f"🔌 Pinging nodes: {input.nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(input.nodes)
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
    if not input.nodes:
        header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"

    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in sorted(messages))

@mcp.tool()
async def run_op(input: RunOpInput, ctx: Context) -> str:
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    return await _run_op(input, ctx, "ook")


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger("server").setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

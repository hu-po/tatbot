"""Base MCP server, usually runs on ook."""
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from tatbot.utils.net import NetworkManager
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('mcp.base', '🔌')

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""
    transport: str = "streamable-http"
    """Transport type for MCP server."""

mcp = FastMCP("tatbot.base")
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
        future_to_node = {
            executor.submit(net._test_node_connection, node): node for node in target_nodes
        }
        for future in concurrent.futures.as_completed(future_to_node):
            _, success, message = future.result()
            messages.append(message)
            if not success:
                all_success = False

    header = "✅ All specified nodes are responding" if all_success else "❌ Some specified nodes are not responding"
    if not nodes:
        header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"

    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in sorted(messages))


# @mcp.tool(description="(Re)start MCP servers on nodes using their respective startup scripts.")
# def start_mcp_servers(
#     nodes: List[str] = ["ojo", "trossen-ai", "rpi1", "rpi2"],
#     timeout: float = 20.0
# ) -> str:
#     log.info(f"🔌 Starting MCP servers on nodes: {nodes}")
#     target_nodes, error = net.get_target_nodes(nodes)
#     if error:
#         return error
#     if not target_nodes:
#         return "No nodes to start MCP servers on."
#     results = []
#     for node in target_nodes:
#         log.info(f"{node.emoji} Starting MCP server on {node.name} ({node.ip})")
#         if net.is_local_node(node):
#             results.append(f"{node.emoji} {node.name}: Skipped (local node)")
#             continue
#         try:
#             client = net.get_ssh_client(node.ip, node.user)
#             # First, check for and kill any existing MCP servers
#             kill_command = f"pkill -f 'mcp-{node.name}.sh' || true"
#             exit_code, out, err = net._run_remote_command(client, kill_command, timeout=timeout)
#             if exit_code == 0:
#                 log.warning(f"{node.emoji} {node.name}: ⚠️ Killed existing MCP server processes")
#             # Wait a moment for processes to fully terminate
#             wait_command = "sleep 2"
#             net._run_remote_command(client, wait_command, timeout=timeout)
#             # Start the new MCP server
#             command = (
#                 f"cd ~/tatbot && "
#                 f"nohup bash scripts/mcp-{node.name}.sh > /tmp/mcp-{node.name}.log 2>&1 & "
#                 f"echo $!"
#             )
#             exit_code, out, err = net._run_remote_command(client, command, timeout=timeout)
#             client.close()
#             if exit_code == 0:
#                 pid = out.strip()
#                 results.append(f"{node.emoji} {node.name}: ✅ MCP server started in background (PID: {pid})")
#             else:
#                 results.append(f"{node.emoji} {node.name}: ❌ Failed to start MCP server\n{err}")
#         except Exception as e:
#             results.append(f"{node.emoji} {node.name}: ❌ Exception occurred: {str(e)}")
#             log.error(f"❌ Failed to start MCP server on {node.name}: {e}")
#     return "\n\n".join(results)


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)
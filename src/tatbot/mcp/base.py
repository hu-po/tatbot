"""Base MCP server"""
import concurrent.futures
import logging
import os
import re
import tarfile
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
            _name, success, message = future.result()
            messages.append(message)
            if not success:
                all_success = False

    header = "✅ All specified nodes are responding" if all_success else "❌ Some specified nodes are not responding"
    if not nodes:
        header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"

    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in sorted(messages))

@mcp.tool(description="Update tatbot repo and Python env on nodes via git pull and uv.")
def update_nodes(nodes: Optional[List[str]] = None, timeout: float = 300.0) -> str:
    log.info(f"🔌 Updating nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to update."

    results = []

    for node in target_nodes:
        emoji = node.emoji

        log.info(f"{emoji} Updating {node.name} ({node.ip})")

        if net.is_local_node(node):
            results.append(f"{emoji} {node.name}: Skipped (local node)")
            continue

        try:
            client = net.get_ssh_client(node.ip, node.user)
            command = (
                "export PATH=\"$HOME/.local/bin:$PATH\" && "
                "git -C ~/tatbot pull && "
                "cd ~/tatbot/src && "
                "deactivate >/dev/null 2>&1 || true && "
                "rm -rf .venv && "
                "rm -f uv.lock && "
                "uv venv --prompt=\"tatbot\" && "
                f"uv pip install {node.deps}"
            )
            exit_code, out, err = net._run_remote_command(client, command, timeout=timeout)
            client.close()
            if exit_code == 0:
                results.append(f"{emoji} {node.name}: Success\n{out}")
            else:
                results.append(f"{emoji} {node.name}: Failed\n{err}")

        except Exception as e:
            results.append(f"{emoji} {node.name}: Exception occurred: {str(e)}")
            log.error(f"Failed to pull on {node.name}: {e}")

    return "\n\n".join(results)


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)
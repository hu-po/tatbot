"""
TODO:

# server mvp
# mcp inspector
# cursor as client (settings add mcp server)
# add to cursor mcp directory https://cursor.directory/mcp

ideas:
- ping nodes
- git pull local tatbot repo, reinstall uv env

- send files to nodes
- run commands on nodes
- get plan data
- get images
- list available nodes
- get tatbot info
- get tech docs
- get configs
- generate plan
- sim control
- run plan on robot
- run camera calibration
- configure robot
- open up viz browser, chrome, turn on screen
- kill all python processes, kill all docker containers
- gradio frontend for tatbot MCP
- git pull on all machines
- uv env install on all machines
- distribute files to all machines
- ojo: start/stop containers, get CPU/GPU usage, pull latest pattern
- rpi1: pause/play live viz, set path and pose w/ live viz, reset live viz, open chrome
- trossen: reset/check realsenses, configure robot, run bot with CLI kwargs(0.3) 

"""
import concurrent.futures
from dataclasses import dataclass
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from _log import get_logger, setup_log_with_config, print_config
from _net import NetworkManager, run_remote_command

log = get_logger('run_mcp')

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""
    transport: str = "streamable-http"
    """Transport type for MCP server."""

mcp = FastMCP("tatbot")
net = NetworkManager()

@mcp.tool()
def ping_nodes(nodes: Optional[List[str]] = None) -> str:
    """
    Tests connectivity to configured nodes and returns a status summary.
    If `nodes` is provided, only pings the specified nodes. Otherwise, pings all nodes.
    """
    log.info(f"Pinging nodes: {nodes or 'all'}")
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

@mcp.tool()
def update_all(nodes: Optional[List[str]] = None) -> str:
    """
    Runs 'git pull' on the tatbot repository on all configured nodes,
    then reinstalls the Python dependencies using uv.
    If `nodes` is provided, only updates the specified nodes.
    """
    log.info(f"Executing git pull and reinstalling dependencies on nodes: {nodes or 'all'}...")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to update."

    results = []

    for node in target_nodes:
        name = node["name"]
        ip = node["ip"]
        user = node["user"]
        emoji = node.get("emoji", "🌐")
        deps = node.get("deps", ".")

        log.info(f"{emoji} Updating {name} ({ip})")

        if net.is_local_node(node):
            # Handle local update logic if necessary
            pass

        try:
            client = net.get_ssh_client(ip, user)
            command = (
                "export PATH=\"$HOME/.local/bin:$PATH\" && "
                "git -C ~/tatbot pull && "
                "cd ~/tatbot/src/0.3 && "
                "deactivate >/dev/null 2>&1 || true && "
                "rm -rf .venv && "
                "rm -f uv.lock && "
                "uv venv && "
                f"uv pip install '{deps}'"
            )
            exit_code, out, err = run_remote_command(client, command, timeout=300.0)
            client.close()
            if exit_code == 0:
                results.append(f"{emoji} {name}: Success\n{out}")
            else:
                results.append(f"{emoji} {name}: Failed\n{err}")

        except Exception as e:
            results.append(f"{emoji} {name}: Exception occurred: {str(e)}")
            log.error(f"Failed to pull on {name}: {e}")

    return "\n\n".join(results)

@mcp.tool()
def node_usage(nodes: Optional[List[str]] = None) -> dict[str, dict]:
    """
    For every node in config/nodes.yaml, report basic CPU/RAM usage.
    Falls back to 'unreachable' if SSH fails.
    If `nodes` is provided, only reports usage for the specified nodes.
    """
    import psutil
    import json

    log.info(f"Getting usage for nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return {"error": error}

    report: dict[str, dict] = {}
    if not target_nodes:
        return report

    for n in target_nodes:
        if net.is_local_node(n):
            # local machine – use psutil directly
            report[n["name"]] = {
                "cpu_percent": psutil.cpu_percent(),
                "mem_percent": psutil.virtual_memory().percent,
            }
            continue
        try:
            client = net.get_ssh_client(n["ip"], n["user"])
            command = (
                "python - << 'EOF'\n"
                "import psutil, json, sys;"
                "print(json.dumps({'cpu': psutil.cpu_percent(),"
                "'mem': psutil.virtual_memory().percent}))\nEOF"
            )
            exit_code, out, err = run_remote_command(client, command)
            client.close()
            if exit_code == 0:
                report[n["name"]] = json.loads(out)
            else:
                log.error(f"Failed to get usage for {n['name']}: {err}")
                report[n["name"]] = {"error": "command failed"}

        except Exception as e:
            log.error(f"Failed to connect or run command on {n['name']}: {e}")
            report[n["name"]] = {"error": "unreachable"}
    return report

def run_mcp(config: MCPConfig):
    log.info("🔌 Starting MCP server")
    mcp.run(transport=config.transport)

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    run_mcp(args)
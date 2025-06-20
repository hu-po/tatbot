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
import json
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from _log import get_logger, setup_log_with_config, print_config
from _net import NetworkManager

log = get_logger('run_mcp')

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""
    transport: str = "streamable-http"
    """Transport type for MCP server."""

mcp = FastMCP("tatbot")
net = NetworkManager()

@mcp.resource("nodes://all")
def get_nodes() -> str:
    return "\n".join(f"{node.emoji} {node.name}" for node in net.nodes)

@mcp.tool(description="Tests connectivity to configured nodes and returns a status summary. If `nodes` is provided, only pings the specified nodes. Otherwise, pings all nodes.")
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

@mcp.tool(description="Runs 'git pull' on the tatbot repository on all configured nodes, then reinstalls the Python dependencies using uv. If `nodes` is provided, only updates the specified nodes.")
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
                "cd ~/tatbot/src/0.3 && "
                "deactivate >/dev/null 2>&1 || true && "
                "rm -rf .venv && "
                "rm -f uv.lock && "
                "uv venv && "
                f"uv pip install '{node.deps}'"
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

@mcp.tool(description="For every node in config/nodes.yaml, report basic CPU/RAM usage. Falls back to 'unreachable' if SSH fails. If `nodes` is provided, only reports usage for the specified nodes.")
def node_cpu_ram_usage(nodes: Optional[List[str]] = None) -> dict[str, dict]:
    import psutil

    log.info(f"Getting usage for nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return {"error": error}

    report: dict[str, dict] = {}
    if not target_nodes:
        return report

    remote_nodes = [n for n in target_nodes if not net.is_local_node(n)]
    remote_node_names = [n.name for n in remote_nodes]

    # Handle local node
    for n in target_nodes:
        if net.is_local_node(n):
            report[n.name] = {
                "cpu_percent": psutil.cpu_percent(),
                "mem_percent": psutil.virtual_memory().percent,
            }

    # Handle remote nodes
    if remote_node_names:
        command = (
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "cd ~/tatbot/src/0.3 && "
            "uv run python - << 'EOF'\n"
            "import psutil, json, sys;"
            "print(json.dumps({'cpu_percent': psutil.cpu_percent(),"
            "'mem_percent': psutil.virtual_memory().percent}))\nEOF"
        )
        results = net.run_command_on_nodes(command, node_names=remote_node_names)

        for name, (exit_code, out, err) in results.items():
            if exit_code == 0:
                try:
                    report[name] = json.loads(out)
                except json.JSONDecodeError:
                    log.error(f"Failed to parse usage JSON from {name}: {out}")
                    report[name] = {"error": "invalid output"}
            elif exit_code == -1 and "Failed to connect" in err:
                report[name] = {"error": "unreachable"}
            else:
                log.error(f"Failed to get usage for {name}: {err}")
                report[name] = {"error": f"command failed: {err}"}

    return report

@mcp.tool(description="Powers off the specified nodes. Requires passwordless sudo for the 'poweroff' command.")
def poweroff_nodes(nodes: Optional[List[str]] = None) -> str:
    log.info(f"🔌 Powering off nodes: {nodes or 'all'}")
    target_nodes, error = net.get_target_nodes(nodes)
    if error:
        return error
    if not target_nodes:
        return "No nodes to power off."

    remote_nodes = [n for n in target_nodes if not net.is_local_node(n)]
    local_nodes = [n for n in target_nodes if net.is_local_node(n)]
    
    report = [f"⚠️ {node.emoji} {node.name}: Skipped (local node)." for node in local_nodes]

    if not remote_nodes:
        if report:
             return "\n".join(sorted(report))
        return "No remote nodes specified to power off."
    
    remote_node_names = [n.name for n in remote_nodes]
    remote_node_map = {n.name: n for n in remote_nodes}

    command = "nohup sudo poweroff > /dev/null 2>&1 &"
    # Use a short timeout because the command may not return on success
    results = net.run_command_on_nodes(command, node_names=remote_node_names, timeout=5.0)

    for name, (exit_code, out, err) in results.items():
        node = remote_node_map[name]
        # -1 exit code from our wrapper means an exception happened (like a timeout or connection drop)
        # which is expected if poweroff succeeds.
        if exit_code == -1 and ('timeout' in err.lower() or 'session timed out' in err.lower() or 'socket is closed' in err.lower()):
            report.append(f"✅ {node.emoji} {name}: Power off command sent, connection lost as expected.")
        elif exit_code == -1 and "Failed to connect" in err:
            report.append(f"🔌 {node.emoji} {name}: Already offline or unreachable.")
        elif exit_code == 0: # This might happen if poweroff returns immediately
             report.append(f"✅ {node.emoji} {name}: Power off command sent.")
        else:
            error_message = err or out
            report.append(f"❌ {node.emoji} {name}: Failed to power off. Exit code: {exit_code}, Error: {error_message}")
            log.error(f"Failed to power off {name}: Code={exit_code}, out={out}, err={err}")
    
    return "\n".join(sorted(report))

def run_mcp(config: MCPConfig):
    log.info("🔌 Starting MCP server")
    mcp.run(transport=config.transport)

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    run_mcp(args)
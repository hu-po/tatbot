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
import os
import subprocess
from dataclasses import dataclass
import logging
from typing import List
import yaml

from mcp.server.fastmcp import FastMCP

from _log import get_logger, setup_log_with_config, print_config
from _net import (
    SetupNetConfig,
    test_nodes,
    load_nodes,
    _is_local_node,
    get_ssh_client,
    run_remote_command,
)

log = get_logger('run_mcp')

@dataclass
class MCPConfig:
    debug: bool = False
    """Enable debug logging."""

mcp = FastMCP("tatbot")

@mcp.tool()
def ping_nodes() -> str:
    """Tests connectivity to all configured nodes and returns a status summary."""
    log.info("Pinging all nodes...")
    config = SetupNetConfig()
    all_success, messages = test_nodes(config)
    header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"
    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in messages)

@mcp.tool()
def update_all() -> str:
    """
    Runs 'git pull' on the tatbot repository on all configured nodes,
    then reinstalls the Python dependencies using uv.
    Assumes the tatbot repository is located at '~/tatbot' on each node.
    """
    log.info("Executing git pull and reinstalling dependencies on all nodes...")
    config = SetupNetConfig()
    nodes = load_nodes(config.yaml_file)
    results = []

    for node in nodes:
        name = node["name"]
        ip = node["ip"]
        user = node["user"]
        emoji = node.get("emoji", "🌐")

        log.info(f"{emoji} Updating {name} ({ip})")

        if _is_local_node(node):
            pass

        try:
            client = get_ssh_client(ip, user, config.key_path)
            command = (
                "export PATH=\"$HOME/.local/bin:$PATH\" && "
                "git -C ~/tatbot pull && "
                "cd ~/tatbot/src/0.3 && "
                "deactivate >/dev/null 2>&1 || true && "
                "rm -rf .venv && "
                "rm -f uv.lock && "
                "uv venv && "
                "uv pip install ."
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
def node_usage() -> dict[str, dict]:
    """
    For every node in config/nodes.yaml, report basic CPU/RAM usage.
    Falls back to 'unreachable' if SSH fails.
    """
    import psutil
    import json

    config = SetupNetConfig()
    nodes = load_nodes(config.yaml_file)
    report: dict[str, dict] = {}
    for n in nodes:
        if _is_local_node(n):
            # local machine – use psutil directly
            report[n["name"]] = {
                "cpu_percent": psutil.cpu_percent(),
                "mem_percent": psutil.virtual_memory().percent,
            }
            continue
        try:
            client = get_ssh_client(n["ip"], n["user"], config.key_path)
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

@mcp.resource("plan/{plan_name}")
def get_plan(plan_name: str) -> dict:
    """
    Provides detailed information about a specific tattoo plan.
    """
    log.info(f"Fetching plan data for '{plan_name}'")
    base_path = "output/plans"
    plan_path = os.path.join(base_path, plan_name)

    if not os.path.isdir(plan_path):
        return {"error": f"Plan '{plan_name}' not found."}

    meta_path = os.path.join(plan_path, "meta.yaml")
    stats_path = os.path.join(plan_path, "pathstats.yaml")
    image_path = os.path.join(plan_path, "image.png")

    response = {"name": plan_name}

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            response["metadata"] = yaml.safe_load(f)
    else:
        response["metadata"] = {"error": "meta.yaml not found"}

    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = yaml.safe_load(f)
            response["path_count"] = stats.get("count", 0)
            response["path_stats"] = stats
    else:
        response["path_count"] = 0
        response["path_stats"] = {"error": "pathstats.yaml not found"}

    if os.path.exists(image_path):
        response["image"] = image_path
    else:
        response["image"] = "not_found"

    return response

def run_mcp(config: MCPConfig):
    log.info("🔌 Starting MCP server")
    # transport can be 'stdio' or 'streamable-http'
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    run_mcp(args)
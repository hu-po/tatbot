"""
TODO:

# server mvp
# mcp inspector
# cursor as client (settings add mcp server)
# add to cursor mcp directory https://cursor.directory/mcp

ideas:
- ping nodes
- send files to nodes
- run commands on nodes
- git pull local tatbot repo, reinstall uv env
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
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def ping_nodes() -> str:
    """Tests connectivity to all configured nodes and returns a status summary."""
    log.info("Pinging all nodes...")
    config = SetupNetConfig()
    all_success, messages = test_nodes(config)
    header = "✅ All nodes are responding" if all_success else "❌ Some nodes are not responding"
    return f"{header}:\n" + "\n".join(f"- {msg}" for msg in messages)

@mcp.tool()
def git_pull_all() -> str:
    """
    Runs 'git pull' on the tatbot repository on all configured nodes.
    Assumes the tatbot repository is located at '~/tatbot' on each node.
    """
    log.info("Executing git pull on all nodes...")
    config = SetupNetConfig()
    nodes = load_nodes(config.yaml_file)
    results = []

    for node in nodes:
        name = node["name"]
        ip = node["ip"]
        user = node["user"]
        emoji = node.get("emoji", "🌐")

        log.info(f"{emoji} Pulling on {name} ({ip})")

        try:
            if _is_local_node(node):
                repo_path = os.path.expanduser("~/tatbot")
                process = subprocess.run(
                    ["git", "-C", repo_path, "pull"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if process.returncode == 0:
                    results.append(
                        f"{emoji} {name}: Success\n{process.stdout.strip()}"
                    )
                else:
                    results.append(f"{emoji} {name}: Failed\n{process.stderr.strip()}")
            else:
                client = get_ssh_client(ip, user, config.key_path)
                exit_code, out, err = run_remote_command(
                    client, "git -C ~/tatbot pull"
                )
                client.close()
                if exit_code == 0:
                    results.append(f"{emoji} {name}: Success\n{out}")
                else:
                    results.append(f"{emoji} {name}: Failed\n{err}")

        except Exception as e:
            results.append(f"{emoji} {name}: Exception occurred: {str(e)}")
            log.error(f"Failed to pull on {name}: {e}")

    return "\n\n".join(results)

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

def run_mcp(config: MCPConfig):
    log.info(f"🔌 Starting MCP server")
    mcp.run()

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    print_config(args)
    run_mcp(args)
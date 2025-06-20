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
from dataclasses import dataclass
import logging
from typing import List

from mcp.server.fastmcp import FastMCP

from _log import get_logger, setup_log_with_config, print_config
from _net import SetupNetConfig, test_nodes

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

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

def run_mcp(config: MCPConfig):
    del config
    log.info(f"🔌 Starting MCP server")
    mcp.run()

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    print_config(args)
    run_mcp(args)
"""MCP server running on rpi2 node."""

import logging
import os

from mcp.server.fastmcp import FastMCP

from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.rpi2", "🔌🍇")

mcp = FastMCP("tatbot.rpi2", host="0.0.0.0", port=8000)


@mcp.tool(description="Get nfs information")
def get_nfs_info() -> str:
    return "nfs info"


@mcp.tool(description="Get the latest scan")
def get_latest_scan() -> str:
    scan_dir = os.path.expanduser("~/tatbot/nfs/scans")
    scans = [f for f in os.listdir(scan_dir) if f.endswith(".yaml")]
    if not scans:
        return "No scans found"
    latest_scan = max(scans, key=lambda x: os.path.getctime(os.path.join(scan_dir, x)))
    return latest_scan


@mcp.tool(description="Get the latest recording")
def get_latest_recording() -> str:
    recording_dir = os.path.expanduser("~/tatbot/nfs/recordings")
    recordings = [f for f in os.listdir(recording_dir) if f.endswith(".yaml")]
    if not recordings:
        return "No recordings found"
    latest_recording = max(recordings, key=lambda x: os.path.getctime(os.path.join(recording_dir, x)))
    return latest_recording


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

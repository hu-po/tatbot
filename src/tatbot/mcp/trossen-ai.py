"""MCP server running on trossen-ai node."""
import logging

from mcp.server.fastmcp import FastMCP

from tatbot.cam.scan import ScanConfig, scan
from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('mcp.trossen-ai', '🔌🦾')

mcp = FastMCP("tatbot.trossen-ai", host="0.0.0.0", port=8000)

@mcp.tool(description="Perform a scan, saving images into the scan directory")
def scan(scene: str) -> str:
    """Performs a scan, saving images into the scan directory."""
    scan(ScanConfig(scene=scene))

@mcp.tool(description="Run visualization on trossen-ai")
def run_viz(viz_type: str, name: str) -> str:
    return "viz ran"

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)
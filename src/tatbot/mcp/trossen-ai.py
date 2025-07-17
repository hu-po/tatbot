"""MCP server running on trossen-ai node."""
import logging

from mcp.server.fastmcp import FastMCP

from tatbot.cam.scan import ScanConfig, scan
from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('mcp.trossen-ai', '🔌🦾')

mcp = FastMCP("tatbot.trossen-ai", host="0.0.0.0", port=8000)

@mcp.tool(description="Run a scan, saving images into the scan directory")
def run_scan(scene: str) -> str:
    """Performs a scan, saving images into the scan directory."""
    try:
        scan_name = scan(ScanConfig(scene=scene))
        log.info(f"✅ Scan completed: {scan_name}")
        return scan_name
    except Exception as e:
        log.error(f"❌ Error calling run_scan: {e}")
        return "failed_scan"

@mcp.tool(description="Run visualization on trossen-ai")
def run_viz(viz_type: str, name: str) -> str:
    return "viz ran"

if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    mcp.run(transport=args.transport)
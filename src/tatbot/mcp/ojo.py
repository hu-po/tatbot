"""MCP server running on ojo node."""

import logging

from mcp.server.fastmcp import FastMCP

from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.ojo", "🔌🦎")

mcp = FastMCP("tatbot.ojo", host="localhost", port=8000)


@mcp.tool()
def manage_policy_server(action: str, policy_type: str = "gr00t") -> str:
    """Starts or stops the policy inference server on ojo.

    Args:
        action: 'start' or 'stop'
        policy_type: 'gr00t' or 'smolvla'
    """
    # Add logic to run the docker containers for the policy servers
    if action == "start":
        # command to start the policy server
        return f"{policy_type} policy server started."
    elif action == "stop":
        # command to stop the policy server
        return f"{policy_type} policy server stopped."
    return "Invalid action."


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

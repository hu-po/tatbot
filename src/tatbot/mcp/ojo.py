"""MCP server running on ojo node."""

import logging

from pydantic import BaseModel, field_validator
from mcp.server.fastmcp import FastMCP

from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.ojo", "🔌🦎")

mcp = FastMCP("tatbot.ojo", host="192.168.1.96", port=8000)


class ManagePolicyInput(BaseModel):
    """Input model for managing policy servers."""
    action: str
    policy_type: str = "gr00t"

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate that action is either start or stop."""
        if v not in ['start', 'stop']:
            raise ValueError("Action must be 'start' or 'stop'")
        return v

    @field_validator('policy_type')
    @classmethod
    def validate_policy_type(cls, v: str) -> str:
        """Validate that policy_type is supported."""
        if v not in ['gr00t', 'smolvla']:
            raise ValueError("Policy type must be 'gr00t' or 'smolvla'")
        return v


@mcp.tool()
def manage_policy_server(input: ManagePolicyInput) -> str:
    """Starts or stops the policy inference server on ojo.

    Args:
        input: ManagePolicyInput with action ('start' or 'stop') and policy_type ('gr00t' or 'smolvla')
    """
    # Add logic to run the docker containers for the policy servers
    if input.action == "start":
        # command to start the policy server
        return f"{input.policy_type} policy server started."
    elif input.action == "stop":
        # command to stop the policy server
        return f"{input.policy_type} policy server stopped."
    return "Invalid action."


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

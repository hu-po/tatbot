"""MCP server running on rpi1 node."""

import logging

from pydantic import BaseModel, field_validator
from mcp.server.fastmcp import FastMCP

from tatbot.mcp.base import MCPConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("mcp.rpi1", "🔌🍓")

mcp = FastMCP("tatbot.rpi1", host="192.168.1.98", port=8000)


class RunVizInput(BaseModel):
    """Input model for running visualization on rpi1."""
    viz_type: str
    name: str

    @field_validator('viz_type')
    @classmethod
    def validate_viz_type(cls, v: str) -> str:
        """Validate visualization type."""
        valid_types = ['stream', 'record', 'plot', 'mesh']  # Add actual viz types
        if v not in valid_types:
            raise ValueError(f"Invalid viz_type: {v}. Valid types: {valid_types}")
        return v


@mcp.tool()
def run_viz(input: RunVizInput) -> str:
    return "viz ran"

    # try:
    #     client = net.get_ssh_client(rpi1_node.ip, rpi1_node.user)
    #     script_content = (
    #         "#!/bin/bash\n"
    #         "echo whoami: > ~/chromium-viz.log\n"
    #         "whoami >> ~/chromium-viz.log\n"
    #         "echo env: >> ~/chromium-viz.log\n"
    #         "env >> ~/chromium-viz.log\n"
    #         "echo killing existing chromium... >> ~/chromium-viz.log\n"
    #         "pkill -f chromium-browser >> ~/chromium-viz.log 2>&1\n"
    #         "echo killing existing python3... >> ~/chromium-viz.log\n"
    #         "pkill -f python3 >> ~/chromium-viz.log 2>&1\n"
    #         "echo exporting display... >> ~/chromium-viz.log\n"
    #         "export DISPLAY=:0\n"
    #         "export XAUTHORITY=/home/rpi1/.Xauthority\n"
    #         "echo launching viz process... >> ~/chromium-viz.log\n"
    #         "cd ~/tatbot/src\n"
    #         "source .venv/bin/activate\n"
    #         f"setsid uv run {viz_command} >> ~/chromium-viz.log 2>&1 &\n"
    #         "echo waiting for viser server on port 8080... >> ~/chromium-viz.log\n"
    #         "for i in {1..20}; do\n"
    #         "    if nc -z localhost 8080; then\n"
    #         "        echo viser server is up! >> ~/chromium-viz.log\n"
    #         "        break\n"
    #         "    fi\n"
    #         "    sleep 1\n"
    #         "done\n"
    #         "echo launching chromium... >> ~/chromium-viz.log\n"
    #         "setsid chromium-browser --kiosk http://localhost:8080 --disable-gpu >> ~/chromium-viz.log 2>&1 &\n"
    #     )
    #     # Write the script to a file on the remote machine
    #     sftp = client.open_sftp()
    #     with sftp.file('/home/rpi1/mcp_chromium_test.sh', 'w') as f:
    #         f.write(script_content)
    #     sftp.chmod('/home/rpi1/mcp_chromium_test.sh', 0o755)
    #     sftp.close()
    #     # Run the script with an interactive shell
    #     command = "bash -i ~/mcp_chromium_test.sh"
    #     exit_code, out, err = net._run_remote_command(client, command, timeout=80)
    #     client.close()
    #     if exit_code == 0:
    #         return f"✅ rpi1: Viz script executed (waits for server, launches Chromium after, kills python3).\n{out}"
    #     else:
    #         return f"❌ rpi1: Viz script failed.\n{err}"
    # except Exception as e:
    #     log.error(f"Failed to run viz script on rpi1: {e}")
    #     return f"❌ rpi1: Exception occurred: {str(e)}"


if __name__ == "__main__":
    args = setup_log_with_config(MCPConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    mcp.run(transport=args.transport)

#!/bin/bash
echo "Starting base MCP server on $HOSTNAME..."
source ~/tatbot/scripts/env.sh
uv pip install .[bot,dev,gen]
uv run -m tatbot.mcp.base
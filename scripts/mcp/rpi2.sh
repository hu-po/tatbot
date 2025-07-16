#!/bin/bash
echo "Closing existing MCP server on rpi2 🍇..."
pkill -f "tatbot.mcp.rpi2" || true
rm -f ~/tatbot/nfs/mcp-logs/rpi2.log
echo "Starting MCP server on rpi2 🍇..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
nohup uv run -m tatbot.mcp.rpi2 > ~/tatbot/nfs/mcp-logs/rpi2.log 2>&1 &
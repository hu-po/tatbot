#!/bin/bash
echo "Closing existing MCP server on rpi2 🍇..."
pkill -9 -f "python.*" || true # assumes all python processes are tatbot related
rm -f ~/tatbot/nfs/mcp-logs/rpi2.log
echo "Starting MCP server on rpi2 🍇..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
nohup uv run -m tatbot.mcp.rpi2 > ~/tatbot/nfs/mcp-logs/rpi2.log 2>&1 &
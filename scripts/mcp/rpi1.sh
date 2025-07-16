#!/bin/bash
echo "Closing existing MCP server on rpi1 🍓..."
pkill -f "tatbot.mcp.rpi1" || true
rm -f ~/tatbot/nfs/logs/rpi1.log
echo "Starting MCP server on rpi1 🍓..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/env.sh
nohup uv run -m tatbot.mcp.rpi1 > ~/tatbot/nfs/logs/rpi1.log 2>&1 &
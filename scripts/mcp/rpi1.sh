#!/bin/bash
echo "Closing existing MCP server on rpi1 🍓..."
pkill -f "python.*" || true # assumes all python processes are tatbot related
rm -f ~/tatbot/nfs/mcp-logs/rpi1.log
echo "Starting MCP server on rpi1 🍓..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[viz]
nohup uv run -m tatbot.mcp.rpi1 > ~/tatbot/nfs/mcp-logs/rpi1.log 2>&1 &
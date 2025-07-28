#!/bin/bash
echo "Closing existing MCP server on rpi1 🍓..."
pkill -9 -f tatbot || true
rm -f ~/tatbot/nfs/mcp-logs/rpi1.log
echo "Starting MCP server on rpi1 🍓..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[viz,img]
nohup uv run -m tatbot.mcp.rpi1 > ~/tatbot/nfs/mcp-logs/rpi1.log 2>&1 &
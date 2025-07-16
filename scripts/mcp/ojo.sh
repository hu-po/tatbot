#!/bin/bash
echo "Closing existing MCP server on ojo 🦎..."
pkill -f "tatbot.mcp.ojo" || true
rm -f ~/tatbot/nfs/logs/ojo.log
echo "Starting MCP server on ojo 🦎..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[map]
nohup uv run -m tatbot.mcp.ojo > ~/tatbot/nfs/logs/ojo.log 2>&1 &
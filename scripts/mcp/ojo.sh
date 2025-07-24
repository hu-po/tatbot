#!/bin/bash
echo "Closing existing MCP server on ojo 🦎..."
pkill -f "python.*" || true # assumes all python processes are tatbot related
rm -f ~/tatbot/nfs/mcp-logs/ojo.log
echo "Starting MCP server on ojo 🦎..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[bot,gpu]
nohup uv run -m tatbot.mcp.ojo > ~/tatbot/nfs/mcp-logs/ojo.log 2>&1 &
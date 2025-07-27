#!/bin/bash
echo "Closing existing MCP server on trossen-ai 🦾..."
pkill -9 -f "python.*" || true # assumes all python processes are tatbot related
rm -f ~/tatbot/nfs/mcp-logs/trossen-ai.log
echo "Starting MCP server on trossen-ai 🦾..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[bot,cam,gen,img]
nohup uv run -m tatbot.mcp.trossen-ai > ~/tatbot/nfs/mcp-logs/trossen-ai.log 2>&1 &
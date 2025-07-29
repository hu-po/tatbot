#!/bin/bash
echo "Closing existing MCP server on oop 🦊 ..."
source ~/tatbot/scripts/kill.sh
rm -f ~/tatbot/nfs/mcp-logs/oop.log
echo "Starting MCP server on oop 🦊 ..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[bot,dev,gen,gpu,img,viz]
nohup uv run -m tatbot.mcp.oop > ~/tatbot/nfs/mcp-logs/oop.log 2>&1 &
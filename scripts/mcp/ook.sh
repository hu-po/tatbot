#!/bin/bash
echo "Closing existing MCP server on ook 🦧 ..."
bash ~/tatbot/scripts/kill.sh
rm -f ~/tatbot/nfs/mcp-logs/ook.log
echo "Starting MCP server on ook 🦧 ..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh
uv pip install .[bot,dev,gen,img,viz]
uv pip install .[gpu]
nohup uv run -m tatbot.mcp.ook > ~/tatbot/nfs/mcp-logs/ook.log 2>&1 &
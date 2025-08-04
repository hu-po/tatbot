#!/bin/bash
echo "☠️  Killing all tatbot MCP server processes"
# Only kill actual MCP server processes, not the launcher scripts
pkill -9 -f "tatbot.mcp.server" || true
echo "✅ Done"
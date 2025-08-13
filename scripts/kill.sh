#!/bin/bash
echo "☠️  Killing all tatbot MCP related processes"
pkill -9 -f "tatbot.mcp" || true
echo "✅ Done"
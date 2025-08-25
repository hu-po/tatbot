#!/bin/bash
echo "☠️  Killing all tatbot MCP related processes"
pkill -9 -f "tatbot.mcp" || true
echo "☠️  Killing all python processes"
pkill -9 -f "python" || true
echo "✅ Done"
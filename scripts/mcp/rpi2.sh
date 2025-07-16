#!/bin/bash
echo "Starting MCP server on rpi2 🍇..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/env.sh
uv run -m tatbot.mcp.rpi2
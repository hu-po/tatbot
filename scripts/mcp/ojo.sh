#!/bin/bash
echo "Starting MCP server on ojo 🦎..."
source ~/tatbot/scripts/env.sh
uv pip install .[map]
uv run -m tatbot.mcp.ojo
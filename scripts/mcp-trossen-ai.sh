#!/bin/bash
echo "Starting MCP server on trossen-ai 🦾..."
source ~/tatbot/scripts/env.sh
uv pip install .[bot,cam,viz]
uv run -m tatbot.mcp.trossen-ai
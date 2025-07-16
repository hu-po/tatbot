#!/bin/bash
echo "Starting MCP server on trossen-ai 🦾..."
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/env.sh
uv pip install .[bot,cam,viz]
uv run -m tatbot.mcp.trossen-ai
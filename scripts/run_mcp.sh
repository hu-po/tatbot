#!/usr/bin/env bash

# Unified MCP server launcher with Hydra configuration
# Usage: ./run_mcp.sh <node_name> [additional_hydra_args...]
# Example: ./run_mcp.sh ook
# Example: ./run_mcp.sh ook mcp.debug=true mcp.port=9000

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <node_name> [additional_hydra_args...]"
    echo "Available nodes: ook, oop, rpi1, rpi2, ojo, trossen-ai"
    exit 1
fi

NODE=$1
shift

echo "🔌 Starting MCP server for node: $NODE"

# Set up environment
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup-env.sh

# Change to tatbot directory
cd ~/tatbot

# Kill any existing MCP server processes
echo "🛑 Closing existing MCP servers..."
bash ~/tatbot/scripts/kill.sh

# Create log directory if it doesn't exist
mkdir -p ~/tatbot/nfs/mcp-logs
rm -f ~/tatbot/nfs/mcp-logs/${NODE}.log

# Get extras from Hydra config using a temporary Python script
echo "📦 Determining required extras for node $NODE..."
EXTRAS=$(python3 - <<EOF
import yaml
import os
from pathlib import Path

config_path = Path("src/conf/mcp/${NODE}.yaml")
if not config_path.exists():
    print("Error: Configuration file for node '$NODE' not found")
    exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

# Handle defaults inheritance
if 'defaults' in config:
    # Load default config first
    with open('src/conf/mcp/default.yaml') as f:
        default_config = yaml.safe_load(f)
    # Merge configs (node config overrides defaults)
    extras = default_config.get('extras', [])
    extras.extend(config.get('extras', []))
    # Remove duplicates while preserving order
    seen = set()
    final_extras = []
    for extra in extras:
        if extra not in seen:
            seen.add(extra)
            final_extras.append(extra)
    print(','.join(final_extras))
else:
    print(','.join(config.get('extras', [])))
EOF
)

if [ $? -ne 0 ]; then
    echo "❌ Failed to determine extras for node $NODE"
    exit 1
fi

# Install required extras
# Trim whitespace and check if EXTRAS is non-empty
EXTRAS_TRIMMED=$(echo "$EXTRAS" | tr -d '[:space:]')
if [ -n "$EXTRAS_TRIMMED" ] && [ "$EXTRAS_TRIMMED" != "" ]; then
    echo "📦 Installing extras: [$EXTRAS]"
    uv pip install ".[$EXTRAS]"
else
    echo "📦 No extras required for node $NODE"
    uv pip install "."
fi

# Start the MCP server with Hydra
echo "🚀 Starting MCP server for $NODE..."
echo "📝 Logs will be written to ~/tatbot/nfs/mcp-logs/${NODE}.log"

nohup uv run python3 -m tatbot.mcp.server node=${NODE} "$@" \
    > ~/tatbot/nfs/mcp-logs/${NODE}.log 2>&1 &

SERVER_PID=$!
echo "✅ MCP server started with PID: $SERVER_PID"
echo "📊 To monitor logs: tail -f ~/tatbot/nfs/mcp-logs/${NODE}.log"
echo "🛑 To stop server: kill $SERVER_PID"
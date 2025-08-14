#!/usr/bin/env bash

# Unified MCP server launcher with Hydra configuration
# Usage: ./run_mcp.sh [<node_name>] [additional_hydra_args...]
# - If <node_name> is omitted, the current username ($USER) will be used if a matching config exists
# Example: ./run_mcp.sh ook
# Example: ./run_mcp.sh ook mcp.debug=true mcp.port=9000

set -euo pipefail

if [ $# -lt 1 ]; then
    # Default to username if a matching node config exists
    DEFAULT_NODE=${USER:-}
    if [ -n "$DEFAULT_NODE" ] && [ -f "$HOME/tatbot/src/conf/mcp/${DEFAULT_NODE}.yaml" ]; then
        NODE="$DEFAULT_NODE"
        echo "ℹ️  No node provided. Assuming node from username: $NODE"
    else
        echo "Usage: $0 <node_name> [additional_hydra_args...]"
        echo "Available nodes: ook, oop, rpi1, rpi2, ojo, eek, hog"
        echo "You can also omit <node_name> if your username matches a node config (e.g., '$USER')."
        exit 1
    fi
else
    NODE=$1
    shift
fi

# Kill any existing MCP server processes
bash ~/tatbot/scripts/kill.sh

echo "🔌 Starting MCP server for node: $NODE"

# Set up environment
export PATH="$HOME/.local/bin:$PATH"
source ~/tatbot/scripts/setup_env.sh

# Change to tatbot directory
cd ~/tatbot

# Create log directory if it doesn't exist
mkdir -p /nfs/tatbot/mcp-logs
rm -f /nfs/tatbot/mcp-logs/${NODE}.log

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
echo "📝 Logs will be written to /nfs/tatbot/mcp-logs/${NODE}.log"

# Convert --mcp.xxx=yyy arguments to mcp.xxx=yyy for Hydra
HYDRA_ARGS=()
for arg in "$@"; do
    if [[ $arg == --mcp.* ]]; then
        # Remove the leading -- for Hydra compatibility
        HYDRA_ARGS+=("${arg#--}")
    else
        HYDRA_ARGS+=("$arg")
    fi
done

nohup uv run python3 -m tatbot.mcp.server node=${NODE} "${HYDRA_ARGS[@]}" \
    > /nfs/tatbot/mcp-logs/${NODE}.log 2>&1 &

SERVER_PID=$!
echo "✅ MCP server started with PID: $SERVER_PID"
echo "📊 To monitor logs: tail -f /nfs/tatbot/mcp-logs/${NODE}.log"
echo "🛑 To stop server: kill $SERVER_PID"
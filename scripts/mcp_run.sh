#!/usr/bin/env bash

# Unified MCP server launcher with Hydra configuration
# Usage: ./mcp_run.sh [<node_name>] [additional_hydra_args...]
# - If <node_name> is omitted, the current username ($USER) will be used if a matching config exists
# Example: ./mcp_run.sh ook
# Example: ./mcp_run.sh ook mcp.debug=true mcp.port=9000

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

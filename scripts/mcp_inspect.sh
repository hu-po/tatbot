#!/usr/bin/env bash

# Wrapper for MCP Inspector with optional node argument
# Usage: ./mcp_inspect.sh [<node_name>] [inspector_args...]
# - If <node_name> is omitted, the current username ($USER) will be used if a matching config exists
# - Additional args are passed through to the inspector

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/.cursor/mcp.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: MCP inspector config not found at $CONFIG_FILE"
    exit 1
fi

NODE=""

if [ $# -ge 1 ] && [[ ! "$1" =~ ^-- ]]; then
    NODE="$1"
    shift
else
    DEFAULT_NODE=${USER:-}
    if [ -n "$DEFAULT_NODE" ] && [ -f "$REPO_ROOT/src/conf/mcp/${DEFAULT_NODE}.yaml" ]; then
        NODE="$DEFAULT_NODE"
        echo "ℹ️  No node provided. Assuming node from username: $NODE"
    else
        echo "Usage: $0 <node_name> [inspector_args...]"
        echo "You can also omit <node_name> if your username matches a node config (e.g., '$USER')."
        exit 1
    fi
fi

echo "🔍 Launching MCP Inspector for node: $NODE"

cd "$REPO_ROOT"

# Use --yes for non-interactive npx behavior
npx --yes @modelcontextprotocol/inspector --config "$CONFIG_FILE" --server "$NODE" "$@"



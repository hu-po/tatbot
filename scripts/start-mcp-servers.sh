#!/bin/bash
source ~/tatbot/scripts/setup-env.sh
NODES=("ojo" "rpi1" "rpi2" "trossen-ai")

for NODE in "${NODES[@]}"; do
  echo "Starting MCP server on $NODE ..."
  ssh "$NODE" "bash ~/tatbot/scripts/mcp/${NODE}.sh"
done

if [[ "$(hostname)" != "ook" ]]; then
  echo "This script should only be run on the ook node."
  exit 1
fi

bash ~/tatbot/scripts/mcp/ook.sh
#!/bin/bash
source ~/tatbot/scripts/setup-env.sh
NODES=("ojo" "rpi1" "rpi2" "trossen-ai")

# remote nodes
for NODE in "${NODES[@]}"; do
  echo "Starting MCP server on $NODE ..."
  ssh "$NODE" "bash ~/tatbot/scripts/mcp/${NODE}.sh"
done

# local nodes
if [[ "$(hostname)" == "ook" ]]; then
  bash ~/tatbot/scripts/mcp/ook.sh
fi

if [[ "$(hostname)" == "oop" ]]; then
  bash ~/tatbot/scripts/mcp/oop.sh
fi
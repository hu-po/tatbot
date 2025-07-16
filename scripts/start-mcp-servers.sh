#!/bin/bash
source ~/tatbot/scripts/setup-env.sh
NODES=("ojo" "rpi1" "rpi2" "trossen-ai")

for NODE in "${NODES[@]}"; do
  echo "Starting MCP server on $NODE ..."
  ssh "$NODE" "bash ~/tatbot/scripts/mcp/${NODE}.sh"
done

echo "Starting base MCP server $HOSTNAME..."
uv pip install .[bot,dev,gen]
uv run -m tatbot.mcp.base --debug
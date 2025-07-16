#!/bin/bash
source ~/tatbot/scripts/env.sh
NODES=("ojo" "rpi1" "rpi2" "trossen-ai")

for NODE in "${NODES[@]}"; do
  echo "Starting MCP server on $NODE"
  ssh -f "$NODE" '
    pkill -f '"$NODE"'.sh || true
    bash ~/tatbot/scripts/mcp/'"$NODE"'.sh
  '
done

echo "Starting base MCP server $HOSTNAME..."
uv pip install .[bot,dev,gen]
uv run -m tatbot.mcp.base --debug
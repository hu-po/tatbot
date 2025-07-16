#!/bin/bash
source ~/tatbot/scripts/env.sh
NODES=("ojo" "rpi1" "rpi2" "trossen-ai")

for NODE in "${NODES[@]}"; do
  echo "Starting MCP server on $NODE"
  ssh -f "$NODE" '
    pkill -f mcp-'"$NODE"'.sh || true
    rm -f ~/tatbot/nfs/logs/mcp-'"$NODE"'.txt
    nohup bash ~/tatbot/scripts/mcp/'"$NODE"'.sh > ~/tatbot/nfs/logs/mcp-'"$NODE"'.txt 2>&1 &
  '
done

echo "Starting base MCP server $HOSTNAME..."
uv pip install .[bot,dev,gen]
uv run -m tatbot.mcp.base --debug
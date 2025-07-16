#!/bin/bash
source ~/tatbot/scripts/env.sh
NODES=("ojo" "rpi1" "rpi2" "trossen-ai")

for NODE in "${NODES[@]}"; do
  echo "Starting MCP server on $NODE ..."
  # Start the remote MCP script in an interactive SSH session that remains
  # attached to this terminal.  Prefix each line of its output with the node
  # name so that interleaved logs are still readable.  Run the SSH command in
  # the background so we launch all nodes concurrently.
  ssh "$NODE" "pkill -f ${NODE}.sh || true; bash ~/tatbot/scripts/mcp/${NODE}.sh" \
    | sed -u "s/^/[${NODE}] /" &
done

echo "Starting base MCP server $HOSTNAME..."
uv pip install .[bot,dev,gen]
uv run -m tatbot.mcp.base --debug
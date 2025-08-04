# MCP (Model Context Protocol)

**MCP has been refactored for unified, dynamic, and type-safe server management.** Individual node scripts are replaced by a single, generic server powered by Hydra and Pydantic.

## Key Changes
- **Unified Server**: A single `tatbot.mcp.server` now runs on all nodes.
- **Hydra Configuration**: Node-specific behavior (host, port, tools) is defined in YAML files under `conf/mcp/`.
- **Dynamic Tools**: Tools are now dynamically registered handlers, enabled/disabled via config.
- **Pydantic Models**: All requests and responses are strongly typed with Pydantic for validation and clarity.

## Starting Servers
To start an MCP server on a specific node, use the unified launcher script:
```bash
# Start the server on the 'ook' node
./scripts/run_mcp.sh ook

# Start on 'trossen-ai' with debug mode enabled
./scripts/run_mcp.sh trossen-ai mcp.debug=true
```
The old `start-mcp-servers.sh` is now **deprecated**. You must start each server individually on its target machine.

## Server Logs
Logs are now written to `~/tatbot/nfs/mcp-logs/<node_name>.log`. For example, the `ook` server's log is at `~/tatbot/nfs/mcp-logs/ook.log`.

## Available Tools
The available tools are defined as handlers in `tatbot/mcp/handlers.py` and enabled per-node in the `conf/mcp/` YAML files.

- `run_op`: Executes a robot operation.
- `ping_nodes`: Pings network nodes to check connectivity.
- `list_scenes`: Lists available scenes from the config.
- `list_nodes`: Lists all configured network nodes.

## Cursor IDE Integration
The workflow in Cursor remains similar:
- Use `ctrl-shift-p` > "View: OpenMCP Settings" to toggle servers.
- If you change the server config (e.g., enable a new tool), you must restart the server and re-toggle it in Cursor.
- If you encounter "Bad Request" errors, a restart of the server and a re-toggle in Cursor usually resolves it.

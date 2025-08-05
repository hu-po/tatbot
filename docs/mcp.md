# MCP (Model Context Protocol)

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

## Server Logs
Logs are written to `~/tatbot/nfs/mcp-logs/<node_name>.log`. For example, the `ook` server's log is at `~/tatbot/nfs/mcp-logs/ook.log`.

## Node Configuration
The behavior of each MCP server is defined by a corresponding YAML file in `src/conf/mcp/`. For example, the `ook` node is configured by `src/conf/mcp/ook.yaml`.

These files control:
- **`host` and `port`**: Network settings for the server.
- **`tools`**: A list of which tools (from `tatbot.mcp.handlers`) are enabled on this node.
- **`extras`**: A list of optional dependency groups (from `pyproject.toml`) that should be installed on this node. The `run_mcp.sh` script automatically reads this list and uses `uv pip install` to ensure the correct dependencies are present before starting the server.

## Available Tools
The available tools are defined as handlers in `tatbot.mcp.handlers` and enabled per-node in the `conf/mcp/` YAML files.

- `run_op`: Executes a robot operation.
- `ping_nodes`: Pings network nodes to check connectivity.
- `list_scenes`: Lists available scenes from the config.
- `list_nodes`: Lists all configured network nodes.
- `list_ops`: Lists available operations, which can vary by node.

## Cursor IDE Integration
The workflow in Cursor remains similar:
- Use `ctrl-shift-p` > "View: OpenMCP Settings" to toggle servers.
- If you change the server config (e.g., enable a new tool), you must restart the server and re-toggle it in Cursor.
- If you encounter "Bad Request" errors, a restart of the server and a re-toggle in Cursor usually resolves it.

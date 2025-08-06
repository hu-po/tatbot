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
- `convert_strokelist_to_batch`: GPU-accelerated stroke trajectory conversion (GPU nodes only).

## Cross-Node GPU Processing

The MCP system enables transparent cross-node GPU acceleration for stroke trajectory conversion. This allows robot operations on non-GPU nodes (like `trossen-ai`) to automatically leverage GPU-accelerated inverse kinematics solving on GPU-enabled nodes (like `ook`).

### Architecture

**GPU Detection & Routing**
- Robot operations (`align`, `stroke`) automatically detect local GPU availability via `check_local_gpu()`
- When no local GPU is available, operations route conversion requests to remote GPU nodes
- `GPUProxy` handles node discovery, load balancing, and communication with GPU-enabled nodes

**NFS Path Translation**
- All nodes share NFS storage but mount at different local paths:
  - `ook`: `/home/ook/tatbot/nfs/`
  - `trossen-ai`: `/home/trossen-ai/tatbot/nfs/`
- `GPUProxy._translate_path_for_node()` converts paths between node-specific mount points
- Files remain on shared NFS throughout the entire process - no data transfer required

**MCP Protocol Communication**
- Cross-node requests use JSON-RPC 2.0 over StreamableHTTP transport
- Proper session establishment with `initialize` → `notifications/initialized` handshake
- Tool namespacing: `{node_name}_{tool_name}` (e.g., `ook_convert_strokelist_to_batch`)
- Retry logic with exponential backoff for fault tolerance

### Workflow Example

1. **Operation Start**: User runs `align` operation on `trossen-ai`
2. **GPU Detection**: `check_local_gpu()` returns `False` on `trossen-ai`
3. **File Creation**: Strokes saved to `/home/trossen-ai/tatbot/nfs/recordings/align-*/strokes.yaml`
4. **Path Translation**: GPUProxy translates to `/home/ook/tatbot/nfs/recordings/align-*/strokes.yaml`
5. **Remote Conversion**: MCP call to `ook_convert_strokelist_to_batch` with translated paths
6. **GPU Processing**: `ook` performs JAX-accelerated inverse kinematics solving
7. **Result Storage**: Strokebatch saved to shared NFS at translated output path
8. **Operation Continue**: `trossen-ai` loads strokebatch and continues robot operation

### Configuration

**GPU Node Setup**
```yaml
# conf/mcp/ook.yaml
host: "0.0.0.0"
port: 8000
extras: ["gpu"]  # Enables GPU dependencies
tools:
  - run_op
  - convert_strokelist_to_batch  # GPU conversion tool
```

**Non-GPU Node Setup**
```yaml  
# conf/mcp/trossen-ai.yaml
host: "0.0.0.0"
port: 8000
extras: []  # No GPU dependencies
tools:
  - run_op  # Will use GPUProxy for remote conversion
```

### Error Handling

- **Connection Failures**: Automatic retry with exponential backoff
- **GPU Node Unavailable**: Falls back to CPU conversion (if enabled)
- **NFS Sync Issues**: Built-in timeout and file existence verification
- **Path Translation**: Robust handling of different mount point formats

### Benefits

- **Transparent**: Robot operations work identically regardless of node GPU capability
- **Efficient**: No data transfer - files remain on shared NFS throughout
- **Scalable**: Easy to add new GPU nodes to the processing pool
- **Fault Tolerant**: Multiple retry mechanisms and fallback options
- **Zero Configuration**: Automatic detection and routing based on node capabilities

## Cursor IDE Integration
The workflow in Cursor remains similar:
- Use `ctrl-shift-p` > "View: OpenMCP Settings" to toggle servers.
- If you change the server config (e.g., enable a new tool), you must restart the server and re-toggle it in Cursor.
- If you encounter "Bad Request" errors, a restart of the server and a re-toggle in Cursor usually resolves it.
